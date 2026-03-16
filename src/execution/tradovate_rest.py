"""Tradovate REST API client for orders and position queries.

Used as the primary order placement interface and as a fallback
when the WebSocket is unavailable. All order calls include
`isAutomated: true` per CME Rule 536-B.

Key endpoints:
  - order/placeorder     → market/limit orders
  - order/placeoso       → bracket orders (entry + stop)
  - order/modifyorder    → move stops, change limits
  - order/cancelorder    → cancel working orders
  - order/liquidateposition → emergency flatten
  - position/list        → current positions (reconciliation)
  - order/list           → working orders
"""

from __future__ import annotations

from typing import Any

import aiohttp
import structlog

from src.core.exceptions import (
    InsufficientMarginError,
    OrderModifyFailedError,
    OrderRejectedError,
    TradovateConnectionError,
)
from src.execution.rate_limiter import RateLimiter
from src.execution.tradovate_auth import TradovateAuth

logger = structlog.get_logger()


class TradovateREST:
    """Async REST client for Tradovate order and position management.

    All methods require a valid TradovateAuth instance that has already
    authenticated. Rate limiting is enforced via the shared RateLimiter.

    Usage:
        rest = TradovateREST(auth, rate_limiter)
        order = await rest.place_market_order("MNQM6", "Buy", 3)
        positions = await rest.get_positions()
    """

    def __init__(
        self,
        auth: TradovateAuth,
        rate_limiter: RateLimiter,
    ) -> None:
        self._auth = auth
        self._limiter = rate_limiter
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._auth.access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _post(
        self,
        endpoint: str,
        payload: dict,
        emergency: bool = False,
    ) -> dict:
        """POST to a Tradovate REST endpoint with rate limiting.

        Raises TradovateConnectionError on non-200 status codes.
        """
        await self._limiter.acquire(emergency=emergency)

        url = f"{self._auth.base_url}/{endpoint}"
        session = self._get_session()

        logger.debug("tradovate_rest.post", endpoint=endpoint, payload=payload)

        try:
            async with session.post(url, json=payload, headers=self._headers()) as resp:
                data = await resp.json()

                # Check for penalty ticket
                if isinstance(data, dict) and data.get("p-ticket"):
                    logger.critical(
                        "tradovate_rest.penalty_ticket",
                        ticket=data["p-ticket"],
                        endpoint=endpoint,
                    )

                if resp.status != 200:
                    logger.error(
                        "tradovate_rest.error",
                        status=resp.status,
                        endpoint=endpoint,
                        response=data,
                    )
                    # Don't silently swallow — raise so callers know
                    error_msg = ""
                    if isinstance(data, dict):
                        error_msg = data.get("errorText", "")
                    raise TradovateConnectionError(
                        f"REST {endpoint} returned {resp.status}: {error_msg or data}"
                    )

                return data

        except aiohttp.ClientError as e:
            raise TradovateConnectionError(
                f"REST request to {endpoint} failed: {e}"
            ) from e

    async def _get(
        self,
        endpoint: str,
        params: dict | None = None,
        emergency: bool = False,
    ) -> Any:
        """GET from a Tradovate REST endpoint with rate limiting."""
        await self._limiter.acquire(emergency=emergency)

        url = f"{self._auth.base_url}/{endpoint}"
        session = self._get_session()

        try:
            async with session.get(url, headers=self._headers(), params=params) as resp:
                return await resp.json()
        except aiohttp.ClientError as e:
            raise TradovateConnectionError(
                f"REST GET {endpoint} failed: {e}"
            ) from e

    # ── Order Placement ──────────────────────────────────────────────────────

    async def place_market_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
    ) -> dict:
        """Place a market order.

        Args:
            symbol: Contract symbol (e.g., "MNQM6")
            action: "Buy" or "Sell"
            quantity: Number of contracts

        Returns:
            Order response dict with orderId, ordStatus, etc.

        Raises:
            OrderRejectedError: If the order is rejected.
        """
        payload = {
            "accountSpec": self._auth.account_spec,
            "accountId": self._auth.account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": "Market",
            "isAutomated": True,
        }

        result = await self._post("order/placeorder", payload)
        self._check_order_rejection(result, "market order")
        return result

    async def place_bracket_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_price: float,
        take_profit_price: float | None = None,
    ) -> dict:
        """Place a bracket order (entry + stop, optionally + take profit).

        Our system uses entry + stop only (LLM manages exits dynamically),
        but take_profit_price is available if needed.

        Args:
            symbol: Contract symbol
            action: "Buy" or "Sell" for the entry
            quantity: Number of contracts
            stop_price: Stop loss price
            take_profit_price: Optional take profit price

        Returns:
            OSO order response dict.
        """
        exit_action = "Sell" if action == "Buy" else "Buy"

        payload: dict = {
            "accountSpec": self._auth.account_spec,
            "accountId": self._auth.account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": "Market",
            "isAutomated": True,
            "bracket1": {
                "action": exit_action,
                "orderType": "Stop",
                "stopPrice": stop_price,
                "isAutomated": True,
            },
        }

        if take_profit_price is not None:
            payload["bracket2"] = {
                "action": exit_action,
                "orderType": "Limit",
                "price": take_profit_price,
                "isAutomated": True,
            }

        result = await self._post("order/placeoso", payload)
        self._check_order_rejection(result, "bracket order")
        return result

    async def place_stop_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        stop_price: float,
    ) -> dict:
        """Place a standalone stop order (fallback when bracket leg fails).

        Args:
            symbol: Contract symbol
            action: "Buy" (to close short) or "Sell" (to close long)
            quantity: Number of contracts
            stop_price: Trigger price
        """
        payload = {
            "accountSpec": self._auth.account_spec,
            "accountId": self._auth.account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": quantity,
            "orderType": "Stop",
            "stopPrice": stop_price,
            "timeInForce": "GTC",
            "isAutomated": True,
        }

        result = await self._post("order/placeorder", payload)
        self._check_order_rejection(result, "stop order")
        return result

    # ── Order Modification ───────────────────────────────────────────────────

    async def modify_order(
        self,
        order_id: int,
        quantity: int,
        order_type: str = "Stop",
        stop_price: float | None = None,
        limit_price: float | None = None,
    ) -> dict:
        """Modify an existing working order.

        IMPORTANT: Tradovate's modifyOrder can return 200 but silently fail.
        Callers should verify the order state after modification.

        Args:
            order_id: The order to modify
            quantity: Order quantity (required even if unchanged)
            order_type: "Stop", "Limit", "StopLimit", "Market"
            stop_price: New stop price (for Stop/StopLimit orders)
            limit_price: New limit price (for Limit/StopLimit orders)

        Returns:
            Modified order response.

        Raises:
            OrderModifyFailedError: If modification fails.
        """
        payload: dict = {
            "orderId": order_id,
            "orderQty": quantity,
            "orderType": order_type,
            "timeInForce": "GTC",  # Always include to avoid silent failure
            "isAutomated": True,
        }

        if stop_price is not None:
            payload["stopPrice"] = stop_price
        if limit_price is not None:
            payload["price"] = limit_price

        result = await self._post("order/modifyorder", payload)

        # Check for modification failure
        if isinstance(result, dict):
            if result.get("errorText"):
                raise OrderModifyFailedError(
                    f"Order {order_id} modify failed: {result['errorText']}"
                )

        return result

    async def modify_order_with_verify(
        self,
        order_id: int,
        quantity: int,
        order_type: str = "Stop",
        stop_price: float | None = None,
        limit_price: float | None = None,
    ) -> dict:
        """Modify an order and verify it actually changed.

        Handles Tradovate's silent failure bug by checking the order
        state after modification and retrying once if needed.
        """
        result = await self.modify_order(
            order_id, quantity, order_type, stop_price, limit_price
        )

        # Verify the modification took effect
        order = await self.get_order(order_id)

        if stop_price is not None and order.get("stopPrice") != stop_price:
            logger.warning(
                "tradovate_rest.modify_silent_failure",
                order_id=order_id,
                expected_stop=stop_price,
                actual_stop=order.get("stopPrice"),
            )
            # Retry once
            result = await self.modify_order(
                order_id, quantity, order_type, stop_price, limit_price
            )
            # Check again
            order = await self.get_order(order_id)
            if stop_price is not None and order.get("stopPrice") != stop_price:
                raise OrderModifyFailedError(
                    f"Order {order_id} modify failed after retry. "
                    f"Expected stop={stop_price}, got={order.get('stopPrice')}"
                )

        return result

    # ── Order Cancellation ───────────────────────────────────────────────────

    async def cancel_order(self, order_id: int) -> dict:
        """Cancel a working order."""
        payload = {
            "orderId": order_id,
            "isAutomated": True,
        }
        return await self._post("order/cancelorder", payload)

    # ── Position Queries ─────────────────────────────────────────────────────

    async def get_positions(self) -> list[dict]:
        """Get all open positions."""
        result = await self._get("position/list")
        if isinstance(result, list):
            return result
        return []

    async def get_orders(self) -> list[dict]:
        """Get all working orders."""
        result = await self._get("order/list")
        if isinstance(result, list):
            return result
        return []

    async def get_order(self, order_id: int) -> dict:
        """Get a specific order by ID.

        Uses Tradovate's /order/item endpoint with the id query parameter.
        Falls back to /order/find if item returns empty.
        """
        result = await self._get("order/item", params={"id": str(order_id)})
        if isinstance(result, dict) and result:
            return result
        # Fallback — some Tradovate environments use /find
        result = await self._get("order/find", params={"name": str(order_id)})
        if isinstance(result, dict):
            return result
        return {}

    async def get_cash_balance(self) -> dict:
        """Get account cash balance snapshot."""
        return await self._get("cashBalance/getcashbalancesnapshot")

    async def get_account_info(self) -> list[dict]:
        """Get account details."""
        result = await self._get("account/list")
        if isinstance(result, list):
            return result
        return []

    # ── Emergency ────────────────────────────────────────────────────────────

    async def liquidate_position(self, account_id: int | None = None) -> dict:
        """EMERGENCY: Flatten all positions at market.

        Bypasses rate limiter. Used by kill switch.
        """
        acct_id = account_id or self._auth.account_id

        payload = {
            "accountId": acct_id,
            "admin": False,
            "isAutomated": True,
        }

        logger.critical(
            "tradovate_rest.liquidating",
            account_id=acct_id,
        )

        return await self._post("order/liquidateposition", payload, emergency=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _check_order_rejection(self, result: dict, order_desc: str) -> None:
        """Check if an order response indicates rejection."""
        if not isinstance(result, dict):
            return

        error = result.get("errorText", "")
        if error:
            if "margin" in error.lower() or "insufficient" in error.lower():
                raise InsufficientMarginError(f"{order_desc} rejected: {error}")
            raise OrderRejectedError(f"{order_desc} rejected: {error}")

        status = result.get("ordStatus", "")
        if status == "Rejected":
            reason = result.get("rejectReason", "unknown")
            raise OrderRejectedError(f"{order_desc} rejected: {reason}")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("tradovate_rest.closed")
