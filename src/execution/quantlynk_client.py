"""QuantLynk webhook client — sends trade signals via HTTP POST.

QuantLynk acts as a bridge between our system and Tradovate.
We send webhook requests, QuantLynk executes them on Tradovate.

Payload format (from QuantVue/QuantLynk):
    {
        "qv_user_id": "user-id",
        "alert_id": "alert-id",
        "quantity": "2",
        "order_type": "market",
        "action": "buy",
        "price": "19825.50"
    }

Actions: "buy", "sell", "flatten"
Order types: "market" (always for us)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import aiohttp
import structlog

from src.core.config import QuantLynkConfig
from src.core.exceptions import OrderRejectedError

logger = structlog.get_logger()


class QuantLynkClient:
    """HTTP webhook client for QuantLynk order execution.

    Usage:
        client = QuantLynkClient(config)
        await client.connect()  # creates HTTP session

        await client.buy(quantity=2, price=19825.50)
        await client.sell(quantity=2, price=19810.00)
        await client.flatten(price=19830.00)

        await client.close()
    """

    def __init__(self, config: QuantLynkConfig) -> None:
        self._config = config
        self._session: Optional[aiohttp.ClientSession] = None

        # Stats
        self._requests_sent: int = 0
        self._requests_failed: int = 0
        self._total_latency_ms: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Create HTTP session for webhook calls."""
        if self._session and not self._session.closed:
            return

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._config.timeout_sec),
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "quantlynk.connected",
            webhook_url=self._config.webhook_url[:30] + "...",
        )

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("quantlynk.closed")

    @property
    def is_connected(self) -> bool:
        return self._session is not None and not self._session.closed

    # ── Order Actions ────────────────────────────────────────────────────

    async def buy(self, quantity: int, price: float) -> dict[str, Any]:
        """Send a buy market order."""
        return await self._send_signal(
            action="buy",
            quantity=quantity,
            price=price,
        )

    async def sell(self, quantity: int, price: float) -> dict[str, Any]:
        """Send a sell market order."""
        return await self._send_signal(
            action="sell",
            quantity=quantity,
            price=price,
        )

    async def flatten(self, price: float) -> dict[str, Any]:
        """Flatten all positions (close everything)."""
        return await self._send_signal(
            action="flatten",
            quantity=0,
            price=price,
        )

    async def partial_close(self, side: str, quantity: int, price: float) -> dict[str, Any]:
        """Close a partial position (not flatten — directional close).

        For long positions: sends a sell order to reduce quantity.
        For short positions: sends a buy order to reduce quantity.

        Args:
            side: "long" or "short" — the current position side.
            quantity: Number of contracts to close.
            price: Current market price (for logging/payload).
        """
        action = "sell" if side.lower() == "long" else "buy"
        return await self._send_signal(
            action=action,
            quantity=quantity,
            price=price,
        )

    # ── Core Webhook Sender ──────────────────────────────────────────────

    async def _send_signal(
        self,
        action: str,
        quantity: int,
        price: float,
    ) -> dict[str, Any]:
        """Send a trade signal to QuantLynk webhook.

        Retries on transient HTTP errors (timeouts, 5xx).
        Raises OrderRejectedError on permanent failures (4xx).

        Returns:
            Dict with status and response details.
        """
        if not self._config.enabled:
            logger.info("quantlynk.disabled", action=action)
            return {"status": "disabled", "action": action}

        if not self._session or self._session.closed:
            await self.connect()

        payload = {
            "qv_user_id": self._config.user_id,
            "alert_id": self._config.alert_id,
            "quantity": str(quantity),
            "order_type": "market",
            "action": action,
            "price": str(price),
        }

        last_error: Optional[Exception] = None

        for attempt in range(1 + self._config.max_retries):
            try:
                t0 = asyncio.get_event_loop().time()

                async with self._session.post(  # type: ignore[union-attr]
                    self._config.webhook_url,
                    json=payload,
                ) as resp:
                    elapsed_ms = int(
                        (asyncio.get_event_loop().time() - t0) * 1000
                    )
                    self._total_latency_ms += elapsed_ms
                    self._requests_sent += 1

                    response_text = await resp.text()

                    if resp.status < 300:
                        logger.info(
                            "quantlynk.signal_sent",
                            action=action,
                            quantity=quantity,
                            price=price,
                            status=resp.status,
                            latency_ms=elapsed_ms,
                            response_body=response_text[:500],
                        )
                        return {
                            "status": "sent",
                            "action": action,
                            "quantity": quantity,
                            "price": price,
                            "http_status": resp.status,
                            "response": response_text[:500],
                            "latency_ms": elapsed_ms,
                        }

                    elif resp.status < 500:
                        # Client error — don't retry (bad request, auth issue)
                        self._requests_failed += 1
                        logger.error(
                            "quantlynk.rejected",
                            action=action,
                            status=resp.status,
                            response=response_text[:200],
                        )
                        raise OrderRejectedError(
                            f"QuantLynk rejected order: HTTP {resp.status} — {response_text[:200]}"
                        )

                    else:
                        # Server error — retryable
                        last_error = Exception(
                            f"QuantLynk server error: HTTP {resp.status}"
                        )
                        logger.warning(
                            "quantlynk.server_error",
                            status=resp.status,
                            attempt=attempt + 1,
                        )

            except aiohttp.ClientError as e:
                last_error = e
                logger.warning(
                    "quantlynk.connection_error",
                    error=str(e),
                    attempt=attempt + 1,
                )

            except OrderRejectedError:
                raise  # don't retry client errors

            # Wait before retry (exponential backoff)
            if attempt < self._config.max_retries:
                delay = 0.5 * (2 ** attempt)
                await asyncio.sleep(delay)

        # All retries exhausted
        self._requests_failed += 1
        error_msg = f"QuantLynk failed after {self._config.max_retries + 1} attempts: {last_error}"
        logger.error("quantlynk.all_retries_failed", action=action, error=str(last_error))
        raise OrderRejectedError(error_msg)

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        avg_latency = (
            self._total_latency_ms / self._requests_sent
            if self._requests_sent > 0
            else 0
        )
        return {
            "requests_sent": self._requests_sent,
            "requests_failed": self._requests_failed,
            "avg_latency_ms": round(avg_latency, 1),
        }
