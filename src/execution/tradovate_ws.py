"""Tradovate WebSocket client for real-time account/order events.

The Tradovate WebSocket uses a custom text-based protocol, NOT pure JSON:
  - Outgoing: "endpoint\\nrequestId\\n\\njsonBody"
  - Incoming: prefix char + payload
    - 'o' = connection opened
    - 'h' = server heartbeat
    - 'a' = data array (JSON after the 'a')

Heartbeat `[]` must be sent every 2.5 seconds or the connection drops.

This client handles:
  - Connection, authorization, heartbeat
  - Request/response matching via request IDs
  - Subscription to account updates (user/syncrequest)
  - Event dispatching to registered callbacks
  - Auto-reconnect with exponential backoff
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Coroutine

import structlog
import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from src.core.exceptions import TradovateConnectionError

logger = structlog.get_logger()

# Type alias for event handlers
EventHandler = Callable[[dict], Coroutine[Any, Any, None]]


class TradovateWS:
    """Async WebSocket client for Tradovate account/order events.

    Usage:
        ws = TradovateWS(ws_url, access_token)
        ws.on_position_update(my_handler)
        ws.on_order_update(my_handler)
        await ws.connect()
        # ... runs in background ...
        await ws.close()
    """

    def __init__(
        self,
        ws_url: str,
        access_token: str,
        heartbeat_interval: float = 2.5,
    ) -> None:
        self._url = ws_url
        self._access_token = access_token
        self._heartbeat_interval = heartbeat_interval
        self._ws: Any = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._running = False
        self._connected = False
        self._heartbeat_task: asyncio.Task | None = None
        self._read_task: asyncio.Task | None = None

        # Event handlers
        self._position_handlers: list[EventHandler] = []
        self._order_handlers: list[EventHandler] = []
        self._fill_handlers: list[EventHandler] = []
        self._cash_balance_handlers: list[EventHandler] = []
        self._generic_handlers: list[EventHandler] = []

        # Reconnection
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._min_reconnect_gap = 5.0
        self._auto_reconnect = True

        # Stats
        self._messages_received = 0
        self._messages_sent = 0
        self._reconnect_count = 0

    # ── Handler Registration ─────────────────────────────────────────────────

    def on_position_update(self, handler: EventHandler) -> None:
        """Register handler for position change events."""
        self._position_handlers.append(handler)

    def on_order_update(self, handler: EventHandler) -> None:
        """Register handler for order status change events."""
        self._order_handlers.append(handler)

    def on_fill(self, handler: EventHandler) -> None:
        """Register handler for fill events."""
        self._fill_handlers.append(handler)

    def on_cash_balance(self, handler: EventHandler) -> None:
        """Register handler for cash balance change events."""
        self._cash_balance_handlers.append(handler)

    def on_event(self, handler: EventHandler) -> None:
        """Register handler for all generic events."""
        self._generic_handlers.append(handler)

    # ── Connection ───────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Connect to Tradovate WebSocket and authorize.

        Starts heartbeat and read loops in the background.
        """
        logger.info("tradovate_ws.connecting", url=self._url)

        try:
            self._ws = await websockets.connect(
                self._url,
                ping_interval=None,  # We handle heartbeat ourselves
                ping_timeout=None,
                close_timeout=5,
            )

            # Wait for open frame
            msg = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            if msg != "o":
                raise TradovateConnectionError(
                    f"Expected 'o' open frame, got: {msg!r}"
                )

            # Authorize
            self._request_id += 1
            auth_msg = f"authorize\n{self._request_id}\n\n{self._access_token}"
            await self._ws.send(auth_msg)

            # Wait for auth response
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            if not response.startswith("a"):
                raise TradovateConnectionError(
                    f"Unexpected auth response: {response!r}"
                )

            data = json.loads(response[1:])
            if not data or data[0].get("s") != 200:
                raise TradovateConnectionError(
                    f"WebSocket auth failed: {data}"
                )

            self._connected = True
            self._running = True
            self._reconnect_delay = 1.0  # Reset backoff on success

            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._read_task = asyncio.create_task(self._read_loop())

            logger.info("tradovate_ws.connected")

            # Subscribe to account updates
            await self.subscribe_account_updates()

        except (websockets.exceptions.WebSocketException, asyncio.TimeoutError) as e:
            raise TradovateConnectionError(
                f"WebSocket connection failed: {e}"
            ) from e

    async def subscribe_account_updates(self) -> dict:
        """Subscribe to real-time account, position, and order events.

        This sends user/syncrequest which pushes all account changes.
        """
        return await self.send_request("user/syncrequest", {})

    # ── Message Sending ──────────────────────────────────────────────────────

    async def send_request(
        self,
        endpoint: str,
        body: dict | None = None,
        timeout: float = 10.0,
    ) -> dict:
        """Send a request and wait for its response.

        Args:
            endpoint: API endpoint (e.g., "order/placeorder")
            body: JSON body
            timeout: Max seconds to wait for response

        Returns:
            Response data dict.
        """
        if not self._connected or not self._ws:
            raise TradovateConnectionError("WebSocket not connected")

        self._request_id += 1
        rid = self._request_id

        body_str = json.dumps(body) if body else ""
        msg = f"{endpoint}\n{rid}\n\n{body_str}"

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = future

        await self._ws.send(msg)
        self._messages_sent += 1

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TradovateConnectionError(
                f"WebSocket request to {endpoint} timed out after {timeout}s"
            )

    async def send_fire_and_forget(self, endpoint: str, body: dict | None = None) -> None:
        """Send a request without waiting for a response."""
        if not self._connected or not self._ws:
            raise TradovateConnectionError("WebSocket not connected")

        self._request_id += 1
        body_str = json.dumps(body) if body else ""
        msg = f"{endpoint}\n{self._request_id}\n\n{body_str}"

        await self._ws.send(msg)
        self._messages_sent += 1

    # ── Internal Loops ───────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send heartbeat every 2.5 seconds to keep connection alive."""
        while self._running and self._ws:
            try:
                await self._ws.send("[]")
                await asyncio.sleep(self._heartbeat_interval)
            except (ConnectionClosed, ConnectionClosedError):
                logger.warning("tradovate_ws.heartbeat_connection_lost")
                break
            except Exception:
                logger.exception("tradovate_ws.heartbeat_error")
                break

    async def _read_loop(self) -> None:
        """Read and dispatch incoming WebSocket messages."""
        try:
            async for raw_msg in self._ws:
                self._messages_received += 1

                if raw_msg == "h":
                    # Server heartbeat — acknowledged, no action needed
                    continue

                if raw_msg == "o":
                    # Duplicate open frame — ignore
                    continue

                if raw_msg.startswith("a"):
                    try:
                        items = json.loads(raw_msg[1:])
                        for item in items:
                            await self._dispatch_message(item)
                    except json.JSONDecodeError:
                        logger.error(
                            "tradovate_ws.json_decode_error",
                            raw=raw_msg[:200],
                        )
                else:
                    logger.debug("tradovate_ws.unknown_frame", raw=raw_msg[:100])

        except ConnectionClosedOK:
            logger.info("tradovate_ws.connection_closed_normally")
        except ConnectionClosedError as e:
            logger.warning("tradovate_ws.connection_closed_error", code=e.code, reason=e.reason)
        except Exception:
            logger.exception("tradovate_ws.read_loop_error")
        finally:
            self._connected = False
            if self._running and self._auto_reconnect:
                asyncio.create_task(self._reconnect())

    async def _dispatch_message(self, item: dict) -> None:
        """Route a parsed message to the appropriate handlers."""
        # Check if this is a response to a pending request
        req_id = item.get("i")
        if req_id and req_id in self._pending:
            self._pending[req_id].set_result(item)
            del self._pending[req_id]
            return

        # Push event — route by event type
        event_type = item.get("e", "")
        event_data = item.get("d", {})

        # Handle generic handlers first
        for handler in self._generic_handlers:
            try:
                await handler(item)
            except Exception:
                logger.exception("tradovate_ws.generic_handler_error")

        # Route specific entity updates
        if event_type == "props":
            await self._handle_props_event(event_data)

    async def _handle_props_event(self, data: dict) -> None:
        """Handle 'props' events (position/order/balance updates)."""
        # Position updates
        if "positions" in data:
            for pos in data["positions"]:
                for handler in self._position_handlers:
                    try:
                        await handler(pos)
                    except Exception:
                        logger.exception("tradovate_ws.position_handler_error")

        # Order updates
        if "orders" in data:
            for order in data["orders"]:
                # Check if this is a fill
                if order.get("ordStatus") == "Filled":
                    for handler in self._fill_handlers:
                        try:
                            await handler(order)
                        except Exception:
                            logger.exception("tradovate_ws.fill_handler_error")

                for handler in self._order_handlers:
                    try:
                        await handler(order)
                    except Exception:
                        logger.exception("tradovate_ws.order_handler_error")

        # Cash balance updates
        if "cashBalances" in data:
            for balance in data["cashBalances"]:
                for handler in self._cash_balance_handlers:
                    try:
                        await handler(balance)
                    except Exception:
                        logger.exception("tradovate_ws.cash_balance_handler_error")

    # ── Reconnection ─────────────────────────────────────────────────────────

    async def _reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        self._reconnect_count += 1

        logger.warning(
            "tradovate_ws.reconnecting",
            delay=self._reconnect_delay,
            attempt=self._reconnect_count,
        )

        await asyncio.sleep(self._reconnect_delay)

        # Exponential backoff
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._max_reconnect_delay,
        )

        try:
            await self.connect()
            logger.info("tradovate_ws.reconnected", attempt=self._reconnect_count)
        except Exception:
            logger.exception("tradovate_ws.reconnect_failed")
            if self._running and self._auto_reconnect:
                asyncio.create_task(self._reconnect())

    def update_token(self, new_token: str) -> None:
        """Update the access token (called after refresh)."""
        self._access_token = new_token

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def stats(self) -> dict:
        return {
            "connected": self._connected,
            "messages_received": self._messages_received,
            "messages_sent": self._messages_sent,
            "reconnect_count": self._reconnect_count,
            "pending_requests": len(self._pending),
        }

    # ── Shutdown ──────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._running = False
        self._auto_reconnect = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()

        # Resolve any pending futures with errors
        for rid, future in self._pending.items():
            if not future.done():
                future.set_exception(
                    TradovateConnectionError("WebSocket closing")
                )
        self._pending.clear()

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        self._connected = False
        logger.info("tradovate_ws.closed")
