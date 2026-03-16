"""Tests for Tradovate WebSocket client.

Tests the protocol parsing, handler registration, event dispatch,
and state management without real WebSocket connections.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.exceptions import TradovateConnectionError
from src.execution.tradovate_ws import TradovateWS


@pytest.fixture
def ws():
    return TradovateWS(
        ws_url="wss://demo.tradovateapi.com/v1/websocket",
        access_token="test-token-123",
        heartbeat_interval=2.5,
    )


class TestInitialization:
    def test_initial_state(self, ws: TradovateWS):
        assert ws.is_connected is False
        assert ws.stats["messages_received"] == 0
        assert ws.stats["messages_sent"] == 0
        assert ws.stats["reconnect_count"] == 0

    def test_ws_url(self, ws: TradovateWS):
        assert ws._url == "wss://demo.tradovateapi.com/v1/websocket"


class TestHandlerRegistration:
    def test_register_position_handler(self, ws: TradovateWS):
        async def handler(data: dict):
            pass

        ws.on_position_update(handler)
        assert len(ws._position_handlers) == 1

    def test_register_order_handler(self, ws: TradovateWS):
        async def handler(data: dict):
            pass

        ws.on_order_update(handler)
        assert len(ws._order_handlers) == 1

    def test_register_fill_handler(self, ws: TradovateWS):
        async def handler(data: dict):
            pass

        ws.on_fill(handler)
        assert len(ws._fill_handlers) == 1

    def test_register_cash_balance_handler(self, ws: TradovateWS):
        async def handler(data: dict):
            pass

        ws.on_cash_balance(handler)
        assert len(ws._cash_balance_handlers) == 1

    def test_register_generic_handler(self, ws: TradovateWS):
        async def handler(data: dict):
            pass

        ws.on_event(handler)
        assert len(ws._generic_handlers) == 1

    def test_multiple_handlers(self, ws: TradovateWS):
        async def h1(data: dict):
            pass

        async def h2(data: dict):
            pass

        ws.on_position_update(h1)
        ws.on_position_update(h2)
        assert len(ws._position_handlers) == 2


class TestMessageDispatch:
    async def test_dispatch_position_update(self, ws: TradovateWS):
        """Props event with positions should trigger position handlers."""
        received = []

        async def handler(data: dict):
            received.append(data)

        ws.on_position_update(handler)

        item = {
            "e": "props",
            "d": {
                "positions": [
                    {"id": 1, "netPos": 3, "netPrice": 19850.0},
                ]
            },
        }
        await ws._dispatch_message(item)

        assert len(received) == 1
        assert received[0]["netPos"] == 3

    async def test_dispatch_order_update(self, ws: TradovateWS):
        received = []

        async def handler(data: dict):
            received.append(data)

        ws.on_order_update(handler)

        item = {
            "e": "props",
            "d": {
                "orders": [
                    {"orderId": 123, "ordStatus": "Working"},
                ]
            },
        }
        await ws._dispatch_message(item)

        assert len(received) == 1
        assert received[0]["orderId"] == 123

    async def test_dispatch_fill_event(self, ws: TradovateWS):
        """Filled orders should trigger both fill and order handlers."""
        fills = []
        orders = []

        async def fill_handler(data: dict):
            fills.append(data)

        async def order_handler(data: dict):
            orders.append(data)

        ws.on_fill(fill_handler)
        ws.on_order_update(order_handler)

        item = {
            "e": "props",
            "d": {
                "orders": [
                    {"orderId": 123, "ordStatus": "Filled", "avgPx": 19850.25},
                ]
            },
        }
        await ws._dispatch_message(item)

        assert len(fills) == 1
        assert len(orders) == 1
        assert fills[0]["avgPx"] == 19850.25

    async def test_dispatch_generic_handler(self, ws: TradovateWS):
        received = []

        async def handler(data: dict):
            received.append(data)

        ws.on_event(handler)

        item = {"e": "some_event", "d": {"key": "value"}}
        await ws._dispatch_message(item)

        assert len(received) == 1

    async def test_response_to_pending_request(self, ws: TradovateWS):
        """Messages with matching request IDs should resolve pending futures."""
        future = asyncio.get_running_loop().create_future()
        ws._pending[5] = future

        item = {"s": 200, "i": 5, "d": {"orderId": 123}}
        await ws._dispatch_message(item)

        assert future.done()
        result = future.result()
        assert result["d"]["orderId"] == 123
        assert 5 not in ws._pending  # Should be removed

    async def test_dispatch_cash_balance_update(self, ws: TradovateWS):
        """Props event with cashBalances should trigger cash balance handlers."""
        received = []

        async def handler(data: dict):
            received.append(data)

        ws.on_cash_balance(handler)

        item = {
            "e": "props",
            "d": {
                "cashBalances": [
                    {"id": 1, "amount": 50000.0, "realizedPnl": 250.0},
                ]
            },
        }
        await ws._dispatch_message(item)

        assert len(received) == 1
        assert received[0]["amount"] == 50000.0

    async def test_dispatch_mixed_props(self, ws: TradovateWS):
        """A single props event can contain positions, orders, AND cashBalances."""
        positions = []
        orders = []
        balances = []

        async def pos_handler(data: dict):
            positions.append(data)

        async def order_handler(data: dict):
            orders.append(data)

        async def bal_handler(data: dict):
            balances.append(data)

        ws.on_position_update(pos_handler)
        ws.on_order_update(order_handler)
        ws.on_cash_balance(bal_handler)

        item = {
            "e": "props",
            "d": {
                "positions": [{"id": 1, "netPos": 3}],
                "orders": [{"orderId": 123, "ordStatus": "Working"}],
                "cashBalances": [{"id": 1, "amount": 49750.0}],
            },
        }
        await ws._dispatch_message(item)

        assert len(positions) == 1
        assert len(orders) == 1
        assert len(balances) == 1

    async def test_handler_error_doesnt_crash(self, ws: TradovateWS):
        """Handler errors should be caught and logged."""
        async def bad_handler(data: dict):
            raise ValueError("handler error")

        async def good_handler(data: dict):
            pass

        ws.on_position_update(bad_handler)
        ws.on_position_update(good_handler)

        item = {"e": "props", "d": {"positions": [{"id": 1}]}}
        # Should not raise
        await ws._dispatch_message(item)


class TestSendRequest:
    async def test_send_request_not_connected(self, ws: TradovateWS):
        """Should raise when not connected."""
        with pytest.raises(TradovateConnectionError, match="not connected"):
            await ws.send_request("order/placeorder", {"symbol": "MNQM6"})

    async def test_send_fire_and_forget_not_connected(self, ws: TradovateWS):
        with pytest.raises(TradovateConnectionError, match="not connected"):
            await ws.send_fire_and_forget("order/placeorder")


class TestTokenUpdate:
    def test_update_token(self, ws: TradovateWS):
        assert ws._access_token == "test-token-123"
        ws.update_token("new-token-456")
        assert ws._access_token == "new-token-456"


class TestClose:
    async def test_close_when_not_connected(self, ws: TradovateWS):
        """Should be safe to close without connecting."""
        await ws.close()
        assert ws.is_connected is False
        assert ws._running is False

    async def test_close_resolves_pending_futures(self, ws: TradovateWS):
        """Pending futures should be rejected on close."""
        future = asyncio.get_running_loop().create_future()
        ws._pending[1] = future

        await ws.close()

        assert future.done()
        with pytest.raises(TradovateConnectionError):
            future.result()

    async def test_close_disables_reconnect(self, ws: TradovateWS):
        await ws.close()
        assert ws._auto_reconnect is False


class TestStats:
    def test_initial_stats(self, ws: TradovateWS):
        stats = ws.stats
        assert stats["connected"] is False
        assert stats["messages_received"] == 0
        assert stats["messages_sent"] == 0
        assert stats["reconnect_count"] == 0
        assert stats["pending_requests"] == 0


class TestProtocolFormat:
    """Verify the request message format matches Tradovate's custom protocol."""

    async def test_request_format(self, ws: TradovateWS):
        """Requests should be: endpoint\\nrequestId\\n\\njsonBody"""
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        ws._ws = mock_ws
        ws._connected = True

        # Create a future that we'll resolve from _dispatch_message
        async def resolve_soon():
            await asyncio.sleep(0.05)
            if 1 in ws._pending:
                ws._pending[1].set_result({"s": 200, "i": 1, "d": {}})

        asyncio.create_task(resolve_soon())

        result = await ws.send_request(
            "order/placeorder",
            {"symbol": "MNQM6", "action": "Buy"},
            timeout=1.0,
        )

        # Verify the sent message format
        sent_msg = mock_ws.send.call_args[0][0]
        parts = sent_msg.split("\n")
        assert parts[0] == "order/placeorder"  # endpoint
        assert parts[1] == "1"  # request ID
        assert parts[2] == ""  # empty line
        body = json.loads(parts[3])  # JSON body
        assert body["symbol"] == "MNQM6"
        assert body["action"] == "Buy"
