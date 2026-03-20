"""Tests for QuantLynk webhook client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import QuantLynkConfig
from src.core.exceptions import OrderRejectedError
from src.execution.quantlynk_client import QuantLynkClient


class TestQuantLynkClient:
    """Tests for the QuantLynk HTTP webhook client."""

    def _make_client(self, **overrides) -> QuantLynkClient:
        defaults = {
            "webhook_url": "https://quantlynk.io/webhook/test-123",
            "user_id": "test-user",
            "alert_id": "test-alert",
            "timeout_sec": 5.0,
            "max_retries": 1,
            "enabled": True,
        }
        defaults.update(overrides)
        config = QuantLynkConfig(**defaults)
        return QuantLynkClient(config)

    def _mock_response(self, status: int = 200, text: str = "OK"):
        """Create a mock aiohttp response."""
        resp = AsyncMock()
        resp.status = status
        resp.text = AsyncMock(return_value=text)
        return resp

    @pytest.fixture
    def client(self):
        return self._make_client()

    # ── Connection ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_connect_creates_session(self, client):
        await client.connect()
        assert client.is_connected
        await client.close()

    @pytest.mark.asyncio
    async def test_close_clears_session(self, client):
        await client.connect()
        await client.close()
        assert not client.is_connected

    # ── Buy ───────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_buy_sends_correct_payload(self, client):
        mock_resp = self._mock_response(200, "OK")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        result = await client.buy(quantity=2, price=19825.50)

        assert result["status"] == "sent"
        assert result["action"] == "buy"
        assert result["quantity"] == 2

        # Verify payload
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["qv_user_id"] == "test-user"
        assert payload["alert_id"] == "test-alert"
        assert payload["quantity"] == "2"
        assert payload["order_type"] == "market"
        assert payload["action"] == "buy"
        assert payload["price"] == "19825.5"

    # ── Sell ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sell_sends_correct_action(self, client):
        mock_resp = self._mock_response(200, "OK")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        result = await client.sell(quantity=3, price=19810.00)

        assert result["status"] == "sent"
        assert result["action"] == "sell"
        payload = mock_session.post.call_args.kwargs.get("json") or \
                  mock_session.post.call_args[1].get("json")
        assert payload["action"] == "sell"
        assert payload["quantity"] == "3"

    # ── Flatten ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_flatten_sends_flatten_action(self, client):
        mock_resp = self._mock_response(200, "OK")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        result = await client.flatten(price=19830.00)

        assert result["status"] == "sent"
        assert result["action"] == "flatten"
        payload = mock_session.post.call_args.kwargs.get("json") or \
                  mock_session.post.call_args[1].get("json")
        assert payload["action"] == "flatten"
        assert payload["quantity"] == "0"

    # ── Disabled ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_disabled_returns_without_sending(self):
        client = self._make_client(enabled=False)
        result = await client.buy(quantity=2, price=19825.50)
        assert result["status"] == "disabled"

    # ── HTTP Error Handling ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_4xx_raises_order_rejected(self, client):
        mock_resp = self._mock_response(400, "Bad Request")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        with pytest.raises(OrderRejectedError, match="QuantLynk rejected"):
            await client.buy(quantity=2, price=19825.50)

    @pytest.mark.asyncio
    async def test_5xx_retries_then_fails(self, client):
        mock_resp = self._mock_response(500, "Internal Server Error")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        with pytest.raises(OrderRejectedError, match="failed after"):
            await client.buy(quantity=2, price=19825.50)

        # Should have retried (1 original + 1 retry = 2 calls)
        assert mock_session.post.call_count == 2

    # ── Stats ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_stats_track_requests(self, client):
        mock_resp = self._mock_response(200, "OK")
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=ctx)
        mock_session.closed = False
        client._session = mock_session

        await client.buy(quantity=2, price=19825.50)
        await client.sell(quantity=1, price=19830.00)

        stats = client.stats
        assert stats["requests_sent"] == 2
        assert stats["requests_failed"] == 0


class TestQuantLynkOrderManager:
    """Tests for the QuantLynk order manager."""

    def _make_manager(self):
        from src.core.events import EventBus
        from src.execution.quantlynk_order_manager import QuantLynkOrderManager

        client = MagicMock()
        client.buy = AsyncMock(return_value={"status": "sent"})
        client.sell = AsyncMock(return_value={"status": "sent"})
        client.flatten = AsyncMock(return_value={"status": "sent"})
        client.stats = {"requests_sent": 0, "requests_failed": 0, "avg_latency_ms": 0}

        bus = EventBus()
        manager = QuantLynkOrderManager(
            client=client,
            event_bus=bus,
            symbol="MNQM6",
        )
        return manager, client, bus

    def _make_action(self, action_type, side=None, quantity=None, stop_distance=None, new_stop_price=None):
        from src.core.types import ActionType, LLMAction, Side
        return LLMAction(
            action=ActionType(action_type),
            side=Side(side) if side else None,
            quantity=quantity,
            stop_distance=stop_distance,
            new_stop_price=new_stop_price,
            confidence=0.8,
            reasoning="test",
        )

    def _make_position(self, side="long", quantity=2, avg_entry=19825.0, pnl=0.0):
        from src.core.types import PositionState, Side
        return PositionState(
            side=Side(side),
            quantity=quantity,
            avg_entry=avg_entry,
            unrealized_pnl=pnl,
        )

    # ── ENTER ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_enter_long_calls_buy(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ENTER", side="long", quantity=2, stop_distance=10.0)

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["status"] == "placed"
        assert result["side"] == "long"
        client.buy.assert_called_once_with(quantity=2, price=19825.0)

    @pytest.mark.asyncio
    async def test_enter_short_calls_sell(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ENTER", side="short", quantity=2, stop_distance=10.0)

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["status"] == "placed"
        assert result["side"] == "short"
        client.sell.assert_called_once_with(quantity=2, price=19825.0)

    @pytest.mark.asyncio
    async def test_enter_sets_stop_price_long(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ENTER", side="long", quantity=2, stop_distance=10.0)

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["stop_price"] == 19810.0
        assert manager.current_stop_price == 19810.0

    @pytest.mark.asyncio
    async def test_enter_sets_stop_price_short(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ENTER", side="short", quantity=2, stop_distance=10.0)

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["stop_price"] == 19840.0

    @pytest.mark.asyncio
    async def test_enter_skipped_when_in_position(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ENTER", side="long", quantity=2)
        position = self._make_position()

        result = await manager.execute(action, position=position, last_price=19825.0)

        assert result["status"] == "skipped"
        client.buy.assert_not_called()

    # ── ADD ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_add_long_calls_buy(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ADD", quantity=2)
        position = self._make_position(side="long")

        result = await manager.execute(action, position=position, last_price=19840.0)

        assert result["status"] == "placed"
        client.buy.assert_called_once_with(quantity=2, price=19840.0)

    @pytest.mark.asyncio
    async def test_add_short_calls_sell(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ADD", quantity=1)
        position = self._make_position(side="short")

        result = await manager.execute(action, position=position, last_price=19810.0)

        client.sell.assert_called_once_with(quantity=1, price=19810.0)

    @pytest.mark.asyncio
    async def test_add_skipped_when_no_position(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("ADD", quantity=2)

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["status"] == "skipped"

    # ── SCALE_OUT ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_scale_out_long_calls_sell(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("SCALE_OUT", quantity=1)
        position = self._make_position(side="long", quantity=4)

        result = await manager.execute(action, position=position, last_price=19850.0)

        assert result["status"] == "placed"
        client.sell.assert_called_once_with(quantity=1, price=19850.0)

    @pytest.mark.asyncio
    async def test_scale_out_short_calls_buy(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("SCALE_OUT", quantity=1)
        position = self._make_position(side="short", quantity=4)

        result = await manager.execute(action, position=position, last_price=19800.0)

        client.buy.assert_called_once_with(quantity=1, price=19800.0)

    # ── MOVE_STOP ────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_move_stop_updates_locally(self):
        manager, client, _ = self._make_manager()
        manager.current_stop_price = 19815.0
        action = self._make_action("MOVE_STOP", new_stop_price=19830.0)
        position = self._make_position()

        result = await manager.execute(action, position=position, last_price=19850.0)

        assert result["status"] == "updated"
        assert result["new_stop"] == 19830.0
        assert manager.current_stop_price == 19830.0
        # No webhook call for stop moves
        client.buy.assert_not_called()
        client.sell.assert_not_called()
        client.flatten.assert_not_called()

    # ── FLATTEN ──────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_flatten_calls_flatten(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("FLATTEN")

        result = await manager.execute(action, position=self._make_position(), last_price=19850.0)

        assert result["status"] == "placed"
        assert result["action"] == "FLATTEN"
        client.flatten.assert_called_once_with(price=19850.0)
        assert manager.current_stop_price is None

    # ── DO_NOTHING ───────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_do_nothing(self):
        manager, client, _ = self._make_manager()
        action = self._make_action("DO_NOTHING")

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["status"] == "no_action"
        client.buy.assert_not_called()

    # ── Simulation Mode ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_simulation_mode_blocks_entries(self):
        manager, client, _ = self._make_manager()
        manager.set_simulation_mode(True)
        action = self._make_action("ENTER", side="long", quantity=2)

        result = await manager.execute(action, position=None, last_price=19825.0)

        assert result["status"] == "simulated"
        client.buy.assert_not_called()

    @pytest.mark.asyncio
    async def test_simulation_mode_allows_flatten(self):
        manager, client, _ = self._make_manager()
        manager.set_simulation_mode(True)
        action = self._make_action("FLATTEN")

        result = await manager.execute(action, position=self._make_position(), last_price=19825.0)

        assert result["status"] == "placed"
        client.flatten.assert_called_once()

    # ── Stop Check ───────────────────────────────────────────────────────

    def test_check_stop_hit_long(self):
        from src.core.types import Side
        manager, _, _ = self._make_manager()
        manager.current_stop_price = 19815.0

        assert manager.check_stop_hit(19814.0, Side.LONG) is True
        assert manager.check_stop_hit(19815.0, Side.LONG) is True
        assert manager.check_stop_hit(19816.0, Side.LONG) is False

    def test_check_stop_hit_short(self):
        from src.core.types import Side
        manager, _, _ = self._make_manager()
        manager.current_stop_price = 19835.0

        assert manager.check_stop_hit(19836.0, Side.SHORT) is True
        assert manager.check_stop_hit(19835.0, Side.SHORT) is True
        assert manager.check_stop_hit(19834.0, Side.SHORT) is False

    def test_check_stop_no_stop_set(self):
        from src.core.types import Side
        manager, _, _ = self._make_manager()

        assert manager.check_stop_hit(19800.0, Side.LONG) is False
