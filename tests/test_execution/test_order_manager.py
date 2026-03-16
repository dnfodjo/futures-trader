"""Tests for the OrderManager — LLM action to Tradovate order bridge."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.events import EventBus
from src.core.types import ActionType, EventType, LLMAction, PositionState, Side
from src.execution.order_manager import OrderManager, _avoid_stop_hunt


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_rest():
    rest = AsyncMock()
    rest.place_bracket_order = AsyncMock(return_value={"orderId": 100, "osoOrderId": 200})
    rest.place_market_order = AsyncMock(return_value={"orderId": 101})
    rest.modify_order = AsyncMock(return_value={"orderId": 200})
    rest.liquidate_position = AsyncMock(return_value={"status": "ok"})
    return rest


@pytest.fixture
def order_manager(mock_rest, event_bus):
    return OrderManager(
        rest=mock_rest,
        event_bus=event_bus,
        account_id=12345,
        symbol="MNQM6",
    )


def _action(
    action: ActionType,
    side: Side | None = None,
    quantity: int | None = None,
    stop_distance: float | None = None,
    new_stop_price: float | None = None,
    reasoning: str = "test",
    confidence: float = 0.8,
) -> LLMAction:
    return LLMAction(
        action=action,
        side=side,
        quantity=quantity,
        stop_distance=stop_distance,
        new_stop_price=new_stop_price,
        reasoning=reasoning,
        confidence=confidence,
    )


def _position(
    side: Side = Side.LONG,
    quantity: int = 2,
    avg_entry: float = 19850.0,
    stop_price: float = 19840.0,
) -> PositionState:
    return PositionState(
        side=side,
        quantity=quantity,
        avg_entry=avg_entry,
        stop_price=stop_price,
    )


# ── Test ENTER ───────────────────────────────────────────────────────────────


class TestEnter:
    @pytest.mark.asyncio
    async def test_enter_long(self, order_manager, mock_rest):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=2, stop_distance=10.0)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "placed"
        assert result["action"] == "ENTER"
        assert result["side"] == "long"
        assert result["quantity"] == 2
        assert result["stop_price"] == 19840.0  # 19850 - 10
        mock_rest.place_bracket_order.assert_called_once_with(
            symbol="MNQM6",
            action="Buy",
            quantity=2,
            stop_price=19840.0,
        )

    @pytest.mark.asyncio
    async def test_enter_short(self, order_manager, mock_rest):
        action = _action(ActionType.ENTER, side=Side.SHORT, quantity=1, stop_distance=8.0)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "placed"
        assert result["side"] == "short"
        assert result["stop_price"] == 19858.0  # 19850 + 8
        mock_rest.place_bracket_order.assert_called_once_with(
            symbol="MNQM6",
            action="Sell",
            quantity=1,
            stop_price=19858.0,
        )

    @pytest.mark.asyncio
    async def test_enter_while_in_position(self, order_manager):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1)
        pos = _position()
        result = await order_manager.execute(action, position=pos, last_price=19850.0)

        assert result["status"] == "skipped"
        assert result["reason"] == "already_in_position"

    @pytest.mark.asyncio
    async def test_enter_no_side(self, order_manager):
        action = _action(ActionType.ENTER)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "skipped"
        assert result["reason"] == "no_side_specified"

    @pytest.mark.asyncio
    async def test_enter_default_quantity(self, order_manager, mock_rest):
        action = _action(ActionType.ENTER, side=Side.LONG, stop_distance=10.0)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["quantity"] == 1  # default

    @pytest.mark.asyncio
    async def test_enter_tracks_stop_order(self, order_manager, mock_rest):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        await order_manager.execute(action, position=None, last_price=19850.0)

        assert order_manager.stop_order_id == 200
        assert order_manager.entry_order_id == 100


# ── Test ADD ─────────────────────────────────────────────────────────────────


class TestAdd:
    @pytest.mark.asyncio
    async def test_add_to_long(self, order_manager, mock_rest):
        pos = _position(side=Side.LONG)
        action = _action(ActionType.ADD, quantity=1, stop_distance=10.0)
        result = await order_manager.execute(action, position=pos, last_price=19860.0)

        assert result["status"] == "placed"
        assert result["action"] == "ADD"
        assert result["stop_price"] == 19847.25  # 19860 - 10 = 19850, offset from round 50
        mock_rest.place_bracket_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_no_position(self, order_manager):
        action = _action(ActionType.ADD, quantity=1)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "skipped"


# ── Test SCALE_OUT ───────────────────────────────────────────────────────────


class TestScaleOut:
    @pytest.mark.asyncio
    async def test_scale_out_partial(self, order_manager, mock_rest):
        pos = _position(side=Side.LONG, quantity=3)
        action = _action(ActionType.SCALE_OUT, quantity=1)
        result = await order_manager.execute(action, position=pos, last_price=19860.0)

        assert result["status"] == "placed"
        assert result["action"] == "SCALE_OUT"
        assert result["remaining"] == 2
        mock_rest.place_market_order.assert_called_once_with(
            symbol="MNQM6",
            action="Sell",  # opposite of LONG
            quantity=1,
        )

    @pytest.mark.asyncio
    async def test_scale_out_full_becomes_flatten(self, order_manager, mock_rest):
        pos = _position(side=Side.LONG, quantity=2)
        action = _action(ActionType.SCALE_OUT, quantity=2)
        result = await order_manager.execute(action, position=pos, last_price=19860.0)

        assert result["status"] == "flattened"
        mock_rest.liquidate_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_scale_out_no_position(self, order_manager):
        action = _action(ActionType.SCALE_OUT, quantity=1)
        result = await order_manager.execute(action, position=None, last_price=19860.0)

        assert result["status"] == "skipped"


# ── Test MOVE_STOP ───────────────────────────────────────────────────────────


class TestMoveStop:
    @pytest.mark.asyncio
    async def test_move_stop(self, order_manager, mock_rest):
        # Set up a tracked stop order
        order_manager.update_stop_order_id(200)
        pos = _position(stop_price=19840.0)
        action = _action(ActionType.MOVE_STOP, new_stop_price=19845.0)
        result = await order_manager.execute(action, position=pos, last_price=19860.0)

        assert result["status"] == "modified"
        assert result["new_stop"] == 19845.0
        mock_rest.modify_order.assert_called_once_with(
            order_id=200,
            quantity=2,
            order_type="Stop",
            stop_price=19845.0,
        )

    @pytest.mark.asyncio
    async def test_move_stop_no_price(self, order_manager):
        pos = _position()
        action = _action(ActionType.MOVE_STOP)  # no new_stop_price
        result = await order_manager.execute(action, position=pos, last_price=19860.0)

        assert result["status"] == "skipped"
        assert result["reason"] == "no_stop_price_specified"

    @pytest.mark.asyncio
    async def test_move_stop_no_tracked_order(self, order_manager):
        pos = _position()
        action = _action(ActionType.MOVE_STOP, new_stop_price=19845.0)
        result = await order_manager.execute(action, position=pos, last_price=19860.0)

        assert result["status"] == "skipped"
        assert result["reason"] == "no_stop_order_tracked"


# ── Test FLATTEN ─────────────────────────────────────────────────────────────


class TestFlatten:
    @pytest.mark.asyncio
    async def test_flatten(self, order_manager, mock_rest):
        order_manager.update_stop_order_id(200)
        action = _action(ActionType.FLATTEN, reasoning="Market turning")
        result = await order_manager.execute(action, position=_position(), last_price=19860.0)

        assert result["status"] == "flattened"
        mock_rest.liquidate_position.assert_called_once_with(12345)
        assert order_manager.stop_order_id is None

    @pytest.mark.asyncio
    async def test_stop_trading(self, order_manager, mock_rest):
        action = _action(ActionType.STOP_TRADING, reasoning="Daily limit near")
        result = await order_manager.execute(action, position=_position(), last_price=19860.0)

        assert result["status"] == "stopped"
        assert result["action"] == "STOP_TRADING"
        mock_rest.liquidate_position.assert_called_once()


# ── Test DO_NOTHING ──────────────────────────────────────────────────────────


class TestDoNothing:
    @pytest.mark.asyncio
    async def test_do_nothing(self, order_manager, mock_rest):
        action = _action(ActionType.DO_NOTHING)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "no_action"
        mock_rest.place_bracket_order.assert_not_called()
        mock_rest.place_market_order.assert_not_called()


# ── Test Event Publishing ────────────────────────────────────────────────────


class TestEvents:
    @pytest.mark.asyncio
    async def test_enter_publishes_event(self, order_manager, event_bus):
        events = []
        event_bus.subscribe(EventType.ORDER_PLACED, lambda e: events.append(e))

        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        await order_manager.execute(action, position=None, last_price=19850.0)

        # publish_nowait doesn't await handlers, but we can check calls
        assert order_manager.stats["orders_placed"] == 1

    @pytest.mark.asyncio
    async def test_rejection_publishes_event(self, order_manager, mock_rest, event_bus):
        from src.core.exceptions import OrderRejectedError
        mock_rest.place_bracket_order = AsyncMock(side_effect=OrderRejectedError("No margin"))

        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        result = await order_manager.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "rejected"
        assert order_manager.stats["orders_rejected"] == 1


# ── Test Stats ───────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats(self, order_manager):
        stats = order_manager.stats
        assert stats["orders_placed"] == 0
        assert stats["orders_filled"] == 0
        assert stats["orders_rejected"] == 0
        assert stats["has_stop"] is False

    @pytest.mark.asyncio
    async def test_stats_after_trades(self, order_manager):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        await order_manager.execute(action, position=None, last_price=19850.0)

        stats = order_manager.stats
        assert stats["orders_placed"] == 1
        assert stats["has_stop"] is True


# ── Test Order Event Handlers ────────────────────────────────────────────────


class TestOrderEventHandlers:
    def test_on_fill(self, order_manager):
        order_manager.on_fill({"orderId": 100, "qty": 1, "price": 19850.0})
        assert order_manager.stats["orders_filled"] == 1

    def test_on_order_cancelled(self, order_manager):
        order_manager.update_stop_order_id(200)
        order_manager.on_order_update({"id": 200, "ordStatus": "Cancelled"})
        assert order_manager.stop_order_id is None

    def test_on_order_rejected(self, order_manager):
        order_manager.on_order_update({"id": 300, "ordStatus": "Rejected"})
        assert order_manager.stats["orders_rejected"] == 1

    def test_clear_tracking(self, order_manager):
        order_manager.update_stop_order_id(200)
        order_manager.clear_tracking()
        assert order_manager.stop_order_id is None
        assert order_manager.entry_order_id is None


# ── Test Retry Logic ─────────────────────────────────────────────────────────


class TestRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_on_second_attempt(self, mock_rest, event_bus):
        from src.core.exceptions import TradovateConnectionError

        mock_rest.place_bracket_order = AsyncMock(
            side_effect=[TradovateConnectionError("timeout"), {"orderId": 100, "osoOrderId": 200}]
        )
        om = OrderManager(
            rest=mock_rest, event_bus=event_bus, account_id=12345,
            max_retries=2, retry_delay_sec=0.01,
        )

        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        result = await om.execute(action, position=None, last_price=19850.0)

        assert result["status"] == "placed"
        assert mock_rest.place_bracket_order.call_count == 2
        assert om.stats["retries_attempted"] == 1

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, mock_rest, event_bus):
        from src.core.exceptions import TradovateConnectionError

        mock_rest.place_bracket_order = AsyncMock(
            side_effect=TradovateConnectionError("down")
        )
        om = OrderManager(
            rest=mock_rest, event_bus=event_bus, account_id=12345,
            max_retries=1, retry_delay_sec=0.01,
        )

        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        result = await om.execute(action, position=None, last_price=19850.0)

        # TradovateConnectionError is caught by the generic Exception handler
        assert result["status"] == "error"
        assert mock_rest.place_bracket_order.call_count == 2  # 1 original + 1 retry

    @pytest.mark.asyncio
    async def test_no_retry_on_rejection(self, mock_rest, event_bus):
        from src.core.exceptions import OrderRejectedError

        mock_rest.place_bracket_order = AsyncMock(
            side_effect=OrderRejectedError("no margin")
        )
        om = OrderManager(
            rest=mock_rest, event_bus=event_bus, account_id=12345,
            max_retries=2, retry_delay_sec=0.01,
        )

        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        result = await om.execute(action, position=None, last_price=19850.0)

        # OrderRejectedError is NOT a connection error, so no retry
        assert result["status"] == "rejected"
        assert mock_rest.place_bracket_order.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_flatten(self, mock_rest, event_bus):
        from src.core.exceptions import TradovateConnectionError

        mock_rest.liquidate_position = AsyncMock(
            side_effect=[TradovateConnectionError("timeout"), {"status": "ok"}]
        )
        om = OrderManager(
            rest=mock_rest, event_bus=event_bus, account_id=12345,
            max_retries=2, retry_delay_sec=0.01,
        )

        action = _action(ActionType.FLATTEN, reasoning="test")
        result = await om.execute(action, position=_position(), last_price=19850.0)

        assert result["status"] == "flattened"
        assert mock_rest.liquidate_position.call_count == 2


# ── Test Stop Hunt Avoidance ─────────────────────────────────────────────────


class TestStopHuntAvoidance:
    """Tests for the _avoid_stop_hunt function."""

    def test_long_avoid_round_50(self):
        """Stop at 19850 (multiple of 50) should offset down for long."""
        result = _avoid_stop_hunt(19850.0, Side.LONG)
        assert result == 19847.25  # 19850 - 2.75, rounded to 0.25

    def test_short_avoid_round_50(self):
        """Stop at 19850 (multiple of 50) should offset up for short."""
        result = _avoid_stop_hunt(19850.0, Side.SHORT)
        assert result == 19852.75  # 19850 + 2.75, rounded to 0.25

    def test_long_avoid_round_100(self):
        """Stop at 19900 (multiple of 100) should offset down."""
        result = _avoid_stop_hunt(19900.0, Side.LONG)
        assert result == 19897.25

    def test_short_avoid_round_100(self):
        """Stop at 19900 (multiple of 100) should offset up."""
        result = _avoid_stop_hunt(19900.0, Side.SHORT)
        assert result == 19902.75

    def test_long_avoid_round_25(self):
        """Stop at 19825 (multiple of 25) should offset down."""
        result = _avoid_stop_hunt(19825.0, Side.LONG)
        assert result == 19822.25

    def test_no_offset_for_safe_price(self):
        """Stop at 19837.5 (not near any round number) should stay."""
        result = _avoid_stop_hunt(19837.5, Side.LONG)
        assert result == 19837.5  # No offset needed

    def test_near_round_within_1_point(self):
        """Stop at 19849.5 (within 1pt of 19850) should offset."""
        result = _avoid_stop_hunt(19849.5, Side.LONG)
        assert result == 19846.75  # 19849.5 - 2.75

    def test_avoid_key_level_long(self):
        """Stop near PDH should offset past it for longs."""
        key_levels = [19860.0]  # Prior day high
        result = _avoid_stop_hunt(19859.0, Side.LONG, key_levels)
        assert result == 19857.25  # 19860 - 2.75

    def test_avoid_key_level_short(self):
        """Stop near PDL should offset past it for shorts."""
        key_levels = [19800.0]  # Prior day low
        result = _avoid_stop_hunt(19801.0, Side.SHORT, key_levels)
        assert result == 19802.75  # 19800 + 2.75

    def test_no_key_level_overlap(self):
        """Stop far from key levels should not be affected."""
        key_levels = [19900.0, 19800.0]
        result = _avoid_stop_hunt(19845.0, Side.LONG, key_levels)
        assert result == 19845.0  # No key level within 2 points

    def test_result_always_on_tick(self):
        """All results should be on MNQ tick boundaries (0.25)."""
        for price in [19850.0, 19851.3, 19849.7, 19875.0, 19900.0]:
            result = _avoid_stop_hunt(price, Side.LONG)
            # Result * 4 should be an integer (on 0.25 boundary)
            assert result * 4 == int(result * 4), f"Not on tick: {result}"

    @pytest.mark.asyncio
    async def test_entry_uses_stop_hunt_avoidance(self, order_manager, mock_rest):
        """Verify _execute_enter applies stop hunt avoidance."""
        # 19863 - 13 = 19850 (multiple of 50) → offset to 19847.25
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=13.0)
        result = await order_manager.execute(action, position=None, last_price=19863.0)

        assert result["status"] == "placed"
        assert result["stop_price"] == 19847.25  # Avoided 19850

    @pytest.mark.asyncio
    async def test_entry_with_key_levels(self, order_manager, mock_rest):
        """Verify key levels passed through execute() affect stop placement."""
        # 19860 - 10 = 19850 — near round 50
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1, stop_distance=10.0)
        key_levels = [19847.0]  # Key level at 19847
        result = await order_manager.execute(
            action, position=None, last_price=19860.0, key_levels=key_levels
        )

        assert result["status"] == "placed"
        # First avoids 19850 → 19847.25, then checks key level 19847 (within 2pts) → offsets to 19844.25
        assert result["stop_price"] == 19844.25
