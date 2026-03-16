"""Tests for the AlertManager — event routing and Telegram dispatch."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.events import EventBus
from src.core.types import (
    ActionType,
    Event,
    EventType,
    LLMAction,
    PositionState,
    Regime,
    SessionSummary,
    Side,
    TradeRecord,
)
from src.notifications.alert_manager import AlertManager
from src.notifications.telegram_client import TelegramClient


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def mock_telegram():
    tg = MagicMock(spec=TelegramClient)
    tg.send = AsyncMock(return_value={"ok": True, "message_id": 1})
    tg.stats = {
        "is_configured": True,
        "messages_sent": 0,
        "messages_throttled": 0,
        "messages_failed": 0,
        "retries_attempted": 0,
    }
    return tg


@pytest.fixture
def manager(mock_telegram, bus):
    return AlertManager(telegram=mock_telegram, event_bus=bus)


def _action(**overrides) -> LLMAction:
    defaults = dict(
        action=ActionType.ENTER,
        side=Side.LONG,
        quantity=3,
        stop_distance=10.0,
        reasoning="Momentum play",
        confidence=0.8,
    )
    defaults.update(overrides)
    return LLMAction(**defaults)


def _trade(**overrides) -> TradeRecord:
    defaults = dict(
        timestamp_entry=datetime(2026, 3, 14, 10, 30, tzinfo=UTC),
        side=Side.LONG,
        entry_quantity=3,
        entry_price=19850.0,
        exit_price=19860.0,
        stop_price=19840.0,
        pnl=140.0,
        hold_time_sec=900,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


# ── Direct Send Methods ──────────────────────────────────────────────────────


class TestDirectSend:
    @pytest.mark.asyncio
    async def test_send_startup(self, manager, mock_telegram):
        await manager.send_startup("demo", 6, 400.0, "MNQM6")
        mock_telegram.send.assert_called_once()
        text = mock_telegram.send.call_args[0][0]
        assert "System starting" in text
        assert "demo" in text
        assert manager.stats["alerts_sent"] == 1

    @pytest.mark.asyncio
    async def test_send_shutdown(self, manager, mock_telegram):
        await manager.send_shutdown("End of day", 280.0)
        text = mock_telegram.send.call_args[0][0]
        assert "System shutdown" in text
        assert "+$280.00" in text

    @pytest.mark.asyncio
    async def test_send_trade_entry(self, manager, mock_telegram):
        await manager.send_trade_entry(
            action=_action(),
            fill_price=19850.0,
            position_qty=3,
            daily_pnl=0.0,
        )
        text = mock_telegram.send.call_args[0][0]
        assert "LONG 3 MNQ" in text

    @pytest.mark.asyncio
    async def test_send_trade_exit(self, manager, mock_telegram):
        await manager.send_trade_exit(
            trade=_trade(),
            daily_pnl=140.0,
            winners=1,
            losers=0,
        )
        text = mock_telegram.send.call_args[0][0]
        assert "+$140.00" in text

    @pytest.mark.asyncio
    async def test_send_heartbeat(self, manager, mock_telegram):
        await manager.send_heartbeat(
            daily_pnl=200.0,
            position=None,
            trades_today=3,
            winners=2,
            losers=1,
        )
        text = mock_telegram.send.call_args[0][0]
        assert "System alive" in text
        assert "flat" in text

    @pytest.mark.asyncio
    async def test_send_session_summary(self, manager, mock_telegram):
        summary = SessionSummary(
            date="2026-03-14",
            total_trades=3,
            winners=2,
            losers=1,
            net_pnl=200.0,
            gross_pnl=220.0,
            commissions=20.0,
        )
        await manager.send_session_summary(summary)
        text = mock_telegram.send.call_args[0][0]
        assert "END OF DAY" in text

    @pytest.mark.asyncio
    async def test_send_risk_alert(self, manager, mock_telegram):
        await manager.send_risk_alert(
            alert_type="Daily loss approaching",
            current_value=-320.0,
            limit_value=-400.0,
        )
        text = mock_telegram.send.call_args[0][0]
        assert "RISK ALERT" in text


# ── Priority Handling ────────────────────────────────────────────────────────


class TestPriority:
    @pytest.mark.asyncio
    async def test_startup_is_priority(self, manager, mock_telegram):
        await manager.send_startup("demo", 6, 400.0, "MNQM6")
        _, kwargs = mock_telegram.send.call_args
        assert kwargs.get("priority") is True

    @pytest.mark.asyncio
    async def test_shutdown_is_priority(self, manager, mock_telegram):
        await manager.send_shutdown("test", 0.0)
        _, kwargs = mock_telegram.send.call_args
        assert kwargs.get("priority") is True

    @pytest.mark.asyncio
    async def test_summary_is_priority(self, manager, mock_telegram):
        summary = SessionSummary(date="2026-03-14")
        await manager.send_session_summary(summary)
        _, kwargs = mock_telegram.send.call_args
        assert kwargs.get("priority") is True

    @pytest.mark.asyncio
    async def test_risk_alert_is_priority(self, manager, mock_telegram):
        await manager.send_risk_alert("test", -300, -400)
        _, kwargs = mock_telegram.send.call_args
        assert kwargs.get("priority") is True


# ── Event Bus Integration ────────────────────────────────────────────────────


class TestEventBusIntegration:
    def test_start_subscribes(self, manager, bus):
        manager.start()
        assert bus.subscribers_for(EventType.ORDER_FILLED) > 0
        assert bus.subscribers_for(EventType.KILL_SWITCH_ACTIVATED) > 0
        assert bus.subscribers_for(EventType.CONNECTION_LOST) > 0
        assert bus.subscribers_for(EventType.GUARDRAIL_TRIGGERED) > 0

    @pytest.mark.asyncio
    async def test_kill_switch_event(self, manager, mock_telegram, bus):
        manager.start()
        event = Event(
            type=EventType.KILL_SWITCH_ACTIVATED,
            data={"reason": "Flash crash detected — flattened all"},
        )
        await manager._on_kill_switch(event)
        text = mock_telegram.send.call_args[0][0]
        assert "KILL SWITCH" in text
        _, kwargs = mock_telegram.send.call_args
        assert kwargs.get("priority") is True

    @pytest.mark.asyncio
    async def test_connection_lost_event(self, manager, mock_telegram):
        event = Event(
            type=EventType.CONNECTION_LOST,
            data={"details": "WebSocket disconnected"},
        )
        await manager._on_connection_lost(event)
        text = mock_telegram.send.call_args[0][0]
        assert "CONNECTION LOST" in text

    @pytest.mark.asyncio
    async def test_connection_restored_event(self, manager, mock_telegram):
        event = Event(
            type=EventType.CONNECTION_RESTORED,
            data={"downtime_sec": 15},
        )
        await manager._on_connection_restored(event)
        text = mock_telegram.send.call_args[0][0]
        assert "CONNECTION RESTORED" in text
        assert "15" in text

    @pytest.mark.asyncio
    async def test_guardrail_event(self, manager, mock_telegram):
        manager.start()
        event = Event(
            type=EventType.GUARDRAIL_TRIGGERED,
            data={"reason": "confidence too low", "action_type": "ENTER"},
        )
        await manager._on_guardrail(event)
        text = mock_telegram.send.call_args[0][0]
        assert "BLOCKED" in text

    @pytest.mark.asyncio
    async def test_guardrail_alert_disabled(self, mock_telegram, bus):
        manager = AlertManager(
            telegram=mock_telegram,
            event_bus=bus,
            alert_on_guardrails=False,
        )
        event = Event(
            type=EventType.GUARDRAIL_TRIGGERED,
            data={"reason": "test", "action_type": "ENTER"},
        )
        await manager._on_guardrail(event)
        mock_telegram.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_daily_limit_event(self, manager, mock_telegram):
        event = Event(
            type=EventType.DAILY_LIMIT_HIT,
            data={"daily_pnl": -400.0, "limit": -400.0},
        )
        await manager._on_daily_limit(event)
        text = mock_telegram.send.call_args[0][0]
        assert "RISK ALERT" in text
        assert "-$400.00" in text

    @pytest.mark.asyncio
    async def test_profit_preservation_event(self, manager, mock_telegram):
        event = Event(
            type=EventType.PROFIT_PRESERVATION,
            data={"tier": 1, "daily_pnl": 200.0, "new_max": 3},
        )
        await manager._on_profit_preservation(event)
        text = mock_telegram.send.call_args[0][0]
        assert "PROFIT PRESERVATION" in text
        assert "3" in text

    @pytest.mark.asyncio
    async def test_position_closed_event(self, manager, mock_telegram):
        trade = _trade(pnl=140.0)
        event = Event(
            type=EventType.POSITION_CHANGED,
            data={
                "closed_trade": trade,
                "daily_pnl": 140.0,
                "winners": 1,
                "losers": 0,
            },
        )
        await manager._on_position_changed(event)
        text = mock_telegram.send.call_args[0][0]
        assert "+$140.00" in text

    @pytest.mark.asyncio
    async def test_position_changed_no_close(self, manager, mock_telegram):
        """Position change without a closed trade should not send."""
        event = Event(
            type=EventType.POSITION_CHANGED,
            data={"action": "add"},
        )
        await manager._on_position_changed(event)
        mock_telegram.send.assert_not_called()


# ── Send Failure Handling ────────────────────────────────────────────────────


class TestSendFailures:
    @pytest.mark.asyncio
    async def test_send_failure_tracked(self, manager, mock_telegram):
        mock_telegram.send = AsyncMock(return_value={"ok": False, "error": "fail"})
        await manager.send_startup("demo", 6, 400.0, "MNQM6")
        assert manager.stats["alerts_failed"] == 1

    @pytest.mark.asyncio
    async def test_send_exception_tracked(self, manager, mock_telegram):
        mock_telegram.send = AsyncMock(side_effect=Exception("network down"))
        await manager.send_startup("demo", 6, 400.0, "MNQM6")
        assert manager.stats["alerts_failed"] == 1


# ── Stats ────────────────────────────────────────────────────────────────────


class TestAlertManagerStats:
    def test_initial_stats(self, manager):
        stats = manager.stats
        assert stats["started"] is False
        assert stats["alerts_sent"] == 0
        assert stats["alerts_failed"] == 0
        assert "telegram_stats" in stats

    @pytest.mark.asyncio
    async def test_stats_after_sends(self, manager, mock_telegram):
        await manager.send_startup("demo", 6, 400.0, "MNQM6")
        await manager.send_shutdown("test", 0.0)
        stats = manager.stats
        assert stats["alerts_sent"] == 2


# ── Stop ─────────────────────────────────────────────────────────────────────


class TestStop:
    def test_stop(self, manager):
        manager.start()
        assert manager._started is True
        manager.stop()
        assert manager._started is False


# ── Heartbeat Loop ───────────────────────────────────────────────────────────


class TestHeartbeatLoop:
    @pytest.mark.asyncio
    async def test_heartbeat_loop_sends(self, mock_telegram, bus):
        """Heartbeat loop should call send_heartbeat periodically."""
        manager = AlertManager(
            telegram=mock_telegram,
            event_bus=bus,
            heartbeat_interval_min=0,  # will be 0 * 60 = 0 seconds
        )

        call_count = 0

        async def get_state():
            nonlocal call_count
            call_count += 1
            return {
                "daily_pnl": 100.0,
                "position": None,
                "trades_today": 2,
                "winners": 1,
                "losers": 1,
            }

        manager.start_heartbeat(get_state)

        # Let it run briefly
        await asyncio.sleep(0.05)
        manager.stop()

        # Should have sent at least one heartbeat
        assert call_count >= 1
        assert mock_telegram.send.call_count >= 1

    @pytest.mark.asyncio
    async def test_heartbeat_cancellation(self, manager):
        """Heartbeat task should cancel cleanly on stop."""
        async def get_state():
            return {"daily_pnl": 0, "position": None, "trades_today": 0, "winners": 0, "losers": 0}

        manager.start_heartbeat(get_state)
        assert manager._heartbeat_task is not None
        assert not manager._heartbeat_task.done()

        manager.stop()
        await asyncio.sleep(0.01)
        assert manager._heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_heartbeat_error_does_not_crash(self, mock_telegram, bus):
        """Errors in get_state_fn should be caught, not crash the loop."""
        manager = AlertManager(
            telegram=mock_telegram,
            event_bus=bus,
            heartbeat_interval_min=0,
        )

        call_count = 0

        async def failing_get_state():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("state unavailable")
            return {"daily_pnl": 0, "position": None, "trades_today": 0, "winners": 0, "losers": 0}

        manager.start_heartbeat(failing_get_state)
        await asyncio.sleep(0.05)
        manager.stop()

        # Loop survived the error and tried again
        assert call_count >= 2


# ── ORDER_FILLED Entry Routing ───────────────────────────────────────────────


class TestOrderFilledRouting:
    @pytest.mark.asyncio
    async def test_entry_fill_sends_notification(self, manager, mock_telegram):
        """ORDER_FILLED with an LLMAction should route to trade entry notification."""
        action = _action()
        event = Event(
            type=EventType.ORDER_FILLED,
            data={
                "action_type": "ENTER",
                "action": action,
                "fill_price": 19850.0,
                "position_qty": 3,
                "daily_pnl": 0.0,
            },
        )
        await manager._on_order_filled(event)
        text = mock_telegram.send.call_args[0][0]
        assert "LONG 3 MNQ" in text
        assert "19,850.00" in text

    @pytest.mark.asyncio
    async def test_add_fill_sends_notification(self, manager, mock_telegram):
        """ORDER_FILLED with action_type=ADD routes to entry notification."""
        action = _action(action=ActionType.ADD, quantity=1)
        event = Event(
            type=EventType.ORDER_FILLED,
            data={
                "action_type": "ADD",
                "action": action,
                "fill_price": 19855.0,
                "position_qty": 4,
                "daily_pnl": 50.0,
            },
        )
        await manager._on_order_filled(event)
        text = mock_telegram.send.call_args[0][0]
        assert "LONG 1 MNQ" in text
        assert "Position now: 4 contracts" in text

    @pytest.mark.asyncio
    async def test_non_entry_fill_ignored(self, manager, mock_telegram):
        """ORDER_FILLED with non-entry action_type should not send."""
        event = Event(
            type=EventType.ORDER_FILLED,
            data={"action_type": "SCALE_OUT", "fill_price": 19860.0},
        )
        await manager._on_order_filled(event)
        mock_telegram.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_fill_without_action_object_ignored(self, manager, mock_telegram):
        """ORDER_FILLED with action_type ENTER but no LLMAction object should not send."""
        event = Event(
            type=EventType.ORDER_FILLED,
            data={"action_type": "ENTER", "action": "not_an_llm_action"},
        )
        await manager._on_order_filled(event)
        mock_telegram.send.assert_not_called()
