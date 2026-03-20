"""Tests for the TradingOrchestrator — lifecycle coordination and decision loop."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import AppConfig, TradingConfig, TradovateConfig, AnthropicConfig, TelegramConfig, DatabentoConfig
from src.core.events import EventBus
from src.core.types import (
    ActionType,
    Event,
    EventType,
    GuardrailResult,
    LLMAction,
    MarketState,
    PositionState,
    SessionPhase,
    SessionSummary,
    Side,
)
from src.orchestrator import OrchestratorState, TradingOrchestrator


# ── Helpers ──────────────────────────────────────────────────────────────────


def _config() -> AppConfig:
    return AppConfig(
        tradovate=TradovateConfig(use_demo=True),
        databento=DatabentoConfig(),
        anthropic=AnthropicConfig(),
        telegram=TelegramConfig(),
        trading=TradingConfig(symbol="MNQM6"),
    )


def _market_state(**overrides) -> MarketState:
    defaults = dict(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        last_price=19850.0,
        bid=19849.75,
        ask=19850.25,
        session_phase=SessionPhase.MORNING,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _enter_action(**overrides) -> LLMAction:
    defaults = dict(
        action=ActionType.ENTER,
        side=Side.LONG,
        quantity=2,
        stop_distance=10.0,
        reasoning="Good setup",
        confidence=0.8,
    )
    defaults.update(overrides)
    return LLMAction(**defaults)


def _position(**overrides) -> PositionState:
    defaults = dict(
        side=Side.LONG,
        quantity=2,
        avg_entry=19850.0,
    )
    defaults.update(overrides)
    return PositionState(**defaults)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def mock_reasoner():
    r = MagicMock()
    r.decide = AsyncMock(return_value=LLMAction(
        action=ActionType.DO_NOTHING,
        reasoning="Waiting for setup",
        confidence=0.3,
    ))
    r.stats = {"decision_count": 0, "do_nothing_count": 0}
    return r


@pytest.fixture
def mock_guardrails():
    g = MagicMock()
    g.check = MagicMock(return_value=GuardrailResult(allowed=True))
    g.stats = {"checks_run": 0, "checks_passed": 0, "checks_blocked": 0, "checks_modified": 0, "block_reasons": {}}
    return g


@pytest.fixture
def mock_order_manager():
    om = MagicMock()
    om.execute = AsyncMock(return_value={"order_id": 123, "status": "filled"})
    om.stats = {"orders_placed": 0}
    return om


@pytest.fixture
def mock_position_tracker():
    pt = MagicMock()
    pt.position = None  # flat
    pt.is_flat = True
    return pt


@pytest.fixture
def mock_session_ctrl():
    sc = MagicMock()
    sc.daily_pnl = 0.0
    sc.consecutive_losers = 0
    sc.effective_max_contracts = 6
    sc.should_stop_trading = False
    sc.stop_reason = ""
    sc.total_trades = 0
    sc.winners = 0
    sc.losers = 0
    sc.win_rate = 0.0
    sc.max_drawdown = 0.0
    sc.session_date = "2026-03-14"
    sc.gross_pnl = 0.0
    sc.commissions = 0.0
    sc.stats = {"daily_pnl": 0.0}
    sc.start_session = MagicMock()
    sc.force_stop = MagicMock()
    sc._last_loss_side = None
    sc._consecutive_same_dir_losses = 0
    return sc


@pytest.fixture
def mock_kill_switch():
    ks = MagicMock()
    ks.is_triggered = False
    ks.trigger_reason = ""
    return ks


@pytest.fixture
def orchestrator(
    bus,
    mock_reasoner,
    mock_guardrails,
    mock_order_manager,
    mock_position_tracker,
    mock_session_ctrl,
    mock_kill_switch,
):
    return TradingOrchestrator(
        config=_config(),
        event_bus=bus,
        reasoner=mock_reasoner,
        guardrail_engine=mock_guardrails,
        order_manager=mock_order_manager,
        position_tracker=mock_position_tracker,
        session_controller=mock_session_ctrl,
        kill_switch=mock_kill_switch,
        state_provider=lambda: _market_state(),
    )


# ── Initialization ──────────────────────────────────────────────────────────


class TestInitialization:
    def test_initial_state(self, orchestrator):
        assert orchestrator.state == OrchestratorState.INITIALIZING
        assert orchestrator.is_running is False

    def test_stats_initial(self, orchestrator):
        stats = orchestrator.stats
        assert stats["decisions_made"] == 0
        assert stats["actions_executed"] == 0
        assert stats["actions_blocked"] == 0
        assert stats["cycle_count"] == 0
        assert stats["errors"] == 0


# ── Decision Cycle ──────────────────────────────────────────────────────────


class TestDecisionCycle:
    @pytest.mark.asyncio
    async def test_do_nothing_skips_guardrails(self, orchestrator, mock_reasoner, mock_guardrails):
        """DO_NOTHING decision should not invoke guardrails or execution."""
        mock_reasoner.decide = AsyncMock(return_value=LLMAction(
            action=ActionType.DO_NOTHING,
            reasoning="No setup",
            confidence=0.2,
        ))

        await orchestrator._decision_cycle()

        mock_reasoner.decide.assert_called_once()
        mock_guardrails.check.assert_not_called()
        assert orchestrator._decisions_made == 1

    @pytest.mark.asyncio
    async def test_stop_trading_sets_session_stop(
        self, orchestrator, mock_reasoner, mock_session_ctrl
    ):
        """STOP_TRADING decision should call session_ctrl.force_stop."""
        mock_reasoner.decide = AsyncMock(return_value=LLMAction(
            action=ActionType.STOP_TRADING,
            reasoning="Cost cap exceeded",
            confidence=1.0,
        ))

        await orchestrator._decision_cycle()

        mock_session_ctrl.force_stop.assert_called_once()
        assert "Cost cap exceeded" in mock_session_ctrl.force_stop.call_args[0][0]

    @pytest.mark.asyncio
    async def test_enter_goes_through_full_pipeline(
        self, orchestrator, mock_reasoner, mock_guardrails, mock_order_manager
    ):
        """ENTER action flows: LLM → guardrails → execution."""
        action = _enter_action()
        mock_reasoner.decide = AsyncMock(return_value=action)
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(allowed=True))

        await orchestrator._decision_cycle()

        mock_reasoner.decide.assert_called_once()
        mock_guardrails.check.assert_called_once()
        mock_order_manager.execute.assert_called_once()
        assert orchestrator._actions_executed == 1

    @pytest.mark.asyncio
    async def test_blocked_action_not_executed(
        self, orchestrator, mock_reasoner, mock_guardrails, mock_order_manager
    ):
        """Blocked guardrail result should prevent execution."""
        mock_reasoner.decide = AsyncMock(return_value=_enter_action())
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(
            allowed=False,
            reason="risk_check: confidence too low",
        ))

        await orchestrator._decision_cycle()

        mock_order_manager.execute.assert_not_called()
        assert orchestrator._actions_blocked == 1

    @pytest.mark.asyncio
    async def test_modified_quantity_applied(
        self, orchestrator, mock_reasoner, mock_guardrails, mock_order_manager
    ):
        """Modified quantity from guardrails should be applied to the action."""
        action = _enter_action(quantity=5)
        mock_reasoner.decide = AsyncMock(return_value=action)
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(
            allowed=True,
            modified_quantity=3,
        ))

        await orchestrator._decision_cycle()

        # Verify the executed action has modified quantity
        executed_action = mock_order_manager.execute.call_args[1]["action"]
        assert executed_action.quantity == 3
        assert executed_action.action == ActionType.ENTER

    @pytest.mark.asyncio
    async def test_no_state_skips_cycle(self, orchestrator, mock_reasoner):
        """If state_provider returns None, decision cycle should skip."""
        orchestrator._state_provider = lambda: None

        await orchestrator._decision_cycle()

        mock_reasoner.decide.assert_not_called()

    @pytest.mark.asyncio
    async def test_execution_error_increments_errors(
        self, orchestrator, mock_reasoner, mock_guardrails, mock_order_manager
    ):
        """Execution failure should increment error count, not crash."""
        mock_reasoner.decide = AsyncMock(return_value=_enter_action())
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(allowed=True))
        mock_order_manager.execute = AsyncMock(side_effect=RuntimeError("Connection lost"))

        await orchestrator._decision_cycle()

        assert orchestrator._errors == 1
        assert orchestrator._actions_executed == 0


# ── Guardrail Integration ──────────────────────────────────────────────────


class TestGuardrailIntegration:
    @pytest.mark.asyncio
    async def test_passes_session_ctrl_values_to_guardrails(
        self, orchestrator, mock_reasoner, mock_guardrails, mock_session_ctrl
    ):
        """Guardrails should receive daily PnL, consecutive losers, etc."""
        mock_session_ctrl.daily_pnl = -250.0
        mock_session_ctrl.consecutive_losers = 3
        mock_session_ctrl.effective_max_contracts = 4

        mock_reasoner.decide = AsyncMock(return_value=_enter_action())
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(allowed=True))

        await orchestrator._decision_cycle()

        call_kwargs = mock_guardrails.check.call_args[1]
        assert call_kwargs["daily_pnl"] == -250.0
        assert call_kwargs["consecutive_losers"] == 3
        assert call_kwargs["effective_max_contracts"] == 4


# ── Trading Loop Control Flow ──────────────────────────────────────────────


class TestTradingLoopControl:
    @pytest.mark.asyncio
    async def test_kill_switch_breaks_loop(
        self, orchestrator, mock_kill_switch
    ):
        """Kill switch activation should exit the trading loop."""
        mock_kill_switch.is_triggered = True
        mock_kill_switch.trigger_reason = "Flash crash"

        # Run one iteration of the loop — should exit immediately
        await orchestrator._trading_loop()

        assert orchestrator._cycle_count == 0  # never ran a cycle

    @pytest.mark.asyncio
    async def test_session_stop_breaks_loop(
        self, orchestrator, mock_session_ctrl
    ):
        """Session stop should exit the trading loop."""
        mock_session_ctrl.should_stop_trading = True
        mock_session_ctrl.stop_reason = "Daily loss limit"

        await orchestrator._trading_loop()

        assert orchestrator._cycle_count == 0

    @pytest.mark.asyncio
    @patch("src.orchestrator.clock")
    async def test_past_trading_end_breaks_loop(self, mock_clock, orchestrator):
        """Outside trading hours should exit the loop."""
        mock_clock.is_trading_hours.return_value = False  # outside trading window
        mock_clock.is_past_hard_flatten.return_value = False

        await orchestrator._trading_loop()

        assert orchestrator._cycle_count == 0


# ── Hard Flatten ────────────────────────────────────────────────────────────


class TestHardFlatten:
    @pytest.mark.asyncio
    async def test_hard_flatten_when_in_position(
        self, orchestrator, mock_position_tracker, mock_order_manager
    ):
        """Hard flatten should execute FLATTEN when a position exists."""
        mock_position_tracker.position = _position()
        orchestrator._last_state = _market_state()

        await orchestrator._hard_flatten("Market close")

        mock_order_manager.execute.assert_called_once()
        executed_action = mock_order_manager.execute.call_args[1]["action"]
        assert executed_action.action == ActionType.FLATTEN

    @pytest.mark.asyncio
    async def test_hard_flatten_no_position_skips(
        self, orchestrator, mock_position_tracker, mock_order_manager
    ):
        """Hard flatten should be a no-op when flat."""
        mock_position_tracker.position = None

        await orchestrator._hard_flatten("Market close")

        mock_order_manager.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_hard_flatten_error_doesnt_crash(
        self, orchestrator, mock_position_tracker, mock_order_manager
    ):
        """Hard flatten execution error should be caught, not propagated."""
        mock_position_tracker.position = _position()
        orchestrator._last_state = _market_state()
        mock_order_manager.execute = AsyncMock(side_effect=RuntimeError("Conn lost"))

        await orchestrator._hard_flatten("Emergency")

        assert orchestrator._errors == 1


# ── Kill Switch Event ──────────────────────────────────────────────────────


class TestKillSwitchEvent:
    @pytest.mark.asyncio
    async def test_kill_switch_event_sets_shutdown(self, orchestrator):
        """Kill switch event should trigger shutdown."""
        event = Event(
            type=EventType.KILL_SWITCH_ACTIVATED,
            data={"reason": "Flash crash detected"},
        )

        await orchestrator._on_kill_switch(event)

        assert orchestrator._shutdown_event.is_set()


# ── Daily Limit Event ──────────────────────────────────────────────────────


class TestDailyLimitEvent:
    @pytest.mark.asyncio
    async def test_daily_limit_forces_stop(self, orchestrator, mock_session_ctrl):
        """Daily limit event should call force_stop on session controller."""
        event = Event(
            type=EventType.DAILY_LIMIT_HIT,
            data={"pnl": -420.0},
        )

        await orchestrator._on_daily_limit(event)

        mock_session_ctrl.force_stop.assert_called_once_with("Daily limit hit via event")


# ── Shutdown ────────────────────────────────────────────────────────────────


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_sets_stopped_state(self, orchestrator):
        """Shutdown should set state to STOPPED."""
        await orchestrator._shutdown()
        assert orchestrator.state == OrchestratorState.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, orchestrator):
        """Calling shutdown twice should not error."""
        await orchestrator._shutdown()
        await orchestrator._shutdown()
        assert orchestrator.state == OrchestratorState.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_flattens_position(
        self, orchestrator, mock_position_tracker, mock_order_manager
    ):
        """Shutdown should flatten any open position."""
        mock_position_tracker.position = _position()
        orchestrator._last_state = _market_state()

        await orchestrator._shutdown()

        mock_order_manager.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self, orchestrator):
        """Shutdown should cancel all background tasks."""
        # Create a dummy long-running task
        async def forever():
            await asyncio.sleep(9999)

        task = asyncio.create_task(forever())
        orchestrator._tasks.append(task)

        await orchestrator._shutdown()

        assert task.cancelled()
        assert orchestrator._tasks == []

    @pytest.mark.asyncio
    async def test_shutdown_with_alert_manager(
        self, orchestrator, bus, mock_session_ctrl
    ):
        """Shutdown should send notifications if alert_manager is present."""
        mock_alert = MagicMock()
        mock_alert.send_session_summary = AsyncMock()
        mock_alert.send_shutdown = AsyncMock()
        orchestrator._alert_manager = mock_alert
        orchestrator._start_time = time.monotonic() - 3600  # 1 hour ago

        await orchestrator._shutdown()

        mock_alert.send_session_summary.assert_called_once()
        mock_alert.send_shutdown.assert_called_once()


# ── Start / Stop ────────────────────────────────────────────────────────────


class TestStartStop:
    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_event(self, orchestrator):
        """Calling stop() should set the shutdown event."""
        await orchestrator.stop()
        assert orchestrator._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_start_calls_session_start(self, orchestrator, mock_session_ctrl):
        """Starting the orchestrator should start the session."""
        # Immediately signal shutdown so start() doesn't block
        orchestrator._shutdown_event.set()

        await orchestrator.start()

        mock_session_ctrl.start_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_startup_notification(self, orchestrator, mock_session_ctrl):
        """Start should send startup notification."""
        mock_alert = MagicMock()
        mock_alert.send_startup = AsyncMock()
        mock_alert.send_session_summary = AsyncMock()
        mock_alert.send_shutdown = AsyncMock()
        orchestrator._alert_manager = mock_alert
        orchestrator._shutdown_event.set()

        await orchestrator.start()

        mock_alert.send_startup.assert_called_once()


# ── Cycle Interval ──────────────────────────────────────────────────────────


class TestCycleInterval:
    @patch("src.orchestrator.clock")
    def test_interval_when_flat_rth(self, mock_clock, orchestrator, mock_position_tracker):
        """Flat position during RTH should use the 30s interval."""
        mock_clock.get_session_phase.return_value = SessionPhase.MORNING
        mock_clock.is_eth.return_value = False
        mock_position_tracker.position = None
        interval = orchestrator._get_cycle_interval()
        assert interval == 30.0  # state_update_interval_no_position_sec

    @patch("src.orchestrator.clock")
    def test_interval_when_flat_eth(self, mock_clock, orchestrator, mock_position_tracker):
        """Flat position during ETH should use the slower 45s interval."""
        mock_clock.get_session_phase.return_value = SessionPhase.ASIAN
        mock_clock.is_eth.return_value = True
        mock_position_tracker.position = None
        interval = orchestrator._get_cycle_interval()
        assert interval == 45.0  # state_update_interval_eth_no_position_sec

    def test_interval_when_in_position(self, orchestrator, mock_position_tracker):
        """In-position should use the faster interval."""
        mock_position_tracker.position = _position()
        interval = orchestrator._get_cycle_interval()
        assert interval == 10.0  # state_update_interval_in_position_sec

    def test_interval_critical_near_stop(self, orchestrator, mock_position_tracker):
        """Near stop (within 5 points) should use the critical interval."""
        # Position with stop at 19835, last_price = 19838 → 3 points away
        pos = _position(stop_price=19835.0)
        mock_position_tracker.position = pos
        orchestrator._last_state = _market_state(last_price=19838.0)
        interval = orchestrator._get_cycle_interval()
        assert interval == 5.0  # state_update_interval_critical_sec

    def test_interval_critical_high_adverse(self, orchestrator, mock_position_tracker):
        """High adverse excursion should trigger critical interval."""
        pos = _position(stop_price=19800.0, max_adverse=-150.0)
        mock_position_tracker.position = pos
        orchestrator._last_state = _market_state(last_price=19845.0)
        interval = orchestrator._get_cycle_interval()
        assert interval == 5.0

    def test_interval_normal_in_position(self, orchestrator, mock_position_tracker):
        """In-position but far from stop should use normal in-position interval."""
        pos = _position(stop_price=19800.0, max_adverse=-20.0)
        mock_position_tracker.position = pos
        orchestrator._last_state = _market_state(last_price=19850.0)
        interval = orchestrator._get_cycle_interval()
        assert interval == 10.0  # Not critical, but still in position


# ── Stats ───────────────────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_after_cycles(
        self, orchestrator, mock_reasoner, mock_guardrails, mock_order_manager
    ):
        """Stats should track decisions, executions, and blocks."""
        # Run one successful cycle
        mock_reasoner.decide = AsyncMock(return_value=_enter_action())
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(allowed=True))
        await orchestrator._decision_cycle()

        # Run one blocked cycle
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(
            allowed=False, reason="blocked",
        ))
        await orchestrator._decision_cycle()

        stats = orchestrator.stats
        assert stats["decisions_made"] == 2
        assert stats["actions_executed"] == 1
        assert stats["actions_blocked"] == 1

    @pytest.mark.asyncio
    async def test_stats_include_subsystem_stats(self, orchestrator):
        """Stats should include session, guardrail, and reasoner sub-stats."""
        stats = orchestrator.stats
        assert "session" in stats
        assert "guardrails" in stats
        assert "reasoner" in stats


# ── Lifecycle States ────────────────────────────────────────────────────────


class TestLifecycleStates:
    def test_is_running_states(self, orchestrator):
        """is_running should be True only for active states."""
        for running_state in (
            OrchestratorState.WAITING_FOR_MARKET,
            OrchestratorState.PRE_MARKET,
            OrchestratorState.LIVE_TRADING,
        ):
            orchestrator._state = running_state
            assert orchestrator.is_running is True

        for stopped_state in (
            OrchestratorState.INITIALIZING,
            OrchestratorState.SHUTTING_DOWN,
            OrchestratorState.STOPPED,
        ):
            orchestrator._state = stopped_state
            assert orchestrator.is_running is False


# ── Pre-Market ──────────────────────────────────────────────────────────────


class TestPreMarket:
    @pytest.mark.asyncio
    @patch("src.orchestrator.clock")
    async def test_pre_market_sets_game_plan(self, mock_clock, orchestrator):
        """Pre-market function result should be stored as game_plan."""
        # Simulate: pre-market time has arrived, trading time has arrived
        mock_clock.seconds_until.return_value = -1  # past both pre-market and trading start
        mock_clock.is_trading_hours.return_value = True
        mock_clock.is_past_hard_flatten.return_value = False

        orchestrator._pre_market_fn = AsyncMock(return_value="Buy dips near VWAP.")

        # Kill switch stops the loop after one check
        orchestrator._kill_switch.is_triggered = True

        await orchestrator._run_lifecycle()

        assert orchestrator._game_plan == "Buy dips near VWAP."

    @pytest.mark.asyncio
    @patch("src.orchestrator.clock")
    async def test_pre_market_failure_doesnt_crash(self, mock_clock, orchestrator):
        """Pre-market failure should log error but not crash orchestrator."""
        mock_clock.seconds_until.return_value = -1
        mock_clock.is_trading_hours.return_value = True
        mock_clock.is_past_hard_flatten.return_value = False

        orchestrator._pre_market_fn = AsyncMock(side_effect=RuntimeError("LLM down"))
        orchestrator._kill_switch.is_triggered = True

        await orchestrator._run_lifecycle()

        assert orchestrator._game_plan == ""
        assert orchestrator._errors == 1


# ── Event Integration ───────────────────────────────────────────────────────


class TestEventIntegration:
    @pytest.mark.asyncio
    async def test_execution_publishes_order_filled_event(
        self, orchestrator, bus, mock_reasoner, mock_guardrails, mock_order_manager
    ):
        """Successful execution should publish ORDER_FILLED event."""
        events_received = []
        bus.subscribe(EventType.ORDER_FILLED, lambda e: events_received.append(e))

        mock_reasoner.decide = AsyncMock(return_value=_enter_action())
        mock_guardrails.check = MagicMock(return_value=GuardrailResult(allowed=True))

        await orchestrator._decision_cycle()

        # Check the event was queued (publish_nowait is synchronous queueing)
        # We need to drain the bus queue
        assert not bus._queue.empty()
        event = bus._queue.get_nowait()
        assert event.type == EventType.ORDER_FILLED
        assert event.data["action"] == "ENTER"


# ── Consecutive LLM Errors ──────────────────────────────────────────────────


class TestConsecutiveLLMErrors:
    @pytest.mark.asyncio
    async def test_single_llm_error_increments_consecutive(
        self, orchestrator, mock_reasoner
    ):
        """A single LLM error should increment consecutive counter."""
        mock_reasoner.decide = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        orchestrator._last_state = _market_state()

        await orchestrator._decision_cycle()

        assert orchestrator._consecutive_errors == 1
        assert orchestrator._errors == 1
        assert not orchestrator._shutdown_event.is_set()  # not yet at threshold

    @pytest.mark.asyncio
    async def test_consecutive_errors_reset_on_success(
        self, orchestrator, mock_reasoner
    ):
        """Successful decision should reset consecutive error counter."""
        # First: cause 2 errors
        mock_reasoner.decide = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        orchestrator._last_state = _market_state()
        await orchestrator._decision_cycle()
        await orchestrator._decision_cycle()
        assert orchestrator._consecutive_errors == 2

        # Then: succeed
        mock_reasoner.decide = AsyncMock(return_value=LLMAction(
            action=ActionType.DO_NOTHING,
            reasoning="Waiting",
            confidence=0.3,
        ))
        await orchestrator._decision_cycle()
        assert orchestrator._consecutive_errors == 0

    @pytest.mark.asyncio
    async def test_three_consecutive_errors_triggers_shutdown(
        self, orchestrator, mock_reasoner
    ):
        """3 consecutive LLM failures should trigger emergency shutdown."""
        mock_reasoner.decide = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        orchestrator._last_state = _market_state()

        for _ in range(3):
            await orchestrator._decision_cycle()

        assert orchestrator._consecutive_errors == 3
        assert orchestrator._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_two_errors_no_shutdown(
        self, orchestrator, mock_reasoner
    ):
        """2 consecutive LLM failures should NOT trigger shutdown."""
        mock_reasoner.decide = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        orchestrator._last_state = _market_state()

        for _ in range(2):
            await orchestrator._decision_cycle()

        assert orchestrator._consecutive_errors == 2
        assert not orchestrator._shutdown_event.is_set()
