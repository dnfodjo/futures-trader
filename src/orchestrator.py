"""Trading orchestrator — top-level coordinator for the autonomous trading system.

Wires together all components and manages the trading session lifecycle:

1. Startup: initialize components, authenticate, sync state, send notifications
2. Pre-market: collect data, run pre-market analysis at 09:25 ET
3. Live trading: state update → LLM reasoning → guardrail check → execution
4. Shutdown: stop entries, flatten, run summary, disconnect, notify

The orchestrator itself never touches orders or LLM calls directly — it
coordinates the components that do.
"""

from __future__ import annotations

import asyncio
import signal
import time
from datetime import UTC, datetime, time as dt_time, timedelta
from typing import Any, Callable, Coroutine, Optional
from zoneinfo import ZoneInfo

import structlog

from src.core import clock
from src.core.config import AppConfig
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
from src.agents.bull_bear_debate import BullBearDebate
from src.agents.reasoner import Reasoner
from src.agents.session_controller import SessionController
from src.execution.kill_switch import KillSwitch
from src.execution.order_manager import OrderManager
from src.execution.quantlynk_order_manager import QuantLynkOrderManager
from src.execution.position_tracker import PositionTracker
from src.execution.tick_stop_monitor import TickStopMonitor
from src.execution.trail_manager import TrailManager
from src.guardrails.guardrail_engine import GuardrailEngine
from src.learning.kelly_calculator import KellyCalculator
from src.learning.regime_tracker import RegimeTracker
from src.learning.postmortem import PostmortemAnalyzer, PostmortemResult, combine_recent_lessons
from src.learning.trade_logger import TradeLogger
from src.notifications.alert_manager import AlertManager
from src.guardrails.apex_rules import ApexRuleGuardrail
from src.guardrails.circuit_breakers import CircuitBreakers
from src.replay.data_recorder import DataRecorder
from src.indicators.confluence import ConfluenceEngine
from src.indicators.order_flow import OrderFlowEngine
from src.execution.risk_manager import RiskManager

# Optional enhanced modules (imported conditionally to avoid hard dependency)
try:
    from src.data.setup_detector import SetupDetector
except ImportError:
    SetupDetector = None  # type: ignore[misc, assignment]

try:
    from src.data.price_action_analyzer import PriceActionAnalyzer
except ImportError:
    PriceActionAnalyzer = None  # type: ignore[misc, assignment]

logger = structlog.get_logger()

ET = ZoneInfo("US/Eastern")

# ── Lifecycle States ────────────────────────────────────────────────────────


class OrchestratorState:
    """Possible states of the orchestrator lifecycle."""

    INITIALIZING = "initializing"
    WAITING_FOR_MARKET = "waiting_for_market"
    PRE_MARKET = "pre_market"
    LIVE_TRADING = "live_trading"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class TradingOrchestrator:
    """Top-level coordinator for the autonomous trading system.

    Manages the full lifecycle: startup → pre-market → trading → shutdown.
    Coordinates all components without directly making trading decisions or
    placing orders.

    Usage:
        orchestrator = TradingOrchestrator(
            config=config,
            event_bus=bus,
            reasoner=reasoner,
            guardrail_engine=guardrails,
            order_manager=order_mgr,
            position_tracker=tracker,
            session_controller=session_ctrl,
            kill_switch=ks,
            alert_manager=alerts,
        )
        await orchestrator.start()
        # ... runs until shutdown signal or market close
        await orchestrator.stop()
    """

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        reasoner: Reasoner,
        guardrail_engine: GuardrailEngine,
        order_manager: OrderManager | QuantLynkOrderManager,
        position_tracker: PositionTracker,
        session_controller: SessionController,
        kill_switch: KillSwitch,
        alert_manager: Optional[AlertManager] = None,
        state_provider: Optional[Callable[[], Optional[MarketState]]] = None,
        pre_market_fn: Optional[Callable[[], Coroutine]] = None,
        summary_fn: Optional[Callable[[SessionController], Coroutine[Any, Any, str]]] = None,
        # ── Phase 10/11 components ───────────────────────────────────────
        data_recorder: Optional[DataRecorder] = None,
        trade_logger: Optional[TradeLogger] = None,
        regime_tracker: Optional[RegimeTracker] = None,
        kelly_calculator: Optional[KellyCalculator] = None,
        trail_manager: Optional[TrailManager] = None,
        tick_stop_monitor: Optional[TickStopMonitor] = None,
        bull_bear_debate: Optional[BullBearDebate] = None,
        setup_detector: Optional[Any] = None,
        price_action_analyzer: Optional[Any] = None,
        circuit_breakers: Optional[CircuitBreakers] = None,
        apex_guardrail: Optional[ApexRuleGuardrail] = None,
        rth_reset_fn: Optional[Callable[[], None]] = None,
        session_stats_fn: Optional[Callable[..., Coroutine]] = None,
        # ── New confluence strategy components ──────────────────────────────
        confluence_engine: Optional[ConfluenceEngine] = None,
        order_flow_engine: Optional[OrderFlowEngine] = None,
        risk_manager: Optional[RiskManager] = None,
        # ── Daily maintenance callbacks ──────────────────────────────────────
        contract_roll_fn: Optional[Callable[[], Coroutine]] = None,
    ) -> None:
        self._config = config
        self._bus = event_bus
        self._reasoner = reasoner
        self._guardrails = guardrail_engine
        self._order_manager = order_manager
        self._position_tracker = position_tracker
        self._session_ctrl = session_controller
        self._kill_switch = kill_switch
        self._alert_manager = alert_manager

        # Callable that returns the latest MarketState (from StateEngine)
        self._state_provider = state_provider

        # Optional pre-market analysis coroutine
        self._pre_market_fn = pre_market_fn

        # Optional end-of-day summary coroutine
        self._summary_fn = summary_fn

        # ── Phase 10/11: Replay, Learning, Enhanced Execution ────────────

        self._data_recorder = data_recorder
        self._trade_logger = trade_logger
        self._regime_tracker = regime_tracker
        self._kelly_calculator = kelly_calculator
        self._trail_manager = trail_manager
        self._tick_stop_monitor = tick_stop_monitor
        self._debate = bull_bear_debate
        self._setup_detector = setup_detector
        self._price_action_analyzer = price_action_analyzer
        self._circuit_breakers = circuit_breakers
        self._apex = apex_guardrail
        self._rth_reset_fn = rth_reset_fn
        self._session_stats_fn = session_stats_fn

        # ── New confluence strategy components ────────────────────────────
        self._confluence_engine = confluence_engine
        self._order_flow_engine = order_flow_engine
        self._risk_manager = risk_manager
        self._use_confluence = (confluence_engine is not None and risk_manager is not None)

        # ── Daily maintenance callbacks ──────────────────────────────────────
        self._contract_roll_fn = contract_roll_fn

        # ── Pre-market context (set by set_pre_market_context) ────────────
        self._pre_market_context = None

        # ── Internal state ──────────────────────────────────────────────────

        self._state = OrchestratorState.INITIALIZING
        self._shutdown_event = asyncio.Event()
        self._game_plan: str = ""
        self._last_decision_time: float = 0.0
        self._last_state: Optional[MarketState] = None
        self._last_regime_observation_id: Optional[int] = None

        # Timing
        self._trading_start = config.trading.trading_start
        self._trading_end = config.trading.trading_end
        self._hard_flatten_time = config.trading.hard_flatten_time
        self._pre_market_time = config.trading.pre_market_analysis_time

        # Stats
        self._decisions_made: int = 0
        self._actions_executed: int = 0
        self._actions_blocked: int = 0
        self._debate_count: int = 0
        self._trail_updates: int = 0
        self._cycle_count: int = 0
        self._errors: int = 0

        # Post-trade thesis tracking: monitor price after exit to see
        # if the directional thesis was correct (helps LLM learn intra-session)
        self._thesis_tracker: list[dict] = []  # pending thesis checks
        self._thesis_results: list[dict] = []  # completed thesis outcomes
        self._consecutive_errors: int = 0

        # ── Post-exit cooldown ─────────────────────────────────────────
        # After any exit: 2-minute cooldown (win), 5-minute cooldown (loss).
        # Prevents re-entering the same chop that caused a stop-out.
        self._last_exit_time: float = 0.0  # monotonic time of last exit
        self._last_exit_side: Optional[Side] = None  # direction of last exited trade
        self._last_exit_was_loss: bool = False  # was the last exit a loss?
        self._consecutive_same_dir_losses: int = 0  # consecutive losses in same direction
        self._last_loss_side: Optional[Side] = None  # direction of consecutive losses
        self._max_consecutive_errors: int = 3  # kill switch threshold
        self._start_time: Optional[float] = None

        # ── Trade quality guardrails (hard-coded, LLM cannot override) ──────
        # These address the core problem: the LLM overtrades and chases.
        self._last_entry_time: float = 0.0  # monotonic time of last ENTRY
        self._trades_this_hour: int = 0  # entries in current clock hour
        self._current_hour: int = -1  # track hour for reset

        # No-trend confirmation delay — wait 90s for score to hold before entering
        self._no_trend_confirm_side: Optional[str] = None
        self._no_trend_confirm_score: int = 0
        self._no_trend_confirm_start: float = 0.0

        # Tasks
        self._tasks: list[asyncio.Task] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def set_pre_market_context(self, context) -> None:
        """Set pre-market context for today's trading session."""
        # Filter out invalid no-trade windows that cross midnight
        # (string comparison "start <= now <= end" fails when end < start)
        if hasattr(context, 'no_trade_windows') and context.no_trade_windows:
            valid_windows = []
            for start, end in context.no_trade_windows:
                if end < start:
                    logger.warning("orchestrator.invalid_no_trade_window",
                                 window=f"{start}-{end}",
                                 msg="Window crosses midnight — removed (not supported)")
                else:
                    valid_windows.append((start, end))
            context.no_trade_windows = valid_windows
        self._pre_market_context = context
        logger.info("orchestrator.pre_market_context_set",
                    risk_level=getattr(context, 'risk_level', 'unknown'),
                    events=getattr(context, 'events', []))

    # ── Public Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the orchestrator and begin the trading session.

        Runs the lifecycle: wait for market → pre-market → live trading.
        Returns when shutdown is triggered (signal, kill switch, or market close).
        """
        self._start_time = time.monotonic()
        self._state = OrchestratorState.INITIALIZING

        # Start session
        self._session_ctrl.start_session()

        # Restore same-direction loss block from persisted state.
        # Without this, restarts clear the direction block and the system
        # immediately re-enters the blocked direction and loses again.
        if self._session_ctrl._last_loss_side is not None:
            side_str = self._session_ctrl._last_loss_side
            try:
                self._last_loss_side = Side(side_str)
            except ValueError:
                self._last_loss_side = None
            self._consecutive_same_dir_losses = self._session_ctrl._consecutive_same_dir_losses
            if self._consecutive_same_dir_losses >= 2:
                logger.info(
                    "orchestrator.direction_block_restored",
                    side=side_str,
                    consecutive=self._consecutive_same_dir_losses,
                    msg=f"Restored: {self._consecutive_same_dir_losses} consecutive {side_str} losses — direction blocked",
                )

        # Check circuit breakers (multi-day/week/month loss limits)
        if self._circuit_breakers:
            cb_state = self._circuit_breakers.evaluate()
            if cb_state.is_shutdown:
                logger.critical(
                    "orchestrator.circuit_breaker_shutdown",
                    reason=cb_state.shutdown_reason,
                )
                if self._alert_manager:
                    await self._alert_manager.send_system_alert(
                        f"Circuit breaker: {cb_state.shutdown_reason}"
                    )
                return  # Don't trade today
            if cb_state.sim_only:
                logger.warning(
                    "orchestrator.circuit_breaker_sim_only",
                    consecutive_red_days=cb_state.consecutive_red_days,
                )
                self._order_manager.set_simulation_mode(True)
                if self._alert_manager:
                    await self._alert_manager.send_system_alert(
                        f"Circuit breaker: {cb_state.consecutive_red_days} "
                        f"consecutive red days → simulation mode (no live orders)"
                    )
            if cb_state.max_contracts_override is not None:
                self._session_ctrl.set_circuit_breaker_max(
                    cb_state.max_contracts_override
                )
                logger.info(
                    "orchestrator.circuit_breaker_size_reduction",
                    max_contracts=cb_state.max_contracts_override,
                )

        # Start data recording
        if self._data_recorder:
            self._data_recorder.start_session()

        # Load Kelly sizing recommendation (non-blocking)
        if self._kelly_calculator and self._trade_logger:
            try:
                trades = self._trade_logger.get_recent_trades(limit=200)
                if trades:
                    kelly_result = self._kelly_calculator.calculate(trades)
                    if kelly_result.is_reliable:
                        logger.info(
                            "orchestrator.kelly_loaded",
                            optimal_contracts=kelly_result.optimal_contracts,
                            half_kelly=kelly_result.half_kelly,
                            ruin_prob=kelly_result.estimated_ruin_probability,
                        )
            except Exception:
                logger.warning("orchestrator.kelly_load_failed", exc_info=True)

        logger.info("orchestrator.started")

        # Send startup notification
        if self._alert_manager:
            await self._alert_manager.send_startup(
                mode="demo" if self._config.tradovate.use_demo else "live",
                max_contracts=self._config.trading.max_contracts,
                daily_loss_limit=self._config.trading.max_daily_loss,
                symbol=self._config.trading.symbol,
            )

        # Subscribe to events
        self._bus.subscribe(EventType.KILL_SWITCH_ACTIVATED, self._on_kill_switch)
        self._bus.subscribe(EventType.DAILY_LIMIT_HIT, self._on_daily_limit)

        # Run the main lifecycle
        try:
            await self._run_lifecycle()
        except asyncio.CancelledError:
            logger.info("orchestrator.cancelled")
        except Exception:
            logger.exception("orchestrator.lifecycle_error")
            self._errors += 1
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        """Request graceful shutdown."""
        logger.info("orchestrator.stop_requested")
        self._shutdown_event.set()

    @property
    def state(self) -> str:
        """Current orchestrator lifecycle state."""
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state in (
            OrchestratorState.WAITING_FOR_MARKET,
            OrchestratorState.PRE_MARKET,
            OrchestratorState.LIVE_TRADING,
        )

    # ── Main Lifecycle ──────────────────────────────────────────────────────

    async def _run_lifecycle(self) -> None:
        """Execute the full trading day lifecycle."""

        # 1. Wait for market open (if before hours)
        self._state = OrchestratorState.WAITING_FOR_MARKET
        logger.info("orchestrator.waiting_for_market")

        # Wait until pre-market time or shutdown
        while not self._shutdown_event.is_set():
            if self._is_pre_market_time():
                break
            if self._is_trading_time():
                break  # Already past pre-market, go straight to trading
            await asyncio.sleep(5)

        if self._shutdown_event.is_set():
            return

        # 2. Pre-market analysis
        self._state = OrchestratorState.PRE_MARKET
        logger.info("orchestrator.pre_market_phase")

        if self._pre_market_fn:
            try:
                result = await self._pre_market_fn()
                if isinstance(result, str):
                    self._game_plan = result
                    logger.info("orchestrator.game_plan_set", length=len(self._game_plan))
            except Exception:
                logger.exception("orchestrator.pre_market_failed")
                self._errors += 1

        # Load postmortem lessons from recent sessions into the Reasoner
        await self._load_postmortem_lessons()

        # Clear session-specific state in the Reasoner for a fresh day
        self._reasoner.clear_session()

        # Wait for trading start
        logger.info(
            "orchestrator.waiting_for_trading_start",
            is_trading_time=self._is_trading_time(),
            trading_start=self._trading_start,
            trading_end=self._trading_end,
            shutdown_set=self._shutdown_event.is_set(),
        )
        while not self._shutdown_event.is_set():
            if self._is_trading_time():
                break
            await asyncio.sleep(5)

        if self._shutdown_event.is_set():
            logger.info("orchestrator.shutdown_before_trading")
            return

        logger.info("orchestrator.trading_time_reached")

        # 3. Start background tasks: heartbeat + position reconciliation
        self._background_tasks: list[asyncio.Task] = []

        # Start heartbeat notifications (every 30 min)
        if self._alert_manager:
            async def _heartbeat_state():
                return {
                    "daily_pnl": self._session_ctrl.daily_pnl,
                    "position": self._position_tracker.position,
                    "trades_today": self._session_ctrl.total_trades,
                    "winners": self._session_ctrl.winners,
                    "losers": self._session_ctrl.losers,
                }
            self._alert_manager.start_heartbeat(_heartbeat_state)
            logger.info("orchestrator.heartbeat_started")

        # Start periodic position reconciliation (every 30s)
        reconcile_task = asyncio.create_task(
            self._reconciliation_loop(), name="position_reconcile"
        )
        self._background_tasks.append(reconcile_task)

        # 4. Live trading loop
        self._state = OrchestratorState.LIVE_TRADING
        # NOTE: reset_for_rth is NOT called on startup.  The historical
        # warmup already provides clean session data.  Calling reset here
        # wipes the live price (session_close) which causes stale-price
        # entries during thin sessions.  Reset only at the actual daily
        # session boundary (handled by the session date change detector).
        # if self._rth_reset_fn:
        #     self._rth_reset_fn()
        # Reset risk manager daily state (trade counts, shutdown flag, cooldown)
        if self._risk_manager:
            self._risk_manager.reset_daily()
        # Reset flash crash detector to prevent false positives from
        # overnight/pre-market price gaps at session open
        if self._kill_switch:
            self._kill_switch.reset_price_window()
        logger.info("orchestrator.live_trading_started")

        await self._trading_loop()

    async def _trading_loop(self) -> None:
        """The core trading loop — runs until shutdown signal or market close."""

        while not self._shutdown_event.is_set():
            # Check hard flatten time FIRST — must flatten before halt
            if self._is_past_hard_flatten():
                await self._hard_flatten("Past hard flatten time — daily halt approaching")
                break

            # Check if we're outside the trading window
            if self._is_past_trading_end():
                logger.info("orchestrator.trading_hours_ended")
                break

            # Check kill switch
            if self._kill_switch.is_triggered:
                logger.warning(
                    "orchestrator.kill_switch_active",
                    reason=self._kill_switch.trigger_reason,
                )
                break

            # Check session controller stop
            if self._session_ctrl.should_stop_trading:
                logger.info(
                    "orchestrator.session_stopped",
                    reason=self._session_ctrl.stop_reason,
                )
                break

            # Apex flatten deadline check — force flat before 4:59 PM ET
            if self._apex and self._last_state:
                if self._apex.should_force_flatten(self._last_state):
                    position = self._position_tracker.position
                    if position is not None:
                        logger.warning("orchestrator.apex_deadline_flatten")
                        await self._hard_flatten(
                            "Apex rule: all positions must be flat by 4:59 PM ET"
                        )
                    # Stop trading — we're past the deadline
                    self._session_ctrl.force_stop("Apex flatten deadline reached")
                    break

            # Session boundary forced exits — flatten 5 min before each boundary
            # Asian ends 2:00 AM ET → flatten at 1:55 AM
            # Pre-RTH ends 9:30 AM ET → flatten at 9:25 AM
            # (RTH close already handled by apex flatten at 4:54 PM)
            position = self._position_tracker.position
            if position is not None:
                now_et = clock.now_et()
                current_time = now_et.time()
                # 1:55 AM ET — 5 min before Asian/London boundary
                if dt_time(1, 55) <= current_time < dt_time(2, 0):
                    logger.warning(
                        "orchestrator.session_boundary_flatten",
                        boundary="asian_end",
                        time=current_time.isoformat(),
                    )
                    await self._hard_flatten(
                        "Session boundary: flatten 5 min before Asian session end (2:00 AM ET)"
                    )
                    continue  # Skip decision cycle after session boundary flatten
                # 9:25 AM ET — 5 min before RTH open
                elif dt_time(9, 25) <= current_time < dt_time(9, 30):
                    logger.warning(
                        "orchestrator.session_boundary_flatten",
                        boundary="pre_rth_end",
                        time=current_time.isoformat(),
                    )
                    await self._hard_flatten(
                        "Session boundary: flatten 5 min before RTH open (9:30 AM ET)"
                    )
                    continue  # Skip decision cycle after session boundary flatten

            # Time-based exit check — force reassessment of stale positions
            position = self._position_tracker.position
            if position is not None:
                hold_min = position.hold_time_min
                force_min = self._config.trading.time_exit_force_min
                if hold_min >= force_min and not position.is_profitable:
                    logger.warning(
                        "orchestrator.time_exit_force",
                        hold_time_min=round(hold_min, 1),
                        unrealized_pnl=position.unrealized_pnl,
                    )
                    await self._hard_flatten(
                        f"Time exit: held {hold_min:.0f}min with no progress "
                        f"(force at {force_min}min)"
                    )

            # Run one decision cycle (confluence-gated or legacy)
            try:
                if self._use_confluence:
                    await self._confluence_decision_cycle()
                else:
                    await self._decision_cycle()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("orchestrator.decision_cycle_error")
                self._errors += 1

            self._cycle_count += 1

            # Adaptive sleep based on position state
            sleep_sec = self._get_cycle_interval()
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=sleep_sec,
                )
                # If we get here, shutdown was requested
                break
            except asyncio.TimeoutError:
                # Normal — sleep period elapsed, continue loop
                pass

    # ── Confluence Decision Cycle (New Strategy) ─────────────────────────────

    async def _confluence_decision_cycle(self) -> None:
        """Confluence-gated decision cycle: state → risk gates → confluence → LLM → execute.

        This replaces the old LLM-first approach. The LLM only gets called when
        the mechanical confluence engine has already qualified the setup.
        """
        import json as _json

        # 1. Get current market state
        state = self._get_market_state()
        if state is None:
            logger.warning("confluence.cycle_skipped_no_state", cycle=self._cycle_count)
            return
        self._last_state = state

        # 2. Kill switch checks (same as legacy)
        if self._kill_switch:
            self._kill_switch.check_flash_crash(state.last_price, time.monotonic())
            if self._kill_switch.is_triggered:
                await self._hard_flatten("Kill switch: flash crash")
                self._shutdown_event.set()
                return
            in_pos = self._position_tracker.position is not None
            if hasattr(state, "timestamp") and state.timestamp:
                self._kill_switch.check_connection(state.timestamp, in_pos)
                if self._kill_switch.is_triggered:
                    await self._hard_flatten("Kill switch: connection loss")
                    self._shutdown_event.set()
                    return
            self._kill_switch.check_daily_loss(self._session_ctrl.daily_pnl)
            if self._kill_switch.is_triggered:
                await self._hard_flatten("Kill switch: daily loss limit")
                self._shutdown_event.set()
                return

        # 3. Record state for replay
        if self._data_recorder:
            self._data_recorder.record_state(state)

        # 4. Get current position and update unrealized P&L
        position = self._position_tracker.position
        if position is not None:
            self._position_tracker.update_unrealized(state.last_price)

        # Sync state.position with live position tracker
        if (position is None) != (state.position is None):
            state = state.model_copy(update={"position": position})
            self._last_state = state

        # 5. Detect if tick_stop_monitor already flattened us
        #    The tick_stop_monitor calls quantlynk.flatten() directly,
        #    bypassing the position tracker.  We must sync the tracker here
        #    so the orchestrator knows the position is closed.
        if (
            self._tick_stop_monitor
            and self._tick_stop_monitor.is_triggered
        ):
            reason = self._tick_stop_monitor.trigger_reason
            logger.info("confluence.tick_stop_flatten_detected", reason=reason,
                        position_was=position.side.value if position else "None")

            # Sync position tracker — the flatten already happened on Tradovate
            if position is not None and self._position_tracker:
                exit_price = self._tick_stop_monitor._trigger_price or state.last_price
                entry_price = self._tick_stop_monitor._entry_price or 0.0
                side = self._tick_stop_monitor._side or "long"

                # Compute PnL from tick stop monitor's known prices
                if side == "long":
                    pnl_pts = exit_price - entry_price
                else:
                    pnl_pts = entry_price - exit_price
                pnl_est = pnl_pts * position.quantity * 2.0  # $2/pt for MNQ

                # Reset position tracker — position is already flat on Tradovate
                self._position_tracker.reset()
                position = None  # Update local reference

                logger.info("confluence.position_tracker_synced",
                            exit_price=exit_price, entry_price=entry_price,
                            pnl_est=round(pnl_est, 2), reason=reason)

                # Record win/loss for cooldown
                self._last_exit_time = time.monotonic()
                self._last_exit_was_loss = pnl_est < 0
                if pnl_est < 0 and self._risk_manager:
                    self._risk_manager.record_loss(datetime.now(tz=UTC))

            self._tick_stop_monitor.deactivate()

        # 5b. Watchdog: if position tracker says we're in a trade but
        #     tick_stop_monitor is inactive (not monitoring), the position
        #     is stale — force reset after 5 minutes of being stuck.
        if (
            position is not None
            and self._tick_stop_monitor
            and not self._tick_stop_monitor._active
            and not self._tick_stop_monitor._triggered
        ):
            stuck_duration = time.monotonic() - self._last_entry_time
            if stuck_duration > 300:  # 5 minutes
                logger.warning(
                    "confluence.watchdog_position_reset",
                    stuck_sec=round(stuck_duration, 0),
                    msg="Position tracker stuck — tick_stop_monitor inactive. Forcing reset.",
                )
                self._position_tracker.reset()
                position = None
                self._last_exit_time = time.monotonic()

        phase = state.session_phase
        daily_pnl = self._session_ctrl.daily_pnl

        # ── IN POSITION: check for mechanical exits + LLM assessment ──────
        if position is not None:
            # Mechanical exit checks from risk manager
            if self._risk_manager:
                should_exit, exit_reason = self._risk_manager.check_exit_needed(
                    phase=phase,
                    daily_pnl=daily_pnl,
                    entropy=state.flow.entropy,
                    position_pnl=position.unrealized_pnl,
                    delta_against_minutes=0,  # TODO: track this
                    absorption_against=(
                        state.flow.absorption_detected
                        and (
                            (position.side == Side.LONG and state.flow.absorption_side == "ask")
                            or (position.side == Side.SHORT and state.flow.absorption_side == "bid")
                        )
                    ),
                )
                if should_exit:
                    # Entropy/mechanical cooldown: skip non-critical exits
                    # if position is less than 120 seconds old.  Daily loss
                    # limit exits ("DAILY_LOSS_LIMIT") always fire immediately.
                    is_critical = "DAILY_LOSS_LIMIT" in exit_reason or "SESSION_BOUNDARY" in exit_reason
                    position_age = time.monotonic() - self._last_entry_time
                    if not is_critical and position_age < 120.0:
                        logger.info(
                            "confluence.mechanical_exit_cooldown",
                            reason=exit_reason,
                            position_age_sec=round(position_age, 1),
                            msg="Skipping mechanical exit — position too young (< 120s)",
                        )
                    else:
                        logger.warning("confluence.mechanical_exit", reason=exit_reason)
                        exit_action = LLMAction(
                            action=ActionType.FLATTEN,
                            reasoning=f"Mechanical exit: {exit_reason}",
                            confidence=1.0,
                            model_used="mechanical",
                        )
                        await self._execute_confluence_action(exit_action, state, position)
                        return

            # LLM exit assessment DISABLED — trail stop monitor manages all exits.
            # The LLM was exiting after 18 seconds with 12pts unrealized profit
            # because delta turned briefly positive.  Mechanical trail is more
            # reliable: it locks in profit and only exits on actual price reversal.
            # Re-enable once LLM exit logic is tuned to require sustained adverse
            # conditions (e.g. 3+ min of adverse delta, not a single snapshot).
            #
            # The tick_stop_monitor handles: initial stop, trailing stop activation,
            # trail tightening, and take-profit.  Risk manager handles session
            # boundary exits and daily P&L gates.
            logger.debug("confluence.in_position_cycle", side=position.side.value,
                         unrealized_pnl=position.unrealized_pnl)

            return  # No new entries while in position

        # ── FLAT: check for new entry opportunities ───────────────────────

        # Hard gate checks from risk manager
        if not self._risk_manager or not self._confluence_engine:
            return

        # Score confluence for both directions
        session_levels = {
            "asian_high": state.levels.asian_high,
            "asian_low": state.levels.asian_low,
            "london_high": state.levels.london_high,
            "london_low": state.levels.london_low,
            "ny_high": state.levels.ny_high,
            "ny_low": state.levels.ny_low,
            "session_high": state.levels.session_high,
            "session_low": state.levels.session_low,
        }

        # Periodic debug: log confluence engine internals every 20 cycles
        if self._confluence_engine and self._cycle_count % 20 == 0:
            logger.info(
                "confluence.debug_state",
                price=state.last_price,
                atr=round(state.atr, 2),
                bars_1m=len(state.recent_1min_bars) if state.recent_1min_bars else 0,
                multi_tf_emas=state.multi_tf_emas,
                bull_obs=len(self._confluence_engine._bull_obs) if hasattr(self._confluence_engine, '_bull_obs') else '?',
                bear_obs=len(self._confluence_engine._bear_obs) if hasattr(self._confluence_engine, '_bear_obs') else '?',
                swing_highs=len(self._confluence_engine._swing_highs) if hasattr(self._confluence_engine, '_swing_highs') else '?',
                swing_lows=len(self._confluence_engine._swing_lows) if hasattr(self._confluence_engine, '_swing_lows') else '?',
                sweeps=len(self._confluence_engine._sweep_levels) if hasattr(self._confluence_engine, '_sweep_levels') else '?',
                bars_5m=len(state.multi_tf_bars.get("5m", [])) if hasattr(state, 'multi_tf_bars') and state.multi_tf_bars else '?',
            )

        # Get bars_5m from state for structure bounce confirmation
        bars_5m = []
        if hasattr(state, 'multi_tf_bars') and state.multi_tf_bars:
            bars_5m = state.multi_tf_bars.get("5m", [])

        # Try both directions — check which side has better confluence
        results: dict[str, dict] = {}

        for side_str in ("long", "short"):
            score_result = self._confluence_engine.score(
                side=side_str,
                last_price=state.last_price,
                bars_1m=state.recent_1min_bars if state.recent_1min_bars else [],
                atr=state.atr,
                multi_tf_emas=state.multi_tf_emas,
                session_levels=session_levels,
                bars_5m=bars_5m,
            )

            if not score_result.get("blocked"):
                results[side_str] = score_result

        if not results:
            return  # No confluence on either side

        # ── Trend-aware side selection ────────────────────────────────
        # Rule: trend is the tiebreaker. Going against trend requires
        # score to be at least 2 points higher. This prevents buying
        # into waterfalls where OB+candle+volume fire but trend is down.
        long_result = results.get("long")
        short_result = results.get("short")
        long_score = long_result.get("score", 0) if long_result else 0
        short_score = short_result.get("score", 0) if short_result else 0

        long_has_trend = bool(
            long_result and long_result.get("factors", {}).get("trend", {}).get("score", 0) > 0
        )
        short_has_trend = bool(
            short_result and short_result.get("factors", {}).get("trend", {}).get("score", 0) > 0
        )

        # Apply trend advantage: the trending side gets +2 effective score for comparison
        long_effective = long_score + (2 if long_has_trend and not short_has_trend else 0)
        short_effective = short_score + (2 if short_has_trend and not long_has_trend else 0)

        if long_effective >= short_effective and long_result:
            best_score = long_score
            best_side = "long"
            best_result = long_result
        elif short_result:
            best_score = short_score
            best_side = "short"
            best_result = short_result
        else:
            return

        # Log if trend override changed the selection
        if long_has_trend != short_has_trend:
            trending_side = "long" if long_has_trend else "short"
            if best_side != trending_side and best_score > 0:
                logger.info(
                    "confluence.counter_trend_entry",
                    best_side=best_side,
                    best_score=best_score,
                    trending_side=trending_side,
                    msg=f"Entering {best_side} against {trending_side} trend (score advantage overcame +2 trend bonus)",
                )

        if best_score is None or best_side is None:
            return  # No confluence on either side

        # Check no_trade_windows from pre-market context
        # NOTE: String comparison assumes windows do NOT cross midnight (e.g., "23:45" to "00:15").
        # All MNQ trading windows are intraday ET, so this is safe.
        if self._pre_market_context and getattr(self._pre_market_context, 'no_trade_windows', None):
            from zoneinfo import ZoneInfo
            from datetime import datetime as _dt
            now_et = _dt.now(ZoneInfo("America/New_York")).strftime("%H:%M")
            for window_start, window_end in self._pre_market_context.no_trade_windows:
                if window_start <= now_et <= window_end:
                    logger.warning("orchestrator.no_trade_window_active",
                                 window=f"{window_start}-{window_end}",
                                 msg=f"Entry blocked by no-trade window {window_start}-{window_end}")
                    return

        # Get session params
        session_params = self._risk_manager.get_session_params(phase)

        # Apply pre-market min_confluence_override (raises floor if set)
        min_confluence = session_params.min_confluence
        if (self._pre_market_context
            and getattr(self._pre_market_context, 'min_confluence_override', None) is not None):
            min_confluence = max(min_confluence, self._pre_market_context.min_confluence_override)

        # Check if confluence meets session minimum
        if best_score < min_confluence:
            return  # Below threshold

        # ── No-trend confirmation delay ──────────────────────────────
        # When trend=0 on the winning side, don't enter immediately.
        # Wait 90-120s of sustained scoring to confirm it's a real
        # setup and not a trap (dead cat bounce in a waterfall).
        # With trend: enter immediately. Without: wait for confirmation.
        winning_has_trend = (
            (best_side == "long" and long_has_trend)
            or (best_side == "short" and short_has_trend)
        )
        if not winning_has_trend:
            now_mono = time.monotonic()
            # Check if we have an active confirmation timer for this side
            if (
                self._no_trend_confirm_side != best_side
                or self._no_trend_confirm_score == 0
            ):
                # New setup or side changed — start confirmation timer
                self._no_trend_confirm_side = best_side
                self._no_trend_confirm_score = best_score
                self._no_trend_confirm_start = now_mono
                logger.info(
                    "confluence.no_trend_confirmation_started",
                    side=best_side,
                    score=best_score,
                    required_sec=90,
                )
                return  # Don't enter yet — wait for confirmation

            # Timer is running — check if score dropped
            if best_score < min_confluence:
                # Score dropped below threshold — cancel confirmation
                self._no_trend_confirm_side = None
                self._no_trend_confirm_score = 0
                self._no_trend_confirm_start = 0.0
                logger.info(
                    "confluence.no_trend_confirmation_cancelled",
                    side=best_side,
                    score=best_score,
                    reason="score_dropped",
                )
                return

            # Check if enough time has passed (90 seconds)
            elapsed = now_mono - self._no_trend_confirm_start
            if elapsed < 90.0:
                logger.info(
                    "confluence.no_trend_confirmation_waiting",
                    side=best_side,
                    score=best_score,
                    elapsed_sec=round(elapsed, 0),
                    remaining_sec=round(90.0 - elapsed, 0),
                )
                return  # Still waiting

            # Confirmation passed — score held for 90+ seconds, allow entry
            logger.info(
                "confluence.no_trend_confirmation_passed",
                side=best_side,
                score=best_score,
                elapsed_sec=round(elapsed, 0),
            )
            # Reset timer
            self._no_trend_confirm_side = None
            self._no_trend_confirm_score = 0
            self._no_trend_confirm_start = 0.0
        else:
            # Has trend — reset any pending no-trend confirmation
            if self._no_trend_confirm_side is not None:
                self._no_trend_confirm_side = None
                self._no_trend_confirm_score = 0
                self._no_trend_confirm_start = 0.0

        # 30m trend agreement — computed here and passed to risk manager
        # which decides whether to hard-block based on speed state.
        ema_30m = state.multi_tf_emas.get("30m", {})
        trend_30m_agrees = True
        if ema_30m.get("ema_9", 0) > 0 and ema_30m.get("ema_50", 0) > 0:
            if best_side == "long" and ema_30m["ema_9"] < ema_30m["ema_50"]:
                trend_30m_agrees = False
            elif best_side == "short" and ema_30m["ema_9"] > ema_30m["ema_50"]:
                trend_30m_agrees = False

        # Volume check — 5-bar rolling avg vs 20-bar SMA
        # Uses rolling window instead of single bar to avoid one quiet minute
        # blocking an otherwise valid entry.  Confluence already scores volume
        # imbalance separately, so this gate just filters truly dead markets.
        bars = state.recent_1min_bars or []
        volume_sufficient = True
        if len(bars) >= 20:
            avg_vol = sum(b.get("volume", 0) for b in bars[-20:]) / 20
            recent_vol = sum(b.get("volume", 0) for b in bars[-5:]) / 5
            volume_sufficient = recent_vol >= avg_vol

        # Extract structure score from best result factors
        best_factors = best_result.get("factors", {})
        structure_score = best_factors.get("structure", {}).get("score", 0)

        # Run all hard gates
        entry_allowed, block_reason = self._risk_manager.check_entry_allowed(
            phase=phase,
            daily_pnl=daily_pnl,
            has_position=False,
            confluence_score=best_score,
            confidence=0.0,  # LLM hasn't been called yet
            entropy=state.flow.entropy,
            speed_state=best_result.get("speed_state", "NORMAL"),
            trend_30m_agrees=trend_30m_agrees,
            volume_sufficient=volume_sufficient,
            structure_score=structure_score,
        )

        if not entry_allowed:
            if "daily_loss" not in block_reason.lower():  # Don't spam daily loss
                logger.info("confluence.entry_blocked", reason=block_reason, score=best_score)
            return

        # ── Confluence passed + risk gates passed → execute entry ──

        # Guard: don't enter with invalid price (can happen on first cycle
        # after restart before any tick arrives)
        if state.last_price <= 0:
            logger.warning("confluence.entry_skipped_no_price", last_price=state.last_price)
            return

        # Map confluence score to a confidence proxy for logging/tracking
        conf_proxy = min(0.50 + (best_score - session_params.min_confluence) * 0.10, 0.85)

        entry_side = Side.LONG if best_side == "long" else Side.SHORT

        # Dynamic stop: use OB zone or HTF structure zone + ATR buffer
        ob_zone = best_result.get("ob_zone")
        structure_zone = best_result.get("structure_zone")
        sl_pts = self._risk_manager.compute_dynamic_stop(
            side=best_side,
            entry_price=state.last_price,
            ob_zone=ob_zone,
            atr=state.atr,
            phase=phase,
            structure_zone=structure_zone,
        )
        logger.info(
            "confluence.dynamic_stop",
            side=best_side,
            dynamic_sl=sl_pts,
            ob_zone=ob_zone is not None,
            structure_zone=structure_zone is not None,
            atr=state.atr,
            phase=phase.value,
        )

        # Apply pre-market context adjustments: widen stops on volatile days
        if self._pre_market_context and getattr(self._pre_market_context, "widen_stops", False):
            sl_pts = sl_pts * 1.25  # 25% wider stops (e.g., 17pt → 21pt)
            logger.info("confluence.widen_stops_applied", original_sl=sl_pts / 1.25, widened_sl=sl_pts)

        # factors values are dicts with {"score": int, "detail": str}
        raw_factors = best_result.get("factors", {})
        factors_str = ", ".join(
            f"{k}={v['score'] if isinstance(v, dict) else v}"
            for k, v in raw_factors.items()
            if (v["score"] if isinstance(v, dict) else v) > 0
        )

        logger.info(
            "confluence.entry_signal",
            side=best_side,
            confluence_score=best_score,
            confidence_proxy=conf_proxy,
            factors=factors_str,
            phase=phase.value,
        )

        # Confluence-based position sizing: better signal = bigger bet
        # Score 3 (min) → 3 contracts, Score 4 → 4, Score 5+ → max (5)
        max_entry = self._config.trading.max_entry_contracts  # default 5
        if best_score >= 5:
            base_quantity = max_entry              # everything aligns — full size
        elif best_score >= 4:
            base_quantity = max(3, max_entry - 1)  # solid setup
        else:
            base_quantity = 3                      # minimum confluence — base size

        entry_quantity = base_quantity

        # Pre-market reduce_size: halve
        if self._pre_market_context and getattr(self._pre_market_context, "reduce_size", False):
            entry_quantity = max(1, entry_quantity // 2)

        # FAST market: logged for awareness, no size reduction.
        # Strong directional moves with confluence = best setups.
        if best_result.get("fast_market"):
            logger.info("confluence.fast_market_detected", quantity=entry_quantity)

        # Apex scaling check: cap by effective max micros if apex is active
        if self._apex and hasattr(self._apex, 'account_state') and self._apex.account_state:
            effective = self._apex.account_state.effective_max_micros
            entry_quantity = min(entry_quantity, effective)

        # Session controller cap (profit preservation tiers)
        if self._session_ctrl:
            entry_quantity = min(entry_quantity, self._session_ctrl.effective_max_contracts)

        # Hard floor
        entry_quantity = max(1, entry_quantity)

        logger.info(
            "confluence.position_size",
            confluence_score=best_score,
            base_quantity=base_quantity,
            final_quantity=entry_quantity,
            max_entry=max_entry,
        )

        entry_action = LLMAction(
            action=ActionType.ENTER,
            side=entry_side,
            quantity=entry_quantity,
            stop_distance=sl_pts,
            reasoning=f"Confluence {best_score}/6 [{factors_str}] passed all risk gates. LLM bypassed (no DOM/OB data).",
            confidence=conf_proxy,
            setup_type=factors_str or "confluence",
            model_used="bypass",
            primary_timeframe="5m",
            confluence_factors=list(best_result.get("factors", {}).keys()),
            order_flow_assessment=f"entropy={state.flow.entropy:.2f}, vpin={state.flow.vpin:.2f}, delta_trend={state.flow.delta_trend}",
            risk_flags=[],
        )

        await self._execute_confluence_action(entry_action, state, position)

    async def _execute_confluence_action(
        self,
        action: LLMAction,
        state: MarketState,
        position: Optional[PositionState],
    ) -> None:
        """Execute a confluence-validated action via order manager."""
        last_price = state.last_price

        try:
            exec_result = await self._order_manager.execute(
                action=action,
                position=position,
                last_price=last_price,
            )
            self._actions_executed += 1

            logger.info(
                "confluence.action_executed",
                action=action.action.value,
                side=action.side.value if action.side else None,
                confidence=action.confidence,
            )

            # Record entry time and session trade count (unconditional on ENTER)
            if action.action == ActionType.ENTER:
                self._last_entry_time = time.monotonic()
                self._trades_this_hour += 1
                if self._risk_manager:
                    self._risk_manager.record_entry(clock.get_session_phase())

            # Activate tick stop monitor on entry
            if action.action == ActionType.ENTER and self._tick_stop_monitor:
                new_position = self._position_tracker.position
                if new_position:
                    stop_price = 0.0
                    if isinstance(self._order_manager, QuantLynkOrderManager):
                        stop_price = self._order_manager.current_stop_price or 0.0

                    fill_price = new_position.avg_entry
                    current_atr = state.atr if state else 0.0
                    current_phase = clock.get_session_phase()
                    is_eth = clock.is_eth(current_phase)
                    trading_cfg = self._config.trading

                    # Compute hard take-profit from session params
                    tp_pts = self._risk_manager.get_tp_points(current_phase) if self._risk_manager else 0.0
                    if tp_pts > 0 and fill_price > 0:
                        if new_position.side.value == "long":
                            tp_price = round(fill_price + tp_pts, 2)
                        else:
                            tp_price = round(fill_price - tp_pts, 2)
                    else:
                        tp_price = 0.0

                    # Build partial close callback
                    partial_fn = None
                    partial_target = trading_cfg.eth_partial_profit_points if is_eth else trading_cfg.partial_profit_points
                    if isinstance(self._order_manager, QuantLynkOrderManager):
                        async def _partial_close_fn(side: str, quantity: int, price: float) -> None:
                            await self._order_manager._client.partial_close(side=side, quantity=quantity, price=price)
                        partial_fn = _partial_close_fn

                    # Dynamic partial quantity: half of entry size, min 1
                    partial_qty = max(1, action.quantity // 2) if action.quantity else 1

                    self._tick_stop_monitor.activate(
                        side=new_position.side.value,
                        entry_price=fill_price,
                        stop_price=stop_price,
                        take_profit_price=tp_price,
                        atr=current_atr,
                        is_eth=is_eth,
                        eth_trail_distance=trading_cfg.eth_trail_distance,
                        eth_trail_activation=trading_cfg.eth_trail_activation_points,
                        eth_mid_tighten_at_profit=trading_cfg.eth_mid_tighten_at_profit,
                        eth_mid_tightened_distance=trading_cfg.eth_mid_tightened_distance,
                        eth_tighten_at_profit=trading_cfg.eth_tighten_at_profit,
                        eth_tightened_distance=trading_cfg.eth_tightened_distance,
                        partial_target_points=partial_target,
                        partial_fn=partial_fn,
                        partial_quantity=partial_qty,
                        breakeven_offset=trading_cfg.partial_breakeven_offset,
                    )
                    logger.info(
                        "confluence.tick_stop_activated",
                        side=new_position.side.value,
                        entry=fill_price,
                        stop=stop_price,
                        take_profit=tp_price,
                        atr=current_atr,
                        partial_target=partial_target,
                    )

            # Full post-exit bookkeeping on flatten (mirrors legacy path)
            if action.action == ActionType.FLATTEN:
                # Deactivate monitors
                if self._tick_stop_monitor:
                    self._tick_stop_monitor.deactivate()
                if self._trail_manager and self._trail_manager.is_active:
                    self._trail_manager.deactivate()

                last_trade = self._position_tracker.last_trade
                if last_trade:
                    # Update session controller (daily P&L, win/loss, limits)
                    try:
                        self._session_ctrl.record_trade(last_trade)
                    except Exception:
                        logger.warning("confluence.session_record_failed", exc_info=True)

                    # Log to trade journal
                    if self._trade_logger:
                        try:
                            self._trade_logger.log_trade(last_trade)
                            logger.info(
                                "confluence.trade_logged",
                                pnl=round(last_trade.pnl, 2),
                                daily_pnl=round(self._session_ctrl.daily_pnl, 2),
                            )
                        except Exception:
                            logger.warning("confluence.trade_log_failed", exc_info=True)

                    # Sync session stats to state engine
                    if self._session_stats_fn:
                        try:
                            await self._session_stats_fn(
                                daily_pnl=self._session_ctrl.daily_pnl,
                                daily_trades=self._session_ctrl.total_trades,
                                daily_winners=self._session_ctrl.winners,
                                daily_losers=self._session_ctrl.losers,
                            )
                        except Exception:
                            logger.warning("confluence.session_stats_sync_failed", exc_info=True)

                    # Send exit notification
                    if self._alert_manager:
                        try:
                            await self._alert_manager.send_trade_exit(
                                trade=last_trade,
                                daily_pnl=self._session_ctrl.daily_pnl,
                                winners=self._session_ctrl.winners,
                                losers=self._session_ctrl.losers,
                            )
                        except Exception:
                            logger.warning("confluence.exit_notification_failed", exc_info=True)

                    # Thesis tracking (for intra-session LLM learning)
                    if last_trade.pnl is not None:
                        self._thesis_tracker.append({
                            "exit_time": time.monotonic(),
                            "exit_price": last_price,
                            "side": last_trade.side.value,
                            "entry_price": last_trade.entry_price,
                            "pnl": last_trade.pnl,
                            "reasoning": last_trade.reasoning_entry,
                            "check_after_sec": 300,
                        })

                    # Cooldown state tracking
                    if last_trade.pnl is not None:
                        self._last_exit_time = time.monotonic()
                        self._last_exit_side = last_trade.side
                        if last_trade.pnl < 0:
                            self._last_exit_was_loss = True
                            if self._last_loss_side == last_trade.side:
                                self._consecutive_same_dir_losses += 1
                            else:
                                self._consecutive_same_dir_losses = 1
                                self._last_loss_side = last_trade.side
                            # Trigger risk manager cooldown
                            if self._risk_manager:
                                self._risk_manager.record_loss(datetime.now(tz=UTC))
                        else:
                            self._last_exit_was_loss = False
                            self._consecutive_same_dir_losses = 0
                            self._last_loss_side = None

                        # Persist direction-loss state to session controller
                        self._session_ctrl._last_loss_side = (
                            self._last_loss_side.value if self._last_loss_side else None
                        )
                        self._session_ctrl._consecutive_same_dir_losses = self._consecutive_same_dir_losses

            # Publish action event
            await self._bus.publish(Event(
                type=EventType.ACTION_DECIDED,
                data={
                    "action": action.action.value,
                    "side": action.side.value if action.side else None,
                    "confidence": action.confidence,
                    "reasoning": action.reasoning[:200],
                },
            ))

        except Exception:
            logger.exception(
                "confluence.execution_error",
                action=action.action.value,
            )
            self._errors += 1

    # ── Legacy Decision Cycle ──────────────────────────────────────────────

    async def _decision_cycle(self) -> None:
        """Execute one full decision cycle: state → reason → guardrail → execute."""

        # 1. Get current market state
        state = self._get_market_state()
        if state is None:
            return  # No state available yet

        self._last_state = state

        # 0a. Kill switch safety checks (flash crash + connection loss)
        if self._kill_switch:
            import time as _time
            self._kill_switch.check_flash_crash(state.last_price, _time.monotonic())
            if self._kill_switch.is_triggered:
                logger.critical("orchestrator.kill_switch_flash_crash")
                await self._hard_flatten("Kill switch: flash crash")
                self._shutdown_event.set()
                return

            # Connection loss check: use state timestamp as last data time
            in_pos = self._position_tracker.position is not None
            if hasattr(state, 'timestamp') and state.timestamp:
                self._kill_switch.check_connection(state.timestamp, in_pos)
                if self._kill_switch.is_triggered:
                    logger.critical("orchestrator.kill_switch_connection_loss")
                    await self._hard_flatten("Kill switch: connection loss")
                    self._shutdown_event.set()
                    return

        # 1a. Check daily loss kill switch — flatten if P&L exceeds limit
        if self._kill_switch:
            self._kill_switch.check_daily_loss(self._session_ctrl.daily_pnl)
            if self._kill_switch.is_triggered:
                logger.critical(
                    "orchestrator.kill_switch_daily_loss",
                    daily_pnl=self._session_ctrl.daily_pnl,
                )
                await self._hard_flatten("Kill switch: daily loss limit")
                self._shutdown_event.set()
                return

        # 1b. Update Apex equity tracking (balance + unrealized P&L)
        if self._apex:
            position = self._position_tracker.position
            equity = (
                self._apex.account_state.starting_balance
                + self._session_ctrl.daily_pnl
                + (position.unrealized_pnl if position else 0.0)
            )
            self._apex.update_equity(equity)

        # 1b. Record state for replay
        if self._data_recorder:
            self._data_recorder.record_state(state)

        # 1b. Track regime classification
        if self._regime_tracker and state.regime:
            try:
                self._last_regime_observation_id = (
                    self._regime_tracker.record_classification(
                        regime=state.regime,
                        price=state.last_price,
                    )
                )
            except Exception:
                logger.debug("orchestrator.regime_tracking_failed", exc_info=True)

        # 2. Get current position and update unrealized P&L
        position = self._position_tracker.position
        if position is not None:
            self._position_tracker.update_unrealized(state.last_price)

        # 2a. Sync state.position with live position tracker (fixes ghost positions)
        # state_engine.last_state may cache a stale position if tick_stop_monitor
        # closed the position between compute cycles.
        if (position is None) != (state.position is None):
            state = state.model_copy(update={"position": position})
            self._last_state = state

        # 2-pre. Tick stop monitor check — did the monitor already flatten?
        # The TickStopMonitor fires on every Databento tick and sends flatten
        # to QuantLynk directly. Here we detect that it triggered and sync
        # the position tracker / session controller.
        if (position is not None
                and self._tick_stop_monitor is not None
                and self._tick_stop_monitor.is_triggered):
            trigger_reason = self._tick_stop_monitor.trigger_reason
            trigger_price = self._tick_stop_monitor._trigger_price
            logger.warning(
                "orchestrator.tick_monitor_triggered",
                reason=trigger_reason,
                trigger_price=trigger_price,
                entry_price=self._tick_stop_monitor._entry_price,
                best_price=self._tick_stop_monitor.best_price,
                ticks_processed=self._tick_stop_monitor._ticks_processed,
            )

            # Sync position tracker — the flatten already happened via QuantLynk
            if self._position_tracker.position is not None:
                fill_action = "Sell" if position.side == Side.LONG else "Buy"
                try:
                    await self._position_tracker.on_fill({
                        "action": fill_action,
                        "qty": position.quantity,
                        "price": trigger_price,
                    })
                except Exception:
                    logger.warning("orchestrator.tick_monitor_fill_sync_failed", exc_info=True)

            # Record the trade
            last_trade = self._position_tracker.last_trade
            if last_trade:
                try:
                    self._session_ctrl.record_trade(last_trade)
                except Exception:
                    logger.warning("orchestrator.tick_monitor_session_record_failed", exc_info=True)
                if self._trade_logger:
                    try:
                        self._trade_logger.log_trade(last_trade)
                        logger.info(
                            "orchestrator.tick_monitor_trade_logged",
                            pnl=round(last_trade.pnl, 2),
                            daily_pnl=round(self._session_ctrl.daily_pnl, 2),
                        )
                    except Exception:
                        logger.warning("orchestrator.tick_monitor_trade_log_failed", exc_info=True)

            # Send notification
            if self._alert_manager and last_trade:
                try:
                    await self._alert_manager.send_trade_exit(
                        trade=last_trade,
                        daily_pnl=self._session_ctrl.daily_pnl,
                        winners=self._session_ctrl.winners,
                        losers=self._session_ctrl.losers,
                    )
                except Exception:
                    logger.warning("orchestrator.tick_monitor_notification_failed", exc_info=True)

            # Track cooldown state for post-stopout protection
            self._last_exit_time = time.monotonic()
            self._last_exit_side = position.side
            if last_trade and last_trade.pnl < 0:
                self._last_exit_was_loss = True
                if self._last_loss_side == position.side:
                    self._consecutive_same_dir_losses += 1
                else:
                    self._consecutive_same_dir_losses = 1
                    self._last_loss_side = position.side
                # Trigger risk manager cooldown on tick-stop loss
                if self._risk_manager:
                    self._risk_manager.record_loss(datetime.now(UTC))
            else:
                self._last_exit_was_loss = False
                self._consecutive_same_dir_losses = 0
                self._last_loss_side = None

            # Persist direction-loss state to session controller (survives restarts)
            self._session_ctrl._last_loss_side = (
                self._last_loss_side.value if self._last_loss_side else None
            )
            self._session_ctrl._consecutive_same_dir_losses = self._consecutive_same_dir_losses

            # Deactivate monitors
            self._tick_stop_monitor.deactivate()
            if self._trail_manager and self._trail_manager.is_active:
                self._trail_manager.deactivate()

            # Clear QuantLynk order manager stop
            if isinstance(self._order_manager, QuantLynkOrderManager):
                self._order_manager.current_stop_price = None

            # Inject synthetic decision memory so LLM knows position was closed
            # Without this, the LLM sees old "ENTER" in decision memory and
            # assumes the position is still open, causing ghost SCALE_OUT.
            from src.agents.reasoner import DecisionMemory

            pnl_str = f"+${last_trade.pnl:.0f}" if last_trade and last_trade.pnl >= 0 else f"-${abs(last_trade.pnl):.0f}" if last_trade else ""
            self._reasoner._recent_decisions.append(
                DecisionMemory(
                    timestamp=time.time(),
                    action="POSITION_CLOSED_BY_TRAIL_STOP",
                    confidence=1.0,
                    reasoning=f"Trail stop closed position at {trigger_price:.2f} ({pnl_str}). You are now FLAT — no open position.",
                    price=trigger_price,
                    model="system",
                )
            )

            return

        # 2-pre-backup. QuantLynk software stop check (backup — if tick monitor
        # is not active or not available, use the old decision-cycle-level check)
        if (position is not None
                and isinstance(self._order_manager, QuantLynkOrderManager)
                and self._tick_stop_monitor is None
                and self._order_manager.check_stop_hit(state.last_price, position.side)):
            logger.warning(
                "orchestrator.software_stop_hit_backup",
                price=state.last_price,
                stop=self._order_manager.current_stop_price,
            )
            await self._hard_flatten("Software stop hit (backup)")
            return

        # 2a. Trail / stop management — two modes:
        #   - TickStopMonitor active (QuantLynk): handles trailing + stop check
        #     on every Databento tick. We just sync the current stop to the
        #     QuantLynk order manager for stats/logging.
        #   - TrailManager only (Tradovate): updates on each decision cycle.
        if position is not None and self._tick_stop_monitor and self._tick_stop_monitor.is_active:
            # Tick monitor handles trailing on every tick. Sync its current
            # stop to the order manager for logging/stats.
            if isinstance(self._order_manager, QuantLynkOrderManager):
                self._order_manager.current_stop_price = self._tick_stop_monitor.current_stop

        elif position is not None and self._trail_manager:
            # Fallback: old trail manager (no tick monitor)
            # Activate trail manager if not yet active (handles delayed
            # position sync — fill arrives via REST reconciliation after
            # the initial execute() call)
            if not self._trail_manager.is_active:
                self._trail_manager.activate(position, state.last_price)

            new_stop = self._trail_manager.update(state.last_price)
            if new_stop is not None:
                self._trail_updates += 1
                try:
                    trail_action = LLMAction(
                        action=ActionType.MOVE_STOP,
                        new_stop_price=new_stop,
                        reasoning="Trailing stop update",
                        confidence=1.0,
                    )
                    await self._order_manager.execute(
                        action=trail_action,
                        position=position,
                        last_price=state.last_price,
                    )
                    self._trail_manager.confirm_modify(success=True)
                except Exception:
                    self._trail_manager.confirm_modify(success=False)
                    logger.warning("orchestrator.trail_modify_failed", exc_info=True)

        # 2b-pre. Asian session block — no new entries during 6PM-2AM ET
        # Asian session has thin liquidity, narrow ranges, and no edge.
        # The system consistently loses money during Asian hours.
        # We still monitor positions and handle stops, just don't take new trades.
        # Only block if NOT currently in a position (still need to manage open trades).
        if position is None:
            current_phase = clock.get_session_phase()
            if current_phase == SessionPhase.ASIAN:
                # Skip the LLM call entirely — save cost and avoid bad entries
                return

        # 2b-pre2. Warm-up period after startup — EMAs need ~90 seconds of data
        # to become meaningful. Without this, the system trades on garbage EMAs
        # immediately after restart, leading to wrong-direction entries.
        # Only block new entries (not position management).
        if position is None and self._start_time is not None:
            warmup_sec = 90  # 90 seconds for EMAs to populate
            uptime = time.monotonic() - self._start_time
            if uptime < warmup_sec:
                remaining = int(warmup_sec - uptime)
                if self._decisions_made < 2:  # Only log first couple times
                    logger.info(
                        "orchestrator.warmup_active",
                        uptime_sec=int(uptime),
                        remaining_sec=remaining,
                        msg=f"WARM-UP: {remaining}s remaining — collecting EMA/indicator data before trading",
                    )
                return

        # 2a2. UNIVERSAL cooldown — 2 MINUTES after a win, 5 MINUTES after a loss.
        # Entries are mechanical (confluence engine, not LLM), so the old
        # 5/8 minute cooldowns designed for LLM re-entry protection were
        # too long. But a short cooldown after loss is still valuable: the
        # market condition that caused the stop-out (whipsaw, sweep) may
        # still be playing out. 2 min after win lets the system catch the
        # next setup; 5 min after loss avoids re-entering the same chop.
        if position is None and self._last_exit_time > 0:
            cooldown_sec = 300 if self._last_exit_was_loss else 120  # 5min loss, 2min win
            elapsed = time.monotonic() - self._last_exit_time
            if elapsed < cooldown_sec:
                remaining = int(cooldown_sec - elapsed)
                kind = "POST-LOSS" if self._last_exit_was_loss else "INTER-TRADE"
                logger.info(
                    "orchestrator.cooldown_active",
                    remaining_sec=remaining,
                    cooldown_type=kind,
                    last_exit_side=self._last_exit_side.value if self._last_exit_side else "none",
                    msg=f"{kind} COOLDOWN: {remaining}s remaining — no entries allowed",
                )
                return

        # 2b. Run setup detector and price action analyzer (pre-processing)
        detected_setups_text = ""
        price_narrative = ""

        if self._setup_detector:
            try:
                setups = self._setup_detector.detect(
                    state,
                    bars_1s=state.recent_bars,
                )
                if setups:
                    detected_setups_text = "\n".join(
                        f"- **{s.setup_type.value}** ({s.side}, confidence={s.confidence:.1f}): "
                        f"{s.description} | Stop: {s.suggested_stop_distance:.0f}pts | "
                        f"Confirms: {', '.join(s.confirming_signals)}"
                        for s in setups
                    )
            except Exception:
                logger.debug("orchestrator.setup_detection_failed", exc_info=True)

        if self._price_action_analyzer:
            try:
                price_narrative = self._price_action_analyzer.analyze(state)
            except Exception:
                logger.debug("orchestrator.price_action_analysis_failed", exc_info=True)

        # 2c. Build extra context (session awareness for the LLM)
        extra_context = self._build_extra_context()

        # 3. Ask LLM for a decision
        decision_start = time.monotonic()
        try:
            action = await self._reasoner.decide(
                state=state,
                game_plan=self._game_plan,
                extra_context=extra_context,
                detected_setups=detected_setups_text,
                price_action_narrative=price_narrative,
            )
        except Exception as e:
            self._consecutive_errors += 1
            self._errors += 1
            logger.error(
                "orchestrator.llm_decision_failed",
                error=str(e),
                consecutive=self._consecutive_errors,
            )

            # Check kill switch for LLM failures while in position
            in_pos = self._position_tracker.position is not None
            if self._kill_switch:
                self._kill_switch.check_llm_failures(self._consecutive_errors, in_pos)
                if self._kill_switch.is_triggered:
                    logger.critical("orchestrator.kill_switch_llm_failures")
                    await self._hard_flatten("Kill switch: LLM failures")
                    self._shutdown_event.set()
                    return

            # Three consecutive LLM failures → emergency shutdown
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.critical(
                    "orchestrator.consecutive_llm_failures",
                    count=self._consecutive_errors,
                )
                self._shutdown_event.set()
            return

        decision_ms = int((time.monotonic() - decision_start) * 1000)

        # Reset consecutive error counter on successful decision
        self._consecutive_errors = 0
        self._decisions_made += 1
        self._last_decision_time = time.monotonic()

        # 3a. Record decision for replay
        if self._data_recorder:
            self._data_recorder.record_decision(
                state=state,
                action=action,
                latency_ms=decision_ms,
            )

        logger.info(
            "orchestrator.decision",
            action=action.action.value,
            confidence=action.confidence,
            reasoning=action.reasoning[:80] if action.reasoning else "",
        )

        # 4. DO_NOTHING and STOP_TRADING skip guardrails
        if action.action == ActionType.DO_NOTHING:
            return

        if action.action == ActionType.STOP_TRADING:
            # Flatten any open position FIRST, then stop trading
            position = self._position_tracker.position
            if position is not None:
                await self._hard_flatten(
                    f"STOP_TRADING: {action.reasoning or 'LLM requested'}"
                )
            self._session_ctrl.force_stop(
                action.reasoning or "LLM requested stop trading"
            )
            return

        # 4a. Debate SKIPPED — single LLM call for speed.
        # The debate added 6-12s latency and blocked ~80% of entries.
        # Reasoner decision goes straight to guardrails.
        # (debate code removed — guardrails provide sufficient filtering)

        # 4b2. Suppress LLM MOVE_STOP when tick_stop_monitor is active.
        # The tick monitor manages the stop on every tick (~50-200/sec) via
        # trailing logic. The LLM's MOVE_STOP fires every 15s and conflicts
        # with the trail — causing the stop to ping-pong. When the tick
        # monitor is active, it OWNS the stop. LLM exits via FLATTEN only.
        if (
            action.action == ActionType.MOVE_STOP
            and self._tick_stop_monitor is not None
            and self._tick_stop_monitor.is_active
        ):
            logger.debug(
                "orchestrator.move_stop_suppressed",
                msg="Tick stop monitor active — trail manages the stop",
            )
            return

        # 4c. Position protection — don't let LLM cut winners short
        # Three layers:
        # 1. Time-based: Block flattens in first 180s unless trade is losing badly
        # 2. Profit-based: If profitable, let trail manage the exit
        # 3. Scale-out minimum: No scale-out before +10pts profit
        if (
            action.action == ActionType.FLATTEN
            and position is not None
        ):
            # Calculate current unrealized P&L
            if position.side == Side.LONG:
                unrealized_pts = state.last_price - position.avg_entry
            else:
                unrealized_pts = position.avg_entry - state.last_price

            # Layer 1: Time-based minimum hold — 3 MINUTES
            # Don't flatten in first 180s unless trade is losing badly (< -8pts)
            # The LLM panic-flattened a 15-second trade at -5.5pts (trade #5 today).
            # That's normal noise — 5pts is a 1-second MNQ move in a fast market.
            # -8pts ($32 on 2 contracts) gives proper room to develop.
            min_hold_sec = 180
            if (
                position.time_in_trade_sec < min_hold_sec
                and unrealized_pts > -8.0
            ):
                logger.info(
                    "orchestrator.min_hold_suppressed_flatten",
                    time_in_trade=position.time_in_trade_sec,
                    min_hold=min_hold_sec,
                    unrealized_pts=round(unrealized_pts, 2),
                    reasoning=action.reasoning[:80] if action.reasoning else "",
                    msg="Too early to flatten — 3min minimum hold not met",
                )
                return

            # Layer 2: Trail-based profit protection
            # If trail is active and position is profitable, let trail manage exit
            if (
                self._tick_stop_monitor is not None
                and self._tick_stop_monitor.is_active
                and unrealized_pts > 0.0
            ):
                logger.info(
                    "orchestrator.trail_protection_suppressed_flatten",
                    unrealized_pts=round(unrealized_pts, 2),
                    side=position.side.value,
                    time_in_trade=position.time_in_trade_sec,
                    reasoning=action.reasoning[:80] if action.reasoning else "",
                    msg="Trail active and profitable — letting trail manage exit",
                )
                return

        # 4c2. Scale-out minimum profit — HARD BLOCK before +10pts
        # The LLM scales out at +4-6pts, turning potential 20pt winners into $17.
        # Professional traders hold for the full move. Scale-out only at milestones.
        if (
            action.action == ActionType.SCALE_OUT
            and position is not None
        ):
            if position.side == Side.LONG:
                unrealized_pts = state.last_price - position.avg_entry
            else:
                unrealized_pts = position.avg_entry - state.last_price

            if unrealized_pts < 10.0:
                logger.info(
                    "orchestrator.scale_out_too_early",
                    unrealized_pts=round(unrealized_pts, 2),
                    min_required=10.0,
                    side=position.side.value,
                    msg="BLOCKED: Cannot scale out before +10pts — let winners run",
                )
                return

        # 4c2. Hard cap on entry size — the LLM requests 4 contracts even when
        # the prompt says 2-3.  Cap at 2 for ANY single entry.  This means:
        # - 2 contracts × 10pt stop = $40 risk per trade (manageable)
        # - 4 contracts × 10pt stop = $80 risk per trade (too much on marginal setups)
        # The guardrails handle max POSITION size; this handles max ENTRY size.
        if action.action == ActionType.ENTER and action.quantity is not None:
            max_entry = self._config.trading.max_entry_contracts
            if action.quantity > max_entry:
                logger.info(
                    "orchestrator.entry_size_capped",
                    requested=action.quantity,
                    capped_to=max_entry,
                    msg=f"Entry size capped from {action.quantity} to {max_entry} contracts",
                )
                action = action.model_copy(update={"quantity": max_entry})

        # 4c2b. Consecutive same-direction loss block — after 2 losses in the
        # SAME direction, block that direction.  This prevents the pattern:
        # "short, stopped, short again, stopped again" = instant -$180.
        if (
            action.action == ActionType.ENTER
            and action.side is not None
            and self._consecutive_same_dir_losses >= 2
            and self._last_loss_side is not None
            and action.side == self._last_loss_side
        ):
            logger.warning(
                "orchestrator.consecutive_loss_direction_blocked",
                blocked_side=self._last_loss_side.value,
                consecutive=self._consecutive_same_dir_losses,
                msg=f"BLOCKED: {self._consecutive_same_dir_losses} consecutive {self._last_loss_side.value} losses — that direction is blocked",
            )
            return

        # 4d. Regime-aware trend guard — blocks counter-trend UNLESS conditions
        # justify it (extended from VWAP, EMAs flat/crossing, or high confidence).
        # The old rigid guard blocked ALL counter-trend entries, which meant:
        # - Couldn't short during selloffs until EMAs caught up (60-80% of move missed)
        # - Couldn't long bounces until EMAs flipped (bounce over by then)
        if action.action == ActionType.ENTER and action.side is not None:
            emas = state.emas
            ema9 = emas.get("ema_9", 0.0)
            ema21 = emas.get("ema_21", 0.0)
            ema50 = emas.get("ema_50", 0.0)

            if ema9 > 0 and ema21 > 0 and ema50 > 0:
                # Measure EMA spread — how "strong" is the trend signal
                ema_spread = abs(ema9 - ema50)  # total spread across EMAs
                strong_trend = ema_spread > 5.0  # >5pts spread = genuine trend

                # Clear bullish: EMA9 > EMA21 > EMA50 with meaningful spread
                bullish = ema9 > ema21 and ema21 > ema50 and strong_trend
                # Clear bearish: EMA9 < EMA21 < EMA50 with meaningful spread
                bearish = ema9 < ema21 and ema21 < ema50 and strong_trend

                # Check if price is extended from VWAP (mean reversion candidate)
                vwap = state.levels.vwap
                vwap_extension = abs(state.last_price - vwap) if vwap > 0 else 0.0

                # Allow counter-trend when:
                # 1. EMAs are flat/crossing (spread < 5pts) — no clear trend
                # 2. Price >25pts from VWAP — extreme mean reversion opportunity
                # NOTE: Confidence override REMOVED. LLM gives 0.75 on garbage
                # trades — it cannot be trusted to self-rate. The trend guard
                # must be deterministic.
                allow_counter = (
                    not strong_trend  # EMAs are flat/crossing
                    or vwap_extension > 25.0  # extreme extension — mean reversion OK
                )

                if bullish and action.side == Side.SHORT and not allow_counter:
                    logger.warning(
                        "orchestrator.trend_guard_blocked",
                        side="short",
                        ema9=round(ema9, 2),
                        ema21=round(ema21, 2),
                        ema50=round(ema50, 2),
                        ema_spread=round(ema_spread, 1),
                        vwap_ext=round(vwap_extension, 1),
                        msg="BLOCKED: Cannot short in strong uptrend (EMAs spread >5pts)",
                    )
                    return

                if bearish and action.side == Side.LONG and not allow_counter:
                    logger.warning(
                        "orchestrator.trend_guard_blocked",
                        side="long",
                        ema9=round(ema9, 2),
                        ema21=round(ema21, 2),
                        ema50=round(ema50, 2),
                        ema_spread=round(ema_spread, 1),
                        vwap_ext=round(vwap_extension, 1),
                        msg="BLOCKED: Cannot go long in strong downtrend (EMAs spread >5pts)",
                    )
                    return

        # 4d2. RSI extreme guard — REMOVED.
        # RSI is a mean-reversion indicator used in a momentum/trend system.
        # When MNQ trends strongly, RSI > 70 or < 30 for extended periods.
        # Blocking entries at RSI extremes prevented entries during the
        # strongest trending moves — exactly when the system SHOULD be trading.
        # The EMA trend filter, direction-aware volume, and structure factor
        # already handle trend-vs-counter-trend filtering far more accurately.
        # Additionally, the trade_quality.py guardrail had an inconsistency:
        # the orchestrator allowed shorts at RSI < 30 with bearish EMAs,
        # but trade_quality blocked them unconditionally — a dead-code bug.
        # RSI is still available in state.rsi for logging/analysis.
        if action.action == ActionType.ENTER and action.side is not None:
            rsi = state.rsi
            if rsi is not None and rsi > 0:
                if (action.side == Side.LONG and rsi > 70) or (action.side == Side.SHORT and rsi < 30):
                    logger.info(
                        "orchestrator.rsi_extreme_info",
                        side=action.side.value,
                        rsi=round(rsi, 1),
                        msg=f"RSI at {round(rsi, 1)} — logged for analysis, no block applied",
                    )

        # 4d3. VWAP extension guard — block entries >25pts from VWAP.
        # Entering far from VWAP = chasing. Price reverts to VWAP, and you
        # get stopped. Exception: strong trend (all EMAs aligned + structure confirms)
        # allows up to 35pts extension for trend continuation entries.
        # Thresholds widened from 15/20 to 25/35 to match MNQ volatility —
        # MNQ regularly trades 20-30pts from VWAP on normal trending days.
        # The old 15pt threshold blocked entries during the strongest moves.
        if action.action == ActionType.ENTER and action.side is not None:
            vwap = state.levels.vwap
            if vwap > 0:
                vwap_dist = abs(state.last_price - vwap)
                ema_align = state.emas.get("alignment", "mixed")
                mkt_struct = getattr(state, "market_structure", {})
                struct_pattern = mkt_struct.get("pattern", "mixed") if isinstance(mkt_struct, dict) else "mixed"

                # Strong trend = aligned EMAs + confirming structure
                strong_trend_long = (
                    ema_align in ("bullish", "bullish_partial")
                    and struct_pattern == "HH_HL"
                    and action.side == Side.LONG
                )
                strong_trend_short = (
                    ema_align in ("bearish", "bearish_partial")
                    and struct_pattern == "LH_LL"
                    and action.side == Side.SHORT
                )
                max_vwap_dist = 35.0 if (strong_trend_long or strong_trend_short) else 25.0

                if vwap_dist > max_vwap_dist:
                    logger.warning(
                        "orchestrator.vwap_extension_blocked",
                        side=action.side.value,
                        price=state.last_price,
                        vwap=round(vwap, 2),
                        distance=round(vwap_dist, 1),
                        max_allowed=max_vwap_dist,
                        msg=f"BLOCKED: {round(vwap_dist, 1)}pts from VWAP (max {max_vwap_dist}pts)",
                    )
                    return

        # 4d4. Momentum chase guard — require pullback after fast move.
        # If price moved 10+ points in the last 5 bars (5 minutes), require
        # at least a 3pt pullback before entering. This prevents chasing
        # momentum spikes where the move is already over.
        if action.action == ActionType.ENTER and action.side is not None:
            recent_bars = getattr(state, "recent_1min_bars", None) or getattr(state, "recent_bars", None) or []
            if len(recent_bars) >= 5:
                last_5 = recent_bars[-5:]
                try:
                    bar_highs = [b.get("high", b.get("h", 0)) if isinstance(b, dict) else getattr(b, "high", 0) for b in last_5]
                    bar_lows = [b.get("low", b.get("l", 0)) if isinstance(b, dict) else getattr(b, "low", 0) for b in last_5]
                    recent_range = max(bar_highs) - min(bar_lows)
                    if recent_range > 10.0:
                        price = state.last_price
                        # For longs: price should have pulled back from the high
                        if action.side == Side.LONG:
                            pullback = max(bar_highs) - price
                            if pullback < 3.0:
                                logger.warning(
                                    "orchestrator.momentum_chase_blocked",
                                    side="long",
                                    recent_range=round(recent_range, 1),
                                    pullback=round(pullback, 1),
                                    msg=f"BLOCKED: {round(recent_range, 1)}pt move in 5 bars, only {round(pullback, 1)}pt pullback (need 3+)",
                                )
                                return
                        # For shorts: price should have bounced from the low
                        elif action.side == Side.SHORT:
                            bounce = price - min(bar_lows)
                            if bounce < 3.0:
                                logger.warning(
                                    "orchestrator.momentum_chase_blocked",
                                    side="short",
                                    recent_range=round(recent_range, 1),
                                    bounce=round(bounce, 1),
                                    msg=f"BLOCKED: {round(recent_range, 1)}pt move in 5 bars, only {round(bounce, 1)}pt bounce (need 3+)",
                                )
                                return
                except Exception:
                    pass  # Don't block on bar parsing errors

        # 4d5. Trades-per-hour limiter — max 3 entries per clock hour.
        # Prevents revenge trading after quick losses. 3/hr allows legitimate
        # burst setups (e.g., London open) while stopping overtrading.
        # Other guards (8 daily max, same-direction block, post-loss confidence)
        # provide additional pacing.
        if action.action == ActionType.ENTER:
            import datetime as _dt
            now_et = _dt.datetime.now(ZoneInfo("US/Eastern"))
            current_hour = now_et.hour
            if current_hour != self._current_hour:
                self._current_hour = current_hour
                self._trades_this_hour = 0
            if self._trades_this_hour >= 3:
                logger.warning(
                    "orchestrator.hourly_trade_limit",
                    trades_this_hour=self._trades_this_hour,
                    hour=current_hour,
                    msg=f"BLOCKED: Already {self._trades_this_hour} trades this hour (max 3/hour)",
                )
                return

        # 4e. Anti-chase guardrail: Block entries near session extremes AND
        # key levels in the wrong direction. Widened to 5pts from 3pts.
        if action.action == ActionType.ENTER and action.side is not None:
            price = state.last_price
            if action.side == Side.LONG:
                # Don't enter long near session high (within 5pts)
                for level_name, level_val in [
                    ("session_high", state.levels.session_high),
                ]:
                    if level_val > 0 and abs(level_val - price) <= 5.0:
                        logger.warning(
                            "orchestrator.anti_chase_blocked",
                            side="long",
                            price=price,
                            level=level_name,
                            level_val=level_val,
                            distance=round(level_val - price, 2),
                            msg=f"BLOCKED: Cannot enter long within 5pts of {level_name}",
                        )
                        return
            elif action.side == Side.SHORT:
                # Don't enter short near session low (within 5pts)
                for level_name, level_val in [
                    ("session_low", state.levels.session_low),
                ]:
                    if level_val > 0 and abs(price - level_val) <= 5.0:
                        logger.warning(
                            "orchestrator.anti_chase_blocked",
                            side="short",
                            price=price,
                            level=level_name,
                            level_val=level_val,
                            distance=round(price - level_val, 2),
                            msg=f"BLOCKED: Cannot enter short within 5pts of {level_name}",
                        )
                        return

        # 5. Run through guardrails
        result = self._guardrails.check(
            action=action,
            state=state,
            position=position,
            daily_pnl=self._session_ctrl.daily_pnl,
            consecutive_losers=self._session_ctrl.consecutive_losers,
            effective_max_contracts=self._session_ctrl.effective_max_contracts,
        )

        if not result.allowed:
            # ── Auto-reverse: ENTER opposite side → convert to FLATTEN ──
            # When the LLM wants to reverse direction (e.g., ENTER SHORT
            # while LONG), it really means "flatten then enter opposite".
            # We convert to FLATTEN here; the LLM will enter the new
            # direction on the next decision cycle.
            if (
                action.action == ActionType.ENTER
                and position is not None
                and action.side is not None
                and action.side != position.side
                and "position_limit" in (result.reason or "")
            ):
                logger.info(
                    "orchestrator.auto_reverse_flatten",
                    current_side=position.side.value,
                    wanted_side=action.side.value,
                    msg="LLM wanted to reverse — converting to FLATTEN",
                )
                action = LLMAction(
                    action=ActionType.FLATTEN,
                    reasoning=f"Auto-reverse: LLM wanted {action.side.value} but in {position.side.value} position. Flattening first.",
                    confidence=action.confidence,
                    model_used=action.model_used,
                    latency_ms=action.latency_ms,
                )
                # Re-run guardrails on the FLATTEN action
                result = self._guardrails.check(
                    action=action,
                    state=state,
                    position=position,
                    daily_pnl=self._session_ctrl.daily_pnl,
                    consecutive_losers=self._session_ctrl.consecutive_losers,
                    effective_max_contracts=self._session_ctrl.effective_max_contracts,
                )
                if not result.allowed:
                    self._actions_blocked += 1
                    logger.info(
                        "orchestrator.auto_reverse_also_blocked",
                        reason=result.reason,
                    )
                    return
                # Fall through to execute the FLATTEN
            # ── Auto-convert: ENTER same side → convert to ADD ──
            # When the LLM sends ENTER while already in a position on the
            # SAME side, it's trying to add to the position. Convert to ADD
            # and re-run guardrails.
            elif (
                action.action == ActionType.ENTER
                and position is not None
                and action.side is not None
                and action.side == position.side
                and "position_limit" in (result.reason or "")
            ):
                logger.info(
                    "orchestrator.auto_convert_add",
                    side=action.side.value,
                    quantity=action.quantity,
                    msg="LLM used ENTER while in position — converting to ADD",
                )
                action = LLMAction(
                    action=ActionType.ADD,
                    side=action.side,
                    quantity=action.quantity,
                    stop_distance=action.stop_distance,
                    reasoning=f"Auto-convert: LLM wanted ENTER {action.side.value} while already {position.side.value}. Converting to ADD.",
                    confidence=action.confidence,
                    setup_type=action.setup_type,
                    model_used=action.model_used,
                    latency_ms=action.latency_ms,
                )
                # Re-run guardrails on the ADD action
                result = self._guardrails.check(
                    action=action,
                    state=state,
                    position=position,
                    daily_pnl=self._session_ctrl.daily_pnl,
                    consecutive_losers=self._session_ctrl.consecutive_losers,
                    effective_max_contracts=self._session_ctrl.effective_max_contracts,
                )
                if not result.allowed:
                    self._actions_blocked += 1
                    logger.info(
                        "orchestrator.auto_convert_add_blocked",
                        reason=result.reason,
                    )
                    return
                # Fall through to execute the ADD
            else:
                self._actions_blocked += 1
                logger.info(
                    "orchestrator.action_blocked",
                    reason=result.reason,
                    action=action.action.value,
                )
                return

        # 5a. Apex rule check (runs AFTER standard guardrails)
        if self._apex:
            current_contracts = position.quantity if position else 0
            apex_result = self._apex.check(
                action=action,
                state=state,
                position=position,
                daily_pnl=self._session_ctrl.daily_pnl,
                current_contracts=current_contracts,
            )
            if not apex_result.allowed:
                self._actions_blocked += 1
                logger.warning(
                    "orchestrator.apex_blocked",
                    reason=apex_result.reason,
                    action=action.action.value,
                )
                return

        # 6. Apply quantity modification if guardrails adjusted it
        if result.modified_quantity is not None:
            action = LLMAction(
                action=action.action,
                side=action.side,
                quantity=result.modified_quantity,
                stop_distance=action.stop_distance,
                new_stop_price=action.new_stop_price,
                reasoning=action.reasoning,
                confidence=action.confidence,
                model_used=action.model_used,
            )

        # 7. Execute the action
        last_price = state.last_price

        # Extract key levels for stop hunt avoidance
        _key_levels: list[float] | None = None
        if state.levels:
            _key_levels = [
                v for v in [
                    state.levels.prior_day_high,
                    state.levels.prior_day_low,
                    state.levels.session_high,
                    state.levels.session_low,
                    state.levels.overnight_high,
                    state.levels.overnight_low,
                    state.levels.vwap,
                    state.levels.poc,
                ] if v > 0
            ]

        try:
            exec_result = await self._order_manager.execute(
                action=action,
                position=position,
                last_price=last_price,
                key_levels=_key_levels,
            )
            self._actions_executed += 1

            logger.info(
                "orchestrator.action_executed",
                action=action.action.value,
                result_keys=list(exec_result.keys()) if exec_result else [],
            )

            # 7a-pre. Track entry timing for trade quality guardrails
            if action.action == ActionType.ENTER:
                self._last_entry_time = time.monotonic()
                self._trades_this_hour += 1
                if self._risk_manager:
                    self._risk_manager.record_entry(clock.get_session_phase())

            # 7a. Activate/deactivate trail manager on position changes
            if self._trail_manager:
                if action.action == ActionType.ENTER:
                    new_position = self._position_tracker.position
                    if new_position:
                        self._trail_manager.activate(new_position, last_price)
                elif action.action == ActionType.FLATTEN:
                    self._trail_manager.deactivate()

            # 7a-tick. Activate/deactivate tick stop monitor (QuantLynk)
            # This fires on every Databento tick — much faster than the
            # decision-cycle-level trail manager.
            if self._tick_stop_monitor:
                if action.action == ActionType.ENTER:
                    new_position = self._position_tracker.position
                    if new_position:
                        # Get stop price from the order manager
                        stop_price = 0.0
                        if isinstance(self._order_manager, QuantLynkOrderManager):
                            stop_price = self._order_manager.current_stop_price or 0.0

                        # Get take profit from the action (if LLM specified one)
                        tp_price = getattr(action, "take_profit_price", 0.0) or 0.0

                        # Pass ATR for dynamic trail distance
                        current_atr = getattr(state, "atr", 0.0) if state else 0.0

                        # Use actual fill price from position tracker, not
                        # the stale state.last_price from before debate.
                        # This prevents stop being anchored to a 20s-old price.
                        fill_price = new_position.avg_entry

                        # Recalculate stop from actual fill if it was based on stale price
                        if stop_price > 0 and fill_price != last_price:
                            stop_dist = abs(last_price - stop_price)
                            if new_position.side == Side.LONG:
                                stop_price = round(fill_price - stop_dist, 2)
                            else:
                                stop_price = round(fill_price + stop_dist, 2)
                            # Also update order manager's tracked stop
                            if isinstance(self._order_manager, QuantLynkOrderManager):
                                self._order_manager.current_stop_price = stop_price

                        # Detect ETH session for tighter trailing
                        current_phase = clock.get_session_phase()
                        is_eth = clock.is_eth(current_phase)
                        trading_cfg = self._config.trading

                        # Build partial close callback
                        partial_fn2 = None
                        partial_target2 = trading_cfg.eth_partial_profit_points if is_eth else trading_cfg.partial_profit_points
                        if isinstance(self._order_manager, QuantLynkOrderManager):
                            async def _partial_close_fn2(side: str, quantity: int, price: float) -> None:
                                await self._order_manager._client.partial_close(side=side, quantity=quantity, price=price)
                            partial_fn2 = _partial_close_fn2

                        # Dynamic partial quantity: half of entry size, min 1
                        partial_qty2 = max(1, action.quantity // 2) if action.quantity else 1

                        self._tick_stop_monitor.activate(
                            side=new_position.side.value,
                            entry_price=fill_price,
                            stop_price=stop_price,
                            take_profit_price=tp_price,
                            atr=current_atr,
                            is_eth=is_eth,
                            eth_trail_distance=trading_cfg.eth_trail_distance,
                            eth_trail_activation=trading_cfg.eth_trail_activation_points,
                            eth_mid_tighten_at_profit=trading_cfg.eth_mid_tighten_at_profit,
                            eth_mid_tightened_distance=trading_cfg.eth_mid_tightened_distance,
                            eth_tighten_at_profit=trading_cfg.eth_tighten_at_profit,
                            eth_tightened_distance=trading_cfg.eth_tightened_distance,
                            partial_target_points=partial_target2,
                            partial_fn=partial_fn2,
                            partial_quantity=partial_qty2,
                            breakeven_offset=trading_cfg.partial_breakeven_offset,
                        )
                        logger.info(
                            "orchestrator.tick_stop_monitor_activated",
                            side=new_position.side.value,
                            entry=fill_price,
                            stale_price=last_price,
                            stop=stop_price,
                            take_profit=tp_price,
                            is_eth=is_eth,
                            partial_target=partial_target2,
                        )
                elif action.action in (ActionType.FLATTEN, ActionType.STOP_TRADING):
                    self._tick_stop_monitor.deactivate()

            # 7b. Log completed trade + update session P&L
            if action.action == ActionType.SCALE_OUT:
                # SCALE_OUT: Track partial P&L without creating a TradeRecord.
                # The position_tracker accumulates partials in realized_pnl;
                # when the position fully closes, record_trade() subtracts these
                # to avoid double-counting.
                pos_after = self._position_tracker.position
                if pos_after is not None:
                    # partial P&L = total realized so far (includes this SCALE_OUT)
                    # minus what we've already tracked
                    partial_pnl = pos_after.realized_pnl - self._session_ctrl.partial_pnl_accumulated
                    try:
                        self._session_ctrl.record_scale_out(partial_pnl)
                    except Exception:
                        logger.warning("orchestrator.scale_out_pnl_failed", exc_info=True)

                    # Sync session stats to state engine
                    if self._session_stats_fn:
                        try:
                            await self._session_stats_fn(
                                daily_pnl=self._session_ctrl.daily_pnl,
                                daily_trades=self._session_ctrl.total_trades,
                                daily_winners=self._session_ctrl.winners,
                                daily_losers=self._session_ctrl.losers,
                            )
                        except Exception:
                            logger.warning("orchestrator.session_stats_sync_failed", exc_info=True)

            elif action.action == ActionType.FLATTEN:
                last_trade = self._position_tracker.last_trade
                if last_trade:
                    # Update session controller (daily P&L, win/loss, limits)
                    try:
                        self._session_ctrl.record_trade(last_trade)
                    except Exception:
                        logger.warning("orchestrator.session_record_failed", exc_info=True)

                    # Log to trade journal
                    if self._trade_logger:
                        try:
                            self._trade_logger.log_trade(last_trade)
                            logger.info(
                                "orchestrator.trade_logged",
                                pnl=round(last_trade.pnl, 2),
                                daily_pnl=round(self._session_ctrl.daily_pnl, 2),
                            )
                        except Exception:
                            logger.warning("orchestrator.trade_log_failed", exc_info=True)

                    # Sync session stats to state engine (for max_daily_trades guardrail)
                    if self._session_stats_fn:
                        try:
                            await self._session_stats_fn(
                                daily_pnl=self._session_ctrl.daily_pnl,
                                daily_trades=self._session_ctrl.total_trades,
                                daily_winners=self._session_ctrl.winners,
                                daily_losers=self._session_ctrl.losers,
                            )
                        except Exception:
                            logger.warning("orchestrator.session_stats_sync_failed", exc_info=True)

            # 7c. Post-trade thesis tracking + cooldown state
            if action.action == ActionType.FLATTEN:
                last_trade = self._position_tracker.last_trade
                if last_trade and last_trade.pnl is not None:
                    self._thesis_tracker.append({
                        "exit_time": time.monotonic(),
                        "exit_price": state.last_price,
                        "side": last_trade.side.value,
                        "entry_price": last_trade.entry_price,
                        "pnl": last_trade.pnl,
                        "reasoning": last_trade.reasoning_entry,
                        "check_after_sec": 300,  # check 5 min after exit
                    })

                    # Track cooldown state for post-stopout protection
                    self._last_exit_time = time.monotonic()
                    self._last_exit_side = last_trade.side
                    if last_trade.pnl < 0:
                        self._last_exit_was_loss = True
                        if self._last_loss_side == last_trade.side:
                            self._consecutive_same_dir_losses += 1
                        else:
                            self._consecutive_same_dir_losses = 1
                            self._last_loss_side = last_trade.side
                    else:
                        self._last_exit_was_loss = False
                        self._consecutive_same_dir_losses = 0
                        self._last_loss_side = None

                    # Persist direction-loss state to session controller (survives restarts)
                    self._session_ctrl._last_loss_side = (
                        self._last_loss_side.value if self._last_loss_side else None
                    )
                    self._session_ctrl._consecutive_same_dir_losses = self._consecutive_same_dir_losses

            # 7d. Record level interaction for Reasoner memory
            self._record_level_interaction(action, state)

            # Publish execution event
            self._bus.publish_nowait(Event(
                type=EventType.ORDER_FILLED,
                data={
                    "action": action.action.value,
                    "result": exec_result,
                },
            ))

        except Exception as e:
            logger.error(
                "orchestrator.execution_failed",
                action=action.action.value,
                error=str(e),
            )
            self._errors += 1

    # ── Level Interaction Memory ────────────────────────────────────────────

    def _record_level_interaction(
        self, action: LLMAction, state: MarketState
    ) -> None:
        """Record a level interaction for the Reasoner's memory.

        When the LLM takes an action (ENTER, FLATTEN, etc.) near a key level,
        record what happened so future decisions can reference it:
        "Last time we were at PDH (19870), price rejected and dropped 15pts."
        """
        from src.agents.reasoner import LevelInteraction

        if action.action not in (ActionType.ENTER, ActionType.FLATTEN, ActionType.SCALE_OUT):
            return

        if not state.levels or state.last_price <= 0:
            return

        # Find the nearest key level
        level_map: dict[str, float] = {
            "PDH": state.levels.prior_day_high,
            "PDL": state.levels.prior_day_low,
            "ONH": state.levels.overnight_high,
            "ONL": state.levels.overnight_low,
            "SessionH": state.levels.session_high,
            "SessionL": state.levels.session_low,
            "VWAP": state.levels.vwap,
            "POC": state.levels.poc,
        }

        nearest_name = ""
        nearest_price = 0.0
        nearest_dist = float("inf")

        for name, price in level_map.items():
            if price > 0:
                dist = abs(state.last_price - price)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_name = name
                    nearest_price = price

        # Only record if within 10 points of a key level
        if nearest_dist > 10.0:
            return

        # Determine outcome description
        if action.action == ActionType.ENTER:
            outcome = f"entered_{action.side.value if action.side else 'unknown'}"
        elif action.action == ActionType.FLATTEN:
            outcome = "flattened"
        else:
            outcome = "scaled_out"

        delta = getattr(state, "cumulative_delta", 0.0) or 0.0

        interaction = LevelInteraction(
            level_name=nearest_name,
            level_price=nearest_price,
            visit_price=state.last_price,
            timestamp=time.time(),
            action_taken=action.action.value,
            outcome=outcome,
            delta_at_visit=delta,
        )

        self._reasoner.record_level_interaction(interaction)
        logger.debug(
            "orchestrator.level_interaction_recorded",
            level=nearest_name,
            price=nearest_price,
            action=action.action.value,
        )

    # ── Context Building ────────────────────────────────────────────────────

    def _build_extra_context(self) -> str:
        """Build situational context for the LLM.

        Provides awareness of:
        - Session P&L state and risk level
        - Recent trade outcomes (loss streaks, win streaks)
        - Time context within the session
        - Profit preservation status
        """
        parts: list[str] = []
        ctrl = self._session_ctrl

        # P&L state and risk awareness
        pnl = ctrl.daily_pnl
        if pnl <= -300:
            parts.append(
                f"WARNING: Daily P&L is ${pnl:.0f}, approaching -$400 limit. "
                f"Only A+ setups (confidence 0.7+). Protect remaining capital."
            )
        elif pnl <= -200:
            parts.append(
                f"CAUTION: Daily P&L is ${pnl:.0f}. Be selective — "
                f"only high-conviction setups with clear risk/reward."
            )
        elif pnl >= 400:
            parts.append(
                f"STRONG DAY: Daily P&L is +${pnl:.0f}. Profit preservation tier 2 active "
                f"(max {ctrl.effective_max_contracts} contracts). Protect this day — "
                f"only take setups with excellent R:R."
            )
        elif pnl >= 200:
            parts.append(
                f"GOOD DAY: Daily P&L is +${pnl:.0f}. Profit preservation tier 1 active "
                f"(max {ctrl.effective_max_contracts} contracts). Don't give back gains."
            )

        # Consecutive losers — informational only (guardrails enforce limits)
        losers = ctrl.consecutive_losers
        if losers >= 2:
            parts.append(
                f"Note: {losers} consecutive losers. Guardrails are active. "
                f"Focus on finding the NEXT high-quality setup — do NOT let "
                f"past losses prevent you from trading when signals align."
            )

        # Trade count — informational only
        trades = ctrl.total_trades
        if trades >= 20:
            parts.append(
                f"Trade count: {trades} today — approaching session limit."
            )

        # Win rate context — informational only
        if trades >= 3:
            winners = ctrl.winners
            win_rate = (winners / trades) * 100 if trades > 0 else 0
            parts.append(
                f"Session stats: {trades} trades, {winners}W/{ctrl.losers}L "
                f"({win_rate:.0f}% win rate)."
            )

        # Time-based exit awareness (stale position warning)
        position = self._position_tracker.position
        if position is not None:
            hold_min = position.hold_time_min
            reassess_min = self._config.trading.time_exit_reassess_min
            force_min = self._config.trading.time_exit_force_min
            if hold_min >= reassess_min:
                parts.append(
                    f"STALE POSITION: Held for {hold_min:.0f}min "
                    f"(reassess at {reassess_min}min, force exit at {force_min}min). "
                    f"If not making progress, strongly consider FLATTEN."
                )

        # Apex drawdown awareness
        if self._apex:
            dd_remaining = self._apex.account_state.drawdown_remaining
            dd_pct = self._apex.account_state.drawdown_remaining_pct
            if dd_pct < 0.50:
                parts.append(
                    f"APEX DANGER: Only ${dd_remaining:.0f} drawdown remaining "
                    f"({dd_pct:.0%}). Protect the account — trade tiny or stop."
                )
            elif dd_pct < 0.70:
                parts.append(
                    f"APEX CAUTION: ${dd_remaining:.0f} drawdown remaining "
                    f"({dd_pct:.0%}). Be conservative."
                )

        # Post-trade thesis tracking: check completed theses and report
        self._check_thesis_outcomes()
        if self._thesis_results:
            recent_theses = self._thesis_results[-3:]  # last 3
            correct = sum(1 for t in recent_theses if t.get("correct", False))
            total = len(recent_theses)
            last = recent_theses[-1]
            parts.append(
                f"THESIS TRACKING: {correct}/{total} recent theses were correct. "
                f"Last: {'CORRECT' if last.get('correct') else 'WRONG'} "
                f"({last.get('side', '?')} from {last.get('entry_price', 0)}, "
                f"target {'reached' if last.get('correct') else 'not reached'} within 5min of exit)."
            )

        if not parts:
            return ""

        return " | ".join(parts)

    def _check_thesis_outcomes(self) -> None:
        """Check pending thesis trackers against current price."""
        if not self._thesis_tracker:
            return
        state = self._get_market_state()
        if not state or state.last_price <= 0:
            return
        now = time.monotonic()
        remaining = []
        for thesis in self._thesis_tracker:
            elapsed = now - thesis["exit_time"]
            if elapsed >= thesis["check_after_sec"]:
                # Check if price moved in the thesis direction
                entry = thesis["entry_price"]
                exit_p = thesis["exit_price"]
                current = state.last_price
                side = thesis["side"]
                # "Correct" means price moved further in the entry direction
                if side == "long":
                    correct = current > exit_p + 5  # price went up 5+ pts from exit
                else:
                    correct = current < exit_p - 5  # price went down 5+ pts from exit
                self._thesis_results.append({
                    "side": side,
                    "entry_price": entry,
                    "exit_price": exit_p,
                    "price_after_5min": current,
                    "pnl": thesis["pnl"],
                    "correct": correct,
                })
                if len(self._thesis_results) > 10:
                    self._thesis_results = self._thesis_results[-10:]
                logger.info(
                    "orchestrator.thesis_result",
                    side=side,
                    correct=correct,
                    exit_price=exit_p,
                    price_now=current,
                    diff=round(current - exit_p, 2),
                )
            else:
                remaining.append(thesis)
        self._thesis_tracker = remaining

    # ── State Access ────────────────────────────────────────────────────────

    def _get_market_state(self) -> Optional[MarketState]:
        """Get the latest market state from the state provider."""
        if self._state_provider:
            return self._state_provider()
        return self._last_state

    def _get_cycle_interval(self) -> float:
        """Determine how long to sleep between decision cycles.

        Four tiers:
        - Flat, ETH: 45s (thin liquidity, save API costs)
        - Flat, RTH: 30s (no urgency)
        - In position: 10s (need to monitor)
        - Critical: 5s (near stop, high adverse excursion, or high volatility)
        """
        position = self._position_tracker.position
        trading = self._config.trading

        if position is None:
            # Slower cycle during extended hours when flat
            phase = clock.get_session_phase()
            if clock.is_eth(phase):
                return trading.state_update_interval_eth_no_position_sec
            return trading.state_update_interval_no_position_sec

        # Check for critical conditions
        if self._is_critical_condition(position):
            return trading.state_update_interval_critical_sec

        return trading.state_update_interval_in_position_sec

    def _is_critical_condition(self, position: PositionState) -> bool:
        """Check if the current position is in a critical state requiring faster updates.

        Critical conditions:
        - Price is within 5 points of the stop
        - Max adverse excursion exceeds $100 per contract
        - Position has been losing for more than 10 minutes
        """
        state = self._last_state
        if state is None or position.stop_price == 0.0:
            return False

        # Near stop: within 5 points
        stop_distance = abs(state.last_price - position.stop_price)
        if stop_distance <= 5.0:
            return True

        # Significant adverse excursion (worse than -$100/contract)
        if position.max_adverse < -100.0:
            return True

        return False

    # ── Time Checks ─────────────────────────────────────────────────────────

    def _is_pre_market_time(self) -> bool:
        """Check if we've reached pre-market analysis time."""
        sec = clock.seconds_until(self._pre_market_time)
        return sec <= 0

    def _is_trading_time(self) -> bool:
        """Check if we're within trading hours."""
        return clock.is_trading_hours(
            start=self._trading_start,
            end=self._trading_end,
        )

    def _is_past_trading_end(self) -> bool:
        """Check if we're outside the trading window.

        With cross-midnight sessions (18:05 → 16:50), we can't simply check
        if we're past a target time. Instead, invert the trading hours check.
        """
        return not clock.is_trading_hours(
            start=self._trading_start,
            end=self._trading_end,
        )

    def _is_past_hard_flatten(self) -> bool:
        """Check if we're past the hard flatten time."""
        return clock.is_past_hard_flatten(self._hard_flatten_time)

    # ── Background Loops ────────────────────────────────────────────────────

    async def _load_postmortem_lessons(self) -> None:
        """Load postmortem lessons from recent sessions into the Reasoner.

        This is the feedback loop: nightly postmortem → next day's LLM context.
        The Reasoner injects these lessons as extra context so the LLM
        learns from its own mistakes and successes across sessions.
        """
        import json as _json

        if not self._trade_logger:
            return

        try:
            # Get recent daily summaries from the trade journal
            recent_summaries = self._trade_logger.get_recent_summaries(limit=3)
            if not recent_summaries:
                logger.info("orchestrator.no_postmortem_data")
                return

            # Convert stored summaries to PostmortemResult objects
            postmortems: list[PostmortemResult] = []
            for summary in recent_summaries:
                raw_postmortem = summary.get("postmortem", "")

                # Try to parse structured JSON from the stored postmortem text
                pm_data: dict[str, Any] = {}
                if raw_postmortem:
                    try:
                        json_start = raw_postmortem.find("{")
                        json_end = raw_postmortem.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            pm_data = _json.loads(raw_postmortem[json_start:json_end])
                    except (_json.JSONDecodeError, TypeError):
                        pass

                pm = PostmortemResult(
                    grade=summary.get("grade", "") or pm_data.get("grade", "C"),
                    what_worked=pm_data.get("what_worked", []),
                    what_didnt_work=pm_data.get("what_didnt_work", []),
                    improvements=pm_data.get("improvements", []),
                    market_observations=pm_data.get("market_observations", []),
                    key_lesson=pm_data.get("key_lesson", ""),
                    tomorrow_focus=pm_data.get("tomorrow_focus", ""),
                )
                postmortems.append(pm)

            # Combine into compact lessons string
            lessons = combine_recent_lessons(postmortems, max_days=3)
            if lessons:
                self._reasoner.set_postmortem_lessons(lessons)
                logger.info(
                    "orchestrator.postmortem_lessons_loaded",
                    days=len(postmortems),
                    length=len(lessons),
                )
        except Exception:
            logger.warning("orchestrator.postmortem_load_failed", exc_info=True)

    async def _reconciliation_loop(self) -> None:
        """Periodically reconcile position state with REST every 30 seconds.

        The WebSocket is the fast path for position updates, but it can miss
        events. This loop is the slow-path safety net — REST is authoritative.
        """
        interval_sec = 30
        try:
            while not self._shutdown_event.is_set():
                await asyncio.sleep(interval_sec)
                if self._shutdown_event.is_set():
                    break

                # Only reconcile if we have a REST client
                rest = getattr(self._order_manager, "_rest", None)
                if rest is None:
                    continue

                try:
                    positions_data = await rest.get_positions()
                    await self._position_tracker.reconcile(positions_data or [])
                except Exception:
                    logger.warning(
                        "orchestrator.reconcile_failed", exc_info=True
                    )
        except asyncio.CancelledError:
            pass

    # ── Kill Switch & Emergency ─────────────────────────────────────────────

    async def _on_kill_switch(self, event: Event) -> None:
        """Handle kill switch activation — flatten immediately."""
        reason = event.data.get("reason", "Unknown")
        logger.critical("orchestrator.kill_switch_triggered", reason=reason)
        await self._hard_flatten(f"kill_switch: {reason}")
        self._shutdown_event.set()

    async def _on_daily_limit(self, event: Event) -> None:
        """Handle daily limit hit event — flatten immediately."""
        logger.warning("orchestrator.daily_limit_hit")
        await self._hard_flatten("daily_limit")
        self._session_ctrl.force_stop("Daily limit hit via event")

    async def _hard_flatten(self, reason: str) -> None:
        """Emergency flatten all positions."""
        position = self._position_tracker.position
        if position is None:
            logger.info("orchestrator.hard_flatten_skipped", reason="no position")
            return

        logger.warning("orchestrator.hard_flatten", reason=reason)

        # Deactivate tick stop monitor so it doesn't also send a flatten
        if self._tick_stop_monitor and self._tick_stop_monitor.is_active:
            self._tick_stop_monitor.deactivate()

        flatten_action = LLMAction(
            action=ActionType.FLATTEN,
            reasoning=f"Hard flatten: {reason}",
            confidence=1.0,
        )

        try:
            await self._order_manager.execute(
                action=flatten_action,
                position=position,
                last_price=self._last_state.last_price if self._last_state else position.avg_entry,
            )

            # Record the trade to session controller and journal
            last_trade = self._position_tracker.last_trade
            if last_trade:
                try:
                    self._session_ctrl.record_trade(last_trade)
                except Exception:
                    logger.warning("orchestrator.hard_flatten_session_record_failed", exc_info=True)
                if self._trade_logger:
                    try:
                        self._trade_logger.log_trade(last_trade)
                    except Exception:
                        logger.warning("orchestrator.hard_flatten_trade_log_failed", exc_info=True)
                # Trigger risk manager cooldown if the flatten was a loss
                if last_trade.pnl < 0 and self._risk_manager:
                    self._risk_manager.record_loss(datetime.now(UTC))

        except Exception:
            logger.exception("orchestrator.hard_flatten_failed")
            self._errors += 1

    # ── Shutdown ────────────────────────────────────────────────────────────

    async def _shutdown(self) -> None:
        """Execute the full shutdown sequence."""
        if self._state == OrchestratorState.STOPPED:
            return

        self._state = OrchestratorState.SHUTTING_DOWN
        logger.info("orchestrator.shutdown_started")

        # 1. Deactivate trail manager and tick stop monitor
        if self._trail_manager and self._trail_manager.is_active:
            self._trail_manager.deactivate()
        if self._tick_stop_monitor and self._tick_stop_monitor.is_active:
            self._tick_stop_monitor.deactivate()

        # 2. Hard flatten if we still have a position
        await self._hard_flatten("Shutdown sequence")

        # 3. Log daily summary to trade journal
        if self._trade_logger:
            try:
                self._trade_logger.log_daily_summary(
                    date=self._session_ctrl.session_date,
                    total_trades=self._session_ctrl.total_trades,
                    winners=self._session_ctrl.winners,
                    losers=self._session_ctrl.losers,
                    gross_pnl=self._session_ctrl.gross_pnl,
                    net_pnl=self._session_ctrl.daily_pnl,
                    commissions=self._session_ctrl.commissions,
                    max_drawdown=self._session_ctrl.max_drawdown,
                )
                logger.info("orchestrator.daily_summary_logged")
            except Exception:
                logger.exception("orchestrator.daily_summary_log_failed")

        # 3a. Record today's result to circuit breakers (for next session)
        if self._circuit_breakers:
            try:
                self._circuit_breakers.record_day(
                    self._session_ctrl.session_date,
                    self._session_ctrl.daily_pnl,
                )
            except Exception:
                logger.debug("orchestrator.circuit_breaker_record_failed", exc_info=True)

        # 4. Run end-of-day summary (postmortem)
        summary = ""
        if self._summary_fn:
            try:
                summary = await self._summary_fn(self._session_ctrl)
            except Exception:
                logger.exception("orchestrator.summary_failed")

        # 4a. Store postmortem grade in trade journal
        if self._trade_logger and summary:
            try:
                self._trade_logger.log_daily_summary(
                    date=self._session_ctrl.session_date,
                    total_trades=self._session_ctrl.total_trades,
                    winners=self._session_ctrl.winners,
                    losers=self._session_ctrl.losers,
                    gross_pnl=self._session_ctrl.gross_pnl,
                    net_pnl=self._session_ctrl.daily_pnl,
                    commissions=self._session_ctrl.commissions,
                    max_drawdown=self._session_ctrl.max_drawdown,
                    postmortem=summary,
                )
            except Exception:
                logger.debug("orchestrator.postmortem_log_failed", exc_info=True)

        # 4b. Check for contract rollover (daily halt window maintenance)
        if self._contract_roll_fn:
            try:
                await self._contract_roll_fn()
            except Exception:
                logger.debug("orchestrator.contract_roll_check_failed", exc_info=True)

        # 5. Flush data recorder
        if self._data_recorder:
            try:
                self._data_recorder.stop_session()
                logger.info("orchestrator.data_recorder_flushed")
            except Exception:
                logger.exception("orchestrator.data_recorder_flush_failed")

        # 6. Close learning subsystem connections
        if self._trade_logger:
            try:
                self._trade_logger.close()
            except Exception:
                pass

        if self._regime_tracker:
            try:
                self._regime_tracker.close()
            except Exception:
                pass

        # 7. Send session summary notification
        if self._alert_manager:
            try:
                session_summary = SessionSummary(
                    date=self._session_ctrl.session_date,
                    total_trades=self._session_ctrl.total_trades,
                    winners=self._session_ctrl.winners,
                    losers=self._session_ctrl.losers,
                    gross_pnl=self._session_ctrl.gross_pnl,
                    commissions=self._session_ctrl.commissions,
                    net_pnl=self._session_ctrl.daily_pnl,
                    max_drawdown=self._session_ctrl.max_drawdown,
                    postmortem=summary,
                )
                await self._alert_manager.send_session_summary(session_summary)
            except Exception:
                logger.exception("orchestrator.summary_notification_failed")

        # 8. Send shutdown notification
        if self._alert_manager:
            try:
                await self._alert_manager.send_shutdown(
                    reason=self._kill_switch.trigger_reason or "Normal shutdown",
                    daily_pnl=self._session_ctrl.daily_pnl,
                )
            except Exception:
                logger.exception("orchestrator.shutdown_notification_failed")

        # 9. Cancel background tasks
        all_tasks = list(self._tasks) + getattr(self, "_background_tasks", [])
        for task in all_tasks:
            if not task.done():
                task.cancel()
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        self._tasks.clear()
        if hasattr(self, "_background_tasks"):
            self._background_tasks.clear()

        self._state = OrchestratorState.STOPPED
        logger.info(
            "orchestrator.shutdown_complete",
            decisions=self._decisions_made,
            executed=self._actions_executed,
            blocked=self._actions_blocked,
            debates=self._debate_count,
            trail_updates=self._trail_updates,
            cycles=self._cycle_count,
            errors=self._errors,
        )

    # ── Stats ───────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        uptime = time.monotonic() - self._start_time if self._start_time else 0
        return {
            "state": self._state,
            "uptime_sec": round(uptime, 1),
            "decisions_made": self._decisions_made,
            "actions_executed": self._actions_executed,
            "actions_blocked": self._actions_blocked,
            "debate_count": self._debate_count,
            "trail_updates": self._trail_updates,
            "cycle_count": self._cycle_count,
            "errors": self._errors,
            "consecutive_errors": self._consecutive_errors,
            "session": self._session_ctrl.stats,
            "guardrails": self._guardrails.stats,
            "reasoner": self._reasoner.stats,
        }
