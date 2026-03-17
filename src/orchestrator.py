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
from datetime import UTC, datetime, timedelta
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
        self._max_consecutive_errors: int = 3  # kill switch threshold
        self._start_time: Optional[float] = None

        # Tasks
        self._tasks: list[asyncio.Task] = []

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
        # Reset tick processor VWAP/delta/volume for RTH session so
        # metrics are not polluted by overnight/pre-market data
        if self._rth_reset_fn:
            self._rth_reset_fn()
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

            # Run one decision cycle
            try:
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

    # ── Decision Cycle ──────────────────────────────────────────────────────

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

        # 4a. Bull/bear debate for ENTER decisions (higher quality entries)
        if action.action == ActionType.ENTER and self._debate:
            try:
                debate_result = await self._debate.quick_decide(
                    state,
                    detected_setups=detected_setups_text,
                    price_action_narrative=price_narrative,
                )
                self._debate_count += 1

                # Debate may override the original action
                action = debate_result.action

                logger.info(
                    "orchestrator.debate_complete",
                    action=action.action.value,
                    confidence=action.confidence,
                    latency_ms=debate_result.total_latency_ms,
                )

                # If debate says DO_NOTHING, respect it
                if action.action == ActionType.DO_NOTHING:
                    return

                # 4b. Stop-breach check: After debate (which takes 9-12s),
                # check if the live price has moved significantly against our
                # intended direction. If price moved more than 60% of the stop
                # distance, abort — we'd enter with too little cushion.
                if (
                    action.action == ActionType.ENTER
                    and action.side is not None
                    and action.stop_distance is not None
                    and state.last_price > 0
                ):
                    decision_price = state.last_price  # price at cycle start
                    stop_dist = action.stop_distance

                    # Get fresh price from tick_stop_monitor (updated every tick)
                    fresh_price = (
                        self._tick_stop_monitor._last_price
                        if self._tick_stop_monitor and self._tick_stop_monitor._last_price > 0
                        else decision_price
                    )

                    if action.side == Side.LONG:
                        price_moved_against = decision_price - fresh_price
                        if price_moved_against > stop_dist * 0.6:
                            logger.warning(
                                "orchestrator.stale_entry_aborted",
                                side="long",
                                decision_price=decision_price,
                                fresh_price=fresh_price,
                                moved_against=round(price_moved_against, 2),
                                stop_dist=stop_dist,
                                reason="Price fell too far during debate — would enter near stop",
                            )
                            return
                    elif action.side == Side.SHORT:
                        price_moved_against = fresh_price - decision_price
                        if price_moved_against > stop_dist * 0.6:
                            logger.warning(
                                "orchestrator.stale_entry_aborted",
                                side="short",
                                decision_price=decision_price,
                                fresh_price=fresh_price,
                                moved_against=round(price_moved_against, 2),
                                stop_dist=stop_dist,
                                reason="Price rose too far during debate — would enter near stop",
                            )
                            return

            except Exception:
                logger.warning("orchestrator.debate_failed", exc_info=True)
                # Fall through with original reasoner decision

        # 4c. Trail protection — don't let LLM flatten winners prematurely
        # Two layers of protection:
        # 1. Time-based: Block flattens in first 60s unless trade is losing
        # 2. Profit-based: If ANY profit exists, let trail manage the exit
        # The LLM consistently cuts winners at +1-3pts after 14-17 seconds
        # when the trail could ride them to +10-20pts.
        if (
            action.action == ActionType.FLATTEN
            and position is not None
        ):
            # Calculate current unrealized P&L
            if position.side == Side.LONG:
                unrealized_pts = state.last_price - position.avg_entry
            else:
                unrealized_pts = position.avg_entry - state.last_price

            # Layer 1: Time-based minimum hold
            # Don't flatten in first 60 seconds unless trade is losing (< -2pts)
            min_hold_sec = 60
            if (
                position.time_in_trade_sec < min_hold_sec
                and unrealized_pts > -2.0
            ):
                logger.info(
                    "orchestrator.min_hold_suppressed_flatten",
                    time_in_trade=position.time_in_trade_sec,
                    min_hold=min_hold_sec,
                    unrealized_pts=round(unrealized_pts, 2),
                    reasoning=action.reasoning[:80] if action.reasoning else "",
                    msg="Too early to flatten — minimum hold period not met",
                )
                return

            # Layer 2: Trail-based profit protection
            # If trail is active and position is profitable at all, let trail manage exit
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
                        )
                        logger.info(
                            "orchestrator.tick_stop_monitor_activated",
                            side=new_position.side.value,
                            entry=fill_price,
                            stale_price=last_price,
                            stop=stop_price,
                            take_profit=tp_price,
                            is_eth=is_eth,
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

            # 7c. Post-trade thesis tracking: record exit for monitoring
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
        """Handle kill switch activation."""
        reason = event.data.get("reason", "Unknown")
        logger.critical("orchestrator.kill_switch_triggered", reason=reason)
        self._shutdown_event.set()

    async def _on_daily_limit(self, event: Event) -> None:
        """Handle daily limit hit event."""
        logger.warning("orchestrator.daily_limit_hit")
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
                last_price=self._last_state.last_price if self._last_state else 0.0,
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
