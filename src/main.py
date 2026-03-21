"""Entry point — bootstraps all components and runs the orchestrator.

Usage:
    python -m src.main              # Normal start (connects to Tradovate + Databento)
    python -m src.main --dry        # Dry run (no broker/data connection, simulated execution)
    python -m src.main --paper      # Paper trading (connects to Tradovate demo + Databento)

The startup sequence:
1. Load config from .env and parse CLI args
2. Initialize structlog
3. Create all components (EventBus, data layer, agents, execution, guardrails)
4. Authenticate with Tradovate (unless --dry)
5. Connect Databento live data stream (unless --dry)
6. Wire WebSocket event handlers
7. Start background tasks (EventBus drain, data streaming, multi-instrument polling)
8. Install signal handlers (SIGINT, SIGTERM)
9. Run orchestrator lifecycle
10. Graceful shutdown (flatten positions, disconnect, send summary)
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import sys
import signal
from pathlib import Path
from typing import Optional

import structlog

from src.core.config import AppConfig, load_config
from src.core.events import EventBus
from src.core.types import Event, EventType
from src.agents.bull_bear_debate import BullBearDebate
from src.agents.llm_client import LLMClient
from src.agents.pre_market_analyst import PreMarketAnalyst
from src.agents.reasoner import Reasoner
from src.agents.session_controller import SessionController
from src.data.databento_client import DatabentoClient
from src.data.economic_calendar import EconomicCalendar
from src.data.multi_instrument import MultiInstrumentPoller
from src.data.state_engine import StateEngine
from src.data.tick_processor import TickProcessor
from src.execution.kill_switch import KillSwitch
from src.execution.order_manager import OrderManager
from src.execution.position_tracker import PositionTracker
from src.execution.quantlynk_client import QuantLynkClient
from src.execution.quantlynk_order_manager import QuantLynkOrderManager
from src.execution.rate_limiter import RateLimiter
from src.execution.tradovate_auth import TradovateAuth
from src.execution.tradovate_rest import TradovateREST
from src.execution.tradovate_ws import TradovateWS
from src.execution.tick_stop_monitor import TickStopMonitor
from src.execution.trail_manager import TrailManager
from src.guardrails.apex_rules import ApexRuleGuardrail
from src.guardrails.circuit_breakers import CircuitBreakers
from src.guardrails.guardrail_engine import GuardrailEngine
from src.learning.kelly_calculator import KellyCalculator
from src.learning.postmortem import PostmortemAnalyzer
from src.learning.regime_tracker import RegimeTracker
from src.learning.trade_logger import TradeLogger
from src.notifications.alert_manager import AlertManager
from src.notifications.telegram_client import TelegramClient
from src.orchestrator import TradingOrchestrator
from src.replay.data_recorder import DataRecorder

# Optional enhanced modules
try:
    from src.data.setup_detector import SetupDetector
except ImportError:
    SetupDetector = None  # type: ignore[misc, assignment]

try:
    from src.data.price_action_analyzer import PriceActionAnalyzer
except ImportError:
    PriceActionAnalyzer = None  # type: ignore[misc, assignment]

logger = structlog.get_logger()

# ── Dual Instance Prevention ────────────────────────────────────────────────

_lock_fd = None  # global so the lock lives for the process lifetime


def _acquire_lock(lock_path: str = "/tmp/futures-trader.lock") -> bool:
    """Acquire an exclusive lock file to prevent dual instances.

    Uses fcntl.flock so the lock is auto-released if the process dies.
    Returns True if lock acquired, False if another instance is running.
    """
    global _lock_fd
    try:
        _lock_fd = open(lock_path, "w")
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_fd.write(f"{os.getpid()}\n")
        _lock_fd.flush()
        return True
    except (IOError, OSError):
        return False


def _release_lock() -> None:
    """Release the lock file."""
    global _lock_fd
    if _lock_fd:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_UN)
            _lock_fd.close()
        except Exception:
            pass
        _lock_fd = None


# ── Clock Drift Check ──────────────────────────────────────────────────────

async def _check_clock_drift(max_drift_sec: float = 5.0) -> bool:
    """Verify system clock is reasonably accurate.

    Makes an HTTP HEAD request to a well-known server and compares
    the Date header to local time. Returns True if within tolerance.
    """
    import aiohttp
    from datetime import datetime, timezone
    from email.utils import parsedate_to_datetime

    try:
        async with aiohttp.ClientSession() as session:
            async with session.head(
                "https://www.google.com",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                server_date_str = resp.headers.get("Date")
                if not server_date_str:
                    logger.warning("clock_drift.no_date_header")
                    return True  # Can't check, assume OK

                server_time = parsedate_to_datetime(server_date_str)
                local_time = datetime.now(timezone.utc)
                drift = abs((local_time - server_time).total_seconds())

                if drift > max_drift_sec:
                    logger.error(
                        "clock_drift.excessive",
                        drift_sec=round(drift, 2),
                        max_allowed=max_drift_sec,
                        server_time=server_time.isoformat(),
                        local_time=local_time.isoformat(),
                    )
                    return False

                logger.info("clock_drift.ok", drift_sec=round(drift, 2))
                return True
    except Exception:
        logger.warning("clock_drift.check_failed", exc_info=True)
        return True  # Can't reach server, don't block startup


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MNQ Futures Trading System")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Dry run mode — no broker connection, simulated execution",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading — connects to Tradovate demo + live Databento data",
    )
    return parser.parse_args()


def _setup_logging(level: str = "INFO") -> None:
    """Configure structlog with console output."""
    import logging

    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _build_components(config: AppConfig, dry_run: bool = False) -> dict:
    """Construct all system components from config.

    Args:
        config: Application configuration.
        dry_run: If True, skip broker/data connections. OrderManager runs
                 in simulation mode.

    Returns a dict of named components for the orchestrator.
    """
    trading = config.trading

    # ── Core ────────────────────────────────────────────────────────────────

    event_bus = EventBus()

    # ── Data Layer ──────────────────────────────────────────────────────────

    tick_processor = TickProcessor(target_symbol=config.trading.symbol)

    multi_instrument = MultiInstrumentPoller()

    calendar = EconomicCalendar(finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""))

    state_engine = StateEngine(
        tick_processor=tick_processor,
        multi_instrument=multi_instrument,
        calendar=calendar,
        event_bus=event_bus,
        symbol=trading.symbol,
        point_value=trading.point_value,
        update_interval_no_position=trading.state_update_interval_no_position_sec,
        update_interval_in_position=trading.state_update_interval_in_position_sec,
        update_interval_critical=trading.state_update_interval_critical_sec,
    )

    # Databento live data client (None in dry run mode)
    databento_client: Optional[DatabentoClient] = None
    if not dry_run and config.databento.api_key:
        databento_client = DatabentoClient(
            db_config=config.databento,
            trading_config=trading,
        )

    # ── LLM / Agents ───────────────────────────────────────────────────────

    llm_client = LLMClient(
        api_key=config.anthropic.api_key,
        haiku_model=config.anthropic.haiku_model,
        sonnet_model=config.anthropic.sonnet_model,
        max_retries=config.anthropic.max_retries,
        timeout_sec=config.anthropic.timeout_sec,
        daily_cost_cap=config.anthropic.daily_cost_cap,
    )

    reasoner = Reasoner(llm_client=llm_client)

    pre_market_analyst = PreMarketAnalyst(llm_client=llm_client)

    session_controller = SessionController(
        max_daily_loss=trading.max_daily_loss,
        commission_per_rt=trading.commission_per_rt,
        point_value=trading.point_value,
        profit_tier1_pnl=trading.profit_preservation_tier1_pnl,
        profit_tier1_max_size=trading.profit_preservation_tier1_max_size,
        profit_tier2_pnl=trading.profit_preservation_tier2_pnl,
        profit_tier2_max_size=trading.profit_preservation_tier2_max_size,
        base_max_contracts=trading.max_contracts,
    )

    # ── Execution ───────────────────────────────────────────────────────────

    rate_limiter = RateLimiter()

    # Determine execution backend: QuantLynk (webhook) or Tradovate (direct API)
    use_quantlynk = (
        config.quantlynk.enabled
        and config.quantlynk.webhook_url
        and config.quantlynk.user_id
    )

    # Tradovate auth + REST + WS are created here but connected later in run()
    tradovate_auth: Optional[TradovateAuth] = None
    tradovate_rest: Optional[TradovateREST] = None
    tradovate_ws: Optional[TradovateWS] = None

    # QuantLynk components (when using webhook execution)
    quantlynk_client: Optional[QuantLynkClient] = None

    position_tracker = PositionTracker(
        event_bus=event_bus,
        symbol=trading.symbol,
        point_value=trading.point_value,
    )

    if use_quantlynk and not dry_run:
        # ── QuantLynk execution backend ──────────────────────────────────
        logger.info(
            "main.execution_backend",
            backend="quantlynk",
            webhook_url=config.quantlynk.webhook_url[:30] + "...",
        )
        quantlynk_client = QuantLynkClient(config.quantlynk)

        order_manager = QuantLynkOrderManager(  # type: ignore[assignment]
            client=quantlynk_client,
            event_bus=event_bus,
            symbol=trading.symbol,
            default_stop_distance=10.0,
            point_value=trading.point_value,
            position_tracker=position_tracker,
        )

    else:
        # ── Tradovate direct API execution backend ───────────────────────
        if not dry_run and config.tradovate.username:
            logger.info("main.execution_backend", backend="tradovate")
            tradovate_auth = TradovateAuth(config.tradovate)
            tradovate_rest = TradovateREST(tradovate_auth, rate_limiter)

        order_manager = OrderManager(
            rest=tradovate_rest,  # type: ignore[arg-type]  # None in dry run
            event_bus=event_bus,
            account_id=0,  # set after auth
            symbol=trading.symbol,
            point_value=trading.point_value,
        )

    # Enable simulation mode in dry run
    if dry_run:
        order_manager.set_simulation_mode(True)

    # Build emergency flatten function for kill switch
    async def _emergency_flatten() -> dict:
        """Emergency flatten via the active order manager."""
        from src.core.types import ActionType, LLMAction
        action = LLMAction(
            action=ActionType.FLATTEN,
            reasoning="Kill switch emergency flatten",
            confidence=1.0,
        )
        return await order_manager.execute(
            action=action,
            position=position_tracker.position,
            last_price=0.0,  # market order — price doesn't matter
        )

    kill_switch = KillSwitch(
        event_bus=event_bus,
        flatten_fn=_emergency_flatten,
        flash_crash_points=trading.flash_crash_threshold_points,
        flash_crash_seconds=trading.flash_crash_window_sec,
        connection_timeout_sec=trading.connection_loss_max_sec,
        llm_failure_threshold=trading.llm_failure_max_consecutive,
        daily_loss_limit=trading.max_daily_loss,
    )

    # ── Guardrails ──────────────────────────────────────────────────────────

    guardrail_engine = GuardrailEngine(
        event_bus=event_bus,
        max_contracts=trading.max_contracts,
        max_stop_distance=trading.max_stop_points,
        max_stop_distance_eth=trading.eth_max_stop_points,
        blackout_minutes=trading.news_blackout_before_min,
        max_daily_trades=trading.max_daily_trades,
        max_contracts_eth=trading.max_contracts_eth,
    )

    # ── Notifications ───────────────────────────────────────────────────────

    telegram: Optional[TelegramClient] = None
    alert_manager: Optional[AlertManager] = None

    _tg_token = config.telegram.bot_token or ""
    _tg_chat = config.telegram.chat_id or ""
    _tg_configured = (
        _tg_token not in ("", "your_telegram_bot_token")
        and _tg_chat not in ("", "your_telegram_chat_id")
    )
    if _tg_configured:
        telegram = TelegramClient(
            bot_token=config.telegram.bot_token,
            chat_id=config.telegram.chat_id,
            throttle_sec=config.telegram.throttle_sec,
        )
        alert_manager = AlertManager(
            telegram=telegram,
            event_bus=event_bus,
        )

    # ── Wire position changes to state engine ─────────────────────────────
    # When tick_stop_monitor closes a position, the state_engine must learn
    # about it immediately so the LLM sees position=None on the next cycle.

    async def _sync_position_to_state_engine(event: Event) -> None:
        """Forward position changes from position_tracker to state_engine."""
        await state_engine.update_position(position_tracker.position)

    event_bus.subscribe(EventType.POSITION_CHANGED, _sync_position_to_state_engine)

    # ── Replay / Recording ─────────────────────────────────────────────────

    data_dir = Path(config.data_dir)

    data_recorder = DataRecorder(
        base_dir=str(data_dir / "recordings"),
    )

    # ── Learning ─────────────────────────────────────────────────────────

    journal_path = config.journal_path

    trade_logger = TradeLogger(db_path=journal_path)

    # Journal reset disabled for production — trade history must persist
    # across restarts for circuit breakers and P&L tracking.
    # Uncomment only for dev/testing:
    # if config.dry_run:
    #     trade_logger.reset_all()
    #     logger.info("main.journal_reset_for_testing")

    postmortem_analyzer = PostmortemAnalyzer(llm_client=llm_client)

    regime_tracker = RegimeTracker(db_path=journal_path)

    kelly_calculator = KellyCalculator(
        min_trades=50,
        max_contracts=trading.max_contracts,
    )

    # ── Enhanced Execution ───────────────────────────────────────────────

    trail_manager = TrailManager(
        trail_distance=15.0,
        batch_points=trading.trail_min_move_points,
        activation_profit_pts=10.0,
    )

    # ── Tick-Level Stop Monitor (QuantLynk only) ────────────────────────
    # Fires on every Databento trade tick. When stop or TP is hit,
    # immediately sends flatten via QuantLynk — no waiting for the
    # 10-30 second LLM decision cycle.
    tick_stop_monitor: Optional[TickStopMonitor] = None
    if use_quantlynk and quantlynk_client is not None:
        async def _tick_flatten(price: float) -> Any:
            """Flatten via QuantLynk — called by TickStopMonitor on stop/TP hit."""
            return await quantlynk_client.flatten(price=price)

        tick_stop_monitor = TickStopMonitor(
            flatten_fn=_tick_flatten,
            target_symbol=config.trading.symbol,
            trail_distance=15.0,
            trail_activation_points=12.0,    # wait for 12pt profit before trailing
            min_stop_distance=7.0,           # never trail closer than 7pts
            tighten_at_profit=20.0,          # tight tier at 20pt profit
            tightened_distance=5.0,          # 5pt trail when in big profit
            mid_tighten_at_profit=12.0,      # mid tier at 12pt profit
            mid_tightened_distance=7.0,      # 7pt trail at moderate profit
        )
        # Restore stop/trail state from disk if process crashed while in position
        if tick_stop_monitor.load_from_disk():
            logger.warning(
                "main.tick_stop_restored_from_disk",
                side=tick_stop_monitor._side,
                stop=tick_stop_monitor._stop_price,
                tp=tick_stop_monitor._take_profit_price,
            )

    bull_bear_debate = BullBearDebate(llm_client=llm_client)

    # ── Enhanced Intelligence (optional) ─────────────────────────────────

    setup_detector = SetupDetector() if SetupDetector else None
    price_action_analyzer = PriceActionAnalyzer() if PriceActionAnalyzer else None

    # ── HTF Structure Levels ────────────────────────────────────────────────

    from src.indicators.structure_levels import StructureLevelManager

    structure_manager = StructureLevelManager()

    # ── Confluence Strategy Components ─────────────────────────────────────

    from src.indicators.confluence import ConfluenceEngine
    from src.indicators.order_flow import OrderFlowEngine
    from src.execution.risk_manager import RiskManager

    confluence_engine = ConfluenceEngine(structure_manager=structure_manager)
    order_flow_engine = OrderFlowEngine()
    risk_manager = RiskManager(daily_loss_limit=trading.max_daily_loss)
    risk_manager.load_persisted_state()

    # Wire order flow engine to state engine for enriched MarketState
    state_engine.set_confluence_engine(confluence_engine)
    state_engine.set_order_flow_engine(order_flow_engine)

    # Wire depth feed (mbp-10) to order flow engine
    if databento_client is not None:
        async def _on_depth(data: dict) -> None:
            """Forward depth updates to order flow engine."""
            if "levels" in data:
                order_flow_engine.update_depth(data["levels"])

        async def _on_trade_for_flow(data: dict) -> None:
            """Forward trades to order flow engine for VPIN/entropy/absorption."""
            if data.get("symbol", "").startswith(trading.symbol[:3]):
                order_flow_engine.update_trade(
                    price=data["price"],
                    size=data["size"],
                    direction=data.get("direction", "unknown"),
                    timestamp=data["timestamp"],
                )

        async def _on_quote_for_flow(data: dict) -> None:
            """Forward BBO quotes to order flow engine as DOM fallback.

            When mbp-10 isn't authorized, this provides a single-level
            bid/ask imbalance from the BBO data that's always available.
            """
            bid_sz = data.get("bid_size", 0)
            ask_sz = data.get("ask_size", 0)
            if bid_sz > 0 or ask_sz > 0:
                order_flow_engine.update_bbo(bid_sz, ask_sz)

        databento_client.on_depth(_on_depth)
        databento_client.on_trade(_on_trade_for_flow)
        databento_client.on_quote(_on_quote_for_flow)

    # ── Circuit Breakers (multi-day/week/month) ────────────────────────

    circuit_breakers = CircuitBreakers(
        base_max_contracts=trading.max_contracts,
        weekly_loss_limit=trading.max_weekly_loss,
        monthly_loss_limit=trading.max_monthly_loss,
    )

    # Load historical daily P&L from trade journal
    if trade_logger:
        try:
            daily_results = trade_logger.get_daily_pnl_history(limit=60)
            if daily_results:
                circuit_breakers.load_history(daily_results)
        except Exception:
            pass  # First run or empty journal — no history

    # ── Apex Trader Funding Compliance ─────────────────────────────────

    apex_guardrail = None
    if trading.apex_enabled:
        from datetime import time as time_type

        parts = trading.apex_flatten_deadline.split(":")
        flatten_time = time_type(int(parts[0]), int(parts[1]))
        apex_guardrail = ApexRuleGuardrail(
            account_type=trading.apex_account_type,
            flatten_deadline_et=flatten_time,
            drawdown_lockout_pct=trading.apex_drawdown_lockout_pct,
        )
        logger.info(
            "main.apex_enabled",
            account_type=trading.apex_account_type,
            flatten_deadline=trading.apex_flatten_deadline,
            drawdown_lockout=trading.apex_drawdown_lockout_pct,
        )

    return {
        "event_bus": event_bus,
        "tick_processor": tick_processor,
        "multi_instrument": multi_instrument,
        "calendar": calendar,
        "state_engine": state_engine,
        "databento_client": databento_client,
        "llm_client": llm_client,
        "reasoner": reasoner,
        "pre_market_analyst": pre_market_analyst,
        "session_controller": session_controller,
        "rate_limiter": rate_limiter,
        "tradovate_auth": tradovate_auth,
        "tradovate_rest": tradovate_rest,
        "tradovate_ws": tradovate_ws,
        "quantlynk_client": quantlynk_client,
        "use_quantlynk": use_quantlynk,
        "position_tracker": position_tracker,
        "order_manager": order_manager,
        "kill_switch": kill_switch,
        "guardrail_engine": guardrail_engine,
        "telegram": telegram,
        "alert_manager": alert_manager,
        "data_recorder": data_recorder,
        "trade_logger": trade_logger,
        "postmortem_analyzer": postmortem_analyzer,
        "regime_tracker": regime_tracker,
        "kelly_calculator": kelly_calculator,
        "trail_manager": trail_manager,
        "tick_stop_monitor": tick_stop_monitor,
        "bull_bear_debate": bull_bear_debate,
        "setup_detector": setup_detector,
        "price_action_analyzer": price_action_analyzer,
        "circuit_breakers": circuit_breakers,
        "apex_guardrail": apex_guardrail,
        "confluence_engine": confluence_engine,
        "order_flow_engine": order_flow_engine,
        "risk_manager": risk_manager,
        "structure_manager": structure_manager,
    }


def _make_pre_market_fn(analyst, state_engine) -> Optional[callable]:
    """Create a zero-arg async wrapper around PreMarketAnalyst.analyze.

    The analyst requires specific parameters (prior_day_high, etc.) that are
    only available from the state engine at pre-market time. This wrapper
    pulls the latest values and passes them in.
    """
    if not hasattr(analyst, "analyze"):
        return None

    async def _pre_market() -> str:
        state = state_engine.last_state
        if state is None:
            return ""

        levels = state.levels if state.levels else None
        return await analyst.analyze(
            prior_day_high=levels.prior_day_high if levels else 0.0,
            prior_day_low=levels.prior_day_low if levels else 0.0,
            prior_day_close=levels.prior_day_close if levels else 0.0,
            overnight_high=levels.overnight_high if levels else 0.0,
            overnight_low=levels.overnight_low if levels else 0.0,
            current_price=state.last_price,
            events=getattr(state, "upcoming_events", None),
            prior_day_summary=state.game_plan or "",
        )

    return _pre_market


async def _connect_tradovate(
    components: dict,
    config: AppConfig,
) -> None:
    """Authenticate with Tradovate and wire up REST + WebSocket.

    After auth completes, updates the OrderManager and PositionTracker
    with real credentials and starts the WebSocket event stream.
    """
    auth: TradovateAuth = components["tradovate_auth"]
    rest: TradovateREST = components["tradovate_rest"]
    order_manager: OrderManager = components["order_manager"]
    position_tracker: PositionTracker = components["position_tracker"]
    event_bus: EventBus = components["event_bus"]

    # 1. Authenticate
    logger.info("main.tradovate_authenticating")
    await auth.authenticate()
    logger.info(
        "main.tradovate_authenticated",
        account_id=auth.account_id,
        user_id=auth.user_id,
    )

    # 2. Wire OrderManager with real REST client and account ID
    order_manager._rest = rest  # type: ignore[attr-defined]
    order_manager._account_id = auth.account_id  # type: ignore[attr-defined]

    # 3. Initialize REST session
    await rest.open()

    # 4. Sync existing positions from REST (recovery case)
    try:
        positions = await rest.get_positions()
        if positions:
            logger.warning(
                "main.existing_positions_found",
                count=len(positions),
                positions=[
                    {
                        "symbol": p.get("contractId"),
                        "qty": p.get("netPos"),
                    }
                    for p in positions
                ],
            )
            # Sync position tracker with existing positions
            for pos in positions:
                position_tracker.sync_from_rest(pos)
    except Exception:
        logger.warning("main.position_sync_failed", exc_info=True)

    # 5. Connect WebSocket for real-time order/position events
    ws = TradovateWS(
        ws_url=config.tradovate.ws_url,
        access_token=auth.access_token,
        heartbeat_interval=config.tradovate.heartbeat_interval_sec,
    )
    components["tradovate_ws"] = ws

    # Register WS event handlers — fills update position tracker
    ws.on_fill(position_tracker.on_fill)

    await ws.connect()
    logger.info("main.tradovate_ws_connected")


async def _connect_databento(
    components: dict,
    config: AppConfig,
) -> None:
    """Connect to Databento and wire data callbacks to the tick processor."""
    client: DatabentoClient = components["databento_client"]
    tick_processor: TickProcessor = components["tick_processor"]
    state_engine: StateEngine = components["state_engine"]

    # Register callbacks: Databento → TickProcessor → StateEngine
    client.on_trade(tick_processor.process_trade)
    client.on_quote(tick_processor.process_quote)

    # Register bar callback for state engine
    tick_processor.on_bar(state_engine.on_bar_completed)

    # Wire TickStopMonitor as a Databento trade handler (QuantLynk only).
    # This fires on EVERY trade tick, checking stop/TP levels and sending
    # flatten immediately — no waiting for the LLM decision cycle.
    tick_stop_monitor: Optional[TickStopMonitor] = components.get("tick_stop_monitor")
    if tick_stop_monitor is not None:
        client.on_trade(tick_stop_monitor.on_trade)
        logger.info("main.tick_stop_monitor_wired")

    await client.connect()
    logger.info("main.databento_connected")


async def run(config: Optional[AppConfig] = None, dry_run: bool = False) -> None:
    """Build components, wire the orchestrator, and run it.

    Args:
        config: Optional pre-built config (for testing).
        dry_run: If True, skip all external connections.
    """
    if config is None:
        config = load_config()

    _setup_logging(config.log_level)

    # ── Dual instance prevention ─────────────────────────────────────────
    if not _acquire_lock():
        logger.critical("main.dual_instance_detected",
                        msg="Another instance is already running. Exiting.")
        return

    # ── Clock drift check ────────────────────────────────────────────────
    if not dry_run:
        clock_ok = await _check_clock_drift(max_drift_sec=5.0)
        if not clock_ok:
            logger.critical("main.clock_drift_too_large",
                            msg="System clock may be inaccurate. Fix NTP and retry.")
            _release_lock()
            return

    use_ql = config.quantlynk.enabled and config.quantlynk.webhook_url and config.quantlynk.user_id
    if dry_run:
        mode = "DRY RUN"
    elif use_ql:
        mode = "QUANTLYNK"
    elif config.tradovate.use_demo:
        mode = "DEMO"
    else:
        mode = "LIVE"
    logger.info("main.starting", symbol=config.trading.symbol, mode=mode)

    components = _build_components(config, dry_run=dry_run)

    state_engine: StateEngine = components["state_engine"]
    trade_logger: TradeLogger = components["trade_logger"]
    postmortem_analyzer: PostmortemAnalyzer = components["postmortem_analyzer"]
    event_bus: EventBus = components["event_bus"]

    # ── Load RVOL baseline (from file or compute) ────────────────────────
    rvol_path = Path(config.data_dir) / "rvol_baseline.json"
    if rvol_path.exists():
        try:
            state_engine.load_rvol_baseline(str(rvol_path))
            logger.info("main.rvol_baseline_loaded", path=str(rvol_path))
        except Exception:
            logger.warning("main.rvol_baseline_load_failed", exc_info=True)
    else:
        logger.info(
            "main.rvol_baseline_missing",
            path=str(rvol_path),
            hint="Run: python -m src.scripts.compute_rvol to generate baseline",
        )

    # Background tasks we manage
    background_tasks: list[asyncio.Task] = []

    try:
        # ── Step 1: Start EventBus drain loop ────────────────────────────────
        bus_task = asyncio.create_task(event_bus.run(), name="event_bus")
        background_tasks.append(bus_task)

        # ── Step 2: Connect execution backend ─────────────────────────────────
        if not dry_run and components.get("use_quantlynk"):
            # QuantLynk: just open the HTTP session (no WebSocket needed)
            ql_client: QuantLynkClient = components["quantlynk_client"]
            await ql_client.connect()
            logger.info("main.quantlynk_connected")

        elif not dry_run and components["tradovate_auth"]:
            # Tradovate: full auth + REST + WebSocket
            await _connect_tradovate(components, config)

            # Start WS event loop in background
            ws = components["tradovate_ws"]
            if ws:
                ws_task = asyncio.create_task(ws.run(), name="tradovate_ws")
                background_tasks.append(ws_task)

        # ── Step 2b: Verify front-month contract via Databento ────────────────
        # Also checked during daily halt (_contract_roll_fn in shutdown).
        # At startup this is a no-op if the symbol already matches.
        if not dry_run and config.databento.api_key:
            try:
                from src.data.databento_client import DatabentoClient as _DBC

                front_month = _DBC.resolve_front_month(api_key=config.databento.api_key)
                if front_month and front_month != config.trading.symbol:
                    # Only roll FORWARD — never backwards to an expired contract
                    if _DBC.is_forward_roll(config.trading.symbol, front_month):
                        old_symbol = config.trading.symbol
                        config.trading.symbol = front_month
                        logger.info(
                            "main.contract_auto_rolled",
                            old_symbol=old_symbol,
                            new_symbol=front_month,
                            msg=f"Auto-rolled contract: {old_symbol} → {front_month}",
                        )
                    else:
                        logger.info(
                            "main.contract_roll_skipped_backward",
                            current=config.trading.symbol,
                            resolved=front_month,
                            msg="Resolved contract is older — keeping current",
                        )
                elif front_month:
                    logger.info("main.contract_symbol_confirmed", symbol=front_month)
            except Exception:
                logger.warning(
                    "main.contract_resolve_error",
                    exc_info=True,
                    msg=f"Front-month resolution failed — using configured symbol {config.trading.symbol}",
                )

        # ── Step 3: Connect to Databento (unless dry run) ────────────────────
        if not dry_run and components["databento_client"]:
            await _connect_databento(components, config)

            # Start data streaming in background
            db_client = components["databento_client"]
            stream_task = asyncio.create_task(
                db_client.stream(), name="databento_stream"
            )
            background_tasks.append(stream_task)

        # ── Step 4: Start multi-instrument polling (ES, TICK, VIX, DXY) ──────
        if not dry_run:
            multi_instrument: MultiInstrumentPoller = components["multi_instrument"]
            await multi_instrument.start()
            logger.info("main.multi_instrument_started")

        # ── Step 4a: Load prior day levels from Databento historical ─────
        if not dry_run and components["databento_client"]:
            try:
                await _load_prior_day_levels(state_engine, config)
            except Exception:
                logger.warning("main.prior_day_levels_failed", exc_info=True)

        # ── Step 4b: Start state engine compute loop ────────────────────
        await state_engine.start()
        se_task = asyncio.create_task(
            _state_engine_watchdog(state_engine), name="state_engine_watchdog"
        )
        background_tasks.append(se_task)
        logger.info("main.state_engine_started")

        # Reload persisted bars on startup so EMAs/OBs warm instantly
        # instead of rebuilding from scratch over 30-90 minutes.
        state_engine.reload_persisted_bars()

        # ── Step 4c: Load HTF structure levels from Databento historical ──
        structure_manager = components.get("structure_manager")
        if not dry_run and components["databento_client"] and config.databento.api_key and structure_manager:
            try:
                from src.data.databento_client import DatabentoClient as _DBC

                htf_bars_1h = await _DBC.fetch_htf_bars(
                    api_key=config.databento.api_key, timeframe="1h", lookback_days=60,
                )
                htf_bars_4h = await _DBC.fetch_htf_bars(
                    api_key=config.databento.api_key, timeframe="4h", lookback_days=180,
                )
                htf_bars_daily = await _DBC.fetch_htf_bars(
                    api_key=config.databento.api_key, timeframe="1d", lookback_days=365,
                )
                htf_bars_weekly = await _DBC.fetch_htf_bars(
                    api_key=config.databento.api_key, timeframe="1w", lookback_days=730,
                )

                structure_manager.compute_levels(htf_bars_1h, "1h")
                structure_manager.compute_levels(htf_bars_4h, "4h")
                structure_manager.compute_levels(htf_bars_daily, "D")
                structure_manager.compute_levels(htf_bars_weekly, "W")

                all_levels = structure_manager.levels
                logger.info(
                    "main.htf_structure_levels_loaded",
                    levels_1h=len([l for l in all_levels if l.timeframe == "1h"]),
                    levels_4h=len([l for l in all_levels if l.timeframe == "4h"]),
                    levels_D=len([l for l in all_levels if l.timeframe == "D"]),
                    levels_W=len([l for l in all_levels if l.timeframe == "W"]),
                )
            except Exception:
                logger.warning("main.htf_fetch_failed", exc_info=True,
                             msg="Running without HTF structure levels")

        # Wire 1h bar updates to structure level manager
        if structure_manager and hasattr(state_engine, 'register_1h_bar_callback'):
            def _on_1h_bar(bar: dict) -> None:
                structure_manager.update_on_bar_close(bar, "1h")
            state_engine.register_1h_bar_callback(_on_1h_bar)

        # ── Step 4d: Pre-Market Context ──────────────────────────────────────
        from src.intelligence.pre_market_context import PreMarketContextGenerator

        pre_market_ctx_gen = PreMarketContextGenerator(
            llm_client=components["llm_client"],
            calendar_path="config/economic_events.json",
        )
        pre_market_context = await pre_market_ctx_gen.generate()

        # ── Step 5: Build orchestrator ───────────────────────────────────────

        # Build end-of-day summary function
        async def _summary_fn(session_ctrl: SessionController) -> str:
            """Run nightly postmortem analysis and return summary text."""
            try:
                trades = trade_logger.get_trades_by_date(session_ctrl.session_date)
                daily_stats = {
                    "date": session_ctrl.session_date,
                    "total_trades": session_ctrl.total_trades,
                    "winners": session_ctrl.winners,
                    "losers": session_ctrl.losers,
                    "gross_pnl": session_ctrl.gross_pnl,
                    "net_pnl": session_ctrl.daily_pnl,
                    "commissions": session_ctrl.commissions,
                    "max_drawdown": session_ctrl.max_drawdown,
                    "win_rate": (
                        session_ctrl.winners / session_ctrl.total_trades
                        if session_ctrl.total_trades > 0
                        else 0.0
                    ),
                }
                result = await postmortem_analyzer.analyze(trades, daily_stats)
                return result.to_summary_text()
            except Exception:
                logger.exception("main.postmortem_failed")
                return ""

        # Build contract roll check callback for daily halt window
        async def _contract_roll_fn() -> None:
            """Check if front-month contract has changed during daily halt."""
            if not config.databento.api_key:
                return
            try:
                from src.data.databento_client import DatabentoClient as _DBC

                front_month = _DBC.resolve_front_month(api_key=config.databento.api_key)
                if not front_month:
                    return
                if front_month != config.trading.symbol:
                    if not _DBC.is_forward_roll(config.trading.symbol, front_month):
                        logger.info(
                            "main.contract_roll_skipped_backward",
                            current=config.trading.symbol,
                            resolved=front_month,
                        )
                        return
                    old_symbol = config.trading.symbol
                    config.trading.symbol = front_month
                    # Also update the live Databento client's subscription list
                    db_client = components.get("databento_client")
                    if db_client:
                        db_client.update_symbol(front_month)
                    logger.info(
                        "main.contract_rolled_at_halt",
                        old_symbol=old_symbol,
                        new_symbol=front_month,
                        msg=f"Contract rolled during halt: {old_symbol} → {front_month}",
                    )
            except Exception:
                logger.debug("main.contract_roll_check_failed", exc_info=True)

        orchestrator = TradingOrchestrator(
            config=config,
            event_bus=event_bus,
            reasoner=components["reasoner"],
            guardrail_engine=components["guardrail_engine"],
            order_manager=components["order_manager"],
            position_tracker=components["position_tracker"],
            session_controller=components["session_controller"],
            kill_switch=components["kill_switch"],
            alert_manager=components["alert_manager"],
            state_provider=lambda: state_engine.last_state,
            pre_market_fn=_make_pre_market_fn(
                components["pre_market_analyst"], state_engine
            ),
            summary_fn=_summary_fn,
            # Phase 10/11 components
            data_recorder=components["data_recorder"],
            trade_logger=trade_logger,
            regime_tracker=components["regime_tracker"],
            kelly_calculator=components["kelly_calculator"],
            trail_manager=components["trail_manager"],
            tick_stop_monitor=components.get("tick_stop_monitor"),
            bull_bear_debate=components["bull_bear_debate"],
            setup_detector=components.get("setup_detector"),
            price_action_analyzer=components.get("price_action_analyzer"),
            circuit_breakers=components.get("circuit_breakers"),
            apex_guardrail=components.get("apex_guardrail"),
            # New confluence strategy components
            confluence_engine=components.get("confluence_engine"),
            order_flow_engine=components.get("order_flow_engine"),
            risk_manager=components.get("risk_manager"),
            rth_reset_fn=lambda: (
                state_engine.reset_for_rth(),
                state_engine.reload_persisted_bars(),
            ),
            session_stats_fn=state_engine.update_session_stats,
            contract_roll_fn=_contract_roll_fn,
        )

        # Set pre-market context on orchestrator
        orchestrator.set_pre_market_context(pre_market_context)

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()

        def _signal_handler(sig: signal.Signals) -> None:
            logger.info("main.signal_received", signal=sig.name)
            loop.create_task(orchestrator.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler, sig)

        # ── Step 6: Run orchestrator (blocks until shutdown) ─────────────────
        await orchestrator.start()

        logger.info("main.exiting", stats=orchestrator.stats)

    finally:
        # ── Graceful cleanup ──────────────────────────────────────────────────

        # Stop state engine
        try:
            await state_engine.stop()
        except Exception:
            pass

        # Stop multi-instrument polling
        multi_instrument = components.get("multi_instrument")
        if multi_instrument:
            try:
                await multi_instrument.stop()
            except Exception:
                pass

        # Close Databento
        databento_client = components.get("databento_client")
        if databento_client:
            try:
                await databento_client.close()
            except Exception:
                pass

        # Close QuantLynk HTTP session
        ql_client = components.get("quantlynk_client")
        if ql_client and hasattr(ql_client, "close"):
            try:
                await ql_client.close()
            except Exception:
                pass

        # Close Tradovate WebSocket
        tradovate_ws = components.get("tradovate_ws")
        if tradovate_ws and hasattr(tradovate_ws, "close"):
            try:
                await tradovate_ws.close()
            except Exception:
                pass

        # Close Tradovate REST session
        tradovate_rest = components.get("tradovate_rest")
        if tradovate_rest and hasattr(tradovate_rest, "close"):
            try:
                await tradovate_rest.close()
            except Exception:
                pass

        # Close Tradovate auth session
        tradovate_auth = components.get("tradovate_auth")
        if tradovate_auth and hasattr(tradovate_auth, "close"):
            try:
                await tradovate_auth.close()
            except Exception:
                pass

        # Stop EventBus
        try:
            await event_bus.stop()
        except Exception:
            pass

        # Cancel background tasks
        for task in background_tasks:
            if not task.done():
                task.cancel()
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

        _release_lock()
        logger.info("main.cleanup_complete")


async def _load_prior_day_levels(
    state_engine: StateEngine,
    config: AppConfig,
) -> None:
    """Fetch prior day levels from Databento historical API.

    Downloads yesterday's 1-minute bars to determine PDH/PDL/PDC,
    and overnight (Globex pre-market) high/low for ONH/ONL.
    """
    from datetime import date, timedelta

    today = date.today()
    yesterday = today - timedelta(days=1)
    # Skip weekends
    while yesterday.weekday() >= 5:  # Saturday=5, Sunday=6
        yesterday -= timedelta(days=1)

    try:
        # Historical API needs parent symbol (MNQ.FUT), not specific contract (MNQM6)
        hist_symbol = config.trading.symbol
        if not hist_symbol.endswith(".FUT"):
            # Strip contract month code: MNQM6 -> MNQ, then add .FUT
            import re
            root = re.sub(r"[FGHJKMNQUVXZ]\d+$", "", hist_symbol)
            hist_symbol = f"{root}.FUT" if root else hist_symbol

        trades = await DatabentoClient.fetch_historical(
            api_key=config.databento.api_key,
            symbol=hist_symbol,
            start=yesterday.isoformat(),
            end=today.isoformat(),
        )

        if not trades:
            logger.warning("main.no_prior_day_data", date=yesterday.isoformat())
            return

        prices = [t["price"] for t in trades if t.get("price", 0) > 0]
        if not prices:
            return

        # Filter outliers using median-based approach.
        # Some Databento historical records contain anomalous prices
        # (settlement marks, corrections, etc.) that would corrupt PDH/PDL.
        # Valid MNQ prices should be within 5% of the median session price.
        sorted_prices = sorted(prices)
        median_price = sorted_prices[len(sorted_prices) // 2]
        tolerance = 0.05  # 5% from median
        lower_bound = median_price * (1 - tolerance)
        upper_bound = median_price * (1 + tolerance)
        filtered_prices = [p for p in prices if lower_bound <= p <= upper_bound]

        if not filtered_prices:
            logger.warning(
                "main.all_prices_filtered_as_outliers",
                raw_count=len(prices),
                median=median_price,
            )
            return

        outlier_count = len(prices) - len(filtered_prices)
        if outlier_count > 0:
            logger.info(
                "main.price_outliers_filtered",
                outliers=outlier_count,
                total=len(prices),
                median=median_price,
                bounds=f"{lower_bound:.2f}-{upper_bound:.2f}",
            )

        pdh = max(filtered_prices)
        pdl = min(filtered_prices)
        # Last valid trade = close (walk backwards through original to find
        # the last trade that survived the outlier filter)
        pdc = next(
            (t["price"] for t in reversed(trades)
             if t.get("price", 0) > 0 and lower_bound <= t["price"] <= upper_bound),
            filtered_prices[-1],
        )

        state_engine.set_prior_day_levels(high=pdh, low=pdl, close=pdc)
        logger.info(
            "main.prior_day_levels_computed",
            pdh=pdh,
            pdl=pdl,
            pdc=pdc,
            trade_count=len(filtered_prices),
        )

        # Overnight levels: trades from 6PM ET yesterday to 9:30 AM ET today
        from datetime import datetime as dt
        from zoneinfo import ZoneInfo
        et = ZoneInfo("US/Eastern")
        overnight_start = dt(yesterday.year, yesterday.month, yesterday.day, 18, 0, tzinfo=et)
        rth_open = dt(today.year, today.month, today.day, 9, 30, tzinfo=et)

        on_prices = [
            t["price"] for t in trades
            if t.get("price", 0) > 0
            and lower_bound <= t["price"] <= upper_bound  # apply same outlier filter
            and "timestamp" in t
            and overnight_start <= t["timestamp"] <= rth_open  # compare datetime to datetime
        ]

        if on_prices:
            state_engine.set_overnight_levels(high=max(on_prices), low=min(on_prices))
            logger.info(
                "main.overnight_levels_set",
                onh=max(on_prices),
                onl=min(on_prices),
                trade_count=len(on_prices),
            )

    except Exception:
        logger.warning("main.prior_day_levels_fetch_failed", exc_info=True)


async def _state_engine_watchdog(state_engine: StateEngine) -> None:
    """Watchdog: if state engine task dies, log and restart it."""
    try:
        while True:
            await asyncio.sleep(60)
            if state_engine._task is None or state_engine._task.done():
                logger.warning("main.state_engine_died_restarting")
                await state_engine.start()
    except asyncio.CancelledError:
        pass


def main() -> None:
    """Synchronous entry point."""
    args = _parse_args()

    # --paper implies Tradovate demo mode (handled by TV_USE_DEMO=true in .env)
    dry_run = args.dry

    try:
        asyncio.run(run(dry_run=dry_run))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
