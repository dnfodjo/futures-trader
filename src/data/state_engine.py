"""State engine — computes the full MarketState from all data sources.

Central computation hub that assembles the complete MarketState snapshot
from TickProcessor metrics, MultiInstrumentPoller cross-market data,
EconomicCalendar events, key levels, and RVOL baseline.

The engine runs on a configurable timer:
- No position:   every 30 seconds (save LLM costs)
- In position:   every 10 seconds
- Critical:      every 5 seconds (near stop/target, high volatility)

Each cycle:
1. Pull snapshot from TickProcessor (delta, VWAP, volume profile, tape speed)
2. Pull snapshot from MultiInstrumentPoller (ES, TICK, VIX, 10Y, DXY)
3. Check EconomicCalendar for upcoming events and blackout windows
4. Compute key levels (PDH/PDL/PDC, ONH/ONL, session H/L, POC, VA)
5. Compute RVOL from baseline
6. Generate price action summary text
7. Assemble full MarketState
8. Publish STATE_UPDATED event via EventBus
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import structlog

from src.core import clock
from src.core.events import EventBus
from src.core.types import (
    CrossMarketContext,
    EconomicEvent,
    Event,
    EventType,
    KeyLevels,
    MarketState,
    OrderFlowData,
    PositionState,
    Regime,
    SessionPhase,
    TradeRecord,
)
from src.data.economic_calendar import EconomicCalendar
from src.data.multi_instrument import MultiInstrumentPoller
from src.data.regime_classifier import RegimeClassifier
from src.data.schemas import OHLCVBar, RVOLBaseline
from src.data.tick_processor import TickProcessor

logger = structlog.get_logger()

ET = ZoneInfo("US/Eastern")

# ── Bar Persistence ──────────────────────────────────────────────────────────
_BAR_PERSIST_DIR = Path("data")
_BAR_PERSIST_INTERVAL = 5  # save every N new bars (not every bar to avoid I/O)

# ── Price Action Summary Constants ───────────────────────────────────────────

_PRICE_ACTION_WINDOW = 30  # max bars to consider for price action summary
_MOMENTUM_THRESHOLD = 5.0  # points of directional move to call "momentum"

# ── Technical Indicator Helpers ──────────────────────────────────────────────


def _compute_ema(prices: list[float], period: int) -> float:
    """Compute EMA from a list of prices. Returns 0 if insufficient data."""
    if len(prices) < period:
        return 0.0
    multiplier = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # SMA seed
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return round(ema, 2)


def _compute_rsi(prices: list[float], period: int = 14) -> float:
    """Compute RSI from a list of close prices. Returns 50 if insufficient data."""
    if len(prices) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    if len(gains) < period:
        return 50.0
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def _compute_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> dict:
    """Compute MACD line, signal line, and histogram.

    Uses incremental EMA calculation for efficiency (O(n) instead of O(n²)).
    """
    if len(prices) < slow + signal_period:
        return {}
    # Compute fast and slow EMA incrementally
    fast_mult = 2.0 / (fast + 1)
    slow_mult = 2.0 / (slow + 1)
    sig_mult = 2.0 / (signal_period + 1)

    fast_ema = sum(prices[:fast]) / fast
    slow_ema = sum(prices[:slow]) / slow

    # Build MACD values from the slow period onward
    macd_values: list[float] = []
    for i in range(slow, len(prices)):
        fast_ema = (prices[i] - fast_ema) * fast_mult + fast_ema
        slow_ema = (prices[i] - slow_ema) * slow_mult + slow_ema
        macd_values.append(fast_ema - slow_ema)

    if not macd_values:
        return {}

    macd_line = round(macd_values[-1], 2)

    # Compute signal line from MACD values
    if len(macd_values) >= signal_period:
        sig_ema = sum(macd_values[:signal_period]) / signal_period
        for mv in macd_values[signal_period:]:
            sig_ema = (mv - sig_ema) * sig_mult + sig_ema
        signal_line = round(sig_ema, 2)
    else:
        signal_line = macd_line

    histogram = round(macd_line - signal_line, 2)
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def _compute_atr(bars: list[dict], period: int = 14) -> float:
    """Compute ATR from OHLCV bar dicts. Returns 0 if insufficient data."""
    if len(bars) < period + 1:
        return 0.0
    true_ranges = []
    for i in range(1, len(bars)):
        high = bars[i].get("high", 0)
        low = bars[i].get("low", 0)
        prev_close = bars[i - 1].get("close", 0)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return 0.0
    atr = sum(true_ranges[:period]) / period
    for tr in true_ranges[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 2)


def _detect_market_structure(bars: list[dict], min_swing_pts: float = 3.0) -> dict:
    """Detect swing highs/lows and classify market structure.

    Returns: {"trend": "up/down/sideways", "last_swing_high": float,
              "last_swing_low": float, "pattern": "HH_HL/LH_LL/mixed",
              "swing_count": int}
    """
    if len(bars) < 10:
        return {}
    # Find swing highs and lows using 3-bar pivot
    swing_highs: list[tuple[int, float]] = []
    swing_lows: list[tuple[int, float]] = []
    for i in range(2, len(bars) - 2):
        h = bars[i].get("high", 0)
        l = bars[i].get("low", 0)
        if (h >= bars[i - 1].get("high", 0) and h >= bars[i - 2].get("high", 0)
                and h >= bars[i + 1].get("high", 0) and h >= bars[i + 2].get("high", 0)):
            if not swing_highs or abs(h - swing_highs[-1][1]) >= min_swing_pts:
                swing_highs.append((i, h))
        if (l <= bars[i - 1].get("low", float("inf")) and l <= bars[i - 2].get("low", float("inf"))
                and l <= bars[i + 1].get("low", float("inf")) and l <= bars[i + 2].get("low", float("inf"))):
            if not swing_lows or abs(l - swing_lows[-1][1]) >= min_swing_pts:
                swing_lows.append((i, l))
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"trend": "sideways", "pattern": "insufficient_data", "swing_count": len(swing_highs) + len(swing_lows)}
    # Classify pattern from last 3 swing highs and lows
    recent_highs = [h for _, h in swing_highs[-3:]]
    recent_lows = [l for _, l in swing_lows[-3:]]
    higher_highs = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))
    higher_lows = all(recent_lows[i] > recent_lows[i - 1] for i in range(1, len(recent_lows)))
    lower_highs = all(recent_highs[i] < recent_highs[i - 1] for i in range(1, len(recent_highs)))
    lower_lows = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))
    if higher_highs and higher_lows:
        trend = "up"
        pattern = "HH_HL"
    elif lower_highs and lower_lows:
        trend = "down"
        pattern = "LH_LL"
    else:
        trend = "sideways"
        pattern = "mixed"
    return {
        "trend": trend,
        "pattern": pattern,
        "last_swing_high": round(swing_highs[-1][1], 2),
        "last_swing_low": round(swing_lows[-1][1], 2),
        "prev_swing_high": round(swing_highs[-2][1], 2) if len(swing_highs) >= 2 else 0.0,
        "prev_swing_low": round(swing_lows[-2][1], 2) if len(swing_lows) >= 2 else 0.0,
        "swing_count": len(swing_highs) + len(swing_lows),
    }


class StateEngine:
    """Computes and publishes the full MarketState on a timer.

    Integrates all data sources into a single MarketState snapshot and
    publishes STATE_UPDATED events for the reasoning engine to consume.

    Usage:
        engine = StateEngine(
            tick_processor=tp,
            multi_instrument=mi,
            calendar=cal,
            event_bus=bus,
            config=trading_config,
        )
        engine.set_prior_day_levels(high=19900, low=19700, close=19800)
        engine.set_overnight_levels(high=19850, low=19750)
        engine.load_rvol_baseline("data/rvol_baseline.json")
        await engine.start()
        # ... runs until stopped
        await engine.stop()
    """

    def __init__(
        self,
        tick_processor: TickProcessor,
        multi_instrument: MultiInstrumentPoller,
        calendar: EconomicCalendar,
        event_bus: EventBus,
        symbol: str = "MNQM6",
        point_value: float = 2.0,
        update_interval_no_position: float = 30.0,
        update_interval_in_position: float = 10.0,
        update_interval_critical: float = 5.0,
        upcoming_events_window_min: int = 60,
    ) -> None:
        self._tick_processor = tick_processor
        self._multi_instrument = multi_instrument
        self._calendar = calendar
        self._event_bus = event_bus

        self._symbol = symbol
        self._point_value = point_value
        self._update_interval_no_position = update_interval_no_position
        self._update_interval_in_position = update_interval_in_position
        self._update_interval_critical = update_interval_critical
        self._upcoming_events_window_min = upcoming_events_window_min

        # Key levels — set externally before trading starts
        self._prior_day_high: float = 0.0
        self._prior_day_low: float = 0.0
        self._prior_day_close: float = 0.0
        self._overnight_high: float = 0.0
        self._overnight_low: float = 0.0

        # RVOL baseline
        self._rvol_baseline = RVOLBaseline()

        # Position state — set externally by position tracker
        self._position: Optional[PositionState] = None

        # Session P&L — set externally by session controller
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_winners: int = 0
        self._daily_losers: int = 0
        self._daily_commissions: float = 0.0

        # Recent trades — managed externally
        self._recent_trades: list[TradeRecord] = []

        # Game plan from pre-market analysis
        self._game_plan: str = ""

        # Regime — auto-classified by RegimeClassifier on each cycle
        self._regime: Regime = Regime.CHOPPY
        self._regime_confidence: float = 0.5
        self._regime_classifier = RegimeClassifier(stability_window=3)

        # Recent bars for price action summary
        self._recent_bars: deque[OHLCVBar] = deque(maxlen=_PRICE_ACTION_WINDOW)

        # 1-minute OHLCV bars for technical indicators
        self._1min_bars: list[dict] = []  # list of {"open": float, "high": float, ...}
        self._current_1min_bar: Optional[dict] = None
        self._1min_bar_start: Optional[datetime] = None

        # Opening range tracking (9:30-9:45 ET)
        self._opening_range_high: float = 0.0
        self._opening_range_low: float = float("inf")
        self._opening_range_set: bool = False

        # Pivot levels (computed from prior day once)
        self._pivot_levels: dict = {}

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_state: Optional[MarketState] = None
        self._update_count = 0
        self._error_count = 0
        self._is_critical = False

        # Lock for thread safety
        self._lock = asyncio.Lock()

    # ── External Setters ─────────────────────────────────────────────────────

    def set_prior_day_levels(
        self,
        high: float,
        low: float,
        close: float,
    ) -> None:
        """Set prior day's high, low, close. Called once at startup.

        Also computes classic pivot levels (P, R1, R2, S1, S2).
        """
        self._prior_day_high = high
        self._prior_day_low = low
        self._prior_day_close = close

        # Compute classic pivot levels
        if high > 0 and low > 0 and close > 0:
            pivot = (high + low + close) / 3
            self._pivot_levels = {
                "pivot": round(pivot, 2),
                "r1": round(2 * pivot - low, 2),
                "r2": round(pivot + (high - low), 2),
                "s1": round(2 * pivot - high, 2),
                "s2": round(pivot - (high - low), 2),
            }
            logger.info(
                "state_engine.pivot_levels_computed",
                pivot=self._pivot_levels["pivot"],
                r1=self._pivot_levels["r1"],
                s1=self._pivot_levels["s1"],
            )

        logger.info(
            "state_engine.prior_day_levels_set",
            pdh=high,
            pdl=low,
            pdc=close,
        )

    def set_overnight_levels(
        self,
        high: float,
        low: float,
    ) -> None:
        """Set overnight (Globex) high and low. Called before RTH open."""
        self._overnight_high = high
        self._overnight_low = low
        logger.info(
            "state_engine.overnight_levels_set",
            onh=high,
            onl=low,
        )

    def load_rvol_baseline(self, filepath: str) -> None:
        """Load RVOL baseline from a pre-computed JSON file."""
        self._rvol_baseline = RVOLBaseline.load_from_file(filepath)
        logger.info(
            "state_engine.rvol_baseline_loaded",
            buckets=len(self._rvol_baseline.volume_by_time),
        )

    def set_rvol_baseline(self, baseline: RVOLBaseline) -> None:
        """Set RVOL baseline directly (useful for testing)."""
        self._rvol_baseline = baseline

    async def update_position(self, position: Optional[PositionState]) -> None:
        """Update the current position state. Called by position tracker."""
        async with self._lock:
            self._position = position

    async def update_session_stats(
        self,
        daily_pnl: float,
        daily_trades: int,
        daily_winners: int,
        daily_losers: int,
        daily_commissions: float = 0.0,
    ) -> None:
        """Update session P&L stats. Called by session controller."""
        async with self._lock:
            self._daily_pnl = daily_pnl
            self._daily_trades = daily_trades
            self._daily_winners = daily_winners
            self._daily_losers = daily_losers
            self._daily_commissions = daily_commissions

    async def update_regime(
        self,
        regime: Regime,
        confidence: float = 0.5,
    ) -> None:
        """Update the market regime classification. Called by LLM or heuristic."""
        async with self._lock:
            self._regime = regime
            self._regime_confidence = confidence

    def set_game_plan(self, game_plan: str) -> None:
        """Set the pre-market game plan. Called once before trading starts."""
        self._game_plan = game_plan

    def add_recent_trade(self, trade: TradeRecord) -> None:
        """Add a completed trade to the recent trades list."""
        self._recent_trades.append(trade)
        # Keep only last 10 trades
        if len(self._recent_trades) > 10:
            self._recent_trades = self._recent_trades[-10:]

    def set_critical(self, is_critical: bool) -> None:
        """Set critical mode (near stop/target, high volatility).

        When critical, state updates happen at the fastest interval.
        """
        self._is_critical = is_critical

    # ── Bar Callback ─────────────────────────────────────────────────────────

    async def on_bar_completed(self, bar: OHLCVBar) -> None:
        """Callback for completed OHLCV bars from TickProcessor.

        Register with: tick_processor.on_bar(engine.on_bar_completed)
        Also aggregates 1-second bars into 1-minute bars for technical indicators.
        """
        self._recent_bars.append(bar)

        # ── Aggregate into 1-minute bars ──
        now = bar.timestamp
        # Check if we need to start a new 1-min bar
        if self._current_1min_bar is None or self._1min_bar_start is None:
            self._1min_bar_start = now
            self._current_1min_bar = {
                "timestamp": now.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        elif (now - self._1min_bar_start).total_seconds() >= 60:
            # Finalize the current 1-min bar
            self._1min_bars.append(self._current_1min_bar)
            # Keep max 120 bars (2 hours of 1-min data) for indicators
            if len(self._1min_bars) > 120:
                self._1min_bars = self._1min_bars[-120:]
            # Persist to disk periodically for restart recovery
            if len(self._1min_bars) % _BAR_PERSIST_INTERVAL == 0:
                self._persist_bars()
            # Start new bar
            self._1min_bar_start = now
            self._current_1min_bar = {
                "timestamp": now.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        else:
            # Update the current 1-min bar
            self._current_1min_bar["high"] = max(self._current_1min_bar["high"], bar.high)
            self._current_1min_bar["low"] = min(self._current_1min_bar["low"], bar.low)
            self._current_1min_bar["close"] = bar.close
            self._current_1min_bar["volume"] = self._current_1min_bar.get("volume", 0) + bar.volume

        # ── Track opening range (9:30-9:45 ET) ──
        if not self._opening_range_set:
            now_et = now.astimezone(ET) if now.tzinfo else now
            hour_min = now_et.hour * 60 + now_et.minute
            if 570 <= hour_min < 585:  # 9:30 to 9:44
                if bar.high > self._opening_range_high:
                    self._opening_range_high = bar.high
                if bar.low < self._opening_range_low:
                    self._opening_range_low = bar.low
            elif hour_min >= 585 and self._opening_range_high > 0:  # 9:45 → lock it
                self._opening_range_set = True
                logger.info(
                    "state_engine.opening_range_set",
                    orh=self._opening_range_high,
                    orl=self._opening_range_low,
                    width=round(self._opening_range_high - self._opening_range_low, 2),
                )

    # ── Compute Loop ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the state computation loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._compute_loop())
        logger.info("state_engine.started")

    async def stop(self) -> None:
        """Stop the state computation loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(
            "state_engine.stopped",
            updates=self._update_count,
            errors=self._error_count,
        )

    async def _compute_loop(self) -> None:
        """Background loop that periodically computes and publishes MarketState."""
        while self._running:
            try:
                state = await self.compute_state()
                self._last_state = state
                self._update_count += 1

                # Publish event
                await self._event_bus.publish(
                    Event(
                        type=EventType.STATE_UPDATED,
                        data={"market_state": state},
                    )
                )

            except asyncio.CancelledError:
                break
            except Exception:
                self._error_count += 1
                logger.exception("state_engine.compute_error")

            # Wait for next cycle
            interval = self._get_update_interval()
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

    def _get_update_interval(self) -> float:
        """Determine the update interval based on current state."""
        if self._is_critical:
            return self._update_interval_critical
        if self._position is not None:
            return self._update_interval_in_position
        return self._update_interval_no_position

    # ── State Computation ────────────────────────────────────────────────────

    async def compute_state(self) -> MarketState:
        """Compute the full MarketState from all data sources.

        This is the core function that assembles everything into a single
        snapshot for the LLM reasoning engine.
        """
        async with self._lock:
            # 1. Pull tick processor snapshot
            tick_snap = self._tick_processor.snapshot()

            # 2. Pull cross-market context
            cross_market = self._multi_instrument.snapshot()

            # 3. Get session phase
            now = clock.now_et()
            session_phase = clock.get_session_phase(now)

            # 4. Get upcoming events and blackout status
            upcoming = self._calendar.upcoming_events(
                within_minutes=self._upcoming_events_window_min, t=now
            )
            in_blackout = self._calendar.is_in_blackout(t=now)

            # 5. Build key levels
            levels = self._build_key_levels(tick_snap)

            # 6. Build order flow with RVOL
            flow = self._build_order_flow(tick_snap, now)

            # 7. Auto-classify regime from available data
            self._auto_classify_regime(
                tick_snap=tick_snap,
                rvol=flow.rvol,
                in_blackout=in_blackout,
                upcoming=upcoming,
            )

            # 8. Generate price action summary
            price_action = self._generate_price_action_summary(tick_snap)

            # 9. Compute technical indicators from 1-min bars
            emas, rsi, macd, atr, market_structure = self._compute_technical_indicators()

            # 10. Assemble full MarketState
            state = MarketState(
                timestamp=datetime.now(tz=UTC),
                symbol=self._symbol,
                # Price
                last_price=tick_snap["last_price"],
                bid=tick_snap["bid"],
                ask=tick_snap["ask"],
                spread=tick_snap["spread"],
                # Session
                session_phase=session_phase,
                regime=self._regime,
                regime_confidence=self._regime_confidence,
                # Levels
                levels=levels,
                # Flow
                flow=flow,
                # Cross-market
                cross_market=cross_market,
                # Position
                position=self._position,
                # Session P&L
                daily_pnl=self._daily_pnl,
                daily_trades=self._daily_trades,
                daily_winners=self._daily_winners,
                daily_losers=self._daily_losers,
                daily_commissions=self._daily_commissions,
                # Calendar
                upcoming_events=upcoming,
                in_blackout=in_blackout,
                # Context
                price_action_summary=price_action,
                recent_bars=[
                    b.model_dump() for b in self._recent_bars
                ],
                recent_1min_bars=self._1min_bars[-10:],  # last 10 1-min bars for LLM
                recent_trades=self._recent_trades[-5:],
                game_plan=self._game_plan,
                # Technical indicators
                emas=emas,
                market_structure=market_structure,
                atr=atr,
                opening_range_high=self._opening_range_high if self._opening_range_high > 0 else 0.0,
                opening_range_low=self._opening_range_low if self._opening_range_low < float("inf") else 0.0,
                pivot_levels=self._pivot_levels,
                rsi=rsi,
                macd=macd,
            )

            return state

    def _auto_classify_regime(
        self,
        tick_snap: dict[str, Any],
        rvol: float,
        in_blackout: bool,
        upcoming: list[EconomicEvent],
    ) -> None:
        """Run the automatic regime classifier on every compute cycle.

        Updates self._regime and self._regime_confidence in place.
        """
        # Check if any upcoming event is high-impact within 15 minutes
        upcoming_high_impact = any(
            e.impact == "high" for e in upcoming
        )

        regime, confidence = self._regime_classifier.classify(
            tick_snap=tick_snap,
            recent_bars=list(self._recent_bars),
            rvol=rvol,
            in_blackout=in_blackout,
            upcoming_high_impact=upcoming_high_impact,
            prior_day_high=self._prior_day_high,
            prior_day_low=self._prior_day_low,
            overnight_high=self._overnight_high,
            overnight_low=self._overnight_low,
        )

        self._regime = regime
        self._regime_confidence = confidence

    def _build_key_levels(self, tick_snap: dict[str, Any]) -> KeyLevels:
        """Build the KeyLevels from prior day + overnight + live session data."""
        vp = self._tick_processor.volume_profile
        va_low, va_high = vp.value_area()

        return KeyLevels(
            prior_day_high=self._prior_day_high,
            prior_day_low=self._prior_day_low,
            prior_day_close=self._prior_day_close,
            overnight_high=self._overnight_high,
            overnight_low=self._overnight_low,
            session_high=tick_snap["session_high"],
            session_low=tick_snap["session_low"],
            session_open=tick_snap["session_open"],
            vwap=tick_snap["vwap"],
            poc=tick_snap["poc"],
            value_area_high=va_high,
            value_area_low=va_low,
        )

    def _build_order_flow(
        self,
        tick_snap: dict[str, Any],
        now: datetime,
    ) -> OrderFlowData:
        """Build OrderFlowData from tick processor snapshot + RVOL."""
        # Compute RVOL
        time_str = now.strftime("%H:%M")
        rvol = self._rvol_baseline.compute_rvol(
            tick_snap["total_volume"], time_str
        )

        return OrderFlowData(
            cumulative_delta=tick_snap["cumulative_delta"],
            delta_1min=tick_snap["delta_1min"],
            delta_5min=tick_snap["delta_5min"],
            delta_trend=tick_snap["delta_trend"],
            rvol=round(rvol, 2),
            volume_1min=self._estimate_1min_volume(),
            large_lot_count_5min=tick_snap["large_lot_count_5min"],
            tape_speed=tick_snap["tape_speed"],
        )

    def _estimate_1min_volume(self) -> int:
        """Estimate volume over the last 1 minute from recent bars.

        Sums volume from the most recent bars that fit within a 60-second
        window. This is approximate since bars are 1-second resolution.
        """
        if not self._recent_bars:
            return 0

        now = datetime.now(tz=UTC)
        cutoff = now - timedelta(seconds=60)
        volume = 0
        for bar in reversed(self._recent_bars):
            if bar.timestamp < cutoff:
                break
            volume += bar.volume
        return volume

    # ── Technical Indicator Computation ─────────────────────────────────────

    def _compute_technical_indicators(
        self,
    ) -> tuple[dict, float, dict, float, dict]:
        """Compute EMAs, RSI, MACD, ATR, and market structure from 1-min bars.

        Returns:
            Tuple of (emas, rsi, macd, atr, market_structure)
        """
        bars = self._1min_bars
        if not bars:
            return {}, 50.0, {}, 0.0, {}

        closes = [b["close"] for b in bars]

        # EMAs (9, 21, 50)
        emas: dict = {}
        ema_9 = _compute_ema(closes, 9)
        ema_21 = _compute_ema(closes, 21)
        ema_50 = _compute_ema(closes, 50)
        if ema_9 > 0:
            emas["ema_9"] = ema_9
        if ema_21 > 0:
            emas["ema_21"] = ema_21
        if ema_50 > 0:
            emas["ema_50"] = ema_50

        # EMA alignment for trend confirmation
        # Compute with whatever EMAs are available (graceful degradation)
        if ema_9 > 0 and ema_21 > 0 and ema_50 > 0:
            if ema_9 > ema_21 > ema_50:
                emas["alignment"] = "bullish"
            elif ema_9 < ema_21 < ema_50:
                emas["alignment"] = "bearish"
            else:
                emas["alignment"] = "mixed"
        elif ema_9 > 0 and ema_21 > 0:
            # Partial alignment from 9/21 only (EMA 50 not yet available)
            if ema_9 > ema_21:
                emas["alignment"] = "bullish_partial"
            elif ema_9 < ema_21:
                emas["alignment"] = "bearish_partial"
            else:
                emas["alignment"] = "mixed_partial"
        elif ema_9 > 0:
            # Minimal alignment from EMA 9 vs current price only
            # (EMA 21 and 50 not yet available — early warmup period)
            if closes[-1] > ema_9:
                emas["alignment"] = "bullish_partial"
            else:
                emas["alignment"] = "bearish_partial"

        # EMA crossover detection (using prior bar's EMAs)
        if len(closes) > 21:
            prev_closes = closes[:-1]
            prev_ema_9 = _compute_ema(prev_closes, 9)
            prev_ema_21 = _compute_ema(prev_closes, 21)
            if prev_ema_9 > 0 and prev_ema_21 > 0 and ema_9 > 0 and ema_21 > 0:
                if prev_ema_9 <= prev_ema_21 and ema_9 > ema_21:
                    emas["crossover"] = "bullish_cross"  # 9 crossed above 21
                elif prev_ema_9 >= prev_ema_21 and ema_9 < ema_21:
                    emas["crossover"] = "bearish_cross"  # 9 crossed below 21

        # RSI (14-period)
        rsi = _compute_rsi(closes, 14)

        # MACD (12/26/9)
        macd = _compute_macd(closes, 12, 26, 9)

        # ATR (14-period)
        atr = _compute_atr(bars, 14)

        # Market structure (swing highs/lows)
        market_structure = _detect_market_structure(bars, min_swing_pts=3.0)

        return emas, rsi, macd, atr, market_structure

    # ── Price Action Summary ─────────────────────────────────────────────────

    def _generate_price_action_summary(
        self,
        tick_snap: dict[str, Any],
    ) -> str:
        """Generate a text summary of recent price action for LLM context.

        This is a programmatic description (not LLM-generated) that captures
        the last few minutes of market behavior in natural language.
        """
        parts: list[str] = []

        last_price = tick_snap["last_price"]
        if last_price == 0.0:
            return "No trades yet."

        # Session context
        session_high = tick_snap["session_high"]
        session_low = tick_snap["session_low"]
        session_range = session_high - session_low if session_high > session_low else 0.0

        if session_range > 0:
            # Position within session range (0=low, 1=high)
            range_pct = (last_price - session_low) / session_range
            if range_pct > 0.85:
                parts.append("Price near session highs")
            elif range_pct < 0.15:
                parts.append("Price near session lows")
            elif 0.45 <= range_pct <= 0.55:
                parts.append("Price at mid-range")

        # VWAP relationship
        vwap = tick_snap["vwap"]
        if vwap > 0:
            vwap_dist = last_price - vwap
            if abs(vwap_dist) < 2.0:
                parts.append("trading at VWAP")
            elif vwap_dist > 0:
                parts.append(f"trading {vwap_dist:.1f}pts above VWAP")
            else:
                parts.append(f"trading {abs(vwap_dist):.1f}pts below VWAP")

        # Recent bar momentum
        momentum_summary = self._describe_recent_momentum()
        if momentum_summary:
            parts.append(momentum_summary)

        # Delta context
        delta_trend = tick_snap["delta_trend"]
        if delta_trend == "positive":
            parts.append("buyers in control (positive delta)")
        elif delta_trend == "negative":
            parts.append("sellers in control (negative delta)")
        elif delta_trend == "flipping":
            parts.append("delta flipping (momentum shift)")

        # Tape speed
        tape_speed = tick_snap["tape_speed"]
        if tape_speed > 50:
            parts.append(f"fast tape ({tape_speed:.0f} trades/sec)")
        elif tape_speed > 20:
            parts.append(f"moderate pace ({tape_speed:.0f} trades/sec)")
        elif tape_speed > 0:
            parts.append(f"slow tape ({tape_speed:.0f} trades/sec)")

        # Large lots
        ll_count = tick_snap["large_lot_count_5min"]
        if ll_count > 5:
            parts.append(f"heavy institutional activity ({ll_count} large lots in 5min)")
        elif ll_count > 2:
            parts.append(f"notable large lot activity ({ll_count} in 5min)")

        # Key level proximity
        level_note = self._describe_level_proximity(last_price)
        if level_note:
            parts.append(level_note)

        if not parts:
            return "Market is quiet."

        return ". ".join(parts) + "."

    def _describe_recent_momentum(self) -> str:
        """Describe price momentum from recent 1-second bars."""
        if len(self._recent_bars) < 5:
            return ""

        # Look at last 30 bars (30 seconds of data)
        recent = list(self._recent_bars)[-30:]
        if not recent:
            return ""

        first_close = recent[0].close
        last_close = recent[-1].close
        net_move = last_close - first_close

        # Count up vs down bars
        up_bars = sum(1 for b in recent if b.is_up)
        down_bars = len(recent) - up_bars

        if abs(net_move) > _MOMENTUM_THRESHOLD:
            if net_move > 0:
                return f"strong upward momentum (+{net_move:.1f}pts in {len(recent)}s)"
            else:
                return f"strong downward momentum ({net_move:.1f}pts in {len(recent)}s)"
        elif abs(net_move) > 2.0:
            if net_move > 0:
                return f"drifting higher (+{net_move:.1f}pts)"
            else:
                return f"drifting lower ({net_move:.1f}pts)"
        elif up_bars > 0 and down_bars > 0:
            ratio = up_bars / len(recent)
            if 0.4 <= ratio <= 0.6:
                return "choppy/rotational price action"

        return ""

    def _describe_level_proximity(self, last_price: float) -> str:
        """Describe proximity to key levels."""
        proximity = 3.0  # within 3 points
        notes = []

        # Check each level
        level_checks = [
            (self._prior_day_high, "PDH"),
            (self._prior_day_low, "PDL"),
            (self._prior_day_close, "PDC"),
            (self._overnight_high, "ONH"),
            (self._overnight_low, "ONL"),
        ]

        for level, name in level_checks:
            if level > 0 and abs(last_price - level) <= proximity:
                notes.append(name)

        # Also check VWAP and POC from tick processor
        vp = self._tick_processor.volume_profile
        poc = vp.poc
        if poc > 0 and abs(last_price - poc) <= proximity:
            notes.append("POC")

        if notes:
            return f"near key levels: {', '.join(notes)}"
        return ""

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def last_state(self) -> Optional[MarketState]:
        """Most recently computed MarketState."""
        return self._last_state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def position(self) -> Optional[PositionState]:
        return self._position

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "update_count": self._update_count,
            "error_count": self._error_count,
            "is_critical": self._is_critical,
            "update_interval": self._get_update_interval(),
            "regime": self._regime.value,
            "regime_confidence": self._regime_confidence,
            "pdh": self._prior_day_high,
            "pdl": self._prior_day_low,
            "pdc": self._prior_day_close,
            "onh": self._overnight_high,
            "onl": self._overnight_low,
            "rvol_buckets": len(self._rvol_baseline.volume_by_time),
            "recent_bars": len(self._recent_bars),
            "daily_pnl": self._daily_pnl,
        }

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset for a new session. Does NOT reset prior day/overnight levels."""
        self._position = None
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_winners = 0
        self._daily_losers = 0
        self._daily_commissions = 0.0
        self._recent_trades.clear()
        self._game_plan = ""
        self._regime = Regime.CHOPPY
        self._regime_confidence = 0.5
        self._regime_classifier = RegimeClassifier(stability_window=3)
        self._recent_bars.clear()
        self._1min_bars.clear()
        self._current_1min_bar = None
        self._1min_bar_start = None
        self._opening_range_high = 0.0
        self._opening_range_low = float("inf")
        self._opening_range_set = False
        self._last_state = None
        self._update_count = 0
        self._error_count = 0
        self._is_critical = False

    def reset_for_rth(self) -> None:
        """Reset tick processor for RTH session open.

        Clears accumulated VWAP, delta, volume profile, 1-min bars,
        opening range, and all technical indicators so that
        RTH session metrics start fresh and are not polluted by
        overnight/pre-market data.  Prior day, overnight levels,
        and pivot levels are preserved.
        """
        self._tick_processor.reset()
        self._recent_bars.clear()
        # Clear 1-min bars and opening range for fresh RTH computation
        self._1min_bars.clear()
        self._current_1min_bar = None
        self._1min_bar_start = None
        self._opening_range_high = 0.0
        self._opening_range_low = float("inf")
        self._opening_range_set = False
        logger.info("state_engine.reset_for_rth")

    def warm_1min_bars(self, bars: list[dict]) -> None:
        """Pre-populate 1-min bars from historical data for instant indicator warmup.

        Call AFTER reset_for_rth() to avoid bars being cleared.
        Each bar dict must have: timestamp, open, high, low, close, volume.
        """
        if not bars:
            return
        self._1min_bars = bars[-120:]  # keep max 120 bars (2 hours)
        # Set the current bar start to the last bar's timestamp so new bars
        # continue from where historical left off
        last_bar = self._1min_bars[-1]
        if "timestamp" in last_bar:
            from datetime import datetime as dt

            ts = last_bar["timestamp"]
            if isinstance(ts, str):
                try:
                    self._1min_bar_start = dt.fromisoformat(ts)
                except (ValueError, TypeError):
                    self._1min_bar_start = None
            elif isinstance(ts, dt):
                self._1min_bar_start = ts
        self._current_1min_bar = None  # force new bar on next tick
        logger.info(
            "state_engine.warm_1min_bars",
            bar_count=len(self._1min_bars),
            first_ts=self._1min_bars[0].get("timestamp", "?") if self._1min_bars else "?",
            last_ts=self._1min_bars[-1].get("timestamp", "?") if self._1min_bars else "?",
        )

    def _persist_bars(self) -> None:
        """Save current 1-min bars to disk for restart recovery.

        Written as a simple JSON file. Only bars from today's RTH session
        are saved. File is named by date so old files don't interfere.
        """
        try:
            today_str = datetime.now(ET).strftime("%Y-%m-%d")
            path = _BAR_PERSIST_DIR / f"1min_bars_{today_str}.json"
            _BAR_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._1min_bars, default=str))
            logger.info(
                "state_engine.bars_persisted",
                bar_count=len(self._1min_bars),
                path=str(path),
            )
        except Exception:
            logger.warning("state_engine.bars_persist_failed", exc_info=True)

    def reload_persisted_bars(self) -> bool:
        """Reload 1-min bars from disk if available for today.

        Call AFTER reset_for_rth() to restore indicator state.

        Returns:
            True if bars were loaded, False otherwise.
        """
        try:
            today_str = datetime.now(ET).strftime("%Y-%m-%d")
            path = _BAR_PERSIST_DIR / f"1min_bars_{today_str}.json"
            if not path.exists():
                logger.info("state_engine.no_persisted_bars", path=str(path))
                return False

            bars = json.loads(path.read_text())
            if not bars or not isinstance(bars, list):
                logger.info("state_engine.persisted_bars_empty", path=str(path))
                return False

            # Validate bar format
            required_keys = {"open", "high", "low", "close"}
            if not required_keys.issubset(bars[0].keys()):
                logger.warning(
                    "state_engine.persisted_bars_invalid_format",
                    keys=list(bars[0].keys()),
                )
                return False

            self.warm_1min_bars(bars)
            logger.info(
                "state_engine.bars_reloaded_from_disk",
                bar_count=len(self._1min_bars),
                path=str(path),
            )
            return True
        except Exception:
            logger.warning("state_engine.bars_reload_failed", exc_info=True)
            return False
