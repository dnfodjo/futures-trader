"""Domain types for the MNQ futures trading system.

Every module imports from here. These are the shared language of the system.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

_ET = ZoneInfo("US/Eastern")


# ── Enums ──────────────────────────────────────────────────────────────────────


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class ActionType(str, Enum):
    ENTER = "ENTER"
    ADD = "ADD"
    SCALE_OUT = "SCALE_OUT"
    MOVE_STOP = "MOVE_STOP"
    FLATTEN = "FLATTEN"
    DO_NOTHING = "DO_NOTHING"
    STOP_TRADING = "STOP_TRADING"


class Regime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    CHOPPY = "choppy"
    BREAKOUT = "breakout"
    NEWS_DRIVEN = "news_driven"
    LOW_VOLUME = "low_volume"


class SessionPhase(str, Enum):
    # Overnight / Globex sessions
    ASIAN = "asian"  # 18:00-02:00 ET (prev day 6 PM → 2 AM)
    LONDON = "london"  # 02:00-08:00 ET
    PRE_RTH = "pre_rth"  # 08:00-09:30 ET (econ data drops at 8:30)

    # Regular Trading Hours (RTH)
    OPEN_DRIVE = "open_drive"  # 09:30-10:00
    MORNING = "morning"  # 10:00-12:00
    MIDDAY = "midday"  # 12:00-14:00
    AFTERNOON = "afternoon"  # 14:00-15:30
    CLOSE = "close"  # 15:30-16:00

    # Post-RTH
    POST_RTH = "post_rth"  # 16:00-16:55 ET
    DAILY_HALT = "daily_halt"  # 17:00-18:00 ET (CME maintenance)

    # Legacy aliases kept for backward compatibility in stored data
    PRE_MARKET = "pre_market"  # maps to PRE_RTH conceptually
    AFTER_HOURS = "after_hours"  # maps to POST_RTH conceptually


class SystemState(str, Enum):
    STARTING = "starting"
    PRE_MARKET = "pre_market"
    TRADING = "trading"
    SHUTDOWN = "shutdown"
    KILLED = "killed"
    ERROR = "error"


class EventType(str, Enum):
    TICK_RECEIVED = "tick_received"
    STATE_UPDATED = "state_updated"
    ACTION_DECIDED = "action_decided"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    ORDER_MODIFIED = "order_modified"
    POSITION_CHANGED = "position_changed"
    GUARDRAIL_TRIGGERED = "guardrail_triggered"
    KILL_SWITCH_ACTIVATED = "kill_switch_activated"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    DAILY_LIMIT_HIT = "daily_limit_hit"
    PROFIT_PRESERVATION = "profit_preservation"
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"


# ── Market Data Models ─────────────────────────────────────────────────────────


class EconomicEvent(BaseModel):
    time: datetime
    name: str
    impact: str  # "high", "medium", "low"
    prior: Optional[str] = None
    forecast: Optional[str] = None


class KeyLevels(BaseModel):
    prior_day_high: float = 0.0
    prior_day_low: float = 0.0
    prior_day_close: float = 0.0
    overnight_high: float = 0.0
    overnight_low: float = 0.0
    session_high: float = 0.0
    session_low: float = 0.0
    session_open: float = 0.0
    vwap: float = 0.0
    poc: float = 0.0  # point of control (highest volume price)
    value_area_high: float = 0.0
    value_area_low: float = 0.0


class OrderFlowData(BaseModel):
    cumulative_delta: float = 0.0
    delta_1min: float = 0.0
    delta_5min: float = 0.0
    delta_trend: str = "neutral"  # "positive", "negative", "neutral", "flipping"
    rvol: float = 1.0  # relative volume vs 20-day avg
    volume_1min: int = 0
    large_lot_count_5min: int = 0
    tape_speed: float = 0.0  # trades per second


class CrossMarketContext(BaseModel):
    es_price: float = 0.0
    es_change_pct: float = 0.0
    tick_index: int = 0  # NYSE TICK
    vix: float = 0.0
    vix_change_pct: float = 0.0
    ten_year_yield: float = 0.0
    dxy: float = 0.0


# ── Position & Trade Models ────────────────────────────────────────────────────


class PositionState(BaseModel):
    side: Side
    quantity: int
    avg_entry: float
    unrealized_pnl: float = 0.0
    stop_price: float = 0.0
    max_favorable: float = 0.0  # best unrealized P&L seen
    max_adverse: float = 0.0  # worst unrealized P&L seen
    time_in_trade_sec: int = 0
    adds_count: int = 0
    entry_time: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    realized_pnl: float = 0.0  # accumulated P&L from partial closes (SCALE_OUT)

    @property
    def pnl_per_contract(self) -> float:
        """Unrealized P&L divided by number of contracts."""
        if self.quantity == 0:
            return 0.0
        return self.unrealized_pnl / self.quantity

    @property
    def risk_per_contract(self) -> float:
        """Point distance from entry to stop, per contract.

        Returns 0 if no stop is set. Positive value = proper stop placement.
        """
        if self.stop_price == 0.0:
            return 0.0
        return abs(self.avg_entry - self.stop_price)

    @property
    def is_profitable(self) -> bool:
        """Whether the position is currently in profit."""
        return self.unrealized_pnl > 0.0

    @property
    def hold_time_min(self) -> float:
        """Hold time in minutes."""
        return self.time_in_trade_sec / 60.0


class TradeRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_entry: datetime
    timestamp_exit: Optional[datetime] = None
    side: Side
    entry_quantity: int
    exit_quantity: int = 0
    entry_price: float
    exit_price: Optional[float] = None
    stop_price: float
    pnl: Optional[float] = None
    commissions: float = 0.0
    hold_time_sec: Optional[int] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    reasoning_entry: str = ""
    reasoning_exit: Optional[str] = None
    regime_at_entry: Regime = Regime.CHOPPY
    session_phase_at_entry: SessionPhase = SessionPhase.MORNING
    debate_result: Optional[dict] = None
    adds: int = 0
    scale_outs: int = 0


# ── LLM Action Models ─────────────────────────────────────────────────────────


class LLMAction(BaseModel):
    """The structured output from the LLM reasoning engine."""

    action: ActionType
    side: Optional[Side] = None
    quantity: Optional[int] = None
    stop_distance: Optional[float] = None  # points from entry for new entries
    new_stop_price: Optional[float] = None  # absolute price for MOVE_STOP
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    setup_type: Optional[str] = None  # which setup pattern is being played
    model_used: str = "haiku"
    latency_ms: int = 0


# ── MarketState (the complete snapshot sent to the LLM) ────────────────────────


class MarketState(BaseModel):
    """Complete market state snapshot assembled every 10-30 seconds.

    This is what the LLM receives as input for its reasoning.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    symbol: str = "MNQM6"

    # Price
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0

    # Session context
    session_phase: SessionPhase = SessionPhase.PRE_MARKET
    regime: Regime = Regime.CHOPPY
    regime_confidence: float = 0.5

    # Key levels
    levels: KeyLevels = Field(default_factory=KeyLevels)

    # Order flow
    flow: OrderFlowData = Field(default_factory=OrderFlowData)

    # Cross-market
    cross_market: CrossMarketContext = Field(default_factory=CrossMarketContext)

    # Current position (None if flat)
    position: Optional[PositionState] = None

    # Session P&L
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_winners: int = 0
    daily_losers: int = 0
    daily_commissions: float = 0.0

    # Calendar
    upcoming_events: list[EconomicEvent] = Field(default_factory=list)
    in_blackout: bool = False

    # Price action summary (text for LLM context)
    price_action_summary: str = ""

    # Recent 1-second OHLCV bars (for setup detection, not sent to LLM directly)
    recent_bars: list[dict] = Field(default_factory=list, exclude=True)

    # Recent 1-minute OHLCV bars — SENT to LLM so it can see candlestick structure
    recent_1min_bars: list[dict] = Field(default_factory=list)

    # ── Technical Indicators (computed by state engine) ──
    # EMAs for trend identification
    emas: dict = Field(default_factory=dict)  # {"ema_9": float, "ema_21": float, "ema_50": float}

    # Market structure — swing highs/lows and trend pattern
    market_structure: dict = Field(default_factory=dict)
    # {"trend": "up/down/sideways", "last_swing_high": float, "last_swing_low": float,
    #  "pattern": "HH_HL" / "LH_LL" / "mixed", "swing_count": int}

    # ATR — Average True Range (14-period on 1-min bars)
    atr: float = 0.0

    # Opening range (9:30-9:45 high/low)
    opening_range_high: float = 0.0
    opening_range_low: float = 0.0

    # Pivot levels from prior day
    pivot_levels: dict = Field(default_factory=dict)
    # {"pivot": float, "r1": float, "r2": float, "s1": float, "s2": float}

    # RSI (14-period on 1-min bars, 0-100 scale)
    rsi: float = 50.0

    # MACD (12/26/9 on 1-min bars)
    macd: dict = Field(default_factory=dict)  # {"macd": float, "signal": float, "histogram": float}

    # Recent trades for context
    recent_trades: list[TradeRecord] = Field(default_factory=list)

    # Session game plan from pre-market analysis
    game_plan: str = ""

    def to_llm_dict(self) -> dict:
        """Serialize to a flat, token-efficient dict for LLM consumption.

        Strips default/zero values to reduce prompt size. The LLM receives
        this as the 'market_state' JSON in every reasoning call.

        Includes computed signals (VWAP distance, level proximity) that
        help the LLM apply the 5-gate filter efficiently.
        """
        # Convert timestamp to ET so the LLM sees the correct Eastern Time
        # (the raw timestamp is UTC, and the LLM was misreading it as ET)
        ts_et = self.timestamp.astimezone(_ET) if self.timestamp.tzinfo else self.timestamp
        d: dict = {
            "timestamp": ts_et.isoformat(),
            "symbol": self.symbol,
            "price": {
                "last": self.last_price,
                "bid": self.bid,
                "ask": self.ask,
                "spread": self.spread,
            },
            "session": {
                "phase": self.session_phase.value,
                "regime": self.regime.value,
                "regime_confidence": self.regime_confidence,
            },
            "levels": {
                k: v
                for k, v in self.levels.model_dump().items()
                if v != 0.0
            },
            "flow": {
                k: v
                for k, v in self.flow.model_dump().items()
                if v not in (0.0, 0, "neutral", 1.0)
            },
            "pnl": {
                "daily": self.daily_pnl,
                "trades": self.daily_trades,
            },
        }

        # ── Computed signals (help the LLM without extra reasoning) ────
        computed: dict = {}

        # VWAP distance and relationship
        vwap = self.levels.vwap
        if vwap > 0 and self.last_price > 0:
            vwap_dist = round(self.last_price - vwap, 2)
            computed["vwap_distance"] = vwap_dist
            if abs(vwap_dist) <= 2:
                computed["vwap_zone"] = "at_vwap"
            elif abs(vwap_dist) <= 5:
                computed["vwap_zone"] = "near_vwap"
            elif abs(vwap_dist) <= 15:
                computed["vwap_zone"] = "extended"
            else:
                computed["vwap_zone"] = "very_extended"

        # Nearest key level and distance + resistance/support context
        if self.last_price > 0:
            nearest_level = None
            nearest_dist = float("inf")
            nearest_name = ""
            level_data = self.levels.model_dump()

            # Classify each level as support or resistance relative to current price
            resistance_levels: list[tuple[str, float, float]] = []  # (name, value, distance)
            support_levels: list[tuple[str, float, float]] = []

            for name, val in level_data.items():
                if val > 0:
                    dist = abs(self.last_price - val)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_level = val
                        nearest_name = name
                    # Classify levels within 15 points
                    # Session high/PDH/ONH are ALWAYS resistance for longs
                    # Session low/PDL/ONL are ALWAYS support for shorts
                    # This ensures warnings fire even when price IS at the level
                    if dist <= 15:
                        always_resistance = {"session_high", "prior_day_high", "overnight_high"}
                        always_support = {"session_low", "prior_day_low", "overnight_low"}
                        if name in always_resistance:
                            resistance_levels.append((name, val, dist))
                        elif name in always_support:
                            support_levels.append((name, val, dist))
                        elif val > self.last_price:
                            resistance_levels.append((name, val, val - self.last_price))
                        else:
                            support_levels.append((name, val, self.last_price - val))

            if nearest_level is not None and nearest_dist < 50:
                computed["nearest_level"] = nearest_name
                computed["nearest_level_distance"] = round(nearest_dist, 2)
                computed["at_decision_point"] = nearest_dist <= 3.0

            # CRITICAL: Resistance/support warnings for entry quality
            # Sort by distance (closest first)
            resistance_levels.sort(key=lambda x: x[2])
            support_levels.sort(key=lambda x: x[2])

            if resistance_levels:
                closest_res = resistance_levels[0]
                computed["nearest_resistance"] = closest_res[0]
                computed["resistance_distance"] = round(closest_res[2], 2)
                if closest_res[2] <= 5.0:
                    computed["resistance_warning"] = (
                        f"CAUTION: {closest_res[0]} resistance at {closest_res[1]:.1f} "
                        f"is only {closest_res[2]:.1f}pts above. "
                        f"DO NOT enter new longs within 5pts of resistance — "
                        f"wait for breakout confirmation or pullback to support."
                    )

            if support_levels:
                closest_sup = support_levels[0]
                computed["nearest_support"] = closest_sup[0]
                computed["support_distance"] = round(closest_sup[2], 2)
                if closest_sup[2] <= 5.0:
                    computed["support_warning"] = (
                        f"CAUTION: {closest_sup[0]} support at {closest_sup[1]:.1f} "
                        f"is only {closest_sup[2]:.1f}pts below. "
                        f"DO NOT enter new shorts within 5pts of support — "
                        f"wait for breakdown confirmation or rally to resistance."
                    )

            # Extension from VWAP warning
            if vwap > 0 and self.last_price > 0:
                vwap_dist_abs = abs(self.last_price - vwap)
                if vwap_dist_abs > 10 and self.last_price > vwap:
                    computed["extension_warning"] = (
                        f"EXTENDED: Price is {vwap_dist_abs:.1f}pts ABOVE VWAP. "
                        f"Do NOT chase longs when extended. Wait for a pullback "
                        f"toward VWAP ({vwap:.1f}) or EMA21 before entering long."
                    )
                elif vwap_dist_abs > 10 and self.last_price < vwap:
                    computed["extension_warning"] = (
                        f"EXTENDED: Price is {vwap_dist_abs:.1f}pts BELOW VWAP. "
                        f"Do NOT chase shorts when extended. Wait for a rally "
                        f"toward VWAP ({vwap:.1f}) or EMA21 before entering short."
                    )

            # Session high/low proximity — explicit "don't chase" signal
            sh = self.levels.session_high
            sl = self.levels.session_low
            if sh > 0 and sl > 0:
                session_range = sh - sl
                if session_range > 5:
                    pct_of_range = (self.last_price - sl) / session_range
                    if pct_of_range > 0.90:
                        computed["session_position"] = "near_session_high"
                        computed["chase_warning"] = (
                            f"Price at {pct_of_range:.0%} of session range "
                            f"({sl:.1f}-{sh:.1f}). TOO LATE for new longs — "
                            f"wait for pullback or breakout with confirmation."
                        )
                    elif pct_of_range < 0.10:
                        computed["session_position"] = "near_session_low"
                        computed["chase_warning"] = (
                            f"Price at {pct_of_range:.0%} of session range "
                            f"({sl:.1f}-{sh:.1f}). TOO LATE for new shorts — "
                            f"wait for bounce or breakdown with confirmation."
                        )

        # Delta interpretation
        delta = self.flow.cumulative_delta
        if delta != 0:
            if abs(delta) > 500:
                computed["delta_strength"] = "strong"
            elif abs(delta) > 200:
                computed["delta_strength"] = "moderate"
            else:
                computed["delta_strength"] = "light"
            computed["delta_direction"] = "buying" if delta > 0 else "selling"

        # Suggested stop levels from market structure (helps LLM place logical stops)
        if self.market_structure:
            swing_low = self.market_structure.get("last_swing_low", 0)
            swing_high = self.market_structure.get("last_swing_high", 0)
            if swing_low > 0 and swing_high > 0:
                computed["suggested_long_stop"] = round(swing_low - 2.0, 2)  # below swing low + buffer
                computed["suggested_short_stop"] = round(swing_high + 2.0, 2)  # above swing high + buffer

        if computed:
            d["computed_signals"] = computed

        # ── Technical Indicators ──
        if self.emas:
            d["emas"] = self.emas
        if self.market_structure:
            d["market_structure"] = self.market_structure
        if self.atr > 0:
            d["atr"] = round(self.atr, 2)
        if self.opening_range_high > 0:
            d["opening_range"] = {
                "high": self.opening_range_high,
                "low": self.opening_range_low,
            }
        if self.pivot_levels:
            d["pivot_levels"] = self.pivot_levels
        if self.rsi != 50.0:
            d["rsi"] = round(self.rsi, 1)
        if self.macd:
            d["macd"] = self.macd

        # Recent 1-min bars — last 10 bars so LLM can see candlestick structure
        if self.recent_1min_bars:
            d["recent_bars_1min"] = self.recent_1min_bars[-10:]

        # Only include cross-market if any values are non-zero
        cm = self.cross_market.model_dump()
        cm_filtered = {k: v for k, v in cm.items() if v not in (0.0, 0)}
        if cm_filtered:
            d["cross_market"] = cm_filtered

        # Position — ALWAYS include so LLM knows definitively if flat or not
        if self.position is not None:
            d["position"] = {
                "side": self.position.side.value,
                "qty": self.position.quantity,
                "avg_entry": self.position.avg_entry,
                "unrealized_pnl": self.position.unrealized_pnl,
                "stop": self.position.stop_price,
                "time_in_trade_sec": self.position.time_in_trade_sec,
                "max_favorable": self.position.max_favorable,
                "max_adverse": self.position.max_adverse,
                "adds": self.position.adds_count,
            }
        else:
            d["position"] = "FLAT"

        # Upcoming events (only if present)
        if self.upcoming_events:
            d["upcoming_events"] = [
                {"time": e.time.isoformat(), "name": e.name, "impact": e.impact}
                for e in self.upcoming_events
            ]

        if self.in_blackout:
            d["in_blackout"] = True

        # Context strings (only if non-empty)
        if self.price_action_summary:
            d["price_action"] = self.price_action_summary
        if self.game_plan:
            d["game_plan"] = self.game_plan

        # Recent trades (abbreviated)
        if self.recent_trades:
            d["recent_trades"] = [
                {
                    "side": t.side.value,
                    "entry": t.entry_price,
                    "exit": t.exit_price,
                    "pnl": t.pnl,
                    "regime": t.regime_at_entry.value,
                }
                for t in self.recent_trades[-5:]  # last 5 max
            ]

        return d


# ── Internal Event Model ───────────────────────────────────────────────────────


class Event(BaseModel):
    type: EventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    data: dict = Field(default_factory=dict)


# ── Guardrail Result ───────────────────────────────────────────────────────────


class GuardrailResult(BaseModel):
    allowed: bool
    reason: str = ""
    modified_quantity: Optional[int] = None  # if size was reduced


# ── Order Tracking ─────────────────────────────────────────────────────────────


class OrderState(BaseModel):
    order_id: int
    symbol: str
    side: Side
    quantity: int
    order_type: str  # "Market", "Limit", "Stop", "StopLimit"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "Working"  # "Working", "Filled", "Cancelled", "Rejected"
    filled_quantity: int = 0
    fill_price: Optional[float] = None
    is_bracket_stop: bool = False
    parent_order_id: Optional[int] = None
    placed_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


# ── Session Summary ────────────────────────────────────────────────────────────


class SessionSummary(BaseModel):
    date: str
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    gross_pnl: float = 0.0
    commissions: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_position_size: int = 0
    regime_changes: int = 0
    regime_accuracy: Optional[float] = None
    session_grade: str = ""
    postmortem: str = ""
    trades: list[TradeRecord] = Field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Win rate as a percentage (0-100). Returns 0 if no trades."""
        if self.total_trades == 0:
            return 0.0
        return (self.winners / self.total_trades) * 100.0

    @property
    def profit_factor(self) -> float:
        """Gross profits / gross losses. Returns 0 if no losers.

        A value > 1.0 means the system is profitable overall.
        """
        gross_wins = sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in self.trades if t.pnl is not None and t.pnl < 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def avg_winner(self) -> float:
        """Average P&L of winning trades."""
        winning = [t.pnl for t in self.trades if t.pnl is not None and t.pnl > 0]
        if not winning:
            return 0.0
        return sum(winning) / len(winning)

    @property
    def avg_loser(self) -> float:
        """Average P&L of losing trades (returned as negative)."""
        losing = [t.pnl for t in self.trades if t.pnl is not None and t.pnl < 0]
        if not losing:
            return 0.0
        return sum(losing) / len(losing)

    @property
    def is_green_day(self) -> bool:
        """Whether today's net P&L is positive."""
        return self.net_pnl > 0.0
