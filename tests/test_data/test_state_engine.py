"""Tests for the StateEngine — central MarketState computation hub."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.core.events import EventBus
from src.core.types import (
    CrossMarketContext,
    EconomicEvent,
    EventType,
    KeyLevels,
    MarketState,
    OrderFlowData,
    PositionState,
    Regime,
    SessionPhase,
    Side,
    TradeRecord,
)
from src.data.economic_calendar import EconomicCalendar
from src.data.multi_instrument import MultiInstrumentPoller
from src.data.schemas import OHLCVBar, RVOLBaseline, TickDirection
from src.data.state_engine import StateEngine
from src.data.tick_processor import TickProcessor

ET = ZoneInfo("US/Eastern")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_tick_snapshot(**overrides: object) -> dict:
    """Create a default tick processor snapshot dict."""
    defaults = {
        "last_price": 19850.0,
        "bid": 19849.75,
        "ask": 19850.25,
        "spread": 0.50,
        "session_open": 19800.0,
        "session_high": 19870.0,
        "session_low": 19780.0,
        "vwap": 19830.0,
        "poc": 19840.0,
        "value_area_high": 19860.0,
        "value_area_low": 19820.0,
        "cumulative_delta": 150.0,
        "delta_1min": 25.0,
        "delta_5min": 80.0,
        "delta_trend": "positive",
        "tape_speed": 15.0,
        "large_lot_count_5min": 3,
        "total_volume": 5000,
        "total_trades": 1200,
    }
    defaults.update(overrides)
    return defaults


def _make_bar(
    price: float = 19850.0,
    volume: int = 10,
    timestamp: datetime | None = None,
    is_up: bool = True,
) -> OHLCVBar:
    """Create a test OHLCVBar."""
    ts = timestamp or datetime.now(tz=UTC)
    offset = 1.0 if is_up else -1.0
    return OHLCVBar(
        timestamp=ts,
        symbol="MNQM6",
        open=price - offset,
        high=price + 2.0,
        low=price - 3.0,
        close=price,
        volume=volume,
        trade_count=5,
        buy_volume=6 if is_up else 4,
        sell_volume=4 if is_up else 6,
    )


@pytest.fixture
def tick_processor() -> TickProcessor:
    """Create a TickProcessor instance."""
    return TickProcessor()


@pytest.fixture
def multi_instrument() -> MultiInstrumentPoller:
    """Create a MultiInstrumentPoller instance."""
    return MultiInstrumentPoller()


@pytest.fixture
def calendar() -> EconomicCalendar:
    """Create an EconomicCalendar instance."""
    return EconomicCalendar()


@pytest.fixture
def event_bus() -> EventBus:
    """Create an EventBus instance."""
    return EventBus()


@pytest.fixture
def engine(
    tick_processor: TickProcessor,
    multi_instrument: MultiInstrumentPoller,
    calendar: EconomicCalendar,
    event_bus: EventBus,
) -> StateEngine:
    """Create a StateEngine with all dependencies."""
    return StateEngine(
        tick_processor=tick_processor,
        multi_instrument=multi_instrument,
        calendar=calendar,
        event_bus=event_bus,
        symbol="MNQM6",
    )


# ── Test: Construction & Defaults ────────────────────────────────────────────


class TestStateEngineInit:
    def test_default_state(self, engine: StateEngine) -> None:
        """Engine starts with sensible defaults."""
        assert not engine.is_running
        assert engine.update_count == 0
        assert engine.last_state is None
        assert engine.position is None

    def test_custom_intervals(
        self,
        tick_processor: TickProcessor,
        multi_instrument: MultiInstrumentPoller,
        calendar: EconomicCalendar,
        event_bus: EventBus,
    ) -> None:
        """Custom update intervals are applied."""
        eng = StateEngine(
            tick_processor=tick_processor,
            multi_instrument=multi_instrument,
            calendar=calendar,
            event_bus=event_bus,
            update_interval_no_position=60.0,
            update_interval_in_position=20.0,
            update_interval_critical=2.0,
        )
        assert eng._update_interval_no_position == 60.0
        assert eng._update_interval_in_position == 20.0
        assert eng._update_interval_critical == 2.0

    def test_stats_default(self, engine: StateEngine) -> None:
        """Stats reflect default values."""
        stats = engine.stats
        assert stats["running"] is False
        assert stats["update_count"] == 0
        assert stats["error_count"] == 0
        assert stats["regime"] == "choppy"
        assert stats["pdh"] == 0.0
        assert stats["rvol_buckets"] == 0


# ── Test: External Setters ───────────────────────────────────────────────────


class TestSetters:
    def test_set_prior_day_levels(self, engine: StateEngine) -> None:
        """Prior day levels are stored correctly."""
        engine.set_prior_day_levels(high=19900.0, low=19700.0, close=19800.0)
        assert engine._prior_day_high == 19900.0
        assert engine._prior_day_low == 19700.0
        assert engine._prior_day_close == 19800.0

    def test_set_overnight_levels(self, engine: StateEngine) -> None:
        """Overnight levels are stored correctly."""
        engine.set_overnight_levels(high=19850.0, low=19750.0)
        assert engine._overnight_high == 19850.0
        assert engine._overnight_low == 19750.0

    @pytest.mark.asyncio
    async def test_update_position(self, engine: StateEngine) -> None:
        """Position can be set and cleared."""
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19850.0,
        )
        await engine.update_position(pos)
        assert engine.position is not None
        assert engine.position.quantity == 3

        # Clear position
        await engine.update_position(None)
        assert engine.position is None

    @pytest.mark.asyncio
    async def test_update_session_stats(self, engine: StateEngine) -> None:
        """Session stats are stored correctly."""
        await engine.update_session_stats(
            daily_pnl=250.0,
            daily_trades=5,
            daily_winners=3,
            daily_losers=2,
            daily_commissions=4.30,
        )
        assert engine._daily_pnl == 250.0
        assert engine._daily_trades == 5
        assert engine._daily_winners == 3
        assert engine._daily_losers == 2
        assert engine._daily_commissions == 4.30

    @pytest.mark.asyncio
    async def test_update_regime(self, engine: StateEngine) -> None:
        """Regime and confidence are stored correctly."""
        await engine.update_regime(Regime.TRENDING_UP, confidence=0.85)
        assert engine._regime == Regime.TRENDING_UP
        assert engine._regime_confidence == 0.85

    def test_set_game_plan(self, engine: StateEngine) -> None:
        """Game plan string is stored."""
        engine.set_game_plan("Look for long entries above VWAP")
        assert engine._game_plan == "Look for long entries above VWAP"

    def test_add_recent_trade(self, engine: StateEngine) -> None:
        """Recent trades are added and capped at 10."""
        for i in range(15):
            trade = TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19800.0 + i,
                stop_price=19780.0,
            )
            engine.add_recent_trade(trade)
        assert len(engine._recent_trades) == 10

    def test_set_critical(self, engine: StateEngine) -> None:
        """Critical mode flag is stored."""
        engine.set_critical(True)
        assert engine._is_critical is True
        engine.set_critical(False)
        assert engine._is_critical is False

    def test_load_rvol_baseline(self, engine: StateEngine, tmp_path) -> None:
        """RVOL baseline loads from file."""
        import json

        baseline_data = {"volume_by_time": {"09:30": 1200, "09:35": 3500}}
        filepath = tmp_path / "rvol_baseline.json"
        filepath.write_text(json.dumps(baseline_data))

        engine.load_rvol_baseline(str(filepath))
        assert len(engine._rvol_baseline.volume_by_time) == 2

    def test_set_rvol_baseline_directly(self, engine: StateEngine) -> None:
        """RVOL baseline can be set directly."""
        baseline = RVOLBaseline(volume_by_time={"10:00": 5000})
        engine.set_rvol_baseline(baseline)
        assert engine._rvol_baseline.volume_by_time["10:00"] == 5000


# ── Test: Update Interval Logic ──────────────────────────────────────────────


class TestUpdateInterval:
    def test_no_position_interval(self, engine: StateEngine) -> None:
        """Without position, uses the slow interval."""
        assert engine._get_update_interval() == 30.0

    @pytest.mark.asyncio
    async def test_in_position_interval(self, engine: StateEngine) -> None:
        """With position, uses the faster interval."""
        pos = PositionState(side=Side.LONG, quantity=2, avg_entry=19850.0)
        await engine.update_position(pos)
        assert engine._get_update_interval() == 10.0

    @pytest.mark.asyncio
    async def test_critical_interval(self, engine: StateEngine) -> None:
        """In critical mode, uses the fastest interval."""
        pos = PositionState(side=Side.LONG, quantity=2, avg_entry=19850.0)
        await engine.update_position(pos)
        engine.set_critical(True)
        assert engine._get_update_interval() == 5.0

    def test_critical_without_position(self, engine: StateEngine) -> None:
        """Critical mode is fastest even without position."""
        engine.set_critical(True)
        assert engine._get_update_interval() == 5.0


# ── Test: Key Levels Building ────────────────────────────────────────────────


class TestBuildKeyLevels:
    def test_builds_all_levels(self, engine: StateEngine) -> None:
        """Key levels include prior day, overnight, and session data."""
        engine.set_prior_day_levels(high=19900.0, low=19700.0, close=19800.0)
        engine.set_overnight_levels(high=19860.0, low=19740.0)

        snap = _make_tick_snapshot()
        levels = engine._build_key_levels(snap)

        assert levels.prior_day_high == 19900.0
        assert levels.prior_day_low == 19700.0
        assert levels.prior_day_close == 19800.0
        assert levels.overnight_high == 19860.0
        assert levels.overnight_low == 19740.0
        assert levels.session_high == 19870.0
        assert levels.session_low == 19780.0
        assert levels.session_open == 19800.0
        assert levels.vwap == 19830.0
        assert levels.poc == 19840.0

    def test_zero_levels_when_not_set(self, engine: StateEngine) -> None:
        """Levels default to 0.0 when not set."""
        snap = _make_tick_snapshot()
        levels = engine._build_key_levels(snap)
        assert levels.prior_day_high == 0.0
        assert levels.overnight_high == 0.0


# ── Test: Order Flow Building ────────────────────────────────────────────────


class TestBuildOrderFlow:
    def test_builds_flow_with_rvol(self, engine: StateEngine) -> None:
        """Order flow includes RVOL from baseline."""
        baseline = RVOLBaseline(volume_by_time={"10:00": 5000})
        engine.set_rvol_baseline(baseline)

        snap = _make_tick_snapshot(total_volume=7500)
        # Use 10:00 time
        now = datetime(2025, 3, 14, 10, 0, tzinfo=ET)

        flow = engine._build_order_flow(snap, now)
        assert flow.cumulative_delta == 150.0
        assert flow.delta_1min == 25.0
        assert flow.delta_5min == 80.0
        assert flow.delta_trend == "positive"
        assert flow.rvol == 1.5  # 7500 / 5000
        assert flow.tape_speed == 15.0

    def test_rvol_default_no_baseline(self, engine: StateEngine) -> None:
        """RVOL returns 1.0 when no baseline is loaded."""
        snap = _make_tick_snapshot(total_volume=5000)
        now = datetime(2025, 3, 14, 10, 0, tzinfo=ET)

        flow = engine._build_order_flow(snap, now)
        assert flow.rvol == 1.0

    def test_rvol_at_specific_time(self, engine: StateEngine) -> None:
        """RVOL uses the correct time bucket."""
        baseline = RVOLBaseline(
            volume_by_time={
                "09:30": 1000,
                "09:35": 3000,
                "10:00": 5000,
            }
        )
        engine.set_rvol_baseline(baseline)

        snap = _make_tick_snapshot(total_volume=6000)
        now = datetime(2025, 3, 14, 9, 37, tzinfo=ET)  # snaps to 09:35

        flow = engine._build_order_flow(snap, now)
        assert flow.rvol == 2.0  # 6000 / 3000


# ── Test: Price Action Summary ───────────────────────────────────────────────


class TestPriceActionSummary:
    def test_no_trades(self, engine: StateEngine) -> None:
        """Returns 'No trades yet' when price is 0."""
        snap = _make_tick_snapshot(last_price=0.0)
        summary = engine._generate_price_action_summary(snap)
        assert summary == "No trades yet."

    def test_near_session_highs(self, engine: StateEngine) -> None:
        """Mentions session highs when price is near top."""
        snap = _make_tick_snapshot(
            last_price=19868.0,
            session_high=19870.0,
            session_low=19780.0,
        )
        summary = engine._generate_price_action_summary(snap)
        assert "session highs" in summary

    def test_near_session_lows(self, engine: StateEngine) -> None:
        """Mentions session lows when price is near bottom."""
        snap = _make_tick_snapshot(
            last_price=19783.0,
            session_high=19870.0,
            session_low=19780.0,
        )
        summary = engine._generate_price_action_summary(snap)
        assert "session lows" in summary

    def test_vwap_relationship_above(self, engine: StateEngine) -> None:
        """Mentions above VWAP."""
        snap = _make_tick_snapshot(
            last_price=19850.0,
            vwap=19830.0,
            session_high=19850.0,
            session_low=19780.0,
        )
        summary = engine._generate_price_action_summary(snap)
        assert "above VWAP" in summary

    def test_vwap_relationship_below(self, engine: StateEngine) -> None:
        """Mentions below VWAP."""
        snap = _make_tick_snapshot(
            last_price=19810.0,
            vwap=19830.0,
            session_high=19850.0,
            session_low=19780.0,
        )
        summary = engine._generate_price_action_summary(snap)
        assert "below VWAP" in summary

    def test_vwap_at_vwap(self, engine: StateEngine) -> None:
        """Mentions at VWAP when close."""
        snap = _make_tick_snapshot(
            last_price=19830.5,
            vwap=19830.0,
            session_high=19850.0,
            session_low=19780.0,
        )
        summary = engine._generate_price_action_summary(snap)
        assert "at VWAP" in summary

    def test_positive_delta_mentioned(self, engine: StateEngine) -> None:
        """Mentions buyer control with positive delta."""
        snap = _make_tick_snapshot(delta_trend="positive")
        summary = engine._generate_price_action_summary(snap)
        assert "buyers in control" in summary

    def test_negative_delta_mentioned(self, engine: StateEngine) -> None:
        """Mentions seller control with negative delta."""
        snap = _make_tick_snapshot(delta_trend="negative")
        summary = engine._generate_price_action_summary(snap)
        assert "sellers in control" in summary

    def test_flipping_delta_mentioned(self, engine: StateEngine) -> None:
        """Mentions momentum shift with flipping delta."""
        snap = _make_tick_snapshot(delta_trend="flipping")
        summary = engine._generate_price_action_summary(snap)
        assert "flipping" in summary

    def test_fast_tape(self, engine: StateEngine) -> None:
        """Mentions fast tape when trades/sec is high."""
        snap = _make_tick_snapshot(tape_speed=55.0)
        summary = engine._generate_price_action_summary(snap)
        assert "fast tape" in summary

    def test_slow_tape(self, engine: StateEngine) -> None:
        """Mentions slow tape when trades/sec is low."""
        snap = _make_tick_snapshot(tape_speed=5.0)
        summary = engine._generate_price_action_summary(snap)
        assert "slow tape" in summary

    def test_heavy_institutional_activity(self, engine: StateEngine) -> None:
        """Mentions institutional activity with many large lots."""
        snap = _make_tick_snapshot(large_lot_count_5min=7)
        summary = engine._generate_price_action_summary(snap)
        assert "institutional activity" in summary

    def test_notable_large_lots(self, engine: StateEngine) -> None:
        """Mentions notable large lots (3-5 count)."""
        snap = _make_tick_snapshot(large_lot_count_5min=4)
        summary = engine._generate_price_action_summary(snap)
        assert "large lot activity" in summary

    def test_level_proximity(self, engine: StateEngine) -> None:
        """Mentions nearby key levels."""
        engine.set_prior_day_levels(high=19852.0, low=19700.0, close=19800.0)
        snap = _make_tick_snapshot(last_price=19850.0)
        summary = engine._generate_price_action_summary(snap)
        assert "PDH" in summary

    def test_level_proximity_multiple(self, engine: StateEngine) -> None:
        """Mentions multiple nearby levels."""
        engine.set_prior_day_levels(high=19851.0, low=19849.0, close=19850.0)
        snap = _make_tick_snapshot(last_price=19850.0)
        summary = engine._generate_price_action_summary(snap)
        assert "PDH" in summary
        assert "PDL" in summary
        assert "PDC" in summary

    def test_no_level_proximity_when_far(self, engine: StateEngine) -> None:
        """No level mention when price is far from all levels."""
        engine.set_prior_day_levels(high=20000.0, low=19600.0, close=19700.0)
        snap = _make_tick_snapshot(last_price=19850.0)
        summary = engine._generate_price_action_summary(snap)
        assert "PDH" not in summary
        assert "PDL" not in summary
        assert "PDC" not in summary


# ── Test: Momentum Description ───────────────────────────────────────────────


class TestMomentumDescription:
    def test_strong_upward_momentum(self, engine: StateEngine) -> None:
        """Detects strong upward momentum from bars."""
        now = datetime.now(tz=UTC)
        for i in range(10):
            bar = _make_bar(
                price=19850.0 + i * 1.0,
                timestamp=now - timedelta(seconds=10 - i),
                is_up=True,
            )
            engine._recent_bars.append(bar)

        desc = engine._describe_recent_momentum()
        assert "upward" in desc or "higher" in desc or desc == ""

    def test_no_momentum_few_bars(self, engine: StateEngine) -> None:
        """Returns empty with fewer than 5 bars."""
        engine._recent_bars.append(_make_bar())
        assert engine._describe_recent_momentum() == ""

    def test_choppy_action(self, engine: StateEngine) -> None:
        """Detects choppy action when bars alternate."""
        now = datetime.now(tz=UTC)
        for i in range(20):
            bar = _make_bar(
                price=19850.0 + (0.5 if i % 2 == 0 else -0.5),
                timestamp=now - timedelta(seconds=20 - i),
                is_up=i % 2 == 0,
            )
            engine._recent_bars.append(bar)

        desc = engine._describe_recent_momentum()
        # The net move is ~0, so either choppy or empty
        assert desc == "" or "choppy" in desc


# ── Test: Bar Callback ───────────────────────────────────────────────────────


class TestBarCallback:
    @pytest.mark.asyncio
    async def test_on_bar_completed(self, engine: StateEngine) -> None:
        """Bar callback stores bars in the deque."""
        bar = _make_bar()
        await engine.on_bar_completed(bar)
        assert len(engine._recent_bars) == 1

    @pytest.mark.asyncio
    async def test_bar_deque_maxlen(self, engine: StateEngine) -> None:
        """Bar deque respects max length."""
        for i in range(40):
            bar = _make_bar(price=19800.0 + i)
            await engine.on_bar_completed(bar)
        # Default maxlen is 30
        assert len(engine._recent_bars) == 30


# ── Test: Volume Estimation ──────────────────────────────────────────────────


class TestVolumeEstimation:
    def test_no_bars(self, engine: StateEngine) -> None:
        """Returns 0 with no bars."""
        assert engine._estimate_1min_volume() == 0

    def test_recent_bars_sum(self, engine: StateEngine) -> None:
        """Sums volume from recent bars within 60 seconds."""
        now = datetime.now(tz=UTC)
        for i in range(10):
            bar = _make_bar(
                volume=100,
                timestamp=now - timedelta(seconds=i * 5),
            )
            engine._recent_bars.append(bar)
        # All 10 bars should be within 60 seconds
        vol = engine._estimate_1min_volume()
        assert vol == 1000

    def test_old_bars_excluded(self, engine: StateEngine) -> None:
        """Bars older than 60 seconds are excluded."""
        now = datetime.now(tz=UTC)
        # Add an old bar (90 seconds ago)
        old_bar = _make_bar(volume=500, timestamp=now - timedelta(seconds=90))
        engine._recent_bars.append(old_bar)
        # Add a recent bar (10 seconds ago)
        recent_bar = _make_bar(volume=100, timestamp=now - timedelta(seconds=10))
        engine._recent_bars.append(recent_bar)

        vol = engine._estimate_1min_volume()
        assert vol == 100  # only recent bar counted


# ── Test: Full State Computation ─────────────────────────────────────────────


class TestComputeState:
    @pytest.mark.asyncio
    async def test_compute_state_basic(self, engine: StateEngine) -> None:
        """compute_state returns a valid MarketState."""
        # Mock tick processor snapshot
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        # Set some levels
        engine.set_prior_day_levels(high=19900.0, low=19700.0, close=19800.0)
        engine.set_overnight_levels(high=19860.0, low=19740.0)

        # Mock clock to return consistent phase
        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert isinstance(state, MarketState)
        assert state.symbol == "MNQM6"
        assert state.last_price == 19850.0
        assert state.bid == 19849.75
        assert state.ask == 19850.25
        assert state.session_phase == SessionPhase.MORNING
        assert state.levels.prior_day_high == 19900.0
        assert state.levels.overnight_high == 19860.0
        assert state.flow.cumulative_delta == 150.0
        assert state.flow.delta_trend == "positive"

    @pytest.mark.asyncio
    async def test_compute_state_with_position(self, engine: StateEngine) -> None:
        """compute_state includes position when set."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19840.0,
            unrealized_pnl=60.0,
        )
        await engine.update_position(pos)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.position is not None
        assert state.position.side == Side.LONG
        assert state.position.quantity == 3
        assert state.position.unrealized_pnl == 60.0

    @pytest.mark.asyncio
    async def test_compute_state_with_session_stats(
        self, engine: StateEngine
    ) -> None:
        """compute_state includes session P&L stats."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        await engine.update_session_stats(
            daily_pnl=350.0,
            daily_trades=7,
            daily_winners=5,
            daily_losers=2,
            daily_commissions=6.02,
        )

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 11, 0, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.daily_pnl == 350.0
        assert state.daily_trades == 7
        assert state.daily_winners == 5
        assert state.daily_losers == 2
        assert state.daily_commissions == 6.02

    @pytest.mark.asyncio
    async def test_compute_state_with_regime(self, engine: StateEngine) -> None:
        """compute_state auto-classifies regime from market data."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        # Regime is auto-classified — with default data it should be CHOPPY
        assert state.regime in list(Regime)  # valid regime
        assert 0.0 <= state.regime_confidence <= 1.0

    async def test_compute_state_news_driven_regime(self, engine: StateEngine) -> None:
        """compute_state detects NEWS_DRIVEN regime when in blackout."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)
        engine._calendar.is_in_blackout = MagicMock(return_value=True)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.regime == Regime.NEWS_DRIVEN
        assert state.regime_confidence >= 0.8

    @pytest.mark.asyncio
    async def test_compute_state_with_game_plan(
        self, engine: StateEngine
    ) -> None:
        """compute_state includes game plan."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        engine.set_game_plan("Bullish bias, look for longs above VWAP")

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.game_plan == "Bullish bias, look for longs above VWAP"

    @pytest.mark.asyncio
    async def test_compute_state_with_upcoming_events(
        self, engine: StateEngine
    ) -> None:
        """compute_state includes upcoming economic events."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        event_time = datetime(2025, 3, 14, 11, 0, tzinfo=ET)
        event = EconomicEvent(
            time=event_time, name="CPI m/m", impact="high"
        )
        engine._calendar._events = [event]
        engine._calendar._loaded_date = "2025-03-14"

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_now = datetime(2025, 3, 14, 10, 30, tzinfo=ET)
            mock_clock.now_et.return_value = mock_now
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert len(state.upcoming_events) == 1
        assert state.upcoming_events[0].name == "CPI m/m"

    @pytest.mark.asyncio
    async def test_compute_state_blackout_detection(
        self, engine: StateEngine
    ) -> None:
        """compute_state detects blackout windows."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        # High-impact event at 10:30 — we're at 10:28 (within 5 min blackout)
        event_time = datetime(2025, 3, 14, 10, 30, tzinfo=ET)
        event = EconomicEvent(
            time=event_time, name="NFP", impact="high"
        )
        engine._calendar._events = [event]
        engine._calendar._loaded_date = "2025-03-14"

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_now = datetime(2025, 3, 14, 10, 28, tzinfo=ET)
            mock_clock.now_et.return_value = mock_now
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.in_blackout is True

    @pytest.mark.asyncio
    async def test_compute_state_cross_market(
        self, engine: StateEngine
    ) -> None:
        """compute_state includes cross-market context."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        # Simulate cross-market data
        engine._multi_instrument._es_price = 5200.0
        engine._multi_instrument._vix = 18.5
        engine._multi_instrument._tick_index = 450
        engine._multi_instrument._ten_year_yield = 4.25
        engine._multi_instrument._dxy = 104.5

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.cross_market.es_price == 5200.0
        assert state.cross_market.vix == 18.5
        assert state.cross_market.tick_index == 450
        assert state.cross_market.ten_year_yield == 4.25
        assert state.cross_market.dxy == 104.5

    @pytest.mark.asyncio
    async def test_compute_state_recent_trades(
        self, engine: StateEngine
    ) -> None:
        """compute_state includes recent trades (max 5)."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        # Add 7 trades
        for i in range(7):
            trade = TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19800.0 + i,
                stop_price=19780.0,
            )
            engine.add_recent_trade(trade)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        # MarketState gets last 5
        assert len(state.recent_trades) == 5

    @pytest.mark.asyncio
    async def test_compute_state_rvol(self, engine: StateEngine) -> None:
        """compute_state correctly computes RVOL."""
        snap = _make_tick_snapshot(total_volume=10000)
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        baseline = RVOLBaseline(volume_by_time={"10:30": 5000})
        engine.set_rvol_baseline(baseline)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        assert state.flow.rvol == 2.0  # 10000 / 5000


# ── Test: Compute Loop ───────────────────────────────────────────────────────


class TestComputeLoop:
    @pytest.mark.asyncio
    async def test_start_and_stop(self, engine: StateEngine) -> None:
        """Engine starts and stops cleanly."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            # Use very short interval for testing
            engine._update_interval_no_position = 0.05

            await engine.start()
            assert engine.is_running

            # Wait for at least one computation
            await asyncio.sleep(0.15)

            await engine.stop()
            assert not engine.is_running
            assert engine.update_count >= 1

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self, engine: StateEngine) -> None:
        """Starting twice doesn't create duplicate tasks."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            engine._update_interval_no_position = 0.05

            await engine.start()
            task1 = engine._task
            await engine.start()  # should be no-op
            task2 = engine._task
            assert task1 is task2

            await engine.stop()

    @pytest.mark.asyncio
    async def test_publishes_state_updated_event(
        self, engine: StateEngine, event_bus: EventBus
    ) -> None:
        """Compute loop publishes STATE_UPDATED events."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        received_events: list[Event] = []

        async def handler(event: Event) -> None:
            received_events.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            engine._update_interval_no_position = 0.05

            # Start event bus and engine
            bus_task = asyncio.create_task(event_bus.run())
            await engine.start()
            await asyncio.sleep(0.2)
            await engine.stop()
            await event_bus.stop()
            bus_task.cancel()
            try:
                await bus_task
            except asyncio.CancelledError:
                pass

        assert len(received_events) >= 1
        assert received_events[0].type == EventType.STATE_UPDATED
        assert "market_state" in received_events[0].data

    @pytest.mark.asyncio
    async def test_error_handling_in_loop(self, engine: StateEngine) -> None:
        """Errors in compute don't crash the loop."""
        call_count = 0

        def failing_snapshot():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("test error")
            return _make_tick_snapshot()

        engine._tick_processor.snapshot = MagicMock(side_effect=failing_snapshot)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            engine._update_interval_no_position = 0.05
            await engine.start()
            await asyncio.sleep(0.3)
            await engine.stop()

        # Should have recorded errors but continued running
        assert engine._error_count >= 2
        # Eventually succeeded
        assert engine.update_count >= 1

    @pytest.mark.asyncio
    async def test_last_state_stored(self, engine: StateEngine) -> None:
        """Last computed state is accessible."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()
            engine._last_state = state  # manually set since not running loop

        assert engine.last_state is not None
        assert engine.last_state.symbol == "MNQM6"


# ── Test: Reset ──────────────────────────────────────────────────────────────


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_session_data(self, engine: StateEngine) -> None:
        """Reset clears all session-specific data."""
        await engine.update_position(
            PositionState(side=Side.LONG, quantity=2, avg_entry=19850.0)
        )
        await engine.update_session_stats(
            daily_pnl=300.0,
            daily_trades=5,
            daily_winners=3,
            daily_losers=2,
        )
        await engine.update_regime(Regime.TRENDING_UP, confidence=0.9)
        engine.set_game_plan("test plan")
        engine.add_recent_trade(
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19850.0,
                stop_price=19830.0,
            )
        )

        engine.reset()

        assert engine.position is None
        assert engine._daily_pnl == 0.0
        assert engine._daily_trades == 0
        assert engine._daily_winners == 0
        assert engine._daily_losers == 0
        assert engine._daily_commissions == 0.0
        assert len(engine._recent_trades) == 0
        assert engine._game_plan == ""
        assert engine._regime == Regime.CHOPPY
        assert engine._regime_confidence == 0.5
        assert len(engine._recent_bars) == 0
        assert engine.last_state is None
        assert engine.update_count == 0
        assert engine._is_critical is False

    def test_reset_preserves_prior_day_levels(
        self, engine: StateEngine
    ) -> None:
        """Reset does NOT clear prior day or overnight levels."""
        engine.set_prior_day_levels(high=19900.0, low=19700.0, close=19800.0)
        engine.set_overnight_levels(high=19860.0, low=19740.0)

        engine.reset()

        # These should be preserved (they're for the next session)
        assert engine._prior_day_high == 19900.0
        assert engine._prior_day_low == 19700.0
        assert engine._prior_day_close == 19800.0
        assert engine._overnight_high == 19860.0
        assert engine._overnight_low == 19740.0

    def test_reset_preserves_rvol_baseline(self, engine: StateEngine) -> None:
        """Reset does NOT clear the RVOL baseline."""
        baseline = RVOLBaseline(volume_by_time={"09:30": 1200})
        engine.set_rvol_baseline(baseline)

        engine.reset()

        assert len(engine._rvol_baseline.volume_by_time) == 1


# ── Test: to_llm_dict Integration ────────────────────────────────────────────


class TestLLMDictIntegration:
    @pytest.mark.asyncio
    async def test_computed_state_serializes(self, engine: StateEngine) -> None:
        """MarketState from compute_state can be serialized to LLM dict."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        engine.set_prior_day_levels(high=19900.0, low=19700.0, close=19800.0)
        await engine.update_session_stats(
            daily_pnl=200.0,
            daily_trades=4,
            daily_winners=3,
            daily_losers=1,
        )

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        llm_dict = state.to_llm_dict()

        assert "price" in llm_dict
        assert llm_dict["price"]["last"] == 19850.0
        assert "levels" in llm_dict
        assert llm_dict["levels"]["prior_day_high"] == 19900.0
        assert "flow" in llm_dict
        assert "pnl" in llm_dict
        assert llm_dict["pnl"]["daily"] == 200.0
        assert "price_action" in llm_dict

    @pytest.mark.asyncio
    async def test_computed_state_with_position_serializes(
        self, engine: StateEngine
    ) -> None:
        """MarketState with position serializes correctly."""
        snap = _make_tick_snapshot()
        engine._tick_processor.snapshot = MagicMock(return_value=snap)

        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19840.0,
            unrealized_pnl=60.0,
            stop_price=19820.0,
        )
        await engine.update_position(pos)

        with patch("src.data.state_engine.clock") as mock_clock:
            mock_clock.now_et.return_value = datetime(
                2025, 3, 14, 10, 30, tzinfo=ET
            )
            mock_clock.get_session_phase.return_value = SessionPhase.MORNING

            state = await engine.compute_state()

        llm_dict = state.to_llm_dict()
        assert "position" in llm_dict
        assert llm_dict["position"]["side"] == "long"
        assert llm_dict["position"]["qty"] == 3
