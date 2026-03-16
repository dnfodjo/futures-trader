"""Tests for the automatic regime classifier."""

import pytest
from datetime import UTC, datetime

from src.core.types import Regime
from src.data.regime_classifier import RegimeClassifier
from src.data.schemas import OHLCVBar


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bar(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: int = 100,
    buy_volume: int = 50,
    sell_volume: int = 50,
) -> OHLCVBar:
    """Create a test bar. If open/high/low not given, derive from close."""
    if open_ is None:
        open_ = close - 0.25
    if high is None:
        high = max(open_, close) + 0.25
    if low is None:
        low = min(open_, close) - 0.25
    return OHLCVBar(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        trade_count=10,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
    )


def _make_tick_snap(
    last_price: float = 19800.0,
    vwap: float = 19795.0,
    session_high: float = 19820.0,
    session_low: float = 19780.0,
    session_open: float = 19790.0,
    cumulative_delta: float = 0.0,
    delta_trend: str = "neutral",
    tape_speed: float = 20.0,
    large_lot_count_5min: int = 0,
    total_volume: int = 5000,
) -> dict:
    return {
        "last_price": last_price,
        "bid": last_price - 0.25,
        "ask": last_price + 0.25,
        "spread": 0.50,
        "session_open": session_open,
        "session_high": session_high,
        "session_low": session_low,
        "vwap": vwap,
        "poc": vwap + 1.0,
        "cumulative_delta": cumulative_delta,
        "delta_1min": cumulative_delta * 0.1,
        "delta_5min": cumulative_delta * 0.5,
        "delta_trend": delta_trend,
        "tape_speed": tape_speed,
        "large_lot_count_5min": large_lot_count_5min,
        "total_volume": total_volume,
    }


def _make_trending_bars(count: int, direction: str = "up", start: float = 19800.0) -> list[OHLCVBar]:
    """Create a sequence of bars showing a clear trend."""
    bars = []
    price = start
    step = 0.5 if direction == "up" else -0.5
    for _ in range(count):
        price += step
        buy_vol = 80 if direction == "up" else 30
        sell_vol = 30 if direction == "up" else 80
        bars.append(_make_bar(
            close=price,
            open_=price - step,
            high=price + 0.5 if direction == "up" else price + 0.25,
            low=price - 0.25 if direction == "up" else price - 0.5,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
        ))
    return bars


def _make_choppy_bars(count: int, center: float = 19800.0) -> list[OHLCVBar]:
    """Create bars that oscillate around a center with no direction."""
    bars = []
    offsets = [0.25, -0.5, 0.75, -0.25, 0.5, -0.75, 0.25, -0.5, 0.25, -0.25]
    for i in range(count):
        offset = offsets[i % len(offsets)]
        price = center + offset
        bars.append(_make_bar(close=price))
    return bars


# ── Test Classes ──────────────────────────────────────────────────────────────


class TestNewsDriven:
    """NEWS_DRIVEN is highest priority — overrides everything."""

    def test_blackout_returns_news_driven(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=_make_choppy_bars(15),
            rvol=1.2,
            in_blackout=True,
            upcoming_high_impact=False,
        )
        assert regime == Regime.NEWS_DRIVEN
        assert conf >= 0.80

    def test_upcoming_high_impact_returns_news_driven(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=_make_choppy_bars(15),
            rvol=1.5,
            in_blackout=False,
            upcoming_high_impact=True,
        )
        assert regime == Regime.NEWS_DRIVEN
        assert conf >= 0.80

    def test_news_overrides_trend(self):
        """Even with strong trend data, blackout = NEWS_DRIVEN."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19830.0,
                vwap=19795.0,
                delta_trend="positive",
            ),
            recent_bars=_make_trending_bars(20, "up"),
            rvol=1.5,
            in_blackout=True,
            upcoming_high_impact=False,
        )
        assert regime == Regime.NEWS_DRIVEN


class TestLowVolume:
    """LOW_VOLUME when RVOL < 0.70."""

    def test_low_rvol_returns_low_volume(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=_make_choppy_bars(15),
            rvol=0.50,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.LOW_VOLUME
        assert conf >= 0.50

    def test_rvol_boundary_not_low_volume(self):
        """RVOL at exactly 0.70 should NOT trigger LOW_VOLUME."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=_make_choppy_bars(15),
            rvol=0.70,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime != Regime.LOW_VOLUME

    def test_zero_rvol_not_low_volume(self):
        """RVOL of 0 (no baseline data) should not trigger LOW_VOLUME."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=_make_choppy_bars(15),
            rvol=0.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime != Regime.LOW_VOLUME

    def test_news_overrides_low_volume(self):
        """NEWS_DRIVEN has priority over LOW_VOLUME."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[],
            rvol=0.30,
            in_blackout=True,
            upcoming_high_impact=False,
        )
        assert regime == Regime.NEWS_DRIVEN


class TestBreakout:
    """BREAKOUT when price breaks key levels with confirmation."""

    def test_breakout_above_pdh(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19903.0,  # 3pts above PDH
                vwap=19880.0,
                delta_trend="positive",
                tape_speed=45.0,
            ),
            recent_bars=_make_trending_bars(15, "up", start=19895.0),
            rvol=1.5,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=19900.0,
        )
        assert regime == Regime.BREAKOUT
        assert conf >= 0.50

    def test_breakout_below_pdl(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19697.0,  # 3pts below PDL
                vwap=19720.0,
                session_high=19730.0,
                session_low=19697.0,
                delta_trend="negative",
                tape_speed=40.0,
            ),
            recent_bars=_make_trending_bars(15, "down", start=19710.0),
            rvol=1.4,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_low=19700.0,
        )
        assert regime == Regime.BREAKOUT
        assert conf >= 0.50

    def test_no_breakout_without_overshoot(self):
        """Price barely touching PDH (< 1pt overshoot) should NOT be breakout."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19900.5,  # only 0.5 pts above PDH
                vwap=19895.0,
                session_high=19900.5,  # session high IS the current price
                session_low=19892.0,  # narrow range — no session breakout
                delta_trend="neutral",
            ),
            recent_bars=_make_choppy_bars(15, center=19898.0),
            rvol=0.9,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=19900.0,
        )
        assert regime != Regime.BREAKOUT

    def test_breakout_above_overnight_high(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19853.0,
                vwap=19830.0,
                delta_trend="positive",
                tape_speed=35.0,
            ),
            recent_bars=_make_trending_bars(15, "up", start=19845.0),
            rvol=1.3,
            in_blackout=False,
            upcoming_high_impact=False,
            overnight_high=19850.0,
        )
        assert regime == Regime.BREAKOUT
        assert conf >= 0.50


class TestTrending:
    """TRENDING_UP/DOWN with sustained directional move."""

    def test_trending_up_detected(self):
        """Strong uptrend: above VWAP, positive delta, higher highs."""
        rc = RegimeClassifier()
        # Create 20 bars trending up with 0.5pt/bar = 10pt net move
        bars = _make_trending_bars(20, "up", start=19800.0)
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19810.0,
                vwap=19800.0,
                delta_trend="positive",
            ),
            recent_bars=bars,
            rvol=1.2,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.TRENDING_UP
        assert conf >= 0.50

    def test_trending_down_detected(self):
        """Strong downtrend: below VWAP, negative delta, lower lows."""
        rc = RegimeClassifier()
        bars = _make_trending_bars(20, "down", start=19800.0)
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19790.0,
                vwap=19800.0,
                session_high=19805.0,
                session_low=19785.0,
                delta_trend="negative",
            ),
            recent_bars=bars,
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.TRENDING_DOWN
        assert conf >= 0.50

    def test_no_trend_with_few_bars(self):
        """Too few bars to determine trend → falls through to choppy."""
        rc = RegimeClassifier()
        bars = _make_trending_bars(5, "up", start=19800.0)
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19805.0,
                vwap=19800.0,
                delta_trend="positive",
            ),
            recent_bars=bars,
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime != Regime.TRENDING_UP

    def test_no_trend_with_small_move(self):
        """Small net move (< 4pts) insufficient for trend classification."""
        rc = RegimeClassifier()
        # 20 bars but tiny 0.1pt steps = 2pt net move
        bars = []
        price = 19800.0
        for _ in range(20):
            price += 0.1
            bars.append(_make_bar(close=price, open_=price - 0.1))

        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=price,
                vwap=19799.0,
                delta_trend="positive",
            ),
            recent_bars=bars,
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime not in (Regime.TRENDING_UP, Regime.TRENDING_DOWN)


class TestChoppy:
    """CHOPPY is the default when no other regime matches."""

    def test_choppy_default_no_data(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(last_price=0.0),
            recent_bars=[],
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.CHOPPY

    def test_choppy_narrow_range_oscillating(self):
        rc = RegimeClassifier()
        bars = _make_choppy_bars(15, center=19800.0)
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19800.0,
                vwap=19799.5,
                session_high=19803.0,
                session_low=19797.0,
                delta_trend="neutral",
            ),
            recent_bars=bars,
            rvol=0.9,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.CHOPPY
        assert conf >= 0.50


class TestStabilityFilter:
    """Tests for the anti-flipping stability mechanism."""

    def test_stability_prevents_immediate_switch(self):
        """After establishing a regime, a single different classification
        should NOT switch — requires N consecutive readings."""
        rc = RegimeClassifier(stability_window=3)

        # First call: establishes CHOPPY from choppy data
        rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19800.0,
                vwap=19799.5,
                session_high=19803.0,
                session_low=19797.0,
                delta_trend="neutral",
            ),
            recent_bars=_make_choppy_bars(15),
            rvol=0.9,
            in_blackout=False,
            upcoming_high_impact=False,
        )

        # Second call: TRENDING_UP detected but should NOT switch yet
        regime2, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19810.0,
                vwap=19800.0,
                delta_trend="positive",
            ),
            recent_bars=_make_trending_bars(20, "up"),
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        # Should stay CHOPPY — not enough consecutive readings
        assert regime2 == Regime.CHOPPY

    def test_stability_switches_after_window(self):
        """After establishing a regime, 3 consecutive different classifications
        should switch the regime."""
        rc = RegimeClassifier(stability_window=3)

        # First call establishes CHOPPY
        rc.classify(
            tick_snap=_make_tick_snap(delta_trend="neutral"),
            recent_bars=_make_choppy_bars(15),
            rvol=0.9,
            in_blackout=False,
            upcoming_high_impact=False,
        )

        # Now feed 5 trending readings — should switch after 3
        snap = _make_tick_snap(
            last_price=19810.0,
            vwap=19800.0,
            delta_trend="positive",
        )
        bars = _make_trending_bars(20, "up")

        regimes = []
        for _ in range(5):
            r, _ = rc.classify(
                tick_snap=snap,
                recent_bars=bars,
                rvol=1.0,
                in_blackout=False,
                upcoming_high_impact=False,
            )
            regimes.append(r)

        # After 3 consecutive, should switch to TRENDING_UP
        assert Regime.TRENDING_UP in regimes

    def test_news_driven_overrides_stability(self):
        """NEWS_DRIVEN with high confidence skips stability filter."""
        rc = RegimeClassifier(stability_window=5)  # high window

        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[],
            rvol=1.0,
            in_blackout=True,  # triggers NEWS_DRIVEN
            upcoming_high_impact=False,
        )
        # Should switch immediately despite stability window
        assert regime == Regime.NEWS_DRIVEN

    def test_breakout_overrides_stability(self):
        """BREAKOUT with high confidence skips stability filter."""
        rc = RegimeClassifier(stability_window=5)

        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19905.0,  # 5pts above PDH
                vwap=19880.0,
                delta_trend="positive",
                tape_speed=50.0,
            ),
            recent_bars=_make_trending_bars(15, "up", start=19895.0),
            rvol=1.6,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=19900.0,
        )
        # High-confidence breakout should override stability
        assert regime == Regime.BREAKOUT

    def test_mixed_signals_keep_current_regime(self):
        """When alternating between different regimes, stay at established regime."""
        rc = RegimeClassifier(stability_window=3)

        # First call: establish CHOPPY
        rc.classify(
            tick_snap=_make_tick_snap(delta_trend="neutral"),
            recent_bars=_make_choppy_bars(15),
            rvol=0.9,
            in_blackout=False,
            upcoming_high_impact=False,
        )

        # Alternating trend signals — should stay CHOPPY
        for i in range(6):
            if i % 2 == 0:
                snap = _make_tick_snap(
                    last_price=19810.0, vwap=19800.0, delta_trend="positive"
                )
                bars = _make_trending_bars(20, "up")
            else:
                snap = _make_tick_snap(
                    last_price=19790.0, vwap=19800.0, delta_trend="negative"
                )
                bars = _make_trending_bars(20, "down")

            regime, _ = rc.classify(
                tick_snap=snap,
                recent_bars=bars,
                rvol=1.0,
                in_blackout=False,
                upcoming_high_impact=False,
            )

        # Should stay CHOPPY because alternating prevents stability
        assert regime == Regime.CHOPPY


class TestAccessors:
    """Test property accessors and stats."""

    def test_initial_state(self):
        rc = RegimeClassifier()
        assert rc.current_regime == Regime.CHOPPY
        assert rc.current_confidence == 0.5
        assert rc.classification_count == 0

    def test_stats(self):
        rc = RegimeClassifier()
        rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[],
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        stats = rc.stats
        assert "regime" in stats
        assert "confidence" in stats
        assert "classifications" in stats
        assert stats["classifications"] == 1

    def test_classification_count_increments(self):
        rc = RegimeClassifier()
        for _ in range(5):
            rc.classify(
                tick_snap=_make_tick_snap(),
                recent_bars=[],
                rvol=1.0,
                in_blackout=False,
                upcoming_high_impact=False,
            )
        assert rc.classification_count == 5


class TestSwingStructure:
    """Test the swing structure scoring helper."""

    def test_strong_uptrend_swing(self):
        """Bars with monotonically increasing highs/lows in 3 segments."""
        rc = RegimeClassifier()
        # 15 bars, each segment of 5 has progressively higher H/L
        bars = []
        for i in range(15):
            price = 19800.0 + i * 1.0  # strong upward progression
            bars.append(_make_bar(
                close=price,
                open_=price - 0.5,
                high=price + 0.5,
                low=price - 1.0,
            ))
        score = rc._swing_structure_score(bars, is_uptrend=True)
        assert score >= 0.10  # at least partial confirmation

    def test_choppy_no_swing(self):
        """Choppy bars should have low swing score."""
        rc = RegimeClassifier()
        bars = _make_choppy_bars(15)
        score = rc._swing_structure_score(bars, is_uptrend=True)
        assert score == 0.0  # no swing structure

    def test_too_few_bars(self):
        """< 9 bars returns 0."""
        rc = RegimeClassifier()
        bars = _make_trending_bars(5)
        score = rc._swing_structure_score(bars, is_uptrend=True)
        assert score == 0.0


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_bars_no_crash(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[],
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.CHOPPY

    def test_zero_price_no_crash(self):
        rc = RegimeClassifier()
        regime, conf = rc.classify(
            tick_snap=_make_tick_snap(last_price=0.0),
            recent_bars=[],
            rvol=0.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.CHOPPY

    def test_all_zero_levels_no_crash(self):
        """No prior-day or overnight levels should not crash."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=_make_trending_bars(20, "up"),
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=0.0,
            prior_day_low=0.0,
            overnight_high=0.0,
            overnight_low=0.0,
        )
        # Should classify based on trend, not crash
        assert regime is not None

    def test_single_bar_no_crash(self):
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[_make_bar(19800.0)],
            rvol=1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        assert regime == Regime.CHOPPY

    def test_negative_rvol_handled(self):
        """Negative RVOL (shouldn't happen but be defensive)."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[],
            rvol=-1.0,
            in_blackout=False,
            upcoming_high_impact=False,
        )
        # Should not crash, negative < 0.70 but also < 0 so skipped
        assert regime is not None


class TestPriorityOrder:
    """Verify the priority order: NEWS > LOW_VOL > BREAKOUT > TREND > CHOPPY."""

    def test_news_over_low_volume(self):
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(),
            recent_bars=[],
            rvol=0.3,  # low volume
            in_blackout=True,  # but also blackout
            upcoming_high_impact=False,
        )
        assert regime == Regime.NEWS_DRIVEN

    def test_low_volume_over_breakout(self):
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19905.0,
                delta_trend="positive",
                tape_speed=50.0,
            ),
            recent_bars=_make_trending_bars(15, "up", start=19895.0),
            rvol=0.4,  # low volume — even with breakout signals
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=19900.0,
        )
        assert regime == Regime.LOW_VOLUME

    def test_breakout_over_trend(self):
        """When both breakout and trend signals present, breakout wins."""
        rc = RegimeClassifier()
        regime, _ = rc.classify(
            tick_snap=_make_tick_snap(
                last_price=19905.0,  # above PDH
                vwap=19880.0,
                delta_trend="positive",
                tape_speed=50.0,
            ),
            recent_bars=_make_trending_bars(20, "up", start=19895.0),
            rvol=1.5,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=19900.0,
        )
        assert regime == Regime.BREAKOUT
