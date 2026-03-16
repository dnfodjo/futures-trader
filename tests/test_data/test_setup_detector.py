"""Tests for SetupDetector — pre-processes market data to detect trade setups."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.core.types import (
    KeyLevels,
    MarketState,
    OrderFlowData,
    Regime,
    SessionPhase,
)
from src.data.setup_detector import DetectedSetup, SetupDetector, SetupType

ET = ZoneInfo("US/Eastern")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_state(**overrides: object) -> MarketState:
    """Create a MarketState with sensible defaults for setup detection tests."""
    defaults: dict = {
        "timestamp": datetime.now(tz=UTC),
        "symbol": "MNQM6",
        "last_price": 19850.0,
        "bid": 19849.75,
        "ask": 19850.25,
        "spread": 0.50,
        "session_phase": SessionPhase.MORNING,
        "regime": Regime.TRENDING_UP,
        "regime_confidence": 0.8,
        "levels": KeyLevels(
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            prior_day_close=19800.0,
            overnight_high=19870.0,
            overnight_low=19750.0,
            session_high=19880.0,
            session_low=19790.0,
            session_open=19810.0,
            vwap=19840.0,
            poc=19835.0,
            value_area_high=19865.0,
            value_area_low=19815.0,
        ),
        "flow": OrderFlowData(
            cumulative_delta=200.0,
            delta_1min=30.0,
            delta_5min=100.0,
            delta_trend="positive",
            rvol=1.2,
            volume_1min=500,
            large_lot_count_5min=2,
            tape_speed=20.0,
        ),
    }
    defaults.update(overrides)

    # Handle nested overrides for levels and flow
    if "levels" in overrides and isinstance(overrides["levels"], dict):
        defaults["levels"] = KeyLevels(**overrides["levels"])
    if "flow" in overrides and isinstance(overrides["flow"], dict):
        defaults["flow"] = OrderFlowData(**overrides["flow"])

    return MarketState(**defaults)


def _make_bar(
    timestamp: datetime | None = None,
    open_: float = 19850.0,
    high: float = 19852.0,
    low: float = 19848.0,
    close: float = 19851.0,
    volume: int = 100,
) -> dict:
    """Create a simple bar dict for bars_1s lists."""
    return {
        "timestamp": timestamp or datetime.now(tz=UTC),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _find_setup(setups: list[DetectedSetup], setup_type: SetupType) -> DetectedSetup | None:
    """Find the first setup of a given type in the results."""
    for s in setups:
        if s.setup_type == setup_type:
            return s
    return None


def _has_setup(setups: list[DetectedSetup], setup_type: SetupType) -> bool:
    """Check if a setup type exists in the results."""
    return _find_setup(setups, setup_type) is not None


# ── Test: SetupType Enum ──────────────────────────────────────────────────────


class TestSetupTypeEnum:
    def test_all_types_exist(self) -> None:
        """All expected setup types are defined."""
        expected = {
            "vwap_pullback",
            "vwap_rejection",
            "level_test",
            "failed_breakout",
            "opening_range_break",
            "delta_divergence",
            "absorption",
            "trend_continuation",
            "exhaustion",
            "mean_reversion",
        }
        actual = {t.value for t in SetupType}
        assert actual == expected

    def test_setup_type_is_string(self) -> None:
        """SetupType values are strings."""
        assert SetupType.VWAP_PULLBACK == "vwap_pullback"
        assert isinstance(SetupType.LEVEL_TEST, str)


# ── Test: DetectedSetup Model ─────────────────────────────────────────────────


class TestDetectedSetup:
    def test_creation(self) -> None:
        """DetectedSetup can be created with all required fields."""
        setup = DetectedSetup(
            setup_type=SetupType.VWAP_PULLBACK,
            side="long",
            confidence=0.75,
            trigger_price=19840.0,
            suggested_stop_distance=2.0,
            description="Price pulled back to VWAP in uptrend",
            confirming_signals=["positive delta", "trending up regime"],
            invalidation="Price breaks below VWAP by 3+ pts",
        )
        assert setup.setup_type == SetupType.VWAP_PULLBACK
        assert setup.side == "long"
        assert setup.confidence == 0.75
        assert setup.trigger_price == 19840.0
        assert len(setup.confirming_signals) == 2

    def test_confidence_clamped(self) -> None:
        """Confidence is between 0.0 and 1.0."""
        setup = DetectedSetup(
            setup_type=SetupType.LEVEL_TEST,
            side="long",
            confidence=0.5,
            trigger_price=19700.0,
            suggested_stop_distance=3.0,
            description="test",
            confirming_signals=[],
            invalidation="test",
        )
        assert 0.0 <= setup.confidence <= 1.0


# ── Test: SetupDetector Construction ──────────────────────────────────────────


class TestSetupDetectorInit:
    def test_default_construction(self) -> None:
        """SetupDetector can be created with no arguments."""
        detector = SetupDetector()
        assert detector is not None

    def test_detect_returns_list(self) -> None:
        """detect() returns a list."""
        detector = SetupDetector()
        state = _make_state()
        result = detector.detect(state, bars_1s=[])
        assert isinstance(result, list)


# ── Test: VWAP Pullback Detection ──────────────────────────────────────────


class TestVWAPPullback:
    def test_long_pullback_to_vwap(self) -> None:
        """Detects long pullback when price was above VWAP and pulls back near it."""
        detector = SetupDetector()
        # VWAP at 19840, price at 19841 (within 2pts), was previously above by >5pts
        # session_high=19880 implies price was well above VWAP
        state = _make_state(
            last_price=19841.0,
            regime=Regime.TRENDING_UP,
            levels=KeyLevels(
                vwap=19840.0,
                session_high=19880.0,
                session_low=19790.0,
                session_open=19810.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=200.0,
                delta_5min=80.0,
                delta_trend="positive",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.VWAP_PULLBACK)
        assert setup is not None
        assert setup.side == "long"
        assert setup.confidence > 0.0

    def test_short_pullback_to_vwap(self) -> None:
        """Detects short pullback when price was below VWAP and pulls back near it."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19839.0,
            regime=Regime.TRENDING_DOWN,
            levels=KeyLevels(
                vwap=19840.0,
                session_high=19880.0,
                session_low=19790.0,
                session_open=19860.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=-200.0,
                delta_5min=-80.0,
                delta_trend="negative",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.VWAP_PULLBACK)
        assert setup is not None
        assert setup.side == "short"

    def test_no_pullback_when_far_from_vwap(self) -> None:
        """No VWAP pullback when price is far from VWAP."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19860.0,
            levels=KeyLevels(vwap=19840.0, session_high=19880.0, session_low=19790.0),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.VWAP_PULLBACK)


# ── Test: Level Test Detection ────────────────────────────────────────────────


class TestLevelTest:
    def test_approaches_prior_day_low_support(self) -> None:
        """Detects level test when price approaches PDL with buying delta."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19701.5,
            levels=KeyLevels(
                prior_day_low=19700.0,
                prior_day_high=19900.0,
                session_high=19800.0,
                session_low=19701.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=50.0,
                delta_1min=20.0,
                delta_trend="positive",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.LEVEL_TEST)
        assert setup is not None
        assert setup.side == "long"

    def test_approaches_prior_day_high_resistance(self) -> None:
        """Detects level test when price approaches PDH with selling delta."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19898.5,
            levels=KeyLevels(
                prior_day_high=19900.0,
                prior_day_low=19700.0,
                session_high=19899.0,
                session_low=19800.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=-50.0,
                delta_1min=-20.0,
                delta_trend="negative",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.LEVEL_TEST)
        assert setup is not None
        assert setup.side == "short"

    def test_no_level_test_when_far(self) -> None:
        """No level test when price is far from all key levels."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19850.0,
            levels=KeyLevels(
                prior_day_high=19900.0,
                prior_day_low=19700.0,
                overnight_high=19870.0,
                overnight_low=19750.0,
                session_high=19880.0,
                session_low=19790.0,
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.LEVEL_TEST)


# ── Test: Failed Breakout Detection ───────────────────────────────────────────


class TestFailedBreakout:
    def test_failed_breakout_above_level(self) -> None:
        """Detects failed breakout when price broke above a level but came back."""
        detector = SetupDetector()
        # Session high reached 19903 (broke above PDH of 19900 by >2pts),
        # but now price is back at 19897 (below PDH)
        state = _make_state(
            last_price=19897.0,
            levels=KeyLevels(
                prior_day_high=19900.0,
                prior_day_low=19700.0,
                session_high=19903.0,
                session_low=19800.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=-100.0,
                delta_1min=-30.0,
                delta_trend="negative",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.FAILED_BREAKOUT)
        assert setup is not None
        assert setup.side == "short"

    def test_failed_breakout_below_level(self) -> None:
        """Detects failed breakout when price broke below a level but came back."""
        detector = SetupDetector()
        # Session low went to 19697 (broke below PDL of 19700 by >2pts),
        # but now price is back at 19703 (above PDL)
        state = _make_state(
            last_price=19703.0,
            levels=KeyLevels(
                prior_day_high=19900.0,
                prior_day_low=19700.0,
                session_high=19800.0,
                session_low=19697.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=100.0,
                delta_1min=30.0,
                delta_trend="positive",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.FAILED_BREAKOUT)
        assert setup is not None
        assert setup.side == "long"

    def test_no_failed_breakout_when_holding_above(self) -> None:
        """No failed breakout when price broke and is still above the level."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19905.0,
            levels=KeyLevels(
                prior_day_high=19900.0,
                session_high=19907.0,
                session_low=19800.0,
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.FAILED_BREAKOUT)


# ── Test: Delta Divergence Detection ──────────────────────────────────────────


class TestDeltaDivergence:
    def test_bearish_divergence_at_highs(self) -> None:
        """Detects bearish divergence: new session high but delta is declining."""
        detector = SetupDetector()
        # Price at session high, but delta is negative (bearish divergence)
        state = _make_state(
            last_price=19879.0,
            levels=KeyLevels(
                session_high=19880.0,
                session_low=19790.0,
                vwap=19840.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=-50.0,
                delta_5min=-30.0,
                delta_trend="negative",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.DELTA_DIVERGENCE)
        assert setup is not None
        assert setup.side == "short"

    def test_bullish_divergence_at_lows(self) -> None:
        """Detects bullish divergence: new session low but delta is positive."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19791.0,
            levels=KeyLevels(
                session_high=19880.0,
                session_low=19790.0,
                vwap=19840.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=50.0,
                delta_5min=30.0,
                delta_trend="positive",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.DELTA_DIVERGENCE)
        assert setup is not None
        assert setup.side == "long"

    def test_no_divergence_when_price_mid_range(self) -> None:
        """No divergence when price is in the middle of the range."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19840.0,
            levels=KeyLevels(
                session_high=19880.0,
                session_low=19790.0,
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.DELTA_DIVERGENCE)


# ── Test: Absorption Detection ────────────────────────────────────────────────


class TestAbsorption:
    def test_absorption_at_support(self) -> None:
        """Detects absorption: high volume at a level with minimal price movement."""
        detector = SetupDetector()
        # High volume (rvol > 2.0) near PDL but price barely moved
        state = _make_state(
            last_price=19701.0,
            levels=KeyLevels(
                prior_day_low=19700.0,
                prior_day_high=19900.0,
                session_high=19800.0,
                session_low=19700.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=100.0,
                delta_1min=40.0,
                delta_trend="positive",
                rvol=2.5,
                volume_1min=1500,
            ),
        )
        # bars show high volume but small range
        bars = [
            _make_bar(open_=19700.5, high=19701.5, low=19699.5, close=19701.0, volume=500),
            _make_bar(open_=19701.0, high=19701.5, low=19700.0, close=19700.5, volume=500),
            _make_bar(open_=19700.5, high=19701.0, low=19700.0, close=19701.0, volume=500),
        ]
        setups = detector.detect(state, bars_1s=bars)
        setup = _find_setup(setups, SetupType.ABSORPTION)
        assert setup is not None
        assert setup.side == "long"

    def test_no_absorption_with_normal_volume(self) -> None:
        """No absorption when volume is normal."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19701.0,
            levels=KeyLevels(prior_day_low=19700.0),
            flow=OrderFlowData(rvol=1.0, volume_1min=200),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.ABSORPTION)


# ── Test: Mean Reversion Detection ────────────────────────────────────────────


class TestMeanReversion:
    def test_mean_reversion_in_choppy_regime(self) -> None:
        """Detects mean reversion when extended from VWAP in choppy market."""
        detector = SetupDetector()
        # Choppy regime, price >10pts above VWAP => mean revert short
        state = _make_state(
            last_price=19855.0,
            regime=Regime.CHOPPY,
            levels=KeyLevels(
                vwap=19840.0,
                session_high=19860.0,
                session_low=19820.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=20.0,
                delta_trend="neutral",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.MEAN_REVERSION)
        assert setup is not None
        assert setup.side == "short"

    def test_mean_reversion_short_below_vwap(self) -> None:
        """Detects mean reversion long when price is extended below VWAP in choppy market."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19825.0,
            regime=Regime.CHOPPY,
            levels=KeyLevels(
                vwap=19840.0,
                session_high=19860.0,
                session_low=19820.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=-20.0,
                delta_trend="neutral",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        setup = _find_setup(setups, SetupType.MEAN_REVERSION)
        assert setup is not None
        assert setup.side == "long"

    def test_no_mean_reversion_in_trending(self) -> None:
        """No mean reversion in trending regime."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19855.0,
            regime=Regime.TRENDING_UP,
            levels=KeyLevels(vwap=19840.0),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.MEAN_REVERSION)


# ── Test: Exhaustion Detection ────────────────────────────────────────────────


class TestExhaustion:
    def test_exhaustion_on_rapid_up_move(self) -> None:
        """Detects exhaustion after rapid upward move with declining delta."""
        detector = SetupDetector()
        now = datetime.now(tz=UTC)
        # Price moved >15pts up rapidly, but delta is declining
        state = _make_state(
            last_price=19870.0,
            levels=KeyLevels(
                session_high=19870.0,
                session_low=19790.0,
                vwap=19840.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=30.0,
                delta_1min=-10.0,
                delta_5min=50.0,
                delta_trend="negative",
            ),
        )
        # bars showing a >15pt move in <5 minutes
        bars = []
        for i in range(60):
            price = 19850.0 + (i * 0.35)
            bars.append(_make_bar(
                timestamp=now - timedelta(seconds=60 - i),
                open_=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price + 0.25,
                volume=100 - i,  # declining volume
            ))
        setups = detector.detect(state, bars_1s=bars)
        setup = _find_setup(setups, SetupType.EXHAUSTION)
        assert setup is not None
        assert setup.side == "short"  # opposite of the up-move

    def test_no_exhaustion_on_small_move(self) -> None:
        """No exhaustion when the move is small."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19845.0,
            levels=KeyLevels(
                session_high=19850.0,
                session_low=19840.0,
            ),
        )
        bars = [_make_bar(close=19845.0, volume=100) for _ in range(10)]
        setups = detector.detect(state, bars_1s=bars)
        assert not _has_setup(setups, SetupType.EXHAUSTION)


# ── Test: Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_bars(self) -> None:
        """Handles empty bars_1s list gracefully."""
        detector = SetupDetector()
        state = _make_state()
        setups = detector.detect(state, bars_1s=[])
        assert isinstance(setups, list)

    def test_zero_vwap(self) -> None:
        """Handles zero VWAP gracefully (no VWAP-based setups)."""
        detector = SetupDetector()
        state = _make_state(
            levels=KeyLevels(vwap=0.0),
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.VWAP_PULLBACK)
        assert not _has_setup(setups, SetupType.MEAN_REVERSION)

    def test_zero_delta(self) -> None:
        """Handles zero delta gracefully."""
        detector = SetupDetector()
        state = _make_state(
            flow=OrderFlowData(
                cumulative_delta=0.0,
                delta_1min=0.0,
                delta_5min=0.0,
                delta_trend="neutral",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        # Should not crash; may still detect some setups
        assert isinstance(setups, list)

    def test_missing_levels_no_crash(self) -> None:
        """Handles all-zero levels without crashing."""
        detector = SetupDetector()
        state = _make_state(
            levels=KeyLevels(),  # all defaults (0.0)
        )
        setups = detector.detect(state, bars_1s=[])
        assert isinstance(setups, list)
        # No level-based setups should fire
        assert not _has_setup(setups, SetupType.LEVEL_TEST)
        assert not _has_setup(setups, SetupType.FAILED_BREAKOUT)


# ── Test: Multiple Setups Simultaneously ──────────────────────────────────────


class TestMultipleSetups:
    def test_multiple_setups_detected(self) -> None:
        """Multiple setups can fire at the same time."""
        detector = SetupDetector()
        # Scenario: choppy regime, price extended above VWAP (mean reversion),
        # near session high with negative delta (delta divergence),
        # also near PDH (level test)
        state = _make_state(
            last_price=19899.0,
            regime=Regime.CHOPPY,
            levels=KeyLevels(
                prior_day_high=19900.0,
                prior_day_low=19700.0,
                session_high=19900.0,
                session_low=19800.0,
                vwap=19840.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=-50.0,
                delta_1min=-20.0,
                delta_5min=-30.0,
                delta_trend="negative",
                rvol=1.5,
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        # Should have at least 2 setups (mean reversion + level test or divergence)
        assert len(setups) >= 2

    def test_all_setups_have_required_fields(self) -> None:
        """Every detected setup has all required fields populated."""
        detector = SetupDetector()
        state = _make_state(
            last_price=19841.0,
            regime=Regime.TRENDING_UP,
            levels=KeyLevels(
                vwap=19840.0,
                session_high=19880.0,
                session_low=19790.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=200.0,
                delta_trend="positive",
            ),
        )
        setups = detector.detect(state, bars_1s=[])
        for setup in setups:
            assert setup.setup_type in SetupType
            assert setup.side in ("long", "short")
            assert 0.0 <= setup.confidence <= 1.0
            assert setup.trigger_price > 0
            assert setup.suggested_stop_distance > 0
            assert len(setup.description) > 0
            assert isinstance(setup.confirming_signals, list)
            assert len(setup.invalidation) > 0


# ── Test: Opening Range Break ─────────────────────────────────────────────────


class TestOpeningRangeBreak:
    def test_orb_upside_break(self) -> None:
        """Detects opening range break to the upside after 9:45 AM."""
        detector = SetupDetector()
        # Set opening range: 9:30-9:45 bars
        now_et = datetime(2025, 3, 14, 9, 50, tzinfo=ET)
        or_bars = []
        for i in range(15):
            ts = datetime(2025, 3, 14, 9, 30 + i, tzinfo=ET)
            or_bars.append(_make_bar(
                timestamp=ts,
                open_=19800.0 + i,
                high=19815.0,  # OR high
                low=19795.0,   # OR low
                close=19800.0 + i,
            ))

        state = _make_state(
            timestamp=now_et.astimezone(UTC),
            last_price=19818.0,  # above OR high
            session_phase=SessionPhase.OPEN_DRIVE,
            levels=KeyLevels(
                session_high=19818.0,
                session_low=19795.0,
                session_open=19800.0,
                vwap=19805.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=200.0,
                delta_trend="positive",
            ),
        )
        setups = detector.detect(state, bars_1s=or_bars)
        setup = _find_setup(setups, SetupType.OPENING_RANGE_BREAK)
        assert setup is not None
        assert setup.side == "long"

    def test_no_orb_before_945(self) -> None:
        """No ORB detection before 9:45 AM (OR is not complete yet)."""
        detector = SetupDetector()
        now_et = datetime(2025, 3, 14, 9, 40, tzinfo=ET)
        bars = [_make_bar(
            timestamp=datetime(2025, 3, 14, 9, 30 + i, tzinfo=ET),
            high=19815.0,
            low=19795.0,
        ) for i in range(10)]

        state = _make_state(
            timestamp=now_et.astimezone(UTC),
            last_price=19818.0,
            session_phase=SessionPhase.OPEN_DRIVE,
        )
        setups = detector.detect(state, bars_1s=bars)
        assert not _has_setup(setups, SetupType.OPENING_RANGE_BREAK)


# ── Test: Trend Continuation ──────────────────────────────────────────────────


class TestTrendContinuation:
    def test_uptrend_continuation(self) -> None:
        """Detects trend continuation in uptrend with 38-62% pullback."""
        detector = SetupDetector()
        # Uptrend: swing from 19800 to 19880 (80pts). Pullback to ~19850 = 37.5% pullback
        state = _make_state(
            last_price=19850.0,
            regime=Regime.TRENDING_UP,
            regime_confidence=0.85,
            levels=KeyLevels(
                session_high=19880.0,
                session_low=19800.0,
                vwap=19840.0,
            ),
            flow=OrderFlowData(
                cumulative_delta=150.0,
                delta_1min=25.0,
                delta_5min=60.0,
                delta_trend="positive",
            ),
        )
        # Bars showing an upswing then pullback
        now = datetime.now(tz=UTC)
        bars = []
        # Up move
        for i in range(50):
            price = 19800.0 + (i * 1.6)
            bars.append(_make_bar(
                timestamp=now - timedelta(seconds=100 - i),
                close=price,
                open_=price - 0.5,
                high=price + 0.5,
                low=price - 1.0,
            ))
        # Pullback
        for i in range(30):
            price = 19880.0 - (i * 1.0)
            bars.append(_make_bar(
                timestamp=now - timedelta(seconds=50 - i),
                close=price,
                open_=price + 0.5,
                high=price + 1.0,
                low=price - 0.5,
            ))

        setups = detector.detect(state, bars_1s=bars)
        setup = _find_setup(setups, SetupType.TREND_CONTINUATION)
        assert setup is not None
        assert setup.side == "long"

    def test_no_trend_continuation_in_choppy(self) -> None:
        """No trend continuation in choppy regime."""
        detector = SetupDetector()
        state = _make_state(
            regime=Regime.CHOPPY,
        )
        setups = detector.detect(state, bars_1s=[])
        assert not _has_setup(setups, SetupType.TREND_CONTINUATION)
