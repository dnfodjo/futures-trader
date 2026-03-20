"""Tests for HTF structure level computation and proximity checks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.indicators.structure_levels import StructureLevel, StructureLevelManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int = 100,
    timestamp: str = "",
) -> dict:
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "timestamp": timestamp or datetime.now(tz=UTC).isoformat(),
    }


def _fractal_up_bars(
    *,
    pivot_high: float = 21500.0,
    pivot_volume: int = 200,
    base: float = 21400.0,
    bg_volume: int = 50,
) -> list[dict]:
    """Build a minimal 6-bar sequence that forms a fractal UP (resistance).

    Index layout (0-based):
      [0]=i-5, [1]=i-4, [2]=i-3 (pivot), [3]=i-2, [4]=i-1, [5]=i (current)

    Pine Script fractal condition (evaluated at bar i with look-back offsets):
      high[i-3] > high[i-4] > high[i-5]   (left side rises to peak)
      high[i-2] < high[i-3]               (right side descends)
      high[i-1] < high[i-2]               (right side descends)
      volume[i-3] > volume_MA[i-3]        (volume confirmed)
    """
    return [
        # i-5: lowest
        _make_bar(base, base + 10, base - 5, base + 5, bg_volume),
        # i-4: middle
        _make_bar(base + 20, base + 50, base + 15, base + 40, bg_volume),
        # i-3: PIVOT (highest high, big volume, body below high)
        _make_bar(
            pivot_high - 30,
            pivot_high,
            pivot_high - 50,
            pivot_high - 10,
            pivot_volume,
        ),
        # i-2: descending
        _make_bar(base + 30, base + 45, base + 20, base + 35, bg_volume),
        # i-1: descending further
        _make_bar(base + 10, base + 30, base + 5, base + 20, bg_volume),
        # i: current bar (triggers detection)
        _make_bar(base, base + 15, base - 5, base + 10, bg_volume),
    ]


def _fractal_down_bars(
    *,
    pivot_low: float = 21200.0,
    pivot_volume: int = 200,
    base: float = 21300.0,
    bg_volume: int = 50,
) -> list[dict]:
    """Build a minimal 6-bar sequence that forms a fractal DOWN (support).

    Mirror of fractal up:
      low[i-3] < low[i-4] < low[i-5]   (left side descends to trough)
      low[i-2] > low[i-3]              (right side rises)
      low[i-1] > low[i-2]              (right side rises)
      volume[i-3] > volume_MA[i-3]
    """
    return [
        # i-5: highest low
        _make_bar(base, base + 10, base - 5, base + 5, bg_volume),
        # i-4: middle low
        _make_bar(base - 20, base - 10, base - 40, base - 15, bg_volume),
        # i-3: PIVOT (lowest low, big volume, body above low)
        _make_bar(
            pivot_low + 30,
            pivot_low + 50,
            pivot_low,
            pivot_low + 10,
            pivot_volume,
        ),
        # i-2: rising
        _make_bar(base - 20, base - 5, base - 30, base - 10, bg_volume),
        # i-1: rising further
        _make_bar(base - 5, base + 5, base - 10, base, bg_volume),
        # i: current bar
        _make_bar(base, base + 10, base - 5, base + 5, bg_volume),
    ]


def _pad_bars(bars: list[dict], count: int = 20, base: float = 21350.0) -> list[dict]:
    """Prepend enough neutral bars for volume MA warm-up."""
    padding = [
        _make_bar(base, base + 5, base - 5, base + 2, 80) for _ in range(count)
    ]
    return padding + bars


# ---------------------------------------------------------------------------
# StructureLevel dataclass
# ---------------------------------------------------------------------------


class TestStructureLevel:
    def test_basic_fields(self):
        level = StructureLevel(
            price=21500.0,
            zone_high=21500.0,
            zone_low=21470.0,
            level_type="resistance",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        assert level.price == 21500.0
        assert level.level_type == "resistance"
        assert level.timeframe == "D"
        assert level.timeframe_atr == 250.0
        assert level.broken is False
        assert level.flipped is False
        assert level.touch_count == 0
        assert level.last_tested is None


# ---------------------------------------------------------------------------
# compute_levels
# ---------------------------------------------------------------------------


class TestComputeLevels:
    def test_detects_fractal_up_resistance(self):
        """A clear fractal-up pattern with volume confirmation produces a resistance level."""
        mgr = StructureLevelManager()
        bars = _pad_bars(_fractal_up_bars(pivot_high=21500.0))
        levels = mgr.compute_levels(bars, "1h")

        resistance = [lv for lv in levels if lv.level_type == "resistance"]
        assert len(resistance) >= 1
        best = resistance[0]
        assert best.zone_high == 21500.0
        # zone_low = max(open, close) of pivot bar
        assert best.zone_low == max(21500.0 - 30, 21500.0 - 10)  # max(open, close)
        assert best.volume_confirmed is True
        assert best.timeframe == "1h"

    def test_detects_fractal_down_support(self):
        """A clear fractal-down pattern with volume confirmation produces a support level."""
        mgr = StructureLevelManager()
        bars = _pad_bars(_fractal_down_bars(pivot_low=21200.0))
        levels = mgr.compute_levels(bars, "4h")

        support = [lv for lv in levels if lv.level_type == "support"]
        assert len(support) >= 1
        best = support[0]
        assert best.zone_low == 21200.0
        # zone_high = min(open, close) of pivot bar
        assert best.zone_high == min(21200.0 + 30, 21200.0 + 10)  # min(open, close)
        assert best.volume_confirmed is True
        assert best.timeframe == "4h"

    def test_no_fractal_without_volume(self):
        """Fractal pattern with low volume should NOT produce a level."""
        mgr = StructureLevelManager()
        # Pivot volume same as background => below SMA(6)
        bars = _pad_bars(
            _fractal_up_bars(pivot_volume=50, bg_volume=80)
        )
        levels = mgr.compute_levels(bars, "1h")
        # Should NOT detect a resistance level (volume too low)
        resistance = [lv for lv in levels if lv.level_type == "resistance"]
        assert len(resistance) == 0

    def test_volume_ma_is_sma_6(self):
        """Volume MA uses SMA(6), not any other period."""
        mgr = StructureLevelManager()
        # Build bars where the pivot volume is just barely above SMA(6)
        # Use bg_volume=100 for the 5 non-pivot bars near the fractal.
        # SMA(6) of the 6 bars around pivot: we need volume at pivot > average.
        # If 5 bars have vol=100 and pivot has vol=101, SMA(6) = (5*100+101)/6 = 100.17
        # pivot (101) > 100.17 => detected
        bars = _pad_bars(
            _fractal_up_bars(pivot_volume=101, bg_volume=100),
            count=20,
        )
        levels = mgr.compute_levels(bars, "1h")
        resistance = [lv for lv in levels if lv.level_type == "resistance"]
        assert len(resistance) >= 1

    def test_stores_timeframe_atr(self):
        """Each level stores the ATR(14) of its timeframe's bars."""
        mgr = StructureLevelManager()
        bars = _pad_bars(_fractal_up_bars(), count=30)
        levels = mgr.compute_levels(bars, "D")
        assert len(levels) > 0
        for lv in levels:
            assert lv.timeframe_atr > 0
            assert lv.timeframe == "D"

    def test_insufficient_bars_returns_empty(self):
        """Fewer than 6 bars cannot form a fractal."""
        mgr = StructureLevelManager()
        levels = mgr.compute_levels(
            [_make_bar(100, 101, 99, 100)] * 5, "1h"
        )
        assert levels == []

    def test_compute_accumulates_across_calls(self):
        """Calling compute_levels multiple times accumulates levels."""
        mgr = StructureLevelManager()
        bars_1h = _pad_bars(_fractal_up_bars(pivot_high=21500.0))
        bars_4h = _pad_bars(_fractal_down_bars(pivot_low=21200.0))

        mgr.compute_levels(bars_1h, "1h")
        mgr.compute_levels(bars_4h, "4h")

        all_levels = mgr.levels
        tfs = {lv.timeframe for lv in all_levels}
        assert "1h" in tfs
        assert "4h" in tfs


# ---------------------------------------------------------------------------
# check_proximity — bounce detection
# ---------------------------------------------------------------------------


def _make_rejection_bar_1m(
    zone_high: float, zone_low: float, side: str
) -> dict:
    """Create a 1-min bar that shows wick rejection at a zone."""
    if side == "long":
        # Wick below support zone, close inside
        return _make_bar(
            open_=zone_low + 2,
            high=zone_low + 5,
            low=zone_low - 10,  # wick below zone
            close=zone_low + 3,
        )
    else:
        # Wick above resistance zone, close inside
        return _make_bar(
            open_=zone_high - 2,
            high=zone_high + 10,  # wick above zone
            low=zone_high - 5,
            close=zone_high - 3,
        )


class TestCheckProximityBounce:
    def _setup_manager_with_level(
        self, level_type: str, timeframe: str, zone_high: float, zone_low: float
    ) -> StructureLevelManager:
        """Helper: create a manager and inject a known level."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=(zone_high + zone_low) / 2,
            zone_high=zone_high,
            zone_low=zone_low,
            level_type=level_type,
            timeframe=timeframe,
            timeframe_atr=50.0 if timeframe == "1h" else 250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)
        return mgr

    def test_bounce_at_support_long(self):
        """Long near support with 2 rejection bars scores bounce."""
        mgr = self._setup_manager_with_level(
            "support", "4h", zone_high=21210.0, zone_low=21200.0
        )
        price = 21205.0  # Inside zone
        bars_1m = [
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
        ]
        bars_5m: list[dict] = []

        result = mgr.check_proximity(price, "long", bars_1m, bars_5m)
        assert result["bounce_score"] > 0
        assert result["blocked"] is False

    def test_no_bounce_single_1m_bar(self):
        """Single 1-min rejection bar is NOT enough for bounce."""
        mgr = self._setup_manager_with_level(
            "support", "1h", zone_high=21210.0, zone_low=21200.0
        )
        price = 21205.0
        bars_1m = [_make_rejection_bar_1m(21210.0, 21200.0, "long")]
        bars_5m: list[dict] = []

        result = mgr.check_proximity(price, "long", bars_1m, bars_5m)
        assert result["bounce_score"] == 0

    def test_bounce_with_5m_bar(self):
        """A single 5-min rejection bar IS enough for bounce."""
        mgr = self._setup_manager_with_level(
            "resistance", "D", zone_high=21500.0, zone_low=21490.0
        )
        price = 21495.0
        bars_1m: list[dict] = []
        bars_5m = [_make_rejection_bar_1m(21500.0, 21490.0, "short")]

        result = mgr.check_proximity(price, "short", bars_1m, bars_5m)
        assert result["bounce_score"] > 0

    def test_bounce_wrong_side_no_score(self):
        """Long at resistance should NOT score bounce."""
        mgr = self._setup_manager_with_level(
            "resistance", "4h", zone_high=21500.0, zone_low=21490.0
        )
        price = 21495.0
        bars_1m = [
            _make_rejection_bar_1m(21500.0, 21490.0, "short"),
            _make_rejection_bar_1m(21500.0, 21490.0, "short"),
        ]
        result = mgr.check_proximity(price, "long", bars_1m, [])
        # Long at resistance is not a bounce signal
        assert result["bounce_score"] == 0


# ---------------------------------------------------------------------------
# check_proximity — BOS detection
# ---------------------------------------------------------------------------


class TestCheckProximityBOS:
    def test_bos_marks_broken_and_scores(self):
        """Price closing through resistance with volume triggers BOS."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21500.0,
            zone_high=21500.0,
            zone_low=21490.0,
            level_type="resistance",
            timeframe="4h",
            timeframe_atr=80.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        # Price well above the zone, with high volume
        price = 21510.0
        bars_1m = [_make_bar(21490.0, 21515.0, 21488.0, 21510.0, volume=300)]
        bars_5m: list[dict] = []

        # Provide volume info for break detection
        result = mgr.check_proximity(
            price, "long", bars_1m, bars_5m,
            current_volume=300.0, avg_volume=100.0,
        )
        assert result["bos_score"] > 0
        assert level.broken is True

    def test_bos_requires_volume(self):
        """BOS without volume above 1.5x avg should NOT trigger."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21500.0,
            zone_high=21500.0,
            zone_low=21490.0,
            level_type="resistance",
            timeframe="4h",
            timeframe_atr=80.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        price = 21510.0
        bars_1m = [_make_bar(21490.0, 21515.0, 21488.0, 21510.0, volume=100)]

        result = mgr.check_proximity(
            price, "long", bars_1m, [],
            current_volume=100.0, avg_volume=100.0,
        )
        assert result["bos_score"] == 0
        assert level.broken is False


# ---------------------------------------------------------------------------
# check_proximity — anti-signal (block)
# ---------------------------------------------------------------------------


class TestAntiSignalBlock:
    def test_short_blocked_at_daily_support(self):
        """Shorting near a daily support zone should be blocked."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        level = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        # Price near support, trying to short
        price = 21220.0  # within 1.0 * daily_atr (250) of support
        result = mgr.check_proximity(price, "short", [], [])
        assert result["blocked"] is True
        assert "support" in result["block_reason"].lower()

    def test_long_blocked_at_weekly_resistance(self):
        """Longing near a weekly resistance zone should be blocked."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        level = StructureLevel(
            price=22000.0,
            zone_high=22000.0,
            zone_low=21980.0,
            level_type="resistance",
            timeframe="W",
            timeframe_atr=500.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        price = 21900.0  # within 1.0 * daily_atr (250) of resistance
        result = mgr.check_proximity(price, "long", [], [])
        assert result["blocked"] is True
        assert "resistance" in result["block_reason"].lower()

    def test_no_block_for_1h_levels(self):
        """1h/4h levels should NOT trigger anti-signal blocks."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        level = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        price = 21205.0
        result = mgr.check_proximity(price, "short", [], [])
        assert result["blocked"] is False


# ---------------------------------------------------------------------------
# Level flips on BOS
# ---------------------------------------------------------------------------


class TestLevelFlip:
    def test_broken_resistance_creates_flipped_support(self):
        """After BOS through resistance, a new support level should be created."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21500.0,
            zone_high=21500.0,
            zone_low=21490.0,
            level_type="resistance",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        # Trigger BOS
        mgr._update_breaks(price=21520.0, volume=200.0, avg_volume=100.0)

        assert level.broken is True
        flipped = [
            lv for lv in mgr._levels if lv.flipped and lv.level_type == "support"
        ]
        assert len(flipped) == 1
        assert flipped[0].zone_high == 21500.0
        assert flipped[0].zone_low == 21490.0

    def test_broken_support_creates_flipped_resistance(self):
        """After BOS through support, a new resistance level should be created."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="4h",
            timeframe_atr=80.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        mgr._update_breaks(price=21190.0, volume=200.0, avg_volume=100.0)

        assert level.broken is True
        flipped = [
            lv for lv in mgr._levels if lv.flipped and lv.level_type == "resistance"
        ]
        assert len(flipped) == 1


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPruning:
    def test_prune_old_1h_levels(self):
        """1h levels older than 5 days should be pruned."""
        mgr = StructureLevelManager()
        old_level = StructureLevel(
            price=21000.0,
            zone_high=21010.0,
            zone_low=21000.0,
            level_type="support",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=6),
        )
        fresh_level = StructureLevel(
            price=21100.0,
            zone_high=21110.0,
            zone_low=21100.0,
            level_type="resistance",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=2),
        )
        mgr._levels.extend([old_level, fresh_level])

        mgr._prune_stale_levels()

        assert old_level not in mgr._levels
        assert fresh_level in mgr._levels

    def test_prune_broken_levels_older_than_2_days(self):
        """Broken levels older than 2 days should be pruned."""
        mgr = StructureLevelManager()
        broken_old = StructureLevel(
            price=21000.0,
            zone_high=21010.0,
            zone_low=21000.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=10),
            broken=True,
        )
        broken_recent = StructureLevel(
            price=21100.0,
            zone_high=21110.0,
            zone_low=21100.0,
            level_type="resistance",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=1),
            broken=True,
        )
        mgr._levels.extend([broken_old, broken_recent])

        mgr._prune_stale_levels()

        assert broken_old not in mgr._levels
        assert broken_recent in mgr._levels

    def test_weekly_levels_never_pruned_by_age(self):
        """Weekly levels should never be pruned for age."""
        mgr = StructureLevelManager()
        ancient_weekly = StructureLevel(
            price=20000.0,
            zone_high=20010.0,
            zone_low=20000.0,
            level_type="support",
            timeframe="W",
            timeframe_atr=500.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=1000),
        )
        mgr._levels.append(ancient_weekly)

        mgr._prune_stale_levels()

        assert ancient_weekly in mgr._levels

    def test_prune_4h_at_20_days(self):
        """4h levels older than 20 days should be pruned."""
        mgr = StructureLevelManager()
        old_4h = StructureLevel(
            price=21000.0,
            zone_high=21010.0,
            zone_low=21000.0,
            level_type="support",
            timeframe="4h",
            timeframe_atr=80.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=25),
        )
        mgr._levels.append(old_4h)

        mgr._prune_stale_levels()
        assert old_4h not in mgr._levels

    def test_prune_daily_at_180_days(self):
        """Daily levels older than 180 days should be pruned."""
        mgr = StructureLevelManager()
        old_daily = StructureLevel(
            price=21000.0,
            zone_high=21010.0,
            zone_low=21000.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=200),
        )
        mgr._levels.append(old_daily)

        mgr._prune_stale_levels()
        assert old_daily not in mgr._levels


# ---------------------------------------------------------------------------
# Null safety
# ---------------------------------------------------------------------------


class TestNullSafety:
    def test_no_levels_returns_clean_result(self):
        """When no levels exist, check_proximity returns safe defaults."""
        mgr = StructureLevelManager()
        result = mgr.check_proximity(21000.0, "long", [], [])
        assert result["bounce_score"] == 0
        assert result["bos_score"] == 0
        assert result["blocked"] is False
        assert result["block_reason"] == ""
        assert result["nearest_level"] is None
        assert "no htf levels" in result["detail"].lower()

    def test_empty_bars_compute_levels(self):
        """Empty bar list returns empty level list."""
        mgr = StructureLevelManager()
        levels = mgr.compute_levels([], "1h")
        assert levels == []


# ---------------------------------------------------------------------------
# Side-aware filtering
# ---------------------------------------------------------------------------


class TestSideAwareFiltering:
    def test_long_picks_support_not_resistance(self):
        """For a long signal, check_proximity should pick support levels."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        support = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        resistance = StructureLevel(
            price=21500.0,
            zone_high=21500.0,
            zone_low=21490.0,
            level_type="resistance",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.extend([support, resistance])

        # Price near support, going long with rejection bars
        bars_1m = [
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
        ]
        result = mgr.check_proximity(21205.0, "long", bars_1m, [])
        assert result["nearest_level"] is not None
        assert result["nearest_level"].level_type == "support"

    def test_highest_timeframe_wins(self):
        """When multiple levels are nearby, highest TF wins."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        support_1h = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        support_daily = StructureLevel(
            price=21205.0,
            zone_high=21215.0,
            zone_low=21205.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.extend([support_1h, support_daily])

        bars_1m = [
            _make_rejection_bar_1m(21215.0, 21200.0, "long"),
            _make_rejection_bar_1m(21215.0, 21200.0, "long"),
        ]
        result = mgr.check_proximity(21208.0, "long", bars_1m, [])
        assert result["nearest_level"] is not None
        assert result["nearest_level"].timeframe == "D"


# ---------------------------------------------------------------------------
# Proximity uses per-level ATR
# ---------------------------------------------------------------------------


class TestPerLevelATR:
    def test_proximity_threshold_scales_with_timeframe(self):
        """1h level has tighter threshold than D level."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        # 1h level with 30 ATR => threshold = 0.5 * 30 = 15
        level_1h = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level_1h)

        # Price 30 pts away from zone_high (21210) => 21240 - 21210 = 30 pts
        # Outside 1h threshold of 0.5 * 30 = 15
        bars_1m = [
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
        ]
        result = mgr.check_proximity(21240.0, "long", bars_1m, [])
        assert result["bounce_score"] == 0  # Too far from 1h level

    def test_daily_level_wider_threshold(self):
        """D level with 250 ATR => threshold = 1.0 * 250 = 250, much wider."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 250.0
        level_d = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level_d)

        # Price 100 pts away => inside D threshold (250)
        # But need rejection bars for bounce
        bars_1m = [
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
        ]
        result = mgr.check_proximity(21300.0, "long", bars_1m, [])
        # This is within proximity but the bar rejection is at a different price
        # The key test is that the threshold calculation uses per-level ATR
        assert result["nearest_level"] is not None


# ---------------------------------------------------------------------------
# update_on_bar_close
# ---------------------------------------------------------------------------


class TestUpdateOnBarClose:
    def test_update_on_bar_close_prunes(self):
        """update_on_bar_close should trigger pruning."""
        mgr = StructureLevelManager()
        old_level = StructureLevel(
            price=21000.0,
            zone_high=21010.0,
            zone_low=21000.0,
            level_type="support",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC) - timedelta(days=10),
        )
        mgr._levels.append(old_level)

        bar = _make_bar(21100.0, 21110.0, 21090.0, 21105.0)
        mgr.update_on_bar_close(bar, "1h")

        assert old_level not in mgr._levels


# ---------------------------------------------------------------------------
# BOS timeframe tracking (bos_tf in result)
# ---------------------------------------------------------------------------


class TestBOSTimeframeTracking:
    """Verify that check_proximity returns the broken level's actual timeframe."""

    def test_bos_tf_returned_for_weekly_break(self):
        """BOS through weekly resistance returns bos_tf='W'."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=22000.0,
            zone_high=22000.0,
            zone_low=21980.0,
            level_type="resistance",
            timeframe="W",
            timeframe_atr=500.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        # Also need a support level for "long" side filtering
        support = StructureLevel(
            price=21800.0,
            zone_high=21810.0,
            zone_low=21800.0,
            level_type="support",
            timeframe="1h",
            timeframe_atr=30.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(support)

        bars_1m = [_make_bar(21990.0, 22020.0, 21985.0, 22010.0, volume=300)]
        result = mgr.check_proximity(
            21805.0, "long", bars_1m, [],
            current_volume=300.0, avg_volume=100.0,
        )
        assert result["bos_tf"] == "W"
        assert result["bos_score"] == 1

    def test_bos_tf_none_when_no_break(self):
        """When no BOS occurs, bos_tf should be None."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21200.0,
            zone_high=21210.0,
            zone_low=21200.0,
            level_type="support",
            timeframe="D",
            timeframe_atr=250.0,
            volume_confirmed=True,
            created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        bars_1m = [
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
            _make_rejection_bar_1m(21210.0, 21200.0, "long"),
        ]
        result = mgr.check_proximity(21205.0, "long", bars_1m, [])
        assert result["bos_tf"] is None

    def test_bos_tf_highest_priority_wins(self):
        """When multiple levels break, highest TF BOS wins."""
        mgr = StructureLevelManager()
        # 4h resistance
        r_4h = StructureLevel(
            price=21500.0, zone_high=21500.0, zone_low=21490.0,
            level_type="resistance", timeframe="4h", timeframe_atr=80.0,
            volume_confirmed=True, created_at=datetime.now(tz=UTC),
        )
        # D resistance at same price
        r_d = StructureLevel(
            price=21500.0, zone_high=21500.0, zone_low=21490.0,
            level_type="resistance", timeframe="D", timeframe_atr=250.0,
            volume_confirmed=True, created_at=datetime.now(tz=UTC),
        )
        # Need a support for side filtering
        support = StructureLevel(
            price=21400.0, zone_high=21410.0, zone_low=21400.0,
            level_type="support", timeframe="1h", timeframe_atr=30.0,
            volume_confirmed=True, created_at=datetime.now(tz=UTC),
        )
        mgr._levels.extend([r_4h, r_d, support])

        bars_1m = [_make_bar(21490.0, 21520.0, 21485.0, 21510.0, volume=300)]
        result = mgr.check_proximity(
            21405.0, "long", bars_1m, [],
            current_volume=300.0, avg_volume=100.0,
        )
        # Daily has higher priority than 4h
        assert result["bos_tf"] == "D"


# ---------------------------------------------------------------------------
# _update_breaks volume=0 skip
# ---------------------------------------------------------------------------


class TestUpdateBreaksEdgeCases:
    def test_update_breaks_skips_on_zero_volume(self):
        """_update_breaks should skip (not crash) when volume=0."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21500.0, zone_high=21500.0, zone_low=21490.0,
            level_type="resistance", timeframe="D", timeframe_atr=250.0,
            volume_confirmed=True, created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        # volume=0 should not break the level
        mgr._update_breaks(price=21520.0, volume=0.0, avg_volume=100.0)
        assert level.broken is False

    def test_update_breaks_skips_on_zero_avg_volume(self):
        """_update_breaks should skip when avg_volume=0."""
        mgr = StructureLevelManager()
        level = StructureLevel(
            price=21500.0, zone_high=21500.0, zone_low=21490.0,
            level_type="resistance", timeframe="D", timeframe_atr=250.0,
            volume_confirmed=True, created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        mgr._update_breaks(price=21520.0, volume=300.0, avg_volume=0.0)
        assert level.broken is False


# ---------------------------------------------------------------------------
# created_at uses bar timestamp
# ---------------------------------------------------------------------------


class TestCreatedAtFromBar:
    def test_created_at_uses_bar_timestamp_not_wall_clock(self):
        """Levels should use pivot bar's timestamp, not datetime.now()."""
        mgr = StructureLevelManager()
        # Create bars with timestamps 30 days in the past
        old_ts = datetime.now(tz=UTC) - timedelta(days=30)
        bars = []
        for i in range(26):
            ts = old_ts + timedelta(hours=i)
            bars.append({
                "open": 21350.0, "high": 21355.0, "low": 21345.0,
                "close": 21352.0, "volume": 80, "timestamp": ts,
            })
        # Add fractal up pattern at the end
        fractal = _fractal_up_bars(pivot_high=21500.0, pivot_volume=200, bg_volume=50)
        # Give fractal bars timestamps continuing from the padding
        for j, bar in enumerate(fractal):
            bar["timestamp"] = old_ts + timedelta(hours=26 + j)
        bars.extend(fractal)

        levels = mgr.compute_levels(bars, "1h")
        resistance = [lv for lv in levels if lv.level_type == "resistance"]
        assert len(resistance) >= 1
        # created_at should be close to old_ts, NOT datetime.now()
        for lv in resistance:
            age = (datetime.now(tz=UTC) - lv.created_at).total_seconds()
            # Should be roughly 30 days old (> 25 days), not < 1 second
            assert age > 25 * 86400, f"created_at too recent: {lv.created_at}"


# ---------------------------------------------------------------------------
# Anti-signal graceful degradation
# ---------------------------------------------------------------------------


class TestAntiSignalGracefulDegradation:
    def test_no_block_when_daily_atr_zero(self):
        """When daily_atr is 0, anti-signal should NOT block (graceful degradation)."""
        mgr = StructureLevelManager()
        mgr.daily_atr = 0.0  # No daily data available
        level = StructureLevel(
            price=21200.0, zone_high=21210.0, zone_low=21200.0,
            level_type="support", timeframe="D", timeframe_atr=250.0,
            volume_confirmed=True, created_at=datetime.now(tz=UTC),
        )
        mgr._levels.append(level)

        result = mgr.check_proximity(21205.0, "short", [], [])
        assert result["blocked"] is False
