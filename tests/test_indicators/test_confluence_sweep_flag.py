"""Tests for sweep 'already used' flag in ConfluenceEngine (Step 6)."""

from __future__ import annotations

import pytest

from src.indicators.confluence import ConfluenceEngine, SweepLevel


# -- Helpers ----------------------------------------------------------------


def _make_bars(
    n: int = 10,
    base: float = 21000.0,
    lows: list[float] | None = None,
    highs: list[float] | None = None,
) -> list[dict]:
    """Create n 1-min bars. Optionally override lows/highs for specific bars."""
    bars = []
    for i in range(n):
        c = base + i
        bar = {
            "timestamp": f"2025-01-15T09:{i:02d}:00-05:00",
            "open": c,
            "high": highs[i] if highs and i < len(highs) else c + 2.0,
            "low": lows[i] if lows and i < len(lows) else c - 2.0,
            "close": c + 0.5,
            "volume": 200,
            "buy_volume": 120,
            "sell_volume": 80,
        }
        bars.append(bar)
    return bars


class TestSweepAlreadyUsedFlag:
    """Step 6: Sweep levels should fire only once until reset."""

    def test_sweep_fires_once(self):
        """First sweep at a level should score 1."""
        engine = ConfluenceEngine()
        # Equal lows sweep level at 20998.0
        engine._sweep_levels = [SweepLevel(level=20998.0, level_type="equal_lows")]

        # Bars where a recent bar dipped below 20998 and current price is just above
        bars = _make_bars(
            n=10,
            base=21000.0,
            lows=[20998.0 - 2] * 3 + [20997.0] + [20998.0 - 2] * 6,  # bar[3] sweeps below
        )
        # Override last bar to show reversal (price back above level)
        bars[-1]["close"] = 20999.0
        last_price = 20999.0  # Just above 20998.0, within proximity

        atr = 4.0  # proximity = max(4.0*0.5, 2.0) = 2.0
        result = engine._score_liquidity_sweep(
            "long", last_price, bars, atr, session_levels={}
        )
        assert result["score"] == 1, f"First sweep should score 1: {result}"

    def test_sweep_does_not_refire(self):
        """Same sweep level on next bar should score 0 (already used)."""
        engine = ConfluenceEngine()
        engine._sweep_levels = [SweepLevel(level=20998.0, level_type="equal_lows")]

        bars = _make_bars(
            n=10,
            base=21000.0,
            lows=[20998.0 - 2] * 3 + [20997.0] + [20998.0 - 2] * 6,
        )
        bars[-1]["close"] = 20999.0
        last_price = 20999.0
        atr = 4.0

        # First call scores 1
        result1 = engine._score_liquidity_sweep("long", last_price, bars, atr, session_levels={})
        assert result1["score"] == 1

        # Second call on the same level: should score 0
        result2 = engine._score_liquidity_sweep("long", last_price, bars, atr, session_levels={})
        assert result2["score"] == 0, f"Re-fire should score 0: {result2}"

    def test_sweep_resets_after_distance(self):
        """Price moves 2*ATR away from level, then returns -- fires again."""
        engine = ConfluenceEngine()
        engine._sweep_levels = [SweepLevel(level=20998.0, level_type="equal_lows")]
        atr = 4.0

        # First sweep fires
        bars = _make_bars(
            n=10,
            base=21000.0,
            lows=[20998.0 - 2] * 3 + [20997.0] + [20998.0 - 2] * 6,
        )
        bars[-1]["close"] = 20999.0
        result1 = engine._score_liquidity_sweep("long", 20999.0, bars, atr, session_levels={})
        assert result1["score"] == 1
        assert engine._sweep_levels[0].swept is True

        # Price moves far away (> 1.5 * ATR = 6.0 from level 20998)
        far_price = 20998.0 + 8.0  # 8 > 6
        # Call with far price to trigger reset
        bars_far = _make_bars(n=10, base=21006.0)
        engine._score_liquidity_sweep("long", far_price, bars_far, atr, session_levels={})
        assert engine._sweep_levels[0].swept is False, "Should reset after price moved away"

        # Now sweep fires again
        bars2 = _make_bars(
            n=10,
            base=21000.0,
            lows=[20998.0 - 2] * 3 + [20997.0] + [20998.0 - 2] * 6,
        )
        bars2[-1]["close"] = 20999.0
        result3 = engine._score_liquidity_sweep("long", 20999.0, bars2, atr, session_levels={})
        assert result3["score"] == 1, f"Should fire again after reset: {result3}"

    def test_session_sweep_fires_once(self):
        """Session level sweep (e.g., asian_low) should fire only once."""
        engine = ConfluenceEngine()
        session_levels = {"asian_low": 20990.0}
        atr = 4.0

        # Bars where a recent bar (in the -4:-1 window) dipped below asian_low
        # With 10 bars, recent_bars = bars[6:9]. Put the sweep at index 7.
        lows = [20992.0] * 7 + [20989.0] + [20992.0] * 2
        bars = _make_bars(n=10, base=21000.0, lows=lows)
        bars[-1]["close"] = 20991.0
        last_price = 20991.0  # Just above 20990, within proximity

        result1 = engine._score_liquidity_sweep("long", last_price, bars, atr, session_levels=session_levels)
        assert result1["score"] == 1, f"Session sweep should fire: {result1}"

        # Second call -- should NOT fire
        result2 = engine._score_liquidity_sweep("long", last_price, bars, atr, session_levels=session_levels)
        assert result2["score"] == 0, f"Session sweep should not re-fire: {result2}"

    def test_session_sweep_resets_on_reset(self):
        """engine.reset() should clear used session sweeps."""
        engine = ConfluenceEngine()
        session_levels = {"asian_low": 20990.0}
        atr = 4.0

        lows = [20992.0] * 7 + [20989.0] + [20992.0] * 2
        bars = _make_bars(n=10, base=21000.0, lows=lows)
        bars[-1]["close"] = 20991.0

        # Fire once
        engine._score_liquidity_sweep("long", 20991.0, bars, atr, session_levels=session_levels)

        # Reset
        engine.reset()

        # Should fire again
        result = engine._score_liquidity_sweep("long", 20991.0, bars, atr, session_levels=session_levels)
        assert result["score"] == 1, f"Should fire after reset: {result}"

    def test_multiple_sweep_levels_independent(self):
        """Sweeping level A should not mark level B as swept."""
        engine = ConfluenceEngine()
        engine._sweep_levels = [
            SweepLevel(level=20998.0, level_type="equal_lows"),
            SweepLevel(level=20980.0, level_type="equal_lows"),
        ]
        atr = 4.0

        # Sweep only level A (20998)
        bars = _make_bars(
            n=10,
            base=21000.0,
            lows=[20998.0 - 2] * 3 + [20997.0] + [20998.0 - 2] * 6,
        )
        bars[-1]["close"] = 20999.0
        engine._score_liquidity_sweep("long", 20999.0, bars, atr, session_levels={})

        # Level A should be swept
        assert engine._sweep_levels[0].swept is True
        # Level B should NOT be swept
        assert engine._sweep_levels[1].swept is False
