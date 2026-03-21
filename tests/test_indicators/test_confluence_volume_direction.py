"""Tests for direction-aware volume scoring in ConfluenceEngine (Step 4)."""

from __future__ import annotations

import pytest

from src.indicators.confluence import ConfluenceEngine, VOLUME_SMA_PERIOD, VOLUME_SPIKE_MULT


# -- Helpers ----------------------------------------------------------------


def _make_volume_bars(
    n: int = VOLUME_SMA_PERIOD + 1,
    base_volume: int = 100,
    last_volume: int = 200,
    last_buy_volume: int = 0,
    last_sell_volume: int = 0,
    include_directional: bool = True,
) -> list[dict]:
    """Create bars with controllable volume profile.

    The last bar has ``last_volume`` total and the specified buy/sell split.
    Preceding bars have ``base_volume`` with a 60/40 buy/sell default.
    """
    bars = []
    base_price = 21000.0
    for i in range(n):
        c = base_price + i
        bar = {
            "timestamp": f"2025-01-15T09:{i:02d}:00-05:00",
            "open": c,
            "high": c + 2.0,
            "low": c - 2.0,
            "close": c + 0.5,
            "volume": base_volume,
            "buy_volume": int(base_volume * 0.6),
            "sell_volume": int(base_volume * 0.4),
        }
        bars.append(bar)

    # Override the last bar
    last = bars[-1]
    last["volume"] = last_volume

    if include_directional:
        last["buy_volume"] = last_buy_volume
        last["sell_volume"] = last_sell_volume
    else:
        # Simulate no directional data at all
        last["buy_volume"] = 0
        last["sell_volume"] = 0

    return bars


class TestDirectionAwareVolume:
    """Step 4: Volume scoring should check directional alignment."""

    def test_long_buy_dominated_scores_1(self):
        """High volume + buy > sell should score 1 for long."""
        bars = _make_volume_bars(
            last_volume=200,  # 2x avg of 100 >= 1.5x
            last_buy_volume=130,
            last_sell_volume=70,
        )
        result = ConfluenceEngine._score_volume("long", bars)
        assert result["score"] == 1, f"Expected 1, got {result}"

    def test_long_sell_dominated_scores_0(self):
        """High volume + sell > buy should score 0 for long."""
        bars = _make_volume_bars(
            last_volume=200,
            last_buy_volume=70,
            last_sell_volume=130,
        )
        result = ConfluenceEngine._score_volume("long", bars)
        assert result["score"] == 0
        assert "sell-dominated" in result["detail"]

    def test_short_sell_dominated_scores_1(self):
        """High volume + sell > buy should score 1 for short."""
        bars = _make_volume_bars(
            last_volume=200,
            last_buy_volume=70,
            last_sell_volume=130,
        )
        result = ConfluenceEngine._score_volume("short", bars)
        assert result["score"] == 1

    def test_short_buy_dominated_scores_0(self):
        """High volume + buy > sell should score 0 for short."""
        bars = _make_volume_bars(
            last_volume=200,
            last_buy_volume=130,
            last_sell_volume=70,
        )
        result = ConfluenceEngine._score_volume("short", bars)
        assert result["score"] == 0
        assert "buy-dominated" in result["detail"]

    def test_no_directional_data_fallback(self):
        """When buy_volume and sell_volume are both 0, skip direction check."""
        bars = _make_volume_bars(
            last_volume=200,
            last_buy_volume=0,
            last_sell_volume=0,
            include_directional=False,
        )
        # Should still score 1 based on total volume alone
        result = ConfluenceEngine._score_volume("long", bars)
        assert result["score"] == 1, f"Expected fallback to total volume scoring, got {result}"

    def test_volume_below_sma_regardless(self):
        """Low volume should score 0 regardless of direction."""
        bars = _make_volume_bars(
            last_volume=50,  # 0.5x avg -- below 1.5x threshold
            last_buy_volume=40,
            last_sell_volume=10,
        )
        result = ConfluenceEngine._score_volume("long", bars)
        assert result["score"] == 0

    def test_equal_buy_sell_scores_1(self):
        """buy_vol == sell_vol should be treated as direction-neutral (OK)."""
        bars = _make_volume_bars(
            last_volume=200,
            last_buy_volume=100,
            last_sell_volume=100,
        )
        result_long = ConfluenceEngine._score_volume("long", bars)
        result_short = ConfluenceEngine._score_volume("short", bars)
        assert result_long["score"] == 1, f"Equal buy/sell should pass for long: {result_long}"
        assert result_short["score"] == 1, f"Equal buy/sell should pass for short: {result_short}"
