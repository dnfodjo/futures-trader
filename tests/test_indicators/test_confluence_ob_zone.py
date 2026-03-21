"""Tests for OB zone data propagation in confluence scoring.

Verifies that when an OB tap is detected, the zone boundaries (low/high/side)
are included in the confluence result dict for use by dynamic stop computation.
"""

import pytest

from src.indicators.confluence import ConfluenceEngine, OrderBlock


class TestOBZoneInConfluenceResult:
    """Verify ob_zone is present in score() return dict."""

    def _make_bars(self, n: int = 25, base_price: float = 21450.0) -> list[dict]:
        """Create minimal bar data for scoring."""
        bars = []
        for i in range(n):
            bars.append({
                "open": base_price - 0.5,
                "high": base_price + 1.0,
                "low": base_price - 1.0,
                "close": base_price + 0.5,
                "volume": 100,
                "buy_volume": 60,
                "sell_volume": 40,
                "timestamp": f"2026-03-20T10:{i:02d}:00Z",
            })
        return bars

    def _make_emas(self, price: float = 21450.0) -> dict:
        """Create multi-TF EMA dict."""
        return {
            "5m": {"ema_9": price + 0.1, "ema_50": price - 1.0},
            "15m": {"ema_9": price + 0.2, "ema_50": price - 0.5},
            "30m": {"ema_9": price + 0.3, "ema_50": price - 0.3},
        }

    def test_ob_tap_includes_zone_boundaries(self):
        """When OB tap scores 2, ob_zone should contain low/high/side."""
        engine = ConfluenceEngine()
        # Inject a bull OB directly
        engine._bull_obs = [
            OrderBlock(side="bull", high=21452.0, low=21445.0),
        ]
        bars = self._make_bars()
        result = engine.score(
            side="long",
            last_price=21450.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=self._make_emas(),
            session_levels={},
        )
        ob_zone = result.get("ob_zone")
        assert ob_zone is not None, "ob_zone should be present when OB tap found"
        assert ob_zone["ob_zone_low"] == 21445.0
        assert ob_zone["ob_zone_high"] == 21452.0
        assert ob_zone["ob_side"] == "bull"

    def test_no_ob_tap_returns_none(self):
        """When no OB tap, ob_zone should be None."""
        engine = ConfluenceEngine()
        # No OBs injected
        bars = self._make_bars()
        result = engine.score(
            side="long",
            last_price=21450.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=self._make_emas(),
            session_levels={},
        )
        assert result.get("ob_zone") is None

    def test_bear_ob_zone_data(self):
        """Bear OB zone data has correct boundaries."""
        engine = ConfluenceEngine()
        engine._bear_obs = [
            OrderBlock(side="bear", high=21460.0, low=21448.0),
        ]
        bars = self._make_bars()
        result = engine.score(
            side="short",
            last_price=21455.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=self._make_emas(21455.0),
            session_levels={},
        )
        ob_zone = result.get("ob_zone")
        assert ob_zone is not None
        assert ob_zone["ob_zone_low"] == 21448.0
        assert ob_zone["ob_zone_high"] == 21460.0
        assert ob_zone["ob_side"] == "bear"

    def test_ob_zone_from_score_order_block_return(self):
        """_score_order_block return dict includes zone data."""
        engine = ConfluenceEngine()
        engine._bull_obs = [
            OrderBlock(side="bull", high=21452.0, low=21445.0),
        ]
        result = engine._score_order_block("long", 21450.0)
        assert result["score"] == 2
        assert result.get("ob_zone_low") == 21445.0
        assert result.get("ob_zone_high") == 21452.0
        assert result.get("ob_side") == "bull"

    def test_no_ob_tap_score_order_block_no_zone_keys(self):
        """_score_order_block with no tap has no zone keys."""
        engine = ConfluenceEngine()
        result = engine._score_order_block("long", 21450.0)
        assert result["score"] == 0
        assert "ob_zone_low" not in result
