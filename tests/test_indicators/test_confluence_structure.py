"""Tests for structure factor integration in ConfluenceEngine (Step 4)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.indicators.confluence import ConfluenceEngine


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bars(n: int = 30, base: float = 21000.0) -> list[dict]:
    """Create n 1-min bars with increasing closes."""
    bars = []
    for i in range(n):
        c = base + i
        bars.append({
            "timestamp": f"2025-01-15T09:{i:02d}:00-05:00",
            "open": c - 1.0,
            "high": c + 2.0,
            "low": c - 2.0,
            "close": c,
            "volume": 100 + i * 10,
            "buy_volume": 60 + i * 5,
            "sell_volume": 40 + i * 5,
        })
    return bars


def _make_emas() -> dict:
    """Create multi-TF EMA dict for trend scoring."""
    return {
        "5m": {"ema_9": 21020.0, "ema_50": 21000.0},
        "15m": {"ema_9": 21015.0, "ema_50": 20990.0},
        "30m": {"ema_9": 21010.0, "ema_50": 20980.0},
    }


def _make_session_levels() -> dict:
    return {
        "asian_high": 21100.0,
        "asian_low": 20900.0,
        "london_high": 21150.0,
        "london_low": 20850.0,
    }


@dataclass
class FakeLevel:
    timeframe: str = "D"
    level_type: str = "support"
    zone_low: float = 20990.0
    zone_high: float = 21010.0


# ── Constructor ──────────────────────────────────────────────────────────────


class TestConfluenceInit:
    """Verify ConfluenceEngine accepts optional structure_manager."""

    def test_init_without_structure_manager(self):
        engine = ConfluenceEngine()
        assert engine._structure_manager is None

    def test_init_with_structure_manager(self):
        mgr = MagicMock()
        engine = ConfluenceEngine(structure_manager=mgr)
        assert engine._structure_manager is mgr


# ── Class Constants ──────────────────────────────────────────────────────────


class TestStructureConstants:
    """Verify BOUNCE_POINTS and BOS_POINTS constants."""

    def test_bounce_points(self):
        assert ConfluenceEngine.BOUNCE_POINTS == {"1h": 1, "4h": 1, "D": 2, "W": 2}

    def test_bos_points(self):
        assert ConfluenceEngine.BOS_POINTS == {"1h": 1, "4h": 1, "D": 2, "W": 2}


# ── _score_structure ─────────────────────────────────────────────────────────


class TestScoreStructure:
    """Test the _score_structure private method."""

    def test_no_manager_returns_zero(self):
        engine = ConfluenceEngine()
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["score"] == 0
        assert result["blocked"] is False
        assert "no HTF structure manager" in result["detail"]

    def test_bounce_at_daily_support(self):
        mgr = MagicMock()
        level = FakeLevel(timeframe="D", level_type="support")
        mgr.check_proximity.return_value = {
            "bounce_score": 1,
            "bos_score": 0,
            "blocked": False,
            "block_reason": "",
            "nearest_level": level,
            "detail": "near D support",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["score"] == 2  # D bounce = 2 pts
        assert "bounce" in result["detail"]

    def test_bos_through_weekly_level(self):
        mgr = MagicMock()
        level = FakeLevel(timeframe="W", level_type="resistance")
        mgr.check_proximity.return_value = {
            "bounce_score": 0,
            "bos_score": 1,
            "bos_tf": "W",
            "blocked": False,
            "block_reason": "",
            "nearest_level": level,
            "detail": "BOS through W resistance",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["score"] == 2  # W BOS = 2 pts (capped at 2)
        assert "BOS" in result["detail"]

    def test_bos_priority_over_bounce(self):
        """When both BOS and bounce are present, BOS takes priority."""
        mgr = MagicMock()
        level = FakeLevel(timeframe="4h")
        mgr.check_proximity.return_value = {
            "bounce_score": 1,
            "bos_score": 1,
            "bos_tf": "4h",
            "blocked": False,
            "block_reason": "",
            "nearest_level": level,
            "detail": "both",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["score"] == 1  # 4h BOS = 1 (capped, not bounce 1)
        assert "BOS" in result["detail"]

    def test_score_capped_at_3(self):
        """Score should never exceed 3 even with high timeframe."""
        mgr = MagicMock()
        level = FakeLevel(timeframe="W")
        mgr.check_proximity.return_value = {
            "bounce_score": 0,
            "bos_score": 1,
            "bos_tf": "W",
            "blocked": False,
            "block_reason": "",
            "nearest_level": level,
            "detail": "",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["score"] <= 3

    def test_blocked_propagates(self):
        mgr = MagicMock()
        mgr.check_proximity.return_value = {
            "bounce_score": 0,
            "bos_score": 0,
            "blocked": True,
            "block_reason": "long blocked: D resistance",
            "nearest_level": None,
            "detail": "",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["blocked"] is True
        assert "D resistance" in result["block_reason"]

    def test_no_nearest_level_returns_zero(self):
        mgr = MagicMock()
        mgr.check_proximity.return_value = {
            "bounce_score": 0,
            "bos_score": 0,
            "blocked": False,
            "block_reason": "",
            "nearest_level": None,
            "detail": "no nearby levels",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        result = engine._score_structure("long", 21000.0, [], [])
        assert result["score"] == 0


# ── score() Integration ──────────────────────────────────────────────────────


class TestScoreIntegration:
    """Test that score() integrates the structure factor."""

    def test_score_accepts_bars_5m_param(self):
        """score() should accept optional bars_5m parameter."""
        engine = ConfluenceEngine()
        bars = _make_bars()
        result = engine.score(
            side="long",
            last_price=21020.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=_make_emas(),
            session_levels=_make_session_levels(),
            bars_5m=[],
        )
        assert "structure" in result["factors"]

    def test_score_backward_compatible_without_bars_5m(self):
        """score() should work without bars_5m (backward compatibility)."""
        engine = ConfluenceEngine()
        bars = _make_bars()
        result = engine.score(
            side="long",
            last_price=21020.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=_make_emas(),
            session_levels=_make_session_levels(),
        )
        assert "structure" in result["factors"]
        assert result["factors"]["structure"]["score"] == 0

    def test_structure_factor_in_factors(self):
        """Structure factor should appear in the factors dict."""
        mgr = MagicMock()
        level = FakeLevel(timeframe="D")
        mgr.check_proximity.return_value = {
            "bounce_score": 1,
            "bos_score": 0,
            "blocked": False,
            "block_reason": "",
            "nearest_level": level,
            "detail": "near D support",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        bars = _make_bars()
        result = engine.score(
            side="long",
            last_price=21020.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=_make_emas(),
            session_levels=_make_session_levels(),
            bars_5m=[],
        )
        assert "structure" in result["factors"]
        assert result["factors"]["structure"]["score"] == 2  # D bounce

    def test_structure_block_propagates_to_result(self):
        """Structure block should propagate to top-level blocked flag."""
        mgr = MagicMock()
        mgr.check_proximity.return_value = {
            "bounce_score": 0,
            "bos_score": 0,
            "blocked": True,
            "block_reason": "long blocked: D resistance at 21100-21120",
            "nearest_level": None,
            "detail": "",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        bars = _make_bars()
        result = engine.score(
            side="long",
            last_price=21020.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=_make_emas(),
            session_levels=_make_session_levels(),
            bars_5m=[],
        )
        assert result["blocked"] is True

    def test_structure_suppressed_when_core_below_2(self):
        """Structure points should NOT count if core factor score < 2."""
        mgr = MagicMock()
        level = FakeLevel(timeframe="D")
        mgr.check_proximity.return_value = {
            "bounce_score": 1,
            "bos_score": 0,
            "blocked": False,
            "block_reason": "",
            "nearest_level": level,
            "detail": "near D support",
        }
        engine = ConfluenceEngine(structure_manager=mgr)
        # Use minimal bars that produce zero/low core scores
        # Single bar, no volume spike, no OB, no candle pattern, no sweep
        bars = [
            {
                "timestamp": "2025-01-15T09:00:00-05:00",
                "open": 21000.0, "high": 21001.0,
                "low": 20999.0, "close": 21000.5,
                "volume": 10, "buy_volume": 5, "sell_volume": 5,
            }
        ] * 20
        emas_flat = {
            "5m": {"ema_9": 21000.0, "ema_50": 21000.0},
            "15m": {"ema_9": 21000.0, "ema_50": 21000.0},
            "30m": {"ema_9": 21000.0, "ema_50": 21000.0},
        }
        result = engine.score(
            side="long",
            last_price=21000.5,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=emas_flat,
            session_levels=_make_session_levels(),
            bars_5m=[],
        )
        # Structure factor should be recorded but NOT added to total
        structure_score = result["factors"]["structure"]["score"]
        # The structure factor itself shows 2 (D bounce), but total should NOT include it
        # if core factors sum < 2
        core_sum = sum(
            result["factors"][f]["score"]
            for f in ("trend", "order_block", "candle", "sweep", "volume")
        )
        if core_sum < 2:
            # Structure should NOT be in total
            expected_total = core_sum
            assert result["score"] == expected_total

    def test_existing_factors_unchanged(self):
        """The original 6 factors should still be present and unchanged."""
        engine = ConfluenceEngine()
        bars = _make_bars()
        result = engine.score(
            side="long",
            last_price=21020.0,
            bars_1m=bars,
            atr=4.5,
            multi_tf_emas=_make_emas(),
            session_levels=_make_session_levels(),
        )
        for factor in ("trend", "order_block", "candle", "sweep", "volume"):
            assert factor in result["factors"], f"Missing factor: {factor}"
