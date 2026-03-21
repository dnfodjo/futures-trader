"""Tests for FAST market handling in ConfluenceEngine.

FAST markets are strong directional moves — the BEST setups for ICT entries.
No score penalty, no size reduction. Other filters (OB decay, direction-aware
volume, sweep single-fire) protect against news-spike false signals.

FAST is logged as an info flag only.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.indicators.confluence import ConfluenceEngine


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bars_with_speed(
    n: int = 30,
    base: float = 21000.0,
    fast: bool = False,
    slow: bool = False,
) -> list[dict]:
    """Create n 1-min bars with controllable speed profile.

    - fast=True:  last 5 bars have 4x the range of the 14-bar baseline -> FAST
    - slow=True:  last 5 bars have 0.1x the range -> SLOW
    - default:    uniform range -> NORMAL
    """
    bars = []
    for i in range(n):
        c = base + i
        # Default range: high-low = 4.0
        range_mult = 1.0
        if fast and i >= n - 5:
            range_mult = 4.0  # 4x range -> speed_ratio > 1.5
        elif slow and i >= n - 5:
            range_mult = 0.1  # tiny range -> speed_ratio < 0.5

        bars.append({
            "open": c,
            "high": c + 2.0 * range_mult,
            "low": c - 2.0 * range_mult,
            "close": c + 1.5 * range_mult,
            "volume": 500,
            "buy_volume": 300,
            "sell_volume": 200,
        })
    return bars


def _make_emas_long() -> dict:
    """EMAs aligned for a long trade (price > ema9 > ema50)."""
    return {
        "5m": {"ema_9": 21025.0, "ema_50": 21010.0},
        "15m": {"ema_9": 21025.0, "ema_50": 21010.0},
        "30m": {"ema_9": 21025.0, "ema_50": 21010.0},
    }


def _make_session_levels() -> dict:
    return {
        "asian_high": 21100.0, "asian_low": 20900.0,
        "london_high": 21050.0, "london_low": 20950.0,
        "ny_high": 0, "ny_low": 0,
    }


def _score_with_speed(fast: bool = False, slow: bool = False) -> dict:
    bars = _make_bars_with_speed(n=30, base=21000.0, fast=fast, slow=slow)
    engine = ConfluenceEngine()
    atr = 4.0
    engine.update(bars, atr)

    return engine.score(
        side="long",
        last_price=bars[-1]["close"],
        bars_1m=bars,
        atr=atr,
        multi_tf_emas=_make_emas_long(),
        session_levels=_make_session_levels(),
    )


# ── Test 1: FAST market has NO score penalty ─────────────────────────────────


class TestFastMarketNoPenalty:
    """FAST speed_state should NOT reduce the score — strong moves are good."""

    def test_fast_market_same_score_as_normal(self):
        """FAST and NORMAL with same factors should produce the same score."""
        normal_result = _score_with_speed(fast=False, slow=False)
        fast_result = _score_with_speed(fast=True, slow=False)

        assert fast_result["speed_state"] == "FAST"
        assert normal_result["speed_state"] == "NORMAL"

        # Same score — no penalty
        assert fast_result["score"] == normal_result["score"], (
            f"FAST score {fast_result['score']} should equal NORMAL score {normal_result['score']}"
        )


# ── Test 2: FAST market flag in result dict ──────────────────────────────────


class TestFastMarketFlag:
    """Result dict should contain fast_market=True when FAST."""

    def test_fast_market_flag_present_when_fast(self):
        result = _score_with_speed(fast=True)
        assert result.get("fast_market") is True

    def test_fast_market_flag_false_when_normal(self):
        result = _score_with_speed(fast=False, slow=False)
        assert result.get("fast_market") is False


# ── Test 3: NORMAL speed unchanged ───────────────────────────────────────────


class TestNormalSpeedNoPenalty:
    """NORMAL speed should not reduce the score."""

    def test_normal_speed_no_penalty(self):
        result = _score_with_speed(fast=False, slow=False)
        assert result["speed_state"] == "NORMAL"
        factors = result["factors"]
        expected_total = sum(f["score"] for f in factors.values())
        assert result["score"] == expected_total


# ── Test 4: SLOW still blocks (no regression) ───────────────────────────────


class TestSlowStillBlocks:
    """SLOW market should still set blocked=True (no regression)."""

    def test_slow_market_blocks(self):
        result = _score_with_speed(slow=True)
        assert result["speed_state"] == "SLOW"
        assert result["blocked"] is True
        assert "slow" in result["block_reason"].lower()

    def test_slow_market_fast_market_flag_false(self):
        """SLOW should NOT set fast_market flag."""
        result = _score_with_speed(slow=True)
        assert result.get("fast_market") is False


# ── Test 5: FAST market does not block ───────────────────────────────────────


class TestFastMarketNotBlocked:
    """FAST should NOT set blocked=True. Strong moves are good for entries."""

    def test_fast_market_not_blocked(self):
        result = _score_with_speed(fast=True)
        assert result["speed_state"] == "FAST"
        assert result["blocked"] is False

    def test_fast_market_risk_flag_present(self):
        """fast_market should still appear in risk_flags for logging."""
        result = _score_with_speed(fast=True)
        assert "fast_market" in result["risk_flags"]
