"""Tests for order block time decay in ConfluenceEngine (Step 5)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from src.indicators.confluence import ConfluenceEngine, OrderBlock


class TestOrderBlockTimeDecay:
    """Step 5: OB scoring should decay based on age."""

    def _make_engine_with_ob(
        self, side: str, ob_low: float, ob_high: float, created_at: float = 0.0
    ) -> ConfluenceEngine:
        """Create an engine with a single pre-loaded order block."""
        engine = ConfluenceEngine()
        ob = OrderBlock(
            side=side,
            high=ob_high,
            low=ob_low,
            created_at=created_at,
        )
        if side == "bull":
            engine._bull_obs.append(ob)
        else:
            engine._bear_obs.append(ob)
        return engine

    def test_fresh_ob_scores_2(self):
        """OB created < 30 min ago should score full 2 points."""
        now = time.time()
        engine = self._make_engine_with_ob(
            "bull", 100.0, 105.0, created_at=now - 10 * 60  # 10 min ago
        )
        result = engine._score_order_block("long", 102.0)
        assert result["score"] == 2, f"Fresh OB should score 2: {result}"

    def test_aging_ob_scores_1(self):
        """OB created 31-60 min ago should score 1 point."""
        now = time.time()
        engine = self._make_engine_with_ob(
            "bull", 100.0, 105.0, created_at=now - 45 * 60  # 45 min ago
        )
        result = engine._score_order_block("long", 102.0)
        assert result["score"] == 1, f"Aging OB should score 1: {result}"

    def test_stale_ob_scores_0(self):
        """OB created > 60 min ago should score 0 (skipped)."""
        now = time.time()
        engine = self._make_engine_with_ob(
            "bull", 100.0, 105.0, created_at=now - 90 * 60  # 90 min ago
        )
        result = engine._score_order_block("long", 102.0)
        assert result["score"] == 0, f"Stale OB should score 0: {result}"

    def test_legacy_ob_no_timestamp_scores_2(self):
        """OB with created_at=0 (legacy/no timestamp) treated as fresh."""
        engine = self._make_engine_with_ob(
            "bull", 100.0, 105.0, created_at=0.0
        )
        result = engine._score_order_block("long", 102.0)
        assert result["score"] == 2, f"Legacy OB (no timestamp) should score 2: {result}"

    def test_ob_cleanup_removes_old(self):
        """OBs > 120 min should be pruned from deques during update()."""
        now = time.time()
        engine = ConfluenceEngine()

        # Add old and new OBs
        old_ob = OrderBlock("bull", 105.0, 100.0, created_at=now - 130 * 60)
        new_ob = OrderBlock("bull", 115.0, 110.0, created_at=now - 10 * 60)
        engine._bull_obs = [old_ob, new_ob]

        old_bear = OrderBlock("bear", 205.0, 200.0, created_at=now - 130 * 60)
        new_bear = OrderBlock("bear", 215.0, 210.0, created_at=now - 10 * 60)
        engine._bear_obs = [old_bear, new_bear]

        # Create enough bars for update() to run
        bars = []
        for i in range(10):
            bars.append({
                "timestamp": f"2025-01-15T09:{i:02d}:00",
                "open": 150.0 + i,
                "high": 152.0 + i,
                "low": 148.0 + i,
                "close": 150.5 + i,
                "volume": 100,
                "buy_volume": 60,
                "sell_volume": 40,
            })

        engine.update(bars, atr=5.0)

        assert len(engine._bull_obs) == 1, f"Should have pruned old bull OB: {engine._bull_obs}"
        assert engine._bull_obs[0].low == 110.0  # the new one remains
        assert len(engine._bear_obs) == 1, f"Should have pruned old bear OB: {engine._bear_obs}"

    def test_ob_age_in_detail_string(self):
        """Detail string should include age in minutes."""
        now = time.time()
        engine = self._make_engine_with_ob(
            "bull", 100.0, 105.0, created_at=now - 20 * 60  # 20 min ago
        )
        result = engine._score_order_block("long", 102.0)
        assert "age=" in result["detail"], f"Detail should contain age: {result['detail']}"
        assert "min" in result["detail"], f"Detail should contain 'min': {result['detail']}"

    def test_ob_zone_returned_regardless_of_decay(self):
        """OB zone data (low/high/side) should be in result even with decay."""
        now = time.time()
        engine = self._make_engine_with_ob(
            "bull", 100.0, 105.0, created_at=now - 45 * 60  # aging, scores 1
        )
        result = engine._score_order_block("long", 102.0)
        assert result["score"] == 1
        assert "ob_zone_low" in result
        assert "ob_zone_high" in result
        assert "ob_side" in result
