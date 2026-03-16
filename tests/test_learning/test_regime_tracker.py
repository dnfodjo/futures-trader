"""Tests for the regime tracker — classification accuracy tracking."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime

import pytest

from src.core.types import Regime
from src.learning.regime_tracker import RegimeTracker


# ── Initialization ────────────────────────────────────────────────────────


class TestInitialization:
    def test_creates_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            assert tracker.get_total_observations() == 0
            tracker.close()


# ── Record Classification ─────────────────────────────────────────────────


class TestRecordClassification:
    def test_record_returns_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_UP, price=19850.0
            )
            assert obs_id > 0
            tracker.close()

    def test_multiple_observations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            tracker.record_classification(Regime.TRENDING_UP, price=19850.0)
            tracker.record_classification(Regime.CHOPPY, price=19855.0)
            tracker.record_classification(Regime.TRENDING_DOWN, price=19840.0)

            assert tracker.get_total_observations() == 3
            tracker.close()

    def test_record_with_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            ts = datetime(2026, 3, 14, 10, 30, tzinfo=UTC)
            obs_id = tracker.record_classification(
                Regime.BREAKOUT, price=19900.0, timestamp=ts
            )
            assert obs_id > 0

            recent = tracker.get_recent_observations(limit=1)
            assert len(recent) == 1
            assert recent[0]["regime"] == "breakout"
            tracker.close()


# ── Update Outcome ────────────────────────────────────────────────────────


class TestUpdateOutcome:
    def test_update_prices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_UP, price=19850.0
            )
            tracker.update_outcome(
                obs_id,
                price_after_5min=19860.0,
                price_after_15min=19875.0,
            )

            recent = tracker.get_recent_observations(limit=1)
            assert recent[0]["price_after_5min"] == 19860.0
            assert recent[0]["price_after_15min"] == 19875.0
            tracker.close()

    def test_update_trade_pnl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_UP, price=19850.0
            )
            tracker.update_outcome(obs_id, trade_pnl=120.0)

            recent = tracker.get_recent_observations(limit=1)
            assert recent[0]["trade_pnl"] == 120.0
            tracker.close()

    def test_update_correctness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_UP, price=19850.0
            )
            tracker.update_outcome(obs_id, was_correct=True)

            recent = tracker.get_recent_observations(limit=1)
            assert recent[0]["was_correct"] == 1
            tracker.close()


# ── Auto Evaluate ─────────────────────────────────────────────────────────


class TestAutoEvaluate:
    def test_trending_up_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_UP, price=19850.0
            )
            # Price went up 5 points → correct
            result = tracker.auto_evaluate(
                obs_id, Regime.TRENDING_UP, 19850.0, 19855.0
            )
            assert result is True
            tracker.close()

    def test_trending_up_incorrect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_UP, price=19850.0
            )
            # Price went down → incorrect
            result = tracker.auto_evaluate(
                obs_id, Regime.TRENDING_UP, 19850.0, 19845.0
            )
            assert result is False
            tracker.close()

    def test_trending_down_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.TRENDING_DOWN, price=19850.0
            )
            result = tracker.auto_evaluate(
                obs_id, Regime.TRENDING_DOWN, 19850.0, 19845.0
            )
            assert result is True
            tracker.close()

    def test_choppy_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.CHOPPY, price=19850.0
            )
            # Small movement = choppy correct
            result = tracker.auto_evaluate(
                obs_id, Regime.CHOPPY, 19850.0, 19852.0
            )
            assert result is True
            tracker.close()

    def test_choppy_incorrect_big_move(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.CHOPPY, price=19850.0
            )
            # Big movement = choppy was wrong
            result = tracker.auto_evaluate(
                obs_id, Regime.CHOPPY, 19850.0, 19870.0
            )
            assert result is False
            tracker.close()

    def test_breakout_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.BREAKOUT, price=19850.0
            )
            # Big move = breakout correct
            result = tracker.auto_evaluate(
                obs_id, Regime.BREAKOUT, 19850.0, 19865.0
            )
            assert result is True
            tracker.close()

    def test_low_volume_always_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            obs_id = tracker.record_classification(
                Regime.LOW_VOLUME, price=19850.0
            )
            result = tracker.auto_evaluate(
                obs_id, Regime.LOW_VOLUME, 19850.0, 19900.0
            )
            assert result is True
            tracker.close()


# ── Accuracy Queries ──────────────────────────────────────────────────────


class TestAccuracyQueries:
    def test_accuracy_by_regime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            # Record 12 trending_up: 8 correct, 4 wrong
            for i in range(12):
                obs_id = tracker.record_classification(
                    Regime.TRENDING_UP, price=19850.0
                )
                tracker.update_outcome(obs_id, was_correct=(i < 8))

            accuracy = tracker.get_accuracy_by_regime(min_observations=10)
            assert "trending_up" in accuracy
            assert abs(accuracy["trending_up"] - 8 / 12) < 0.01
            tracker.close()

    def test_accuracy_below_min_observations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            # Only 3 observations — below min_observations=10 threshold
            for _ in range(3):
                obs_id = tracker.record_classification(
                    Regime.CHOPPY, price=19850.0
                )
                tracker.update_outcome(obs_id, was_correct=True)

            accuracy = tracker.get_accuracy_by_regime(min_observations=10)
            assert "choppy" not in accuracy
            tracker.close()


# ── Distribution ──────────────────────────────────────────────────────────


class TestDistribution:
    def test_regime_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            for _ in range(5):
                tracker.record_classification(Regime.TRENDING_UP, price=19850.0)
            for _ in range(3):
                tracker.record_classification(Regime.CHOPPY, price=19850.0)

            dist = tracker.get_regime_distribution()
            assert dist["trending_up"] == 5
            assert dist["choppy"] == 3
            tracker.close()


# ── P&L by Regime ────────────────────────────────────────────────────────


class TestRegimePnl:
    def test_regime_pnl_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")
            for pnl in [100, 50, -30]:
                obs_id = tracker.record_classification(
                    Regime.TRENDING_UP, price=19850.0
                )
                tracker.update_outcome(obs_id, trade_pnl=float(pnl))

            pnl_stats = tracker.get_regime_pnl()
            assert "trending_up" in pnl_stats
            assert pnl_stats["trending_up"]["total_pnl"] == 120.0
            assert pnl_stats["trending_up"]["count"] == 3
            tracker.close()


# ── Worst Regimes ────────────────────────────────────────────────────────


class TestWorstRegimes:
    def test_worst_regimes_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = RegimeTracker(db_path=f"{tmpdir}/test.db")

            # 5 trending_up with positive P&L
            for _ in range(5):
                obs_id = tracker.record_classification(
                    Regime.TRENDING_UP, price=19850.0
                )
                tracker.update_outcome(obs_id, trade_pnl=50.0)

            # 5 choppy with negative P&L
            for _ in range(5):
                obs_id = tracker.record_classification(
                    Regime.CHOPPY, price=19850.0
                )
                tracker.update_outcome(obs_id, trade_pnl=-40.0)

            worst = tracker.get_worst_regimes(min_trades=5)
            assert len(worst) == 2
            # Worst (most negative avg) should be first
            assert worst[0]["regime"] == "choppy"
            tracker.close()
