"""Tests for the Kelly calculator — optimal position sizing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.types import Regime, Side, TradeRecord
from src.learning.kelly_calculator import KellyCalculator, KellyResult
from src.replay.decision_scorer import TradeStats


def _trade(**overrides) -> TradeRecord:
    defaults = dict(
        timestamp_entry=datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),
        timestamp_exit=datetime(2026, 3, 14, 15, 0, 0, tzinfo=UTC),
        side=Side.LONG,
        entry_quantity=2,
        exit_quantity=2,
        entry_price=19850.0,
        exit_price=19870.0,
        stop_price=19830.0,
        pnl=80.0,
        commissions=1.72,
        hold_time_sec=1800,
        max_favorable_excursion=100.0,
        max_adverse_excursion=-20.0,
        reasoning_entry="Test",
        regime_at_entry=Regime.TRENDING_UP,
    )
    defaults.update(overrides)
    return TradeRecord(**defaults)


# ── KellyResult ───────────────────────────────────────────────────────────


class TestKellyResult:
    def test_to_dict(self):
        result = KellyResult(
            full_kelly=0.25,
            half_kelly=0.125,
            quarter_kelly=0.0625,
            optimal_contracts=2,
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            total_trades=100,
            is_reliable=True,
        )
        d = result.to_dict()
        assert d["full_kelly"] == 0.25
        assert d["half_kelly"] == 0.125
        assert d["optimal_contracts"] == 2
        assert d["is_reliable"] is True

    def test_defaults(self):
        result = KellyResult()
        assert result.full_kelly == 0.0
        assert result.optimal_contracts == 1
        assert result.is_reliable is False


# ── Kelly Calculation ────────────────────────────────────────────────────


class TestKellyCalculation:
    def test_compute_kelly_profitable_system(self):
        # W=0.55, avg_win=150, avg_loss=-100, R=1.5
        # Kelly = 0.55 - 0.45/1.5 = 0.55 - 0.3 = 0.25
        result = KellyCalculator._compute_kelly(0.55, 150.0, -100.0)
        assert abs(result - 0.25) < 0.001

    def test_compute_kelly_breakeven(self):
        # W=0.5, avg_win=100, avg_loss=-100, R=1.0
        # Kelly = 0.5 - 0.5/1.0 = 0.0
        result = KellyCalculator._compute_kelly(0.5, 100.0, -100.0)
        assert abs(result) < 0.001

    def test_compute_kelly_losing_system(self):
        # W=0.4, avg_win=100, avg_loss=-100, R=1.0
        # Kelly = 0.4 - 0.6/1.0 = -0.2 → clamped to 0
        result = KellyCalculator._compute_kelly(0.4, 100.0, -100.0)
        assert result == 0.0

    def test_compute_kelly_high_win_rate(self):
        # W=0.7, avg_win=100, avg_loss=-100, R=1.0
        # Kelly = 0.7 - 0.3/1.0 = 0.4
        result = KellyCalculator._compute_kelly(0.7, 100.0, -100.0)
        assert abs(result - 0.4) < 0.001

    def test_compute_kelly_high_ratio(self):
        # W=0.4, avg_win=300, avg_loss=-100, R=3.0
        # Kelly = 0.4 - 0.6/3.0 = 0.4 - 0.2 = 0.2
        result = KellyCalculator._compute_kelly(0.4, 300.0, -100.0)
        assert abs(result - 0.2) < 0.001

    def test_compute_kelly_edge_zero_win_rate(self):
        result = KellyCalculator._compute_kelly(0.0, 100.0, -100.0)
        assert result == 0.0

    def test_compute_kelly_edge_one_win_rate(self):
        result = KellyCalculator._compute_kelly(1.0, 100.0, -100.0)
        assert result == 0.0

    def test_compute_kelly_edge_zero_avg_loss(self):
        result = KellyCalculator._compute_kelly(0.55, 100.0, 0.0)
        assert result == 0.0


# ── Calculate from Trades ────────────────────────────────────────────────


class TestCalculateFromTrades:
    def test_insufficient_trades(self):
        calc = KellyCalculator(min_trades=50)
        trades = [_trade() for _ in range(10)]
        result = calc.calculate(trades)
        assert result.is_reliable is False
        assert result.optimal_contracts == 1
        assert "Only 10 trades" in result.confidence_note

    def test_empty_trades(self):
        calc = KellyCalculator()
        result = calc.calculate([])
        assert result.total_trades == 0
        assert result.is_reliable is False

    def test_sufficient_trades(self):
        calc = KellyCalculator(min_trades=5, max_contracts=6)

        # Create 10 trades: 6 winners, 4 losers
        trades = []
        for i in range(10):
            if i < 6:
                trades.append(_trade(pnl=100.0))
            else:
                trades.append(_trade(pnl=-80.0))

        result = calc.calculate(trades)
        assert result.is_reliable is True
        assert result.total_trades == 10
        assert result.full_kelly > 0
        assert result.half_kelly > 0
        assert result.optimal_contracts >= 1


# ── Calculate from Stats ─────────────────────────────────────────────────


class TestCalculateFromStats:
    def test_calculate_from_stats(self):
        calc = KellyCalculator(min_trades=5, max_contracts=6)
        stats = TradeStats(
            total_trades=100,
            winners=55,
            losers=45,
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            profit_factor=1.5 * (55 / 45),
        )

        result = calc.calculate_from_stats(stats)
        assert result.is_reliable is True
        assert abs(result.full_kelly - 0.25) < 0.01
        assert abs(result.half_kelly - 0.125) < 0.01
        assert result.optimal_contracts >= 1
        assert result.optimal_contracts <= 6

    def test_calculate_from_empty_stats(self):
        calc = KellyCalculator()
        stats = TradeStats()
        result = calc.calculate_from_stats(stats)
        assert result.total_trades == 0
        assert result.is_reliable is False


# ── Ruin Probability ─────────────────────────────────────────────────────


class TestRuinProbability:
    def test_ruin_strong_system(self):
        # High win rate = low ruin
        ruin = KellyCalculator._estimate_ruin_probability(0.6, 150.0, -100.0)
        assert ruin < 0.01

    def test_ruin_weak_system(self):
        # 50% win rate with equal avg win/loss = certain ruin
        ruin = KellyCalculator._estimate_ruin_probability(0.5, 100.0, -100.0)
        assert ruin >= 0.99

    def test_ruin_losing_system(self):
        ruin = KellyCalculator._estimate_ruin_probability(0.4, 100.0, -100.0)
        assert ruin >= 0.99

    def test_ruin_edge_zero_win_rate(self):
        ruin = KellyCalculator._estimate_ruin_probability(0.0, 100.0, -100.0)
        assert ruin >= 0.99


# ── Daily Limit Recommendation ───────────────────────────────────────────


class TestDailyLimit:
    def test_recommend_clamped(self):
        calc = KellyCalculator()
        stats = TradeStats(avg_loss=-50.0)
        # 50 * 2 * 4 = 400 — within range
        limit = calc._recommend_daily_limit(stats, optimal_contracts=2)
        assert 200.0 <= limit <= 600.0

    def test_recommend_min_clamp(self):
        calc = KellyCalculator()
        stats = TradeStats(avg_loss=-10.0)
        # 10 * 1 * 4 = 40 → clamped to 200
        limit = calc._recommend_daily_limit(stats, optimal_contracts=1)
        assert limit == 200.0

    def test_recommend_max_clamp(self):
        calc = KellyCalculator()
        stats = TradeStats(avg_loss=-100.0)
        # 100 * 6 * 4 = 2400 → clamped to 600
        limit = calc._recommend_daily_limit(stats, optimal_contracts=6)
        assert limit == 600.0


# ── Properties ───────────────────────────────────────────────────────────


class TestProperties:
    def test_needs_recalculation_initially(self):
        calc = KellyCalculator()
        assert calc.needs_recalculation is True

    def test_last_result_none_initially(self):
        calc = KellyCalculator()
        assert calc.last_result is None

    def test_last_calculated_after_calculation(self):
        calc = KellyCalculator(min_trades=1)
        calc.calculate([_trade()])
        assert calc.last_calculated is not None
        assert calc.last_result is not None
