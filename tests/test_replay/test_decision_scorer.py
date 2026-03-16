"""Tests for the decision scorer — trade outcome analysis."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.types import Regime, Side, TradeRecord
from src.replay.decision_scorer import DecisionScorer, TradeStats


def _trade(
    pnl: float,
    side: Side = Side.LONG,
    regime: Regime = Regime.TRENDING_UP,
    hold_time: int = 300,
    mfe: float = 0.0,
    mae: float = 0.0,
    commissions: float = 1.72,
) -> TradeRecord:
    return TradeRecord(
        timestamp_entry=datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC),
        timestamp_exit=datetime(2026, 3, 14, 10, 5, 0, tzinfo=UTC),
        side=side,
        entry_quantity=2,
        exit_quantity=2,
        entry_price=19850.0,
        exit_price=19860.0 if pnl > 0 else 19840.0,
        stop_price=19835.0,
        pnl=pnl,
        commissions=commissions,
        hold_time_sec=hold_time,
        max_favorable_excursion=mfe,
        max_adverse_excursion=mae,
        regime_at_entry=regime,
    )


# ── Basic Scoring ────────────────────────────────────────────────────────


class TestBasicScoring:
    def test_empty_trades(self):
        scorer = DecisionScorer()
        stats = scorer.score([])
        assert stats.total_trades == 0
        assert stats.win_rate == 0.0

    def test_all_winners(self):
        trades = [_trade(100), _trade(150), _trade(80)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.total_trades == 3
        assert stats.winners == 3
        assert stats.losers == 0
        assert stats.win_rate == 1.0
        assert stats.gross_wins == 330.0
        assert stats.gross_losses == 0.0
        assert stats.net_pnl == 330.0

    def test_all_losers(self):
        trades = [_trade(-80), _trade(-120), _trade(-60)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.total_trades == 3
        assert stats.winners == 0
        assert stats.losers == 3
        assert stats.win_rate == 0.0
        assert stats.gross_losses == 260.0

    def test_mixed_trades(self):
        trades = [_trade(100), _trade(-50), _trade(80), _trade(-30)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.total_trades == 4
        assert stats.winners == 2
        assert stats.losers == 2
        assert stats.win_rate == 0.5
        assert stats.net_pnl == 100.0  # 100 + (-50) + 80 + (-30)

    def test_breakeven_trade(self):
        trades = [_trade(0.0)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.breakeven == 1
        assert stats.winners == 0
        assert stats.losers == 0

    def test_trades_without_pnl_excluded(self):
        """Trades with pnl=None should be excluded from scoring."""
        trades = [
            _trade(100),
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19850.0,
                stop_price=19835.0,
                pnl=None,  # incomplete
            ),
        ]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.total_trades == 1


# ── Detailed Metrics ─────────────────────────────────────────────────────


class TestDetailedMetrics:
    def test_avg_win_loss(self):
        trades = [_trade(100), _trade(200), _trade(-60), _trade(-40)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.avg_win == 150.0  # (100 + 200) / 2
        assert stats.avg_loss == -50.0  # (-60 + -40) / 2

    def test_largest_win_loss(self):
        trades = [_trade(100), _trade(200), _trade(-60), _trade(-40)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.largest_win == 200.0
        assert stats.largest_loss == -60.0

    def test_profit_factor(self):
        trades = [_trade(200), _trade(-100)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.profit_factor == 2.0  # 200 / 100

    def test_profit_factor_no_losses(self):
        trades = [_trade(100)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.profit_factor == float("inf")

    def test_expectancy(self):
        trades = [_trade(100), _trade(-50), _trade(80), _trade(-30)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.expectancy == 25.0  # 100 / 4

    def test_commissions_tracked(self):
        trades = [_trade(100, commissions=1.72), _trade(-50, commissions=1.72)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.total_commissions == pytest.approx(3.44)


# ── Hold Time ────────────────────────────────────────────────────────────


class TestHoldTime:
    def test_avg_hold_time(self):
        trades = [_trade(100, hold_time=300), _trade(-50, hold_time=600)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.avg_hold_time_sec == 450.0

    def test_hold_time_by_win_loss(self):
        trades = [
            _trade(100, hold_time=200),
            _trade(80, hold_time=300),
            _trade(-50, hold_time=600),
        ]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.avg_hold_time_winners_sec == 250.0  # (200 + 300) / 2
        assert stats.avg_hold_time_losers_sec == 600.0


# ── MFE/MAE ──────────────────────────────────────────────────────────────


class TestMfeMae:
    def test_avg_mfe_mae(self):
        trades = [_trade(100, mfe=150, mae=-30), _trade(-50, mfe=20, mae=-80)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.avg_mfe == 85.0   # (150 + 20) / 2
        assert stats.avg_mae == -55.0  # (-30 + -80) / 2


# ── Max Drawdown ─────────────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_max_drawdown_simple(self):
        # Cumulative: 100, 150, 50, 80 → peak 150, lowest after peak 50 → DD 100
        trades = [_trade(100), _trade(50), _trade(-100), _trade(30)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.max_drawdown == 100.0

    def test_max_drawdown_no_losses(self):
        trades = [_trade(100), _trade(50)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.max_drawdown == 0.0

    def test_max_drawdown_only_losses(self):
        trades = [_trade(-50), _trade(-30)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.max_drawdown == 80.0  # cumulative: -50, -80 → peak 0, DD 80


# ── Consecutive Streaks ──────────────────────────────────────────────────


class TestConsecutiveStreaks:
    def test_max_consecutive_winners(self):
        trades = [_trade(100), _trade(50), _trade(80), _trade(-20)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.max_consecutive_winners == 3

    def test_max_consecutive_losers(self):
        trades = [_trade(100), _trade(-50), _trade(-30), _trade(-20), _trade(80)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.max_consecutive_losers == 3


# ── Sharpe Ratio ─────────────────────────────────────────────────────────


class TestSharpeRatio:
    def test_sharpe_positive(self):
        trades = [_trade(100), _trade(80), _trade(120), _trade(90)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        # All positive with low variance → high Sharpe
        assert stats.sharpe_ratio > 3.0

    def test_sharpe_single_trade(self):
        """Single trade cannot compute Sharpe (need variance)."""
        trades = [_trade(100)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)

        assert stats.sharpe_ratio == 0.0


# ── Scoring by Regime ────────────────────────────────────────────────────


class TestScoreByRegime:
    def test_groups_by_regime(self):
        trades = [
            _trade(100, regime=Regime.TRENDING_UP),
            _trade(50, regime=Regime.TRENDING_UP),
            _trade(-30, regime=Regime.CHOPPY),
        ]
        scorer = DecisionScorer()
        by_regime = scorer.score_by_regime(trades)

        assert "trending_up" in by_regime
        assert "choppy" in by_regime
        assert by_regime["trending_up"].total_trades == 2
        assert by_regime["choppy"].total_trades == 1


# ── Scoring by Side ─────────────────────────────────────────────────────


class TestScoreBySide:
    def test_groups_by_side(self):
        trades = [
            _trade(100, side=Side.LONG),
            _trade(-30, side=Side.SHORT),
            _trade(50, side=Side.LONG),
        ]
        scorer = DecisionScorer()
        by_side = scorer.score_by_side(trades)

        assert "long" in by_side
        assert "short" in by_side
        assert by_side["long"].total_trades == 2
        assert by_side["short"].total_trades == 1


# ── TradeStats Serialization ─────────────────────────────────────────────


class TestTradeStatsSerialization:
    def test_to_dict(self):
        trades = [_trade(100), _trade(-50)]
        scorer = DecisionScorer()
        stats = scorer.score(trades)
        d = stats.to_dict()

        assert "total_trades" in d
        assert "win_rate" in d
        assert "profit_factor" in d
        assert "sharpe_ratio" in d
        assert d["total_trades"] == 2
