"""Tests for the Monte Carlo simulator — future P&L distribution modeling."""

from __future__ import annotations

import pytest

from src.replay.decision_scorer import TradeStats
from src.replay.monte_carlo import MonteCarloConfig, MonteCarloResult, MonteCarloSimulator


# ── Basic Simulation ─────────────────────────────────────────────────────


class TestBasicSimulation:
    def test_profitable_system(self):
        """A system with 60% win rate and 2:1 RR should be profitable."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(
            num_simulations=1000,
            trades_per_simulation=100,
            seed=42,
        )

        result = sim.simulate(
            win_rate=0.60,
            avg_win=200.0,
            avg_loss=-100.0,
            config=config,
        )

        assert result.num_simulations == 1000
        assert result.trades_per_sim == 100
        assert result.mean_pnl > 0
        assert result.prob_profitable > 0.8

    def test_losing_system(self):
        """A system with 30% win rate and 1:1 RR should lose money."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(
            num_simulations=1000,
            trades_per_simulation=100,
            seed=42,
        )

        result = sim.simulate(
            win_rate=0.30,
            avg_win=100.0,
            avg_loss=-100.0,
            config=config,
        )

        assert result.mean_pnl < 0
        assert result.prob_profitable < 0.2

    def test_seed_produces_deterministic_results(self):
        """Same seed should produce identical results."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=100, seed=123)

        r1 = sim.simulate(win_rate=0.55, avg_win=150, avg_loss=-100, config=config)
        r2 = sim.simulate(win_rate=0.55, avg_win=150, avg_loss=-100, config=config)

        assert r1.mean_pnl == r2.mean_pnl
        assert r1.median_pnl == r2.median_pnl


# ── P&L Distribution ─────────────────────────────────────────────────────


class TestPnlDistribution:
    def test_percentiles_ordered(self):
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=500, seed=42)

        result = sim.simulate(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            config=config,
        )

        assert result.pnl_percentile_5 <= result.pnl_percentile_25
        assert result.pnl_percentile_25 <= result.median_pnl
        assert result.median_pnl <= result.pnl_percentile_75
        assert result.pnl_percentile_75 <= result.pnl_percentile_95
        assert result.min_pnl <= result.pnl_percentile_5
        assert result.pnl_percentile_95 <= result.max_pnl


# ── Drawdown Distribution ────────────────────────────────────────────────


class TestDrawdownDistribution:
    def test_drawdown_non_negative(self):
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=500, seed=42)

        result = sim.simulate(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            config=config,
        )

        assert result.mean_max_drawdown >= 0
        assert result.worst_drawdown >= result.mean_max_drawdown

    def test_drawdown_increases_with_more_trades(self):
        """More trades should lead to deeper drawdowns on average."""
        sim = MonteCarloSimulator()

        r_short = sim.simulate(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            config=MonteCarloConfig(
                num_simulations=500, trades_per_simulation=20, seed=42,
            ),
        )
        r_long = sim.simulate(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            config=MonteCarloConfig(
                num_simulations=500, trades_per_simulation=200, seed=42,
            ),
        )

        assert r_long.mean_max_drawdown >= r_short.mean_max_drawdown


# ── Kelly Criterion ──────────────────────────────────────────────────────


class TestKellyCriterion:
    def test_kelly_profitable_system(self):
        """60% win rate with 2:1 RR should have positive Kelly."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=100, seed=42)

        result = sim.simulate(
            win_rate=0.60,
            avg_win=200.0,
            avg_loss=-100.0,
            config=config,
        )

        # Kelly = 0.60 - (0.40 / 2.0) = 0.40
        assert result.kelly_fraction == pytest.approx(0.40, abs=0.01)
        assert result.half_kelly_fraction == pytest.approx(0.20, abs=0.01)

    def test_kelly_breakeven_system(self):
        """50% win rate with 1:1 RR should have Kelly = 0."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=100, seed=42)

        result = sim.simulate(
            win_rate=0.50,
            avg_win=100.0,
            avg_loss=-100.0,
            config=config,
        )

        assert result.kelly_fraction == pytest.approx(0.0, abs=0.01)

    def test_kelly_negative_clamped_to_zero(self):
        """Losing system should have Kelly clamped at 0."""
        kelly = MonteCarloSimulator._compute_kelly(0.30, 100.0, -100.0)
        # Kelly = 0.30 - (0.70 / 1.0) = -0.40 → clamped to 0
        assert kelly == 0.0


# ── From TradeStats ──────────────────────────────────────────────────────


class TestFromTradeStats:
    def test_simulate_from_stats(self):
        stats = TradeStats(
            total_trades=50,
            winners=30,
            losers=20,
            win_rate=0.60,
            avg_win=150.0,
            avg_loss=-80.0,
        )
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=100, seed=42)

        result = sim.simulate_from_stats(stats, config=config)

        assert result.num_simulations == 100
        assert result.win_rate == 0.60
        assert result.avg_win == 150.0
        assert result.avg_loss == -80.0
        assert result.mean_pnl > 0

    def test_simulate_from_empty_stats(self):
        stats = TradeStats()
        sim = MonteCarloSimulator()

        result = sim.simulate_from_stats(stats)
        assert result.num_simulations == 0


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_win_rate(self):
        sim = MonteCarloSimulator()
        result = sim.simulate(win_rate=0.0, avg_win=100.0, avg_loss=-100.0)
        assert result.num_simulations == 0  # early return

    def test_one_win_rate(self):
        sim = MonteCarloSimulator()
        result = sim.simulate(win_rate=1.0, avg_win=100.0, avg_loss=-100.0)
        assert result.num_simulations == 0  # early return

    def test_zero_avg_win(self):
        sim = MonteCarloSimulator()
        result = sim.simulate(win_rate=0.5, avg_win=0.0, avg_loss=-100.0)
        assert result.num_simulations == 0

    def test_positive_avg_loss_gets_negated(self):
        """avg_loss > 0 should be treated as negative."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=100, seed=42)

        result = sim.simulate(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=100.0,  # positive — should be negated
            config=config,
        )

        assert result.avg_loss == -100.0  # negated to negative in simulation
        assert result.num_simulations == 100


# ── Ruin Probabilities ───────────────────────────────────────────────────


class TestRuinProbabilities:
    def test_high_risk_system_has_ruin(self):
        """Low win rate with bad RR should have high ruin probability."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(
            num_simulations=500,
            trades_per_simulation=50,
            daily_loss_limit=200.0,
            seed=42,
        )

        result = sim.simulate(
            win_rate=0.35,
            avg_win=100.0,
            avg_loss=-120.0,
            config=config,
        )

        assert result.prob_daily_ruin > 0.1

    def test_strong_system_low_ruin(self):
        """High win rate with good RR should have low ruin."""
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(
            num_simulations=500,
            trades_per_simulation=50,
            daily_loss_limit=1000.0,
            seed=42,
        )

        result = sim.simulate(
            win_rate=0.65,
            avg_win=200.0,
            avg_loss=-80.0,
            config=config,
        )

        assert result.prob_daily_ruin < 0.05


# ── Serialization ────────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict(self):
        sim = MonteCarloSimulator()
        config = MonteCarloConfig(num_simulations=100, seed=42)

        result = sim.simulate(
            win_rate=0.55,
            avg_win=150.0,
            avg_loss=-100.0,
            config=config,
        )

        d = result.to_dict()
        assert "pnl" in d
        assert "drawdown" in d
        assert "probabilities" in d
        assert "kelly" in d
        assert "inputs" in d
        assert d["inputs"]["win_rate"] == 0.55
