"""Monte Carlo simulator — models future P&L distribution from observed stats.

Given observed trading statistics (win rate, avg win, avg loss), runs N
simulated trade sequences to estimate:

- Probability of hitting daily/weekly/monthly profit targets
- Probability of ruin (hitting max drawdown before target)
- Optimal Kelly fraction
- Expected P&L distribution (mean, median, percentiles)
- Drawdown distribution

This is the primary tool for sizing decisions and risk management.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.replay.decision_scorer import TradeStats


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    num_simulations: int = 10_000
    trades_per_simulation: int = 100  # e.g., ~5 trades/day × 20 trading days
    commission_per_rt: float = 0.86  # round-trip per contract
    avg_contracts: float = 2.0  # average contracts per trade
    max_contracts: int = 6  # max position size from guardrails
    daily_loss_limit: float = 400.0
    weekly_loss_limit: float = 800.0
    monthly_loss_limit: float = 2000.0
    daily_target: float = 300.0
    weekly_target: float = 1000.0
    monthly_target: float = 4000.0
    seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Result of a single simulation path."""

    final_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    total_commissions: float = 0.0
    num_trades: int = 0
    hit_daily_limit: bool = False
    hit_weekly_limit: bool = False
    hit_monthly_limit: bool = False
    hit_daily_target: bool = False
    hit_weekly_target: bool = False
    hit_monthly_target: bool = False


@dataclass
class MonteCarloResult:
    """Aggregated Monte Carlo simulation results."""

    num_simulations: int = 0
    trades_per_sim: int = 0

    # P&L distribution
    mean_pnl: float = 0.0
    median_pnl: float = 0.0
    std_pnl: float = 0.0
    pnl_percentile_5: float = 0.0
    pnl_percentile_25: float = 0.0
    pnl_percentile_75: float = 0.0
    pnl_percentile_95: float = 0.0
    min_pnl: float = 0.0
    max_pnl: float = 0.0

    # Drawdown distribution
    mean_max_drawdown: float = 0.0
    median_max_drawdown: float = 0.0
    worst_drawdown: float = 0.0
    drawdown_percentile_95: float = 0.0

    # Probabilities
    prob_profitable: float = 0.0
    prob_daily_target: float = 0.0
    prob_weekly_target: float = 0.0
    prob_monthly_target: float = 0.0
    prob_daily_ruin: float = 0.0
    prob_weekly_ruin: float = 0.0
    prob_monthly_ruin: float = 0.0

    # Kelly criterion
    kelly_fraction: float = 0.0
    half_kelly_fraction: float = 0.0
    kelly_optimal_contracts: float = 0.0

    # Input parameters (for reference)
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_simulations": self.num_simulations,
            "trades_per_sim": self.trades_per_sim,
            "pnl": {
                "mean": round(self.mean_pnl, 2),
                "median": round(self.median_pnl, 2),
                "std": round(self.std_pnl, 2),
                "p5": round(self.pnl_percentile_5, 2),
                "p25": round(self.pnl_percentile_25, 2),
                "p75": round(self.pnl_percentile_75, 2),
                "p95": round(self.pnl_percentile_95, 2),
                "min": round(self.min_pnl, 2),
                "max": round(self.max_pnl, 2),
            },
            "drawdown": {
                "mean": round(self.mean_max_drawdown, 2),
                "median": round(self.median_max_drawdown, 2),
                "worst": round(self.worst_drawdown, 2),
                "p95": round(self.drawdown_percentile_95, 2),
            },
            "probabilities": {
                "profitable": round(self.prob_profitable, 4),
                "daily_target": round(self.prob_daily_target, 4),
                "weekly_target": round(self.prob_weekly_target, 4),
                "monthly_target": round(self.prob_monthly_target, 4),
                "daily_ruin": round(self.prob_daily_ruin, 4),
                "weekly_ruin": round(self.prob_weekly_ruin, 4),
                "monthly_ruin": round(self.prob_monthly_ruin, 4),
            },
            "kelly": {
                "full": round(self.kelly_fraction, 4),
                "half": round(self.half_kelly_fraction, 4),
                "optimal_contracts": round(self.kelly_optimal_contracts, 1),
            },
            "inputs": {
                "win_rate": round(self.win_rate, 4),
                "avg_win": round(self.avg_win, 2),
                "avg_loss": round(self.avg_loss, 2),
                "profit_factor": round(self.profit_factor, 2),
            },
        }


class MonteCarloSimulator:
    """Simulates future trading outcomes using observed statistics.

    Usage:
        sim = MonteCarloSimulator()

        # From TradeStats:
        result = sim.simulate_from_stats(trade_stats)

        # From raw parameters:
        result = sim.simulate(win_rate=0.55, avg_win=150, avg_loss=-100)
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None) -> None:
        self._config = config or MonteCarloConfig()

    def simulate_from_stats(
        self,
        stats: TradeStats,
        config: Optional[MonteCarloConfig] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation using observed TradeStats.

        Args:
            stats: TradeStats from DecisionScorer.
            config: Optional override config.

        Returns:
            MonteCarloResult with full distribution analysis.
        """
        if stats.total_trades == 0 or stats.win_rate == 0.0:
            return MonteCarloResult()

        return self.simulate(
            win_rate=stats.win_rate,
            avg_win=stats.avg_win,
            avg_loss=stats.avg_loss,
            win_std=0.0,  # Could be computed from trade data
            loss_std=0.0,
            config=config,
        )

    def simulate(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        win_std: float = 0.0,
        loss_std: float = 0.0,
        config: Optional[MonteCarloConfig] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation with given parameters.

        Args:
            win_rate: Probability of winning (0-1).
            avg_win: Average winning trade P&L (positive).
            avg_loss: Average losing trade P&L (negative).
            win_std: Standard deviation of win amounts (0 = fixed).
            loss_std: Standard deviation of loss amounts (0 = fixed).
            config: Optional override config.

        Returns:
            MonteCarloResult with full distribution analysis.
        """
        cfg = config or self._config
        rng = np.random.default_rng(cfg.seed)

        # Validate inputs
        if win_rate <= 0 or win_rate >= 1:
            return MonteCarloResult(win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss)
        if avg_win <= 0:
            return MonteCarloResult(win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss)
        if avg_loss >= 0:
            avg_loss = -abs(avg_loss) if avg_loss != 0 else -1.0  # ensure negative

        # Default std to 30% of mean if not provided
        if win_std <= 0:
            win_std = abs(avg_win) * 0.3
        if loss_std <= 0:
            loss_std = abs(avg_loss) * 0.3

        commission_per_trade = cfg.commission_per_rt * cfg.avg_contracts

        # Run simulations
        sim_results: list[SimulationResult] = []

        for _ in range(cfg.num_simulations):
            result = self._run_single_simulation(
                rng=rng,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                win_std=win_std,
                loss_std=loss_std,
                commission_per_trade=commission_per_trade,
                num_trades=cfg.trades_per_simulation,
                daily_limit=cfg.daily_loss_limit,
                weekly_limit=cfg.weekly_loss_limit,
                monthly_limit=cfg.monthly_loss_limit,
                daily_target=cfg.daily_target,
                weekly_target=cfg.weekly_target,
                monthly_target=cfg.monthly_target,
            )
            sim_results.append(result)

        return self._aggregate_results(
            sim_results, cfg, win_rate, avg_win, avg_loss
        )

    def _run_single_simulation(
        self,
        rng: np.random.Generator,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        win_std: float,
        loss_std: float,
        commission_per_trade: float,
        num_trades: int,
        daily_limit: float,
        weekly_limit: float,
        monthly_limit: float,
        daily_target: float,
        weekly_target: float,
        monthly_target: float,
    ) -> SimulationResult:
        """Run a single simulation path."""
        result = SimulationResult(num_trades=num_trades)

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        total_commissions = 0.0

        for _ in range(num_trades):
            # Generate trade outcome
            is_winner = rng.random() < win_rate

            if is_winner:
                pnl = max(0.01, rng.normal(avg_win, win_std))
            else:
                pnl = min(-0.01, rng.normal(avg_loss, loss_std))

            # Subtract commission
            pnl -= commission_per_trade
            total_commissions += commission_per_trade

            cumulative += pnl

            # Track peak and drawdown
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

            # Check limits (stop simulation if hit)
            if cumulative <= -monthly_limit:
                result.hit_monthly_limit = True
                break
            if cumulative <= -weekly_limit:
                result.hit_weekly_limit = True
            if cumulative <= -daily_limit:
                result.hit_daily_limit = True

            # Check targets
            if cumulative >= monthly_target:
                result.hit_monthly_target = True
            if cumulative >= weekly_target:
                result.hit_weekly_target = True
            if cumulative >= daily_target:
                result.hit_daily_target = True

        result.final_pnl = cumulative
        result.max_drawdown = max_dd
        result.peak_pnl = peak
        result.total_commissions = total_commissions

        return result

    def _aggregate_results(
        self,
        sim_results: list[SimulationResult],
        config: MonteCarloConfig,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> MonteCarloResult:
        """Aggregate individual simulation results into summary statistics."""
        n = len(sim_results)
        if n == 0:
            return MonteCarloResult()

        final_pnls = np.array([r.final_pnl for r in sim_results])
        max_drawdowns = np.array([r.max_drawdown for r in sim_results])

        result = MonteCarloResult()
        result.num_simulations = n
        result.trades_per_sim = config.trades_per_simulation

        # P&L distribution
        result.mean_pnl = float(np.mean(final_pnls))
        result.median_pnl = float(np.median(final_pnls))
        result.std_pnl = float(np.std(final_pnls))
        result.pnl_percentile_5 = float(np.percentile(final_pnls, 5))
        result.pnl_percentile_25 = float(np.percentile(final_pnls, 25))
        result.pnl_percentile_75 = float(np.percentile(final_pnls, 75))
        result.pnl_percentile_95 = float(np.percentile(final_pnls, 95))
        result.min_pnl = float(np.min(final_pnls))
        result.max_pnl = float(np.max(final_pnls))

        # Drawdown distribution
        result.mean_max_drawdown = float(np.mean(max_drawdowns))
        result.median_max_drawdown = float(np.median(max_drawdowns))
        result.worst_drawdown = float(np.max(max_drawdowns))
        result.drawdown_percentile_95 = float(np.percentile(max_drawdowns, 95))

        # Probabilities
        result.prob_profitable = float(np.mean(final_pnls > 0))
        result.prob_daily_target = float(
            np.mean([r.hit_daily_target for r in sim_results])
        )
        result.prob_weekly_target = float(
            np.mean([r.hit_weekly_target for r in sim_results])
        )
        result.prob_monthly_target = float(
            np.mean([r.hit_monthly_target for r in sim_results])
        )
        result.prob_daily_ruin = float(
            np.mean([r.hit_daily_limit for r in sim_results])
        )
        result.prob_weekly_ruin = float(
            np.mean([r.hit_weekly_limit for r in sim_results])
        )
        result.prob_monthly_ruin = float(
            np.mean([r.hit_monthly_limit for r in sim_results])
        )

        # Kelly criterion
        result.kelly_fraction = self._compute_kelly(win_rate, avg_win, avg_loss)
        result.half_kelly_fraction = result.kelly_fraction / 2
        result.kelly_optimal_contracts = max(
            1.0, result.half_kelly_fraction * config.max_contracts
        )

        # Input references
        result.win_rate = win_rate
        result.avg_win = avg_win
        result.avg_loss = avg_loss
        result.profit_factor = (
            abs(avg_win) / abs(avg_loss) if avg_loss != 0 else 0.0
        )

        return result

    @staticmethod
    def _compute_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute Kelly criterion fraction.

        Kelly % = W - (1-W) / R
        Where:
            W = win rate
            R = avg_win / abs(avg_loss) (win/loss ratio)

        Returns:
            Kelly fraction (0-1). Negative means don't trade.
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0

        r = abs(avg_win) / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / r

        # Clamp to [0, 1]
        return max(0.0, min(1.0, kelly))
