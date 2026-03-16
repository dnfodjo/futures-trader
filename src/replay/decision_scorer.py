"""Decision scorer — evaluates LLM trading decisions against actual outcomes.

Given a list of TradeRecords (from the trade journal), computes:
- Win rate, average win, average loss
- Profit factor (gross wins / gross losses)
- Sharpe ratio (daily return basis)
- Max drawdown
- Performance breakdown by regime, session phase, and action type
- Hold time statistics

These metrics feed into the Monte Carlo simulator and Kelly calculator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.core.types import Regime, SessionPhase, Side, TradeRecord


@dataclass
class TradeStats:
    """Aggregated statistics from a set of trades."""

    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0
    win_rate: float = 0.0

    gross_wins: float = 0.0
    gross_losses: float = 0.0
    net_pnl: float = 0.0
    total_commissions: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    profit_factor: float = 0.0
    expectancy: float = 0.0

    avg_hold_time_sec: float = 0.0
    avg_hold_time_winners_sec: float = 0.0
    avg_hold_time_losers_sec: float = 0.0

    avg_mfe: float = 0.0  # max favorable excursion
    avg_mae: float = 0.0  # max adverse excursion

    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_consecutive_losers: int = 0
    max_consecutive_winners: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winners": self.winners,
            "losers": self.losers,
            "breakeven": self.breakeven,
            "win_rate": round(self.win_rate, 4),
            "gross_wins": round(self.gross_wins, 2),
            "gross_losses": round(self.gross_losses, 2),
            "net_pnl": round(self.net_pnl, 2),
            "total_commissions": round(self.total_commissions, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2),
            "avg_hold_time_sec": round(self.avg_hold_time_sec, 0),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_consecutive_losers": self.max_consecutive_losers,
            "max_consecutive_winners": self.max_consecutive_winners,
        }


class DecisionScorer:
    """Scores LLM trading decisions by analyzing trade outcomes.

    Usage:
        scorer = DecisionScorer()
        stats = scorer.score(trades)
        by_regime = scorer.score_by_regime(trades)
        by_phase = scorer.score_by_session_phase(trades)
    """

    def score(self, trades: list[TradeRecord]) -> TradeStats:
        """Compute comprehensive statistics from a list of trades.

        Args:
            trades: List of completed TradeRecords (must have pnl set).

        Returns:
            TradeStats with all computed metrics.
        """
        if not trades:
            return TradeStats()

        completed = [t for t in trades if t.pnl is not None]
        if not completed:
            return TradeStats()

        stats = TradeStats()
        stats.total_trades = len(completed)

        # Win/loss classification
        pnls: list[float] = []
        win_pnls: list[float] = []
        loss_pnls: list[float] = []
        hold_times: list[int] = []
        winner_hold_times: list[int] = []
        loser_hold_times: list[int] = []
        mfes: list[float] = []
        maes: list[float] = []

        for t in completed:
            pnl = t.pnl  # type: ignore[assignment]  # checked above
            pnls.append(pnl)
            mfes.append(t.max_favorable_excursion)
            maes.append(t.max_adverse_excursion)

            if t.hold_time_sec is not None:
                hold_times.append(t.hold_time_sec)

            if pnl > 0:
                stats.winners += 1
                stats.gross_wins += pnl
                win_pnls.append(pnl)
                if t.hold_time_sec is not None:
                    winner_hold_times.append(t.hold_time_sec)
            elif pnl < 0:
                stats.losers += 1
                stats.gross_losses += abs(pnl)
                loss_pnls.append(pnl)
                if t.hold_time_sec is not None:
                    loser_hold_times.append(t.hold_time_sec)
            else:
                stats.breakeven += 1

            stats.total_commissions += t.commissions

        # Basic metrics
        stats.win_rate = stats.winners / stats.total_trades if stats.total_trades > 0 else 0.0
        stats.net_pnl = sum(pnls)
        stats.avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
        stats.avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0  # negative
        stats.largest_win = max(win_pnls) if win_pnls else 0.0
        stats.largest_loss = min(loss_pnls) if loss_pnls else 0.0  # most negative

        # Profit factor
        stats.profit_factor = (
            stats.gross_wins / stats.gross_losses
            if stats.gross_losses > 0
            else float("inf") if stats.gross_wins > 0 else 0.0
        )

        # Expectancy (avg $ per trade)
        stats.expectancy = stats.net_pnl / stats.total_trades

        # Hold time
        stats.avg_hold_time_sec = sum(hold_times) / len(hold_times) if hold_times else 0.0
        stats.avg_hold_time_winners_sec = (
            sum(winner_hold_times) / len(winner_hold_times) if winner_hold_times else 0.0
        )
        stats.avg_hold_time_losers_sec = (
            sum(loser_hold_times) / len(loser_hold_times) if loser_hold_times else 0.0
        )

        # MFE/MAE
        stats.avg_mfe = sum(mfes) / len(mfes) if mfes else 0.0
        stats.avg_mae = sum(maes) / len(maes) if maes else 0.0

        # Sharpe ratio (per-trade)
        stats.sharpe_ratio = self._compute_sharpe(pnls)

        # Max drawdown
        stats.max_drawdown = self._compute_max_drawdown(pnls)

        # Consecutive streaks
        stats.max_consecutive_winners = self._max_consecutive(pnls, positive=True)
        stats.max_consecutive_losers = self._max_consecutive(pnls, positive=False)

        return stats

    def score_by_regime(
        self, trades: list[TradeRecord]
    ) -> dict[str, TradeStats]:
        """Score trades grouped by market regime at entry.

        Returns:
            Dict mapping regime name → TradeStats.
        """
        groups: dict[str, list[TradeRecord]] = {}
        for t in trades:
            regime = t.regime_at_entry.value
            groups.setdefault(regime, []).append(t)

        return {regime: self.score(group) for regime, group in groups.items()}

    def score_by_side(
        self, trades: list[TradeRecord]
    ) -> dict[str, TradeStats]:
        """Score trades grouped by side (long/short).

        Returns:
            Dict mapping side name → TradeStats.
        """
        groups: dict[str, list[TradeRecord]] = {}
        for t in trades:
            side = t.side.value
            groups.setdefault(side, []).append(t)

        return {side: self.score(group) for side, group in groups.items()}

    # ── Internal Calculations ────────────────────────────────────────────

    @staticmethod
    def _compute_sharpe(pnls: list[float], risk_free: float = 0.0) -> float:
        """Compute Sharpe ratio from a list of per-trade P&Ls.

        Uses per-trade returns (not daily), so this is a simplified Sharpe.
        """
        if len(pnls) < 2:
            return 0.0

        mean = sum(pnls) / len(pnls) - risk_free
        variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance)

        if std == 0:
            return 0.0

        return mean / std

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        """Compute maximum drawdown from cumulative P&L.

        Returns:
            Max drawdown as a positive dollar amount.
        """
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    @staticmethod
    def _max_consecutive(pnls: list[float], positive: bool) -> int:
        """Count the maximum consecutive winners or losers."""
        max_streak = 0
        current_streak = 0

        for pnl in pnls:
            if (positive and pnl > 0) or (not positive and pnl < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak
