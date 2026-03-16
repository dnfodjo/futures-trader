"""Kelly calculator — optimal position sizing from observed trade statistics.

Computes Kelly criterion after 50+ trades to determine optimal contract sizing:

    Kelly % = W - (1-W) / R

Where:
    W = win rate
    R = avg_win / abs(avg_loss)

Uses HALF-Kelly for safety (reduces to ~75% of optimal long-term growth rate
while dramatically reducing the probability of ruin).

Updates weekly from the trade journal and integrates with the guardrails
to dynamically set max position size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.types import TradeRecord
from src.replay.decision_scorer import DecisionScorer, TradeStats

logger = structlog.get_logger()

_MIN_TRADES_FOR_KELLY = 50


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation."""

    full_kelly: float = 0.0
    half_kelly: float = 0.0
    quarter_kelly: float = 0.0
    optimal_contracts: int = 1
    max_contracts_allowed: int = 6

    # Inputs used
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0

    # Confidence
    is_reliable: bool = False  # True if >= MIN_TRADES
    confidence_note: str = ""

    # Risk metrics
    estimated_ruin_probability: float = 0.0
    recommended_daily_loss_limit: float = 400.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_kelly": round(self.full_kelly, 4),
            "half_kelly": round(self.half_kelly, 4),
            "quarter_kelly": round(self.quarter_kelly, 4),
            "optimal_contracts": self.optimal_contracts,
            "max_contracts_allowed": self.max_contracts_allowed,
            "win_rate": round(self.win_rate, 4),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "is_reliable": self.is_reliable,
            "confidence_note": self.confidence_note,
            "estimated_ruin_probability": round(self.estimated_ruin_probability, 4),
            "recommended_daily_loss_limit": round(self.recommended_daily_loss_limit, 2),
        }


class KellyCalculator:
    """Calculates optimal position sizing using Kelly criterion.

    Usage:
        calculator = KellyCalculator(max_contracts=6)
        result = calculator.calculate(trades)
        print(f"Optimal contracts: {result.optimal_contracts}")

        # Or from pre-computed stats:
        result = calculator.calculate_from_stats(trade_stats)
    """

    def __init__(
        self,
        max_contracts: int = 6,
        min_trades: int = _MIN_TRADES_FOR_KELLY,
        account_value: float = 10_000.0,
        point_value: float = 2.0,  # MNQ = $2/point
    ) -> None:
        self._max_contracts = max_contracts
        self._min_trades = min_trades
        self._account_value = account_value
        self._point_value = point_value
        self._scorer = DecisionScorer()
        self._last_result: Optional[KellyResult] = None
        self._last_calculated: Optional[datetime] = None

    def calculate(self, trades: list[TradeRecord]) -> KellyResult:
        """Calculate Kelly criterion from raw trade records.

        Args:
            trades: List of completed TradeRecords.

        Returns:
            KellyResult with sizing recommendations.
        """
        stats = self._scorer.score(trades)
        return self.calculate_from_stats(stats)

    def calculate_from_stats(self, stats: TradeStats) -> KellyResult:
        """Calculate Kelly criterion from pre-computed TradeStats.

        Args:
            stats: TradeStats from DecisionScorer.

        Returns:
            KellyResult with sizing recommendations.
        """
        result = KellyResult()
        result.total_trades = stats.total_trades
        result.max_contracts_allowed = self._max_contracts

        if stats.total_trades == 0:
            result.confidence_note = "No trades to analyze."
            return result

        result.win_rate = stats.win_rate
        result.avg_win = stats.avg_win
        result.avg_loss = stats.avg_loss
        result.profit_factor = stats.profit_factor

        # Check reliability
        if stats.total_trades < self._min_trades:
            result.is_reliable = False
            result.confidence_note = (
                f"Only {stats.total_trades} trades — need {self._min_trades} "
                f"for reliable Kelly. Using conservative default."
            )
            result.optimal_contracts = 1
            result.full_kelly = 0.0
            result.half_kelly = 0.0
            result.quarter_kelly = 0.0
            self._last_result = result
            return result

        result.is_reliable = True

        # Compute Kelly
        result.full_kelly = self._compute_kelly(
            stats.win_rate, stats.avg_win, stats.avg_loss
        )
        result.half_kelly = result.full_kelly / 2
        result.quarter_kelly = result.full_kelly / 4

        # Compute optimal contracts using half-Kelly
        result.optimal_contracts = self._kelly_to_contracts(result.half_kelly)

        # Estimate ruin probability
        result.estimated_ruin_probability = self._estimate_ruin_probability(
            stats.win_rate, stats.avg_win, stats.avg_loss
        )

        # Adjust daily loss limit based on performance
        result.recommended_daily_loss_limit = self._recommend_daily_limit(
            stats, result.optimal_contracts
        )

        result.confidence_note = (
            f"Based on {stats.total_trades} trades. "
            f"Kelly suggests {result.full_kelly:.1%}, using half-Kelly {result.half_kelly:.1%}."
        )

        self._last_result = result
        self._last_calculated = datetime.now(tz=UTC)

        logger.info(
            "kelly.calculated",
            trades=stats.total_trades,
            full_kelly=round(result.full_kelly, 4),
            half_kelly=round(result.half_kelly, 4),
            optimal_contracts=result.optimal_contracts,
            ruin_prob=round(result.estimated_ruin_probability, 4),
        )

        return result

    @staticmethod
    def _compute_kelly(
        win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Compute Kelly criterion fraction.

        Kelly % = W - (1-W) / R
        Where:
            W = win rate
            R = avg_win / abs(avg_loss)

        Returns:
            Kelly fraction clamped to [0, 1]. Negative means don't trade.
        """
        if avg_loss == 0 or avg_win == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0

        r = abs(avg_win) / abs(avg_loss)
        kelly = win_rate - (1 - win_rate) / r

        return max(0.0, min(1.0, kelly))

    def _kelly_to_contracts(self, kelly_fraction: float) -> int:
        """Convert Kelly fraction to number of contracts.

        Uses the account value and average risk per trade to size.

        Returns:
            Integer contracts, clamped to [1, max_contracts].
        """
        if kelly_fraction <= 0:
            return 1  # always trade at least 1

        # Raw Kelly contracts = kelly_fraction * max_contracts
        raw = kelly_fraction * self._max_contracts
        contracts = max(1, min(self._max_contracts, round(raw)))

        return contracts

    @staticmethod
    def _estimate_ruin_probability(
        win_rate: float, avg_win: float, avg_loss: float
    ) -> float:
        """Estimate the probability of ruin (hitting max drawdown).

        Uses the simplified formula:
            P(ruin) = ((1-W)/W)^(bankroll/avg_loss)

        This is an approximation — the Monte Carlo simulator gives
        more accurate results.

        Returns:
            Estimated probability of ruin (0-1).
        """
        if win_rate <= 0 or win_rate >= 1 or avg_loss == 0:
            return 1.0 if win_rate <= 0.5 else 0.0

        # Simplified ruin formula
        q = 1 - win_rate
        ratio = q / win_rate

        if ratio >= 1.0:
            return 1.0  # negative expectancy = certain ruin

        # Assume 20 "units" of bankroll (conservative)
        try:
            ruin = ratio ** 20
        except (OverflowError, ValueError):
            ruin = 0.0

        return max(0.0, min(1.0, ruin))

    def _recommend_daily_limit(
        self,
        stats: TradeStats,
        optimal_contracts: int,
    ) -> float:
        """Recommend a daily loss limit based on observed performance.

        Heuristic: daily limit = 4 × avg_loss × contracts.
        This allows ~4 consecutive losers before shutdown.
        Clamped to [$200, $600] for MNQ.

        Args:
            stats: TradeStats with avg_loss.
            optimal_contracts: Number of contracts.

        Returns:
            Recommended daily loss limit in dollars.
        """
        if stats.avg_loss == 0:
            return 400.0

        raw = abs(stats.avg_loss) * optimal_contracts * 4
        return max(200.0, min(600.0, raw))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def last_result(self) -> Optional[KellyResult]:
        """Get the most recent Kelly calculation result."""
        return self._last_result

    @property
    def last_calculated(self) -> Optional[datetime]:
        """When Kelly was last calculated."""
        return self._last_calculated

    @property
    def needs_recalculation(self) -> bool:
        """Whether Kelly should be recalculated.

        Returns True if never calculated or last calculation was >7 days ago.
        """
        if self._last_calculated is None:
            return True

        elapsed = (datetime.now(tz=UTC) - self._last_calculated).days
        return elapsed >= 7
