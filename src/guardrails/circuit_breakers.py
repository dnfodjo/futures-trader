"""Circuit breakers — multi-day and multi-week loss protections.

These protect the account from catastrophic drawdowns across sessions.
They operate at a higher level than daily guardrails and can disable
trading entirely for days or weeks.

Rules enforced:
- 2 consecutive red days → 50% max position (e.g., 3 instead of 6)
- 3 consecutive red days → simulation only (no live orders)
- Weekly max loss ($800 default) → shutdown for rest of week
- Monthly max loss ($2,000 default) → shutdown for rest of month

The circuit breaker state is loaded from the trade journal at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class CircuitBreakerState:
    """Current state of all circuit breakers."""

    consecutive_red_days: int = 0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_contracts_override: Optional[int] = None
    sim_only: bool = False
    shutdown_reason: Optional[str] = None
    is_shutdown: bool = False


class CircuitBreakers:
    """Multi-day and multi-week loss protections.

    Usage:
        cb = CircuitBreakers(base_max_contracts=6)
        cb.load_history(daily_pnl_list)  # from trade journal
        state = cb.evaluate()
        if state.is_shutdown:
            # don't trade today
        elif state.sim_only:
            # switch to simulation mode
        elif state.max_contracts_override:
            # use reduced max contracts
    """

    def __init__(
        self,
        base_max_contracts: int = 6,
        max_consecutive_red_days: int = 3,
        weekly_loss_limit: float = 800.0,
        monthly_loss_limit: float = 2000.0,
    ) -> None:
        self._base_max = base_max_contracts
        self._max_red_days = max_consecutive_red_days
        self._weekly_limit = weekly_loss_limit
        self._monthly_limit = monthly_loss_limit

        # State from historical data
        self._consecutive_red_days: int = 0
        self._weekly_pnl: float = 0.0
        self._monthly_pnl: float = 0.0
        self._daily_results: list[tuple[str, float]] = []  # (date_str, net_pnl)

    def load_history(
        self,
        daily_results: list[tuple[str, float]],
    ) -> None:
        """Load historical daily P&L results from the trade journal.

        Args:
            daily_results: List of (date_string, net_pnl) tuples, sorted
                           chronologically (oldest first).
        """
        self._daily_results = list(daily_results)
        self._compute_state()

    def _compute_state(self) -> None:
        """Recompute circuit breaker state from daily results."""
        if not self._daily_results:
            self._consecutive_red_days = 0
            self._weekly_pnl = 0.0
            self._monthly_pnl = 0.0
            return

        today = date.today()

        # Consecutive red days (counting backwards from most recent)
        self._consecutive_red_days = 0
        for _, pnl in reversed(self._daily_results):
            if pnl < 0:
                self._consecutive_red_days += 1
            else:
                break

        # Weekly P&L (current calendar week: Monday-Friday)
        week_start = today - timedelta(days=today.weekday())  # Monday
        self._weekly_pnl = 0.0
        for date_str, pnl in self._daily_results:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d >= week_start:
                    self._weekly_pnl += pnl
            except (ValueError, TypeError):
                continue

        # Monthly P&L (current calendar month)
        month_start = today.replace(day=1)
        self._monthly_pnl = 0.0
        for date_str, pnl in self._daily_results:
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d >= month_start:
                    self._monthly_pnl += pnl
            except (ValueError, TypeError):
                continue

    def evaluate(self) -> CircuitBreakerState:
        """Evaluate all circuit breakers and return current state.

        Returns:
            CircuitBreakerState with shutdown/sim/size override info.
        """
        state = CircuitBreakerState(
            consecutive_red_days=self._consecutive_red_days,
            weekly_pnl=self._weekly_pnl,
            monthly_pnl=self._monthly_pnl,
        )

        # Monthly loss limit — most severe
        if self._monthly_pnl <= -self._monthly_limit:
            state.is_shutdown = True
            state.shutdown_reason = (
                f"Monthly loss limit hit: ${self._monthly_pnl:.2f} "
                f"(limit: -${self._monthly_limit:.2f})"
            )
            logger.critical(
                "circuit_breaker.monthly_shutdown",
                monthly_pnl=self._monthly_pnl,
                limit=self._monthly_limit,
            )
            return state

        # Weekly loss limit
        if self._weekly_pnl <= -self._weekly_limit:
            state.is_shutdown = True
            state.shutdown_reason = (
                f"Weekly loss limit hit: ${self._weekly_pnl:.2f} "
                f"(limit: -${self._weekly_limit:.2f})"
            )
            logger.critical(
                "circuit_breaker.weekly_shutdown",
                weekly_pnl=self._weekly_pnl,
                limit=self._weekly_limit,
            )
            return state

        # 3+ consecutive red days → simulation only
        if self._consecutive_red_days >= self._max_red_days:
            state.sim_only = True
            state.max_contracts_override = self._base_max
            logger.warning(
                "circuit_breaker.sim_only",
                consecutive_red_days=self._consecutive_red_days,
            )
            return state

        # 2 consecutive red days → 50% max position
        if self._consecutive_red_days >= 2:
            reduced = max(1, self._base_max // 2)
            state.max_contracts_override = reduced
            logger.warning(
                "circuit_breaker.size_reduction",
                consecutive_red_days=self._consecutive_red_days,
                max_contracts=reduced,
            )
            return state

        return state

    def record_day(self, date_str: str, net_pnl: float) -> None:
        """Record today's result and recompute state.

        Call this at the end of each trading day.
        """
        self._daily_results.append((date_str, net_pnl))
        self._compute_state()

    @property
    def consecutive_red_days(self) -> int:
        return self._consecutive_red_days

    @property
    def weekly_pnl(self) -> float:
        return self._weekly_pnl

    @property
    def monthly_pnl(self) -> float:
        return self._monthly_pnl

    @property
    def stats(self) -> dict:
        state = self.evaluate()
        return {
            "consecutive_red_days": self._consecutive_red_days,
            "weekly_pnl": self._weekly_pnl,
            "monthly_pnl": self._monthly_pnl,
            "is_shutdown": state.is_shutdown,
            "sim_only": state.sim_only,
            "max_contracts_override": state.max_contracts_override,
            "shutdown_reason": state.shutdown_reason,
        }
