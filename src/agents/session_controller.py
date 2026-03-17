"""Session controller — tracks daily P&L, trade stats, and profit preservation.

Pure Python logic (no LLM calls). Responsible for:
- Tracking daily P&L from trade completions
- Enforcing profit preservation tiers (reduce max size at thresholds)
- Counting wins/losses and consecutive streaks
- Computing max drawdown within the day
- Providing session stats to the MarketState for LLM context
- Tracking commissions
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.types import TradeRecord

logger = structlog.get_logger()


class SessionController:
    """Manages daily trading session state and profit preservation.

    Usage:
        controller = SessionController(
            max_daily_loss=400.0,
            profit_tier1_pnl=200.0,
            profit_tier1_max_size=3,
        )
        controller.start_session("2025-03-14")
        controller.record_trade(trade)
        max_size = controller.effective_max_contracts  # reduced at profit tiers
        should_stop = controller.should_stop_trading
    """

    def __init__(
        self,
        max_daily_loss: float = 400.0,
        commission_per_rt: float = 0.86,
        point_value: float = 2.0,
        profit_tier1_pnl: float = 200.0,
        profit_tier1_max_size: int = 6,
        profit_tier2_pnl: float = 400.0,
        profit_tier2_max_size: int = 4,
        base_max_contracts: int = 10,
    ) -> None:
        self._max_daily_loss = max_daily_loss
        self._commission_per_rt = commission_per_rt
        self._point_value = point_value

        # Profit preservation tiers
        self._tier1_pnl = profit_tier1_pnl
        self._tier1_max = profit_tier1_max_size
        self._tier2_pnl = profit_tier2_pnl
        self._tier2_max = profit_tier2_max_size
        self._base_max = base_max_contracts

        # Session state
        self._session_date: str = ""
        self._daily_pnl: float = 0.0
        self._gross_pnl: float = 0.0
        self._commissions: float = 0.0
        self._trades: list[TradeRecord] = []
        self._winners: int = 0
        self._losers: int = 0
        self._scratches: int = 0  # breakeven trades
        self._consecutive_losers: int = 0
        self._max_consecutive_losers: int = 0
        self._peak_pnl: float = 0.0
        self._max_drawdown: float = 0.0
        self._is_stopped: bool = False
        self._stop_reason: str = ""
        self._circuit_breaker_max: Optional[int] = None

    # ── Session Lifecycle ────────────────────────────────────────────────────

    def start_session(self, date_str: str = "") -> None:
        """Start a new trading session. Resets all daily stats."""
        self._session_date = date_str or datetime.now(tz=UTC).strftime("%Y-%m-%d")
        self._daily_pnl = 0.0
        self._gross_pnl = 0.0
        self._commissions = 0.0
        self._trades.clear()
        self._winners = 0
        self._losers = 0
        self._scratches = 0
        self._consecutive_losers = 0
        self._max_consecutive_losers = 0
        self._peak_pnl = 0.0
        self._max_drawdown = 0.0
        self._is_stopped = False
        self._stop_reason = ""

        logger.info("session_controller.started", date=self._session_date)

    # ── Trade Recording ──────────────────────────────────────────────────────

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade and update all session stats.

        Args:
            trade: The completed TradeRecord with P&L populated.
        """
        self._trades.append(trade)

        pnl = trade.pnl or 0.0
        commission = trade.commissions if trade.commissions is not None else self._commission_per_rt

        self._gross_pnl += pnl
        self._commissions += commission
        self._daily_pnl = self._gross_pnl - self._commissions

        # Win/Loss tracking
        if pnl > 0:
            self._winners += 1
            self._consecutive_losers = 0
        elif pnl < 0:
            self._losers += 1
            self._consecutive_losers += 1
            self._max_consecutive_losers = max(
                self._max_consecutive_losers, self._consecutive_losers
            )
        else:
            self._scratches += 1

        # Peak P&L and drawdown
        if self._daily_pnl > self._peak_pnl:
            self._peak_pnl = self._daily_pnl

        drawdown = self._peak_pnl - self._daily_pnl
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

        # Check stop conditions
        self._check_stop_conditions()

        logger.info(
            "session_controller.trade_recorded",
            pnl=pnl,
            daily_pnl=round(self._daily_pnl, 2),
            trades=len(self._trades),
            wl=f"{self._winners}W/{self._losers}L",
        )

    def _check_stop_conditions(self) -> None:
        """Check if any stop trading conditions are met."""
        if self._daily_pnl <= -self._max_daily_loss:
            self._is_stopped = True
            self._stop_reason = (
                f"Daily loss limit hit: ${self._daily_pnl:.2f} "
                f"(limit: -${self._max_daily_loss:.2f})"
            )
            logger.warning(
                "session_controller.daily_limit_hit",
                pnl=self._daily_pnl,
                limit=self._max_daily_loss,
            )

    # ── Profit Preservation ──────────────────────────────────────────────────

    def set_circuit_breaker_max(self, max_contracts: int) -> None:
        """Override max contracts due to circuit breaker (multi-day losses).

        This is the tightest constraint — takes precedence over everything.
        """
        self._circuit_breaker_max = max_contracts

    @property
    def effective_max_contracts(self) -> int:
        """Maximum contracts allowed based on current daily P&L.

        Implements profit preservation tiers:
        - At +$200 daily P&L → max 3 contracts
        - At +$400 daily P&L → max 2 contracts

        Also respects circuit breaker override (multi-day loss reduction).
        Returns the minimum of all constraints.
        """
        profit_max = self._base_max
        if self._daily_pnl >= self._tier2_pnl:
            profit_max = self._tier2_max
        elif self._daily_pnl >= self._tier1_pnl:
            profit_max = self._tier1_max

        if self._circuit_breaker_max is not None:
            return min(profit_max, self._circuit_breaker_max)
        return profit_max

    @property
    def profit_preservation_active(self) -> bool:
        """Whether profit preservation is currently reducing max size."""
        return self.effective_max_contracts < self._base_max

    @property
    def profit_preservation_tier(self) -> int:
        """Current profit preservation tier (0 = none, 1, 2)."""
        if self._daily_pnl >= self._tier2_pnl:
            return 2
        if self._daily_pnl >= self._tier1_pnl:
            return 1
        return 0

    # ── Stop Conditions ──────────────────────────────────────────────────────

    @property
    def should_stop_trading(self) -> bool:
        """Whether trading should stop for the day."""
        return self._is_stopped

    @property
    def stop_reason(self) -> str:
        """Reason trading was stopped, if applicable."""
        return self._stop_reason

    def force_stop(self, reason: str) -> None:
        """Manually stop trading for the day."""
        self._is_stopped = True
        self._stop_reason = reason
        logger.warning("session_controller.force_stopped", reason=reason)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def session_date(self) -> str:
        return self._session_date

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def gross_pnl(self) -> float:
        return self._gross_pnl

    @property
    def commissions(self) -> float:
        return self._commissions

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def winners(self) -> int:
        return self._winners

    @property
    def losers(self) -> int:
        return self._losers

    @property
    def scratches(self) -> int:
        return self._scratches

    @property
    def win_rate(self) -> float:
        """Win rate as percentage (0-100). Returns 0 if no trades."""
        decided = self._winners + self._losers
        if decided == 0:
            return 0.0
        return (self._winners / decided) * 100.0

    @property
    def consecutive_losers(self) -> int:
        return self._consecutive_losers

    @property
    def max_consecutive_losers(self) -> int:
        return self._max_consecutive_losers

    @property
    def peak_pnl(self) -> float:
        return self._peak_pnl

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def trades(self) -> list[TradeRecord]:
        return self._trades

    @property
    def pnl_per_trade(self) -> float:
        """Average net P&L per trade."""
        if not self._trades:
            return 0.0
        return self._daily_pnl / len(self._trades)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "session_date": self._session_date,
            "daily_pnl": round(self._daily_pnl, 2),
            "gross_pnl": round(self._gross_pnl, 2),
            "commissions": round(self._commissions, 2),
            "total_trades": len(self._trades),
            "winners": self._winners,
            "losers": self._losers,
            "scratches": self._scratches,
            "win_rate": round(self.win_rate, 1),
            "consecutive_losers": self._consecutive_losers,
            "max_consecutive_losers": self._max_consecutive_losers,
            "peak_pnl": round(self._peak_pnl, 2),
            "max_drawdown": round(self._max_drawdown, 2),
            "effective_max_contracts": self.effective_max_contracts,
            "profit_preservation_tier": self.profit_preservation_tier,
            "is_stopped": self._is_stopped,
            "stop_reason": self._stop_reason,
        }
