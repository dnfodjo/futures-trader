"""Apex Trader Funding compliance guardrails.

Prevents account violations that would fail an Apex evaluation or blow
a funded PA (Performance Account). These are HARD guardrails — no override.

Apex rules that can blow your account (in order of severity):

1. TRAILING DRAWDOWN — Account equity (including open P&L) falls below the
   trailing threshold. This is real-time intraday for evals, EOD for PAs.
   The threshold TRAILS UP with your peak balance — it never goes down.

2. NO OVERNIGHT HOLDS — ALL positions must be flat by 4:59 PM ET daily.
   No warnings, no appeals — immediate disqualification.

3. DAILY LOSS LIMIT (EOD accounts) — Cannot lose more than the specified
   amount in a single session. Account paused (frozen), not terminated.

4. CONSISTENCY RULE — No single day's profit can exceed 50% of total
   profit since last payout (or account activation). Required for payouts.

5. MAX CONTRACTS — Cannot exceed the account's contract limit. Immediate
   violation. Scaling rule: half contracts until profit buffer reached.

Account sizes and their limits (Apex 2026):

| Account | Trail DD | Max Micros | Scaling | Safety Net | DLL     |
|---------|----------|------------|---------|------------|---------|
| $25K    | $1,500   | 40         | —       | $26,600    | —       |
| $50K    | $2,500   | 100 (50*)  | $2,600  | $50,100    | $1,000  |
| $100K   | $3,000   | 160        | —       | $103,100   | $1,500  |
| $150K   | $5,000   | 200        | —       | $155,100   | —       |

* 50k account: 50 micros until $2,600 profit buffer, then full 100.
Safety net = balance where trailing drawdown stops trailing.
DLL = Daily Loss Limit (soft breach — frozen for day, not terminated).

Sources:
- https://support.apextraderfunding.com/hc/en-us/articles/40463582041371
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Optional
from zoneinfo import ZoneInfo

import structlog

from src.core.types import (
    ActionType,
    GuardrailResult,
    LLMAction,
    MarketState,
    PositionState,
)

logger = structlog.get_logger()

ET = ZoneInfo("US/Eastern")

# ── Apex Account Presets ──────────────────────────────────────────────────────

APEX_ACCOUNTS = {
    "25k": {
        "starting_balance": 25_000.0,
        "trailing_drawdown": 1_500.0,
        "max_micros": 40,
        "safety_net": 26_600.0,
        "daily_loss_limit": None,  # intraday accounts — no DLL
    },
    "50k": {
        "starting_balance": 50_000.0,
        "trailing_drawdown": 2_500.0,
        "max_micros": 100,  # 100 micros, scaling starts at 50
        "max_micros_scaling": 50,  # half until $2,600 profit buffer
        "scaling_unlock_profit": 2_600.0,  # $2,500 drawdown + $100
        "safety_net": 50_100.0,  # drawdown stops trailing at $50,100
        "daily_loss_limit": 1_000.0,  # soft DLL — frozen for day, not terminated
    },
    "100k": {
        "starting_balance": 100_000.0,
        "trailing_drawdown": 3_000.0,
        "max_micros": 160,
        "safety_net": 103_100.0,
        "daily_loss_limit": 1_500.0,  # EOD 100k has DLL
    },
    "150k": {
        "starting_balance": 150_000.0,
        "trailing_drawdown": 5_000.0,
        "max_micros": 200,
        "safety_net": 155_100.0,
        "daily_loss_limit": None,
    },
}


@dataclass
class ApexAccountState:
    """Tracks the running state for Apex rule compliance."""

    # Account configuration
    starting_balance: float = 50_000.0
    trailing_drawdown: float = 2_500.0
    max_micros: int = 100
    max_micros_scaling: int = 50  # half contracts during scaling
    scaling_unlock_profit: float = 2_600.0  # profit needed to unlock full contracts
    safety_net: float = 50_100.0
    daily_loss_limit: Optional[float] = 1_000.0

    # Running state
    current_balance: float = 50_000.0
    peak_balance: float = 50_000.0
    drawdown_floor: float = 47_500.0  # starting_balance - trailing_drawdown
    drawdown_locked: bool = False  # True once peak >= safety_net
    scaling_unlocked: bool = False  # True once profit >= scaling_unlock_profit
    total_profit_since_payout: float = 0.0
    best_single_day_profit: float = 0.0
    payouts_completed: int = 0

    def update_balance(self, new_balance: float) -> None:
        """Update balance and recalculate trailing drawdown floor.

        Call this after every trade or at regular intervals with
        account equity (balance + open P&L).
        """
        self.current_balance = new_balance

        if new_balance > self.peak_balance:
            self.peak_balance = new_balance

            # Check if scaling rule can be unlocked (profit buffer reached)
            profit = self.peak_balance - self.starting_balance
            if profit >= self.scaling_unlock_profit and not self.scaling_unlocked:
                self.scaling_unlocked = True
                logger.info(
                    "apex.scaling_unlocked",
                    profit=round(profit, 2),
                    max_contracts=self.max_micros,
                )

            # Check if we've reached the safety net (drawdown stops trailing)
            if self.peak_balance >= self.safety_net:
                if not self.drawdown_locked:
                    # Lock the floor at safety_net - trailing_drawdown
                    self.drawdown_floor = self.safety_net - self.trailing_drawdown
                    self.drawdown_locked = True
                    logger.info(
                        "apex.drawdown_locked",
                        safety_net=self.safety_net,
                        floor=self.drawdown_floor,
                    )
            else:
                # Drawdown trails up with peak
                self.drawdown_floor = self.peak_balance - self.trailing_drawdown

    def record_day_profit(self, day_pnl: float) -> None:
        """Record end-of-day P&L for consistency rule tracking."""
        if day_pnl > 0:
            self.total_profit_since_payout += day_pnl
            if day_pnl > self.best_single_day_profit:
                self.best_single_day_profit = day_pnl
        else:
            # Losses don't reduce total profit for consistency calculation
            # (Apex tracks positive profit days)
            pass

    def record_payout(self) -> None:
        """Record a completed payout — resets consistency tracking."""
        self.payouts_completed += 1
        self.total_profit_since_payout = 0.0
        self.best_single_day_profit = 0.0

    @property
    def drawdown_remaining(self) -> float:
        """How much room until the trailing drawdown floor."""
        return self.current_balance - self.drawdown_floor

    @property
    def drawdown_remaining_pct(self) -> float:
        """Drawdown remaining as percentage of trailing drawdown."""
        if self.trailing_drawdown == 0:
            return 1.0
        return self.drawdown_remaining / self.trailing_drawdown

    @property
    def effective_max_micros(self) -> int:
        """Max contracts allowed right now (respects scaling rule)."""
        if self.max_micros_scaling > 0 and not self.scaling_unlocked:
            return self.max_micros_scaling
        return self.max_micros

    @property
    def consistency_ok(self) -> bool:
        """Check 50% consistency rule.

        No single day's profit can exceed 50% of total profit
        since last payout. After 6 payouts, this rule is waived.
        """
        if self.payouts_completed >= 6:
            return True  # waived after 6 payouts
        if self.total_profit_since_payout <= 0:
            return True  # no profit yet — rule doesn't apply
        ratio = self.best_single_day_profit / self.total_profit_since_payout
        return ratio <= 0.50


class ApexRuleGuardrail:
    """Enforces Apex Trader Funding rules as a guardrail layer.

    This guardrail checks:
    1. Trailing drawdown proximity — blocks new entries when close to floor
    2. Position flat deadline — forces flatten before 4:59 PM ET
    3. Max contract limits — blocks entries exceeding Apex limits
    4. Daily loss limit — blocks entries when daily loss approaches limit
    5. Consistency rule — adjusts behavior when a day is getting too profitable

    Usage:
        apex = ApexRuleGuardrail(account_type="50k")
        apex.update_equity(current_balance + unrealized_pnl)

        result = apex.check(action, state, position, daily_pnl)
        if not result.allowed:
            # action blocked by Apex rules
    """

    def __init__(
        self,
        account_type: str = "50k",
        flatten_deadline_et: time = time(16, 54),  # 4:54 PM ET (5min buffer)
        drawdown_warning_pct: float = 0.40,  # warn at 40% of drawdown used
        drawdown_lockout_pct: float = 0.75,  # block entries at 75% drawdown used
        custom_account: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            account_type: One of "25k", "50k", "100k", "150k"
            flatten_deadline_et: Time in ET to force flatten (before 4:59 PM)
            drawdown_warning_pct: Percentage of drawdown used to trigger warnings
            drawdown_lockout_pct: Percentage of drawdown used to block new entries
            custom_account: Override account params (for non-standard accounts)
        """
        if custom_account:
            acct = custom_account
        elif account_type in APEX_ACCOUNTS:
            acct = APEX_ACCOUNTS[account_type]
        else:
            raise ValueError(
                f"Unknown Apex account type: {account_type}. "
                f"Valid: {list(APEX_ACCOUNTS.keys())}"
            )

        self._account_state = ApexAccountState(
            starting_balance=acct["starting_balance"],
            trailing_drawdown=acct["trailing_drawdown"],
            max_micros=acct["max_micros"],
            max_micros_scaling=acct.get("max_micros_scaling", 0),
            scaling_unlock_profit=acct.get("scaling_unlock_profit", 0.0),
            safety_net=acct["safety_net"],
            daily_loss_limit=acct.get("daily_loss_limit"),
            current_balance=acct["starting_balance"],
            peak_balance=acct["starting_balance"],
            drawdown_floor=acct["starting_balance"] - acct["trailing_drawdown"],
        )

        self._flatten_deadline = flatten_deadline_et
        self._drawdown_warning_pct = drawdown_warning_pct
        self._drawdown_lockout_pct = drawdown_lockout_pct

        # Stats
        self._blocks: int = 0
        self._warnings: int = 0
        self._deadline_flattens: int = 0

    def update_equity(self, equity: float) -> None:
        """Update the account equity (balance + open P&L).

        Should be called frequently — ideally every state update cycle.
        """
        self._account_state.update_balance(equity)

    def check(
        self,
        action: LLMAction,
        state: MarketState,
        position: Optional[PositionState] = None,
        daily_pnl: float = 0.0,
        current_contracts: int = 0,
    ) -> GuardrailResult:
        """Check an action against all Apex rules.

        Args:
            action: The proposed trading action.
            state: Current market state.
            position: Current position (None if flat).
            daily_pnl: Today's realized P&L.
            current_contracts: Currently held contracts.

        Returns:
            GuardrailResult with allowed=True/False.
        """
        # DO_NOTHING, STOP_TRADING, FLATTEN always pass
        if action.action in (
            ActionType.DO_NOTHING,
            ActionType.STOP_TRADING,
            ActionType.FLATTEN,
        ):
            return GuardrailResult(allowed=True)

        # MOVE_STOP always allowed (protecting position)
        if action.action == ActionType.MOVE_STOP:
            return GuardrailResult(allowed=True)

        # ── Check 1: Flatten deadline ─────────────────────────────────
        result = self._check_flatten_deadline(state)
        if result is not None:
            return result

        # ── Check 2: Trailing drawdown proximity ─────────────────────
        result = self._check_drawdown_proximity(daily_pnl)
        if result is not None:
            return result

        # ── Check 3: Max contracts ───────────────────────────────────
        result = self._check_max_contracts(action, current_contracts)
        if result is not None:
            return result

        # ── Check 4: Daily loss limit ────────────────────────────────
        result = self._check_daily_loss_limit(daily_pnl)
        if result is not None:
            return result

        return GuardrailResult(allowed=True)

    def should_force_flatten(self, state: MarketState) -> bool:
        """Check if we need to force-flatten for Apex deadline compliance.

        This is separate from check() because FLATTEN should never be blocked.
        The orchestrator should call this and initiate a FLATTEN if True.
        """
        now_et = state.timestamp.astimezone(ET)
        current_time = now_et.time()
        return current_time >= self._flatten_deadline

    def _check_flatten_deadline(self, state: MarketState) -> Optional[GuardrailResult]:
        """Block new entries within 5 minutes of Apex closing deadline."""
        now_et = state.timestamp.astimezone(ET)
        current_time = now_et.time()

        # Block new entries after the flatten deadline
        if current_time >= self._flatten_deadline:
            self._blocks += 1
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"apex_rule: BLOCKED — past Apex flatten deadline "
                    f"({self._flatten_deadline.strftime('%H:%M')} ET). "
                    f"All positions must be flat by 4:59 PM ET."
                ),
            )
        return None

    def _check_drawdown_proximity(
        self,
        daily_pnl: float,
    ) -> Optional[GuardrailResult]:
        """Block entries when approaching the trailing drawdown floor."""
        acct = self._account_state
        remaining = acct.drawdown_remaining
        dd = acct.trailing_drawdown

        if dd <= 0:
            return None

        used_pct = 1.0 - (remaining / dd)

        # Hard lockout — too close to the floor
        if used_pct >= self._drawdown_lockout_pct:
            self._blocks += 1
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"apex_rule: BLOCKED — trailing drawdown {used_pct:.0%} used. "
                    f"Only ${remaining:.0f} remaining above floor (${acct.drawdown_floor:.0f}). "
                    f"New entries blocked to protect account."
                ),
            )

        return None

    def _check_max_contracts(
        self,
        action: LLMAction,
        current_contracts: int,
    ) -> Optional[GuardrailResult]:
        """Enforce Apex max contract limits (scaling-aware).

        During scaling phase (before profit buffer reached), the max
        is halved (e.g. 50 micros instead of 100 for 50k accounts).
        """
        acct = self._account_state
        effective_max = acct.effective_max_micros
        requested = action.quantity or 1

        total_after = current_contracts + requested

        if total_after > effective_max:
            self._blocks += 1
            scaling_note = ""
            if not acct.scaling_unlocked and acct.max_micros_scaling > 0:
                profit_needed = acct.scaling_unlock_profit - max(
                    0.0, acct.peak_balance - acct.starting_balance
                )
                scaling_note = (
                    f" Scaling active — need ${profit_needed:.0f} more profit "
                    f"to unlock full {acct.max_micros} contracts."
                )
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"apex_rule: BLOCKED — would exceed Apex max contracts. "
                    f"Current: {current_contracts}, requested: +{requested}, "
                    f"max allowed: {effective_max}.{scaling_note}"
                ),
            )

        return None

    def _check_daily_loss_limit(
        self,
        daily_pnl: float,
    ) -> Optional[GuardrailResult]:
        """Check Apex daily loss limit (only for EOD accounts with DLL)."""
        dll = self._account_state.daily_loss_limit
        if dll is None:
            return None  # no DLL for this account type

        if daily_pnl <= -dll:
            self._blocks += 1
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"apex_rule: BLOCKED — daily loss ${abs(daily_pnl):.0f} "
                    f"exceeds Apex daily loss limit ${dll:.0f}. "
                    f"Trading paused until next session."
                ),
            )

        # Warning zone — within 50% of DLL
        if daily_pnl <= -(dll * 0.50):
            self._warnings += 1

        return None

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def account_state(self) -> ApexAccountState:
        return self._account_state

    @property
    def stats(self) -> dict[str, Any]:
        acct = self._account_state
        return {
            "account_balance": acct.current_balance,
            "peak_balance": acct.peak_balance,
            "drawdown_floor": acct.drawdown_floor,
            "drawdown_remaining": acct.drawdown_remaining,
            "drawdown_remaining_pct": round(acct.drawdown_remaining_pct, 2),
            "drawdown_locked": acct.drawdown_locked,
            "max_micros": acct.max_micros,
            "effective_max_micros": acct.effective_max_micros,
            "scaling_unlocked": acct.scaling_unlocked,
            "consistency_ok": acct.consistency_ok,
            "blocks": self._blocks,
            "warnings": self._warnings,
            "deadline_flattens": self._deadline_flattens,
        }
