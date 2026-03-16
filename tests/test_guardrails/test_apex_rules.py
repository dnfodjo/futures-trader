"""Comprehensive tests for Apex Trader Funding compliance guardrails.

Tests cover:
- ApexAccountState: balance tracking, trailing drawdown, scaling, consistency
- ApexRuleGuardrail: flatten deadline, drawdown lockout, max contracts, DLL
- Scaling rule enforcement (half contracts until profit buffer)
- Edge cases: exactly at limits, safety net lock, payout reset
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pytest

from src.core.types import (
    ActionType,
    GuardrailResult,
    LLMAction,
    MarketState,
)
from src.guardrails.apex_rules import (
    APEX_ACCOUNTS,
    ApexAccountState,
    ApexRuleGuardrail,
)

ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_action(
    action: ActionType = ActionType.ENTER,
    quantity: int = 1,
    **kwargs,
) -> LLMAction:
    return LLMAction(
        action=action,
        quantity=quantity,
        reasoning="test",
        confidence=0.7,
        **kwargs,
    )


def _make_state(hour: int = 10, minute: int = 0) -> MarketState:
    """Create a MarketState with a specific ET time."""
    ts = datetime(2026, 3, 15, hour, minute, 0, tzinfo=ET)
    return MarketState(timestamp=ts, last_price=19850.0)


# ── ApexAccountState Tests ───────────────────────────────────────────────────


class TestApexAccountState:
    """Tests for the account state tracking dataclass."""

    def test_initial_state_defaults(self):
        state = ApexAccountState()
        assert state.starting_balance == 50_000.0
        assert state.trailing_drawdown == 2_500.0
        assert state.max_micros == 100
        assert state.max_micros_scaling == 50
        assert state.scaling_unlock_profit == 2_600.0
        assert state.safety_net == 50_100.0
        assert state.daily_loss_limit == 1_000.0
        assert state.current_balance == 50_000.0
        assert state.peak_balance == 50_000.0
        assert state.drawdown_floor == 47_500.0
        assert not state.drawdown_locked
        assert not state.scaling_unlocked

    def test_drawdown_remaining(self):
        state = ApexAccountState()
        # Starting: balance 50k, floor 47.5k → remaining $2,500
        assert state.drawdown_remaining == 2_500.0
        assert state.drawdown_remaining_pct == 1.0

    def test_drawdown_remaining_after_loss(self):
        state = ApexAccountState()
        state.update_balance(49_000.0)
        # Balance 49k, floor still 47.5k → remaining $1,500
        assert state.drawdown_remaining == 1_500.0
        assert state.drawdown_remaining_pct == pytest.approx(0.6)

    def test_drawdown_trails_up_with_profit(self):
        """Use 25k account where safety net ($26,600) gives room to observe trailing."""
        state = ApexAccountState(
            starting_balance=25_000.0,
            trailing_drawdown=1_500.0,
            max_micros=40,
            max_micros_scaling=0,
            scaling_unlock_profit=0.0,
            safety_net=26_600.0,
            daily_loss_limit=None,
            current_balance=25_000.0,
            peak_balance=25_000.0,
            drawdown_floor=23_500.0,
        )
        state.update_balance(25_500.0)
        # Peak is now 25.5k, below safety net of 26.6k, floor trails: 25.5k - 1.5k = 24k
        assert state.peak_balance == 25_500.0
        assert state.drawdown_floor == 24_000.0
        assert state.drawdown_remaining == 1_500.0  # always full dd when at peak

    def test_drawdown_never_trails_down(self):
        """Use 25k account to observe trailing behavior before safety net lock."""
        state = ApexAccountState(
            starting_balance=25_000.0,
            trailing_drawdown=1_500.0,
            max_micros=40,
            max_micros_scaling=0,
            scaling_unlock_profit=0.0,
            safety_net=26_600.0,
            daily_loss_limit=None,
            current_balance=25_000.0,
            peak_balance=25_000.0,
            drawdown_floor=23_500.0,
        )
        # Go up (still below safety net 26,600)
        state.update_balance(25_500.0)
        assert state.drawdown_floor == 24_000.0
        # Go down — floor should NOT move down
        state.update_balance(24_500.0)
        assert state.drawdown_floor == 24_000.0  # stays at 24k
        assert state.drawdown_remaining == 500.0

    def test_safety_net_locks_drawdown(self):
        """Once peak reaches safety net, drawdown floor stops trailing."""
        state = ApexAccountState()
        # Push to exactly safety net
        state.update_balance(50_100.0)
        assert state.drawdown_locked is True
        # Floor locked at safety_net - trailing_drawdown
        assert state.drawdown_floor == 47_600.0

        # Go higher — floor should NOT trail up anymore
        state.update_balance(51_000.0)
        assert state.drawdown_floor == 47_600.0  # still locked

    def test_safety_net_above(self):
        """Peak above safety net still locks."""
        state = ApexAccountState()
        state.update_balance(52_000.0)
        assert state.drawdown_locked is True
        assert state.drawdown_floor == 47_600.0

    def test_scaling_not_unlocked_initially(self):
        state = ApexAccountState()
        assert state.scaling_unlocked is False
        assert state.effective_max_micros == 50  # scaling limit

    def test_scaling_unlocked_at_profit_buffer(self):
        state = ApexAccountState()
        # Need $2,600 profit → balance of $52,600
        state.update_balance(52_600.0)
        assert state.scaling_unlocked is True
        assert state.effective_max_micros == 100  # full limit

    def test_scaling_not_unlocked_just_below_threshold(self):
        state = ApexAccountState()
        state.update_balance(52_599.0)
        assert state.scaling_unlocked is False
        assert state.effective_max_micros == 50

    def test_scaling_stays_unlocked_after_pullback(self):
        """Once scaling is unlocked, it stays unlocked even if profit drops."""
        state = ApexAccountState()
        state.update_balance(52_600.0)
        assert state.scaling_unlocked is True
        # Pull back below the threshold
        state.update_balance(51_000.0)
        assert state.scaling_unlocked is True  # stays unlocked
        assert state.effective_max_micros == 100

    def test_no_scaling_for_accounts_without_rule(self):
        """25k and 150k accounts have no scaling rule."""
        state = ApexAccountState(
            starting_balance=25_000.0,
            trailing_drawdown=1_500.0,
            max_micros=40,
            max_micros_scaling=0,
            scaling_unlock_profit=0.0,
            safety_net=26_600.0,
            daily_loss_limit=None,
            current_balance=25_000.0,
            peak_balance=25_000.0,
            drawdown_floor=23_500.0,
        )
        assert state.effective_max_micros == 40  # full limit immediately


class TestConsistencyRule:
    """Tests for the 50% consistency rule."""

    def test_no_profit_is_ok(self):
        state = ApexAccountState()
        assert state.consistency_ok is True

    def test_single_day_all_profit(self):
        """One day with all profit → 100% > 50% → not ok."""
        state = ApexAccountState()
        state.record_day_profit(500.0)
        # 500/500 = 100% > 50%
        assert state.consistency_ok is False

    def test_two_equal_days(self):
        """Two equal profit days → 50% each → exactly at limit."""
        state = ApexAccountState()
        state.record_day_profit(300.0)
        state.record_day_profit(300.0)
        # Best day: 300, total: 600, ratio: 50% → exactly at limit
        assert state.consistency_ok is True

    def test_well_distributed(self):
        """Multiple days well distributed → OK."""
        state = ApexAccountState()
        for _ in range(5):
            state.record_day_profit(200.0)
        # Best: 200, total: 1000, ratio: 20%
        assert state.consistency_ok is True

    def test_one_big_day_ruins_it(self):
        """One big day among small ones → not ok."""
        state = ApexAccountState()
        state.record_day_profit(100.0)
        state.record_day_profit(100.0)
        state.record_day_profit(600.0)  # big day
        # Best: 600, total: 800, ratio: 75% > 50%
        assert state.consistency_ok is False

    def test_losses_dont_affect_total(self):
        """Losses don't reduce total profit for consistency calculation."""
        state = ApexAccountState()
        state.record_day_profit(200.0)
        state.record_day_profit(-100.0)  # loss — ignored
        state.record_day_profit(200.0)
        # Total: 400 (losses ignored), best: 200, ratio: 50%
        assert state.consistency_ok is True

    def test_waived_after_six_payouts(self):
        state = ApexAccountState()
        state.payouts_completed = 6
        state.record_day_profit(1000.0)
        # Even 100% ratio — waived after 6 payouts
        assert state.consistency_ok is True

    def test_payout_resets_tracking(self):
        state = ApexAccountState()
        state.record_day_profit(500.0)  # 100% — not ok
        assert state.consistency_ok is False
        state.record_payout()
        assert state.consistency_ok is True  # reset
        assert state.total_profit_since_payout == 0.0
        assert state.best_single_day_profit == 0.0
        assert state.payouts_completed == 1


# ── ApexRuleGuardrail Tests ──────────────────────────────────────────────────


class TestApexAccountPresets:
    """Verify account presets are correctly defined."""

    def test_25k_preset(self):
        acct = APEX_ACCOUNTS["25k"]
        assert acct["starting_balance"] == 25_000.0
        assert acct["trailing_drawdown"] == 1_500.0
        assert acct["max_micros"] == 40
        assert acct["safety_net"] == 26_600.0
        assert acct["daily_loss_limit"] is None

    def test_50k_preset(self):
        acct = APEX_ACCOUNTS["50k"]
        assert acct["starting_balance"] == 50_000.0
        assert acct["trailing_drawdown"] == 2_500.0
        assert acct["max_micros"] == 100
        assert acct["max_micros_scaling"] == 50
        assert acct["scaling_unlock_profit"] == 2_600.0
        assert acct["safety_net"] == 50_100.0
        assert acct["daily_loss_limit"] == 1_000.0

    def test_100k_preset(self):
        acct = APEX_ACCOUNTS["100k"]
        assert acct["starting_balance"] == 100_000.0
        assert acct["trailing_drawdown"] == 3_000.0
        assert acct["daily_loss_limit"] == 1_500.0

    def test_150k_preset(self):
        acct = APEX_ACCOUNTS["150k"]
        assert acct["starting_balance"] == 150_000.0
        assert acct["trailing_drawdown"] == 5_000.0
        assert acct["daily_loss_limit"] is None


class TestApexGuardrailInit:
    """Tests for guardrail initialization."""

    def test_default_50k(self):
        g = ApexRuleGuardrail()
        assert g.account_state.starting_balance == 50_000.0
        assert g.account_state.max_micros == 100
        assert g.account_state.max_micros_scaling == 50

    def test_25k_account(self):
        g = ApexRuleGuardrail(account_type="25k")
        assert g.account_state.starting_balance == 25_000.0

    def test_invalid_account_raises(self):
        with pytest.raises(ValueError, match="Unknown Apex account type"):
            ApexRuleGuardrail(account_type="999k")

    def test_custom_account(self):
        custom = {
            "starting_balance": 75_000.0,
            "trailing_drawdown": 3_000.0,
            "max_micros": 120,
            "safety_net": 78_100.0,
        }
        g = ApexRuleGuardrail(custom_account=custom)
        assert g.account_state.starting_balance == 75_000.0
        assert g.account_state.max_micros == 120


class TestPassthroughActions:
    """Actions that should always be allowed."""

    def setup_method(self):
        self.g = ApexRuleGuardrail()
        self.state = _make_state(10, 0)

    def test_do_nothing_allowed(self):
        result = self.g.check(_make_action(ActionType.DO_NOTHING), self.state)
        assert result.allowed is True

    def test_stop_trading_allowed(self):
        result = self.g.check(_make_action(ActionType.STOP_TRADING), self.state)
        assert result.allowed is True

    def test_flatten_allowed(self):
        result = self.g.check(_make_action(ActionType.FLATTEN), self.state)
        assert result.allowed is True

    def test_move_stop_allowed(self):
        result = self.g.check(_make_action(ActionType.MOVE_STOP), self.state)
        assert result.allowed is True

    def test_flatten_allowed_even_past_deadline(self):
        """FLATTEN must never be blocked, even past deadline."""
        state = _make_state(16, 58)  # 4:58 PM ET
        result = self.g.check(_make_action(ActionType.FLATTEN), state)
        assert result.allowed is True


class TestFlattenDeadline:
    """Tests for the position flat deadline check."""

    def setup_method(self):
        self.g = ApexRuleGuardrail(
            flatten_deadline_et=time(16, 54),  # 4:54 PM ET
        )

    def test_entry_allowed_before_deadline(self):
        state = _make_state(16, 53)  # 4:53 PM — 1 min before
        result = self.g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_entry_blocked_at_deadline(self):
        state = _make_state(16, 54)  # exactly 4:54 PM
        result = self.g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is False
        assert "flatten deadline" in result.reason

    def test_entry_blocked_after_deadline(self):
        state = _make_state(16, 57)  # 4:57 PM
        result = self.g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is False

    def test_add_blocked_after_deadline(self):
        state = _make_state(16, 55)
        result = self.g.check(_make_action(ActionType.ADD), state)
        assert result.allowed is False

    def test_morning_entry_allowed(self):
        state = _make_state(10, 30)  # 10:30 AM
        result = self.g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_should_force_flatten_before_deadline(self):
        state = _make_state(16, 53)
        assert self.g.should_force_flatten(state) is False

    def test_should_force_flatten_at_deadline(self):
        state = _make_state(16, 54)
        assert self.g.should_force_flatten(state) is True

    def test_should_force_flatten_not_after_halt(self):
        """Flatten should NOT trigger after daily halt (17:00+) or in new session (18:00+)."""
        # During daily halt — NOT a flatten window
        state_halt = _make_state(17, 30)
        assert self.g.should_force_flatten(state_halt) is False

        # New session (Asian) — definitely not flatten
        state_asian = _make_state(18, 30)
        assert self.g.should_force_flatten(state_asian) is False

        # Overnight — not flatten
        state_overnight = _make_state(2, 0)
        assert self.g.should_force_flatten(state_overnight) is False

    def test_flatten_deadline_blocks_entry_only_pre_halt(self):
        """Entry blocking should only apply in pre-halt window, not after 18:00."""
        # 4:55 PM — should block
        state_pre_halt = _make_state(16, 55)
        result = self.g.check(_make_action(ActionType.ENTER), state_pre_halt)
        assert result.allowed is False

        # 6:30 PM — new session, should allow
        state_new_session = _make_state(18, 30)
        result = self.g.check(_make_action(ActionType.ENTER), state_new_session)
        assert result.allowed is True

    def test_custom_earlier_deadline(self):
        g = ApexRuleGuardrail(flatten_deadline_et=time(16, 45))
        state = _make_state(16, 46)
        result = g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is False


class TestDrawdownProximity:
    """Tests for the trailing drawdown lockout."""

    def setup_method(self):
        self.g = ApexRuleGuardrail(drawdown_lockout_pct=0.75)
        self.state = _make_state(10, 0)

    def test_entry_allowed_with_full_drawdown(self):
        # At starting balance, 0% used
        result = self.g.check(_make_action(ActionType.ENTER), self.state)
        assert result.allowed is True

    def test_entry_allowed_at_50pct_drawdown(self):
        # Lose $1,250 of $2,500 drawdown → 50% used
        self.g.update_equity(48_750.0)
        result = self.g.check(_make_action(ActionType.ENTER), self.state)
        assert result.allowed is True

    def test_entry_blocked_at_75pct_drawdown(self):
        # Lose $1,875 of $2,500 → 75% used → lockout
        self.g.update_equity(48_125.0)
        result = self.g.check(_make_action(ActionType.ENTER), self.state)
        assert result.allowed is False
        assert "trailing drawdown" in result.reason

    def test_entry_blocked_at_90pct_drawdown(self):
        # Lose $2,250 of $2,500 → 90% used
        self.g.update_equity(47_750.0)
        result = self.g.check(_make_action(ActionType.ENTER), self.state)
        assert result.allowed is False

    def test_drawdown_lockout_after_profit_and_loss(self):
        """Go up past safety net (floor locks), then come back down."""
        # Profit: balance 51k → safety net ($50,100) hit → floor locks at $47,600
        self.g.update_equity(51_000.0)
        assert self.g.account_state.drawdown_locked is True
        assert self.g.account_state.drawdown_floor == 47_600.0
        # Now drop close to the locked floor
        # Floor is 47,600, DD is 2,500
        # At 48,000 → remaining = 400, used = 2100/2500 = 84%
        self.g.update_equity(48_000.0)
        result = self.g.check(_make_action(ActionType.ENTER), self.state)
        assert result.allowed is False

    def test_custom_lockout_percentage(self):
        """More aggressive lockout at 50%."""
        g = ApexRuleGuardrail(drawdown_lockout_pct=0.50)
        # Lose $1,250 of $2,500 → exactly 50%
        g.update_equity(48_750.0)
        state = _make_state(10, 0)
        result = g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is False


class TestMaxContracts:
    """Tests for Apex max contract enforcement (scaling-aware)."""

    def setup_method(self):
        self.g = ApexRuleGuardrail(account_type="50k")
        self.state = _make_state(10, 0)

    def test_single_contract_allowed(self):
        result = self.g.check(
            _make_action(ActionType.ENTER, quantity=1),
            self.state,
            current_contracts=0,
        )
        assert result.allowed is True

    def test_scaling_limit_enforced_initially(self):
        """Before profit buffer, max is 50 micros (scaling rule)."""
        # Try to enter with 51 contracts — should be blocked
        result = self.g.check(
            _make_action(ActionType.ENTER, quantity=51),
            self.state,
            current_contracts=0,
        )
        assert result.allowed is False
        assert "Scaling active" in result.reason

    def test_scaling_limit_at_exactly_50(self):
        """Exactly at scaling limit → allowed."""
        result = self.g.check(
            _make_action(ActionType.ENTER, quantity=10),
            self.state,
            current_contracts=40,
        )
        assert result.allowed is True

    def test_scaling_limit_exceeded_by_one(self):
        """One over scaling limit → blocked."""
        result = self.g.check(
            _make_action(ActionType.ENTER, quantity=1),
            self.state,
            current_contracts=50,
        )
        assert result.allowed is False

    def test_full_limit_after_scaling_unlocked(self):
        """After profit buffer reached, full 100 micros allowed."""
        # Unlock scaling: profit $2,600 → balance $52,600
        self.g.update_equity(52_600.0)
        assert self.g.account_state.scaling_unlocked is True

        result = self.g.check(
            _make_action(ActionType.ENTER, quantity=51),
            self.state,
            current_contracts=0,
        )
        assert result.allowed is True

    def test_full_limit_blocked_at_101(self):
        """Even with scaling unlocked, can't exceed 100."""
        self.g.update_equity(52_600.0)
        result = self.g.check(
            _make_action(ActionType.ENTER, quantity=1),
            self.state,
            current_contracts=100,
        )
        assert result.allowed is False
        assert "max contracts" in result.reason

    def test_add_contracts_current_plus_new(self):
        """ADD action checks current + requested."""
        result = self.g.check(
            _make_action(ActionType.ADD, quantity=5),
            self.state,
            current_contracts=46,
        )
        # 46 + 5 = 51 > 50 (scaling limit)
        assert result.allowed is False

    def test_25k_account_no_scaling(self):
        """25k account has no scaling rule."""
        g = ApexRuleGuardrail(account_type="25k")
        state = _make_state(10, 0)
        # Full 40 micros allowed immediately
        result = g.check(
            _make_action(ActionType.ENTER, quantity=40),
            state,
            current_contracts=0,
        )
        assert result.allowed is True

    def test_25k_blocked_over_40(self):
        g = ApexRuleGuardrail(account_type="25k")
        state = _make_state(10, 0)
        result = g.check(
            _make_action(ActionType.ENTER, quantity=1),
            state,
            current_contracts=40,
        )
        assert result.allowed is False


class TestDailyLossLimit:
    """Tests for the daily loss limit check."""

    def setup_method(self):
        self.g = ApexRuleGuardrail(account_type="50k")  # DLL = $1,000
        self.state = _make_state(10, 0)

    def test_no_loss_allowed(self):
        result = self.g.check(
            _make_action(ActionType.ENTER),
            self.state,
            daily_pnl=0.0,
        )
        assert result.allowed is True

    def test_small_loss_allowed(self):
        result = self.g.check(
            _make_action(ActionType.ENTER),
            self.state,
            daily_pnl=-400.0,
        )
        assert result.allowed is True

    def test_at_dll_blocked(self):
        """Exactly at DLL → blocked."""
        result = self.g.check(
            _make_action(ActionType.ENTER),
            self.state,
            daily_pnl=-1_000.0,
        )
        assert result.allowed is False
        assert "daily loss" in result.reason

    def test_over_dll_blocked(self):
        result = self.g.check(
            _make_action(ActionType.ENTER),
            self.state,
            daily_pnl=-1_200.0,
        )
        assert result.allowed is False

    def test_profit_allowed(self):
        result = self.g.check(
            _make_action(ActionType.ENTER),
            self.state,
            daily_pnl=500.0,
        )
        assert result.allowed is True

    def test_no_dll_for_25k_account(self):
        """25k account has no DLL — should never block on loss limit."""
        g = ApexRuleGuardrail(account_type="25k")
        state = _make_state(10, 0)
        # Even $5,000 loss (which would breach drawdown separately)
        result = g.check(
            _make_action(ActionType.ENTER),
            state,
            daily_pnl=-5_000.0,
        )
        # DLL check passes (no DLL), but drawdown proximity would catch this
        # separately. For this test we only care about DLL.
        # Since drawdown would fire first, let's test a moderate loss
        # that wouldn't trigger drawdown
        result2 = g.check(
            _make_action(ActionType.ENTER),
            state,
            daily_pnl=-800.0,
        )
        assert result2.allowed is True  # no DLL block

    def test_warning_at_50pct_dll(self):
        """Warning zone is tracked but doesn't block."""
        result = self.g.check(
            _make_action(ActionType.ENTER),
            self.state,
            daily_pnl=-500.0,  # 50% of $1,000 DLL
        )
        assert result.allowed is True
        assert self.g._warnings >= 1


class TestEquityUpdate:
    """Tests for equity update and drawdown tracking."""

    def test_update_equity(self):
        g = ApexRuleGuardrail()
        g.update_equity(50_500.0)
        assert g.account_state.current_balance == 50_500.0
        assert g.account_state.peak_balance == 50_500.0

    def test_drawdown_floor_locks_at_safety_net(self):
        """50k safety net is $50,100 — floor locks immediately on small profit."""
        g = ApexRuleGuardrail()
        g.update_equity(51_000.0)
        # Safety net ($50,100) hit → floor locks at 50,100 - 2,500 = 47,600
        assert g.account_state.drawdown_locked is True
        assert g.account_state.drawdown_floor == 47_600.0

    def test_drawdown_locks_at_safety_net(self):
        g = ApexRuleGuardrail()
        g.update_equity(50_100.0)  # safety net for 50k
        assert g.account_state.drawdown_locked is True
        # Floor: 50,100 - 2,500 = 47,600
        assert g.account_state.drawdown_floor == 47_600.0


class TestStats:
    """Tests for the stats property."""

    def test_initial_stats(self):
        g = ApexRuleGuardrail()
        s = g.stats
        assert s["account_balance"] == 50_000.0
        assert s["peak_balance"] == 50_000.0
        assert s["drawdown_remaining"] == 2_500.0
        assert s["drawdown_remaining_pct"] == 1.0
        assert s["drawdown_locked"] is False
        assert s["max_micros"] == 100
        assert s["effective_max_micros"] == 50  # scaling active
        assert s["scaling_unlocked"] is False
        assert s["consistency_ok"] is True
        assert s["blocks"] == 0
        assert s["warnings"] == 0

    def test_stats_after_block(self):
        g = ApexRuleGuardrail()
        # Trigger a DLL block
        state = _make_state(10, 0)
        g.check(_make_action(ActionType.ENTER), state, daily_pnl=-1_000.0)
        assert g.stats["blocks"] == 1

    def test_stats_after_scaling_unlock(self):
        g = ApexRuleGuardrail()
        g.update_equity(52_600.0)
        assert g.stats["effective_max_micros"] == 100
        assert g.stats["scaling_unlocked"] is True


class TestCheckPriorityOrder:
    """Tests that checks fire in the correct priority order."""

    def test_deadline_fires_before_drawdown(self):
        """Flatten deadline check should fire before drawdown check."""
        g = ApexRuleGuardrail()
        # Both conditions active: past deadline AND near drawdown
        g.update_equity(48_000.0)  # 80% drawdown used
        state = _make_state(16, 55)  # past deadline
        result = g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is False
        assert "flatten deadline" in result.reason  # deadline fires first

    def test_drawdown_fires_before_contracts(self):
        """Drawdown check should fire before contract check."""
        g = ApexRuleGuardrail()
        g.update_equity(48_000.0)  # 80% drawdown used
        state = _make_state(10, 0)
        # Try to add 200 contracts (way over limit)
        result = g.check(
            _make_action(ActionType.ENTER, quantity=200),
            state,
            current_contracts=0,
        )
        assert result.allowed is False
        assert "trailing drawdown" in result.reason  # drawdown fires first


class TestIntegrationScenarios:
    """End-to-end scenarios simulating real trading sessions."""

    def test_profitable_day_unlocks_scaling(self):
        """Simulate a profitable day that unlocks scaling."""
        g = ApexRuleGuardrail(account_type="50k")
        state = _make_state(10, 0)

        # Start: scaling active, max 50 micros
        assert g.account_state.effective_max_micros == 50

        # Make money gradually
        for equity in [50_500, 51_000, 51_500, 52_000, 52_600]:
            g.update_equity(float(equity))

        # Scaling now unlocked
        assert g.account_state.scaling_unlocked is True
        assert g.account_state.effective_max_micros == 100

        # Can now trade 60 contracts
        result = g.check(
            _make_action(ActionType.ENTER, quantity=60),
            state,
            current_contracts=0,
        )
        assert result.allowed is True

    def test_drawdown_danger_zone_blocks_entries(self):
        """Simulate approaching the drawdown floor after safety net locks."""
        g = ApexRuleGuardrail(account_type="50k")
        state = _make_state(10, 0)

        # Profit: balance 51k → safety net ($50,100) hit → floor locks at $47,600
        g.update_equity(51_000.0)
        assert g.account_state.drawdown_floor == 47_600.0

        # Drop close to the locked floor
        # At 48,000 → remaining = 400, used = 2,100/2,500 = 84% > 75% lockout
        g.update_equity(48_000.0)
        result = g.check(_make_action(ActionType.ENTER), state)
        assert result.allowed is False

    def test_end_of_day_flatten_cycle(self):
        """Simulate approaching the end-of-day flatten deadline."""
        g = ApexRuleGuardrail(account_type="50k")

        # 4:50 PM — still trading
        state_450 = _make_state(16, 50)
        assert g.should_force_flatten(state_450) is False
        result = g.check(_make_action(ActionType.ENTER), state_450)
        assert result.allowed is True

        # 4:54 PM — deadline hit
        state_454 = _make_state(16, 54)
        assert g.should_force_flatten(state_454) is True
        result = g.check(_make_action(ActionType.ENTER), state_454)
        assert result.allowed is False

        # FLATTEN still allowed at 4:58 PM
        state_458 = _make_state(16, 58)
        result = g.check(_make_action(ActionType.FLATTEN), state_458)
        assert result.allowed is True

    def test_full_session_lifecycle(self):
        """Simulate a complete trading session."""
        g = ApexRuleGuardrail(account_type="50k")
        state = _make_state(10, 0)

        # 1. Enter small position (within scaling limit)
        result = g.check(
            _make_action(ActionType.ENTER, quantity=5),
            state,
            current_contracts=0,
        )
        assert result.allowed is True

        # 2. Make some money
        g.update_equity(50_300.0)

        # 3. Add more
        result = g.check(
            _make_action(ActionType.ADD, quantity=5),
            state,
            current_contracts=5,
        )
        assert result.allowed is True  # 10 contracts, under 50 scaling limit

        # 4. Scale out
        result = g.check(
            _make_action(ActionType.SCALE_OUT, quantity=3),
            state,
            current_contracts=10,
        )
        # SCALE_OUT is not in passthrough but also not an ENTER/ADD
        # so it goes through all checks but doesn't hit contract limit
        # because it's reducing position
        assert result.allowed is True

        # 5. Move stop — always allowed
        result = g.check(_make_action(ActionType.MOVE_STOP), state)
        assert result.allowed is True

        # 6. End of day — flatten
        state_eod = _make_state(16, 55)
        result = g.check(_make_action(ActionType.FLATTEN), state_eod)
        assert result.allowed is True

    def test_dll_soft_breach_blocks_but_doesnt_fail(self):
        """DLL is a soft breach — blocks new entries but doesn't fail account."""
        g = ApexRuleGuardrail(account_type="50k")
        state = _make_state(10, 0)

        # Hit DLL
        result = g.check(
            _make_action(ActionType.ENTER),
            state,
            daily_pnl=-1_000.0,
        )
        assert result.allowed is False
        assert "paused" in result.reason  # not "terminated"

        # But can still flatten
        result = g.check(
            _make_action(ActionType.FLATTEN),
            state,
            daily_pnl=-1_000.0,
        )
        assert result.allowed is True
