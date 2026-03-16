"""Tests for position limit guardrails."""

from __future__ import annotations

import pytest

from src.core.types import ActionType, LLMAction, PositionState, Side
from src.guardrails.position_limits import PositionLimitGuardrail


def _action(
    action: ActionType,
    side: Side | None = None,
    quantity: int | None = None,
    stop_distance: float | None = None,
    confidence: float = 0.8,
) -> LLMAction:
    return LLMAction(
        action=action,
        side=side,
        quantity=quantity,
        stop_distance=stop_distance,
        reasoning="test",
        confidence=confidence,
    )


def _position(
    side: Side = Side.LONG,
    quantity: int = 2,
    avg_entry: float = 19850.0,
    adds_count: int = 0,
) -> PositionState:
    return PositionState(
        side=side,
        quantity=quantity,
        avg_entry=avg_entry,
        adds_count=adds_count,
    )


@pytest.fixture
def guard():
    return PositionLimitGuardrail(max_contracts=6, max_adds=3)


# ── ENTER Tests ──────────────────────────────────────────────────────────────


class TestEnter:
    def test_enter_allowed_when_flat(self, guard):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=2)
        result = guard.check(action, position=None)
        assert result.allowed is True

    def test_enter_blocked_when_in_position(self, guard):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1)
        result = guard.check(action, position=_position())
        assert result.allowed is False
        assert "cannot ENTER" in result.reason

    def test_enter_clamped_to_max(self, guard):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=10)
        result = guard.check(action, position=None)
        assert result.allowed is True
        assert result.modified_quantity == 6

    def test_enter_within_max(self, guard):
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=4)
        result = guard.check(action, position=None)
        assert result.allowed is True
        assert result.modified_quantity is None

    def test_enter_respects_override(self, guard):
        # Profit preservation reduced max to 3
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=5)
        result = guard.check(action, position=None, max_contracts_override=3)
        assert result.allowed is True
        assert result.modified_quantity == 3


# ── ADD Tests ────────────────────────────────────────────────────────────────


class TestAdd:
    def test_add_allowed(self, guard):
        pos = _position(quantity=2, adds_count=0)
        action = _action(ActionType.ADD, quantity=1)
        result = guard.check(action, position=pos)
        assert result.allowed is True

    def test_add_blocked_no_position(self, guard):
        action = _action(ActionType.ADD, quantity=1)
        result = guard.check(action, position=None)
        assert result.allowed is False
        assert "cannot ADD" in result.reason

    def test_add_blocked_max_adds(self, guard):
        pos = _position(quantity=3, adds_count=3)  # already at max adds
        action = _action(ActionType.ADD, quantity=1)
        result = guard.check(action, position=pos)
        assert result.allowed is False
        assert "max adds" in result.reason

    def test_add_clamped_to_remaining(self, guard):
        pos = _position(quantity=5)  # 5 of 6 max
        action = _action(ActionType.ADD, quantity=3)
        result = guard.check(action, position=pos)
        assert result.allowed is True
        assert result.modified_quantity == 1  # only 1 more allowed

    def test_add_blocked_at_max_contracts(self, guard):
        pos = _position(quantity=6)  # already at max
        action = _action(ActionType.ADD, quantity=1)
        result = guard.check(action, position=pos)
        assert result.allowed is False
        assert "already at max" in result.reason


# ── SCALE_OUT Tests ──────────────────────────────────────────────────────────


class TestScaleOut:
    def test_scale_out_allowed(self, guard):
        pos = _position(quantity=3)
        action = _action(ActionType.SCALE_OUT, quantity=1)
        result = guard.check(action, position=pos)
        assert result.allowed is True

    def test_scale_out_no_position(self, guard):
        action = _action(ActionType.SCALE_OUT, quantity=1)
        result = guard.check(action, position=None)
        assert result.allowed is False

    def test_scale_out_clamped_to_position(self, guard):
        pos = _position(quantity=2)
        action = _action(ActionType.SCALE_OUT, quantity=5)
        result = guard.check(action, position=pos)
        assert result.allowed is True
        assert result.modified_quantity == 2  # clamped to full position


# ── Passthrough Tests ────────────────────────────────────────────────────────


class TestPassthrough:
    def test_do_nothing_passes(self, guard):
        action = _action(ActionType.DO_NOTHING)
        result = guard.check(action, position=None)
        assert result.allowed is True

    def test_flatten_passes(self, guard):
        action = _action(ActionType.FLATTEN)
        result = guard.check(action, position=_position())
        assert result.allowed is True

    def test_move_stop_passes(self, guard):
        action = _action(ActionType.MOVE_STOP)
        result = guard.check(action, position=_position())
        assert result.allowed is True


# ── Zero Override (is not None vs or) ───────────────────────────────────────


class TestZeroOverride:
    def test_zero_override_blocks_all_entries(self):
        """max_contracts_override=0 should block all entries.

        Previous bug: `max_contracts_override or self._max_contracts` would
        fall back to self._max_contracts when override is 0 (falsy).
        Fix: use `is not None` check instead.
        """
        guard = PositionLimitGuardrail(max_contracts=6)
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=1)
        result = guard.check(action, position=None, max_contracts_override=0)
        # Should be clamped to 0 contracts → no room → block
        assert result.allowed is True
        assert result.modified_quantity == 0  # clamped to 0

    def test_none_override_uses_default(self):
        """None override should fall back to the default max_contracts."""
        guard = PositionLimitGuardrail(max_contracts=6)
        action = _action(ActionType.ENTER, side=Side.LONG, quantity=3)
        result = guard.check(action, position=None, max_contracts_override=None)
        assert result.allowed is True
        assert result.modified_quantity is None  # 3 <= 6, no clamping
