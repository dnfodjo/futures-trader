"""Tests for risk check guardrails."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.types import ActionType, LLMAction, MarketState, PositionState, SessionPhase, Side
from src.guardrails.risk_checks import RiskCheckGuardrail


def _action(
    action: ActionType = ActionType.ENTER,
    side: Side = Side.LONG,
    quantity: int = 1,
    stop_distance: float | None = 10.0,
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


def _state(**overrides) -> MarketState:
    defaults = dict(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        last_price=19850.0,
        bid=19849.75,
        ask=19850.25,
        session_phase=SessionPhase.MORNING,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _position(
    side: Side = Side.LONG,
    avg_entry: float = 19850.0,
    quantity: int = 2,
) -> PositionState:
    return PositionState(side=side, quantity=quantity, avg_entry=avg_entry)


@pytest.fixture
def guard():
    return RiskCheckGuardrail(
        min_stop_distance=3.0,
        max_stop_distance=25.0,
        min_confidence=0.3,
        min_entry_spacing_pts=5.0,
        max_spread_pts=3.0,
    )


# ── Confidence Threshold ────────────────────────────────────────────────────


class TestConfidence:
    def test_above_threshold(self, guard):
        action = _action(confidence=0.6)
        result = guard.check(action, _state())
        assert result.allowed is True

    def test_below_threshold(self, guard):
        action = _action(confidence=0.2)
        result = guard.check(action, _state())
        assert result.allowed is False
        assert "confidence" in result.reason

    def test_at_threshold(self, guard):
        action = _action(confidence=0.3)
        result = guard.check(action, _state())
        assert result.allowed is True


# ── Stop Distance ────────────────────────────────────────────────────────────


class TestStopDistance:
    def test_valid_stop(self, guard):
        action = _action(stop_distance=10.0)
        result = guard.check(action, _state())
        assert result.allowed is True

    def test_stop_too_tight(self, guard):
        action = _action(stop_distance=2.0)
        result = guard.check(action, _state())
        assert result.allowed is False
        assert "below minimum" in result.reason

    def test_stop_too_wide(self, guard):
        action = _action(stop_distance=30.0)
        result = guard.check(action, _state())
        assert result.allowed is False
        assert "exceeds" in result.reason and "maximum" in result.reason

    def test_stop_at_minimum(self, guard):
        action = _action(stop_distance=3.0)
        result = guard.check(action, _state())
        assert result.allowed is True

    def test_stop_at_maximum(self, guard):
        # Use 10.0 which is valid in both ETH (max 12) and RTH (max 25)
        action = _action(stop_distance=10.0)
        result = guard.check(action, _state())
        assert result.allowed is True

    def test_no_stop_distance_passes(self, guard):
        action = _action(stop_distance=None)
        result = guard.check(action, _state())
        assert result.allowed is True


# ── Entry Spacing ────────────────────────────────────────────────────────────


class TestEntrySpacing:
    def test_add_far_from_entry(self, guard):
        pos = _position(avg_entry=19840.0)  # 10pts away
        state = _state(last_price=19850.0)
        action = _action(action=ActionType.ADD)
        result = guard.check(action, state, position=pos)
        assert result.allowed is True

    def test_add_too_close_to_entry(self, guard):
        pos = _position(avg_entry=19848.0)  # only 2pts away
        state = _state(last_price=19850.0)
        action = _action(action=ActionType.ADD)
        result = guard.check(action, state, position=pos)
        assert result.allowed is False
        assert "too close" in result.reason

    def test_enter_ignores_spacing(self, guard):
        # ENTER doesn't check spacing (no existing position to space from)
        action = _action(action=ActionType.ENTER)
        result = guard.check(action, _state())
        assert result.allowed is True


# ── Spread Check ─────────────────────────────────────────────────────────────


class TestSpreadCheck:
    def test_normal_spread(self, guard):
        state = _state(bid=19849.75, ask=19850.25)  # 0.5pt spread
        action = _action()
        result = guard.check(action, state)
        assert result.allowed is True

    def test_wide_spread_blocked(self, guard):
        state = _state(bid=19847.0, ask=19853.0)  # 6pt spread
        action = _action()
        result = guard.check(action, state)
        assert result.allowed is False
        assert "illiquid" in result.reason

    def test_spread_at_max(self, guard):
        state = _state(bid=19848.5, ask=19851.5)  # exactly 3pt
        action = _action()
        result = guard.check(action, state)
        assert result.allowed is True


# ── Passthrough ──────────────────────────────────────────────────────────────


class TestPassthrough:
    def test_scale_out_passes(self, guard):
        action = _action(action=ActionType.SCALE_OUT, confidence=0.1)  # low confidence
        result = guard.check(action, _state())
        assert result.allowed is True

    def test_flatten_passes(self, guard):
        action = _action(action=ActionType.FLATTEN, confidence=0.1)
        result = guard.check(action, _state())
        assert result.allowed is True

    def test_do_nothing_passes(self, guard):
        action = _action(action=ActionType.DO_NOTHING, confidence=0.0)
        result = guard.check(action, _state())
        assert result.allowed is True
