"""Tests for session rule guardrails."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.core.types import (
    ActionType,
    EconomicEvent,
    LLMAction,
    MarketState,
    SessionPhase,
    Side,
)
from src.guardrails.session_rules import SessionRuleGuardrail


def _action(action: ActionType, confidence: float = 0.8) -> LLMAction:
    return LLMAction(
        action=action,
        side=Side.LONG,
        quantity=1,
        stop_distance=10.0,
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


@pytest.fixture
def guard():
    return SessionRuleGuardrail(
        max_consecutive_losers=4,
        daily_loss_limit=400.0,
        blackout_minutes=5,
    )


# ── Daily Loss Limit ────────────────────────────────────────────────────────


class TestDailyLoss:
    def test_allows_above_limit(self, guard):
        result = guard.check(_action(ActionType.ENTER), _state(), daily_pnl=-200.0)
        assert result.allowed is True

    def test_blocks_at_limit(self, guard):
        result = guard.check(_action(ActionType.ENTER), _state(), daily_pnl=-400.0)
        assert result.allowed is False
        assert "daily loss limit" in result.reason

    def test_blocks_below_limit(self, guard):
        result = guard.check(_action(ActionType.ENTER), _state(), daily_pnl=-500.0)
        assert result.allowed is False

    def test_positive_pnl_allowed(self, guard):
        result = guard.check(_action(ActionType.ENTER), _state(), daily_pnl=300.0)
        assert result.allowed is True


# ── Consecutive Losers ───────────────────────────────────────────────────────


class TestConsecutiveLosers:
    def test_allows_below_max(self, guard):
        result = guard.check(
            _action(ActionType.ENTER), _state(), consecutive_losers=2,
        )
        assert result.allowed is True

    def test_reduces_to_min_size_at_max(self, guard):
        """4+ consecutive losers reduces to 1 contract (not blocked)."""
        result = guard.check(
            _action(ActionType.ENTER), _state(), consecutive_losers=4,
        )
        assert result.allowed is True
        assert result.modified_quantity == 1

    def test_reduces_to_min_size_above_max(self, guard):
        """6 consecutive losers also reduces to 1 contract."""
        result = guard.check(
            _action(ActionType.ENTER), _state(), consecutive_losers=6,
        )
        assert result.allowed is True
        assert result.modified_quantity == 1


# ── Session Phase Blocking ───────────────────────────────────────────────────


class TestSessionPhase:
    def test_allows_during_morning(self, guard):
        state = _state(session_phase=SessionPhase.MORNING)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_allows_during_open_drive(self, guard):
        state = _state(session_phase=SessionPhase.OPEN_DRIVE)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_allows_during_close(self, guard):
        """Close phase is no longer blocked — we trade all sessions now."""
        state = _state(session_phase=SessionPhase.CLOSE)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_blocks_during_daily_halt(self, guard):
        state = _state(session_phase=SessionPhase.DAILY_HALT)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is False
        assert "daily_halt" in result.reason

    def test_allows_asian_session(self, guard):
        state = _state(session_phase=SessionPhase.ASIAN)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_allows_london_session(self, guard):
        state = _state(session_phase=SessionPhase.LONDON)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_allows_pre_rth(self, guard):
        state = _state(session_phase=SessionPhase.PRE_RTH)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_allows_post_rth(self, guard):
        state = _state(session_phase=SessionPhase.POST_RTH)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_allows_afternoon(self, guard):
        state = _state(session_phase=SessionPhase.AFTERNOON)
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True


# ── News Blackout ────────────────────────────────────────────────────────────


class TestNewsBlackout:
    def test_blocks_near_high_impact_event(self, guard):
        now = datetime.now(tz=UTC)
        event = EconomicEvent(
            time=now + timedelta(minutes=3),
            name="CPI",
            impact="high",
        )
        state = _state(upcoming_events=[event])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is False
        assert "blackout" in result.reason

    def test_allows_after_blackout_window(self, guard):
        now = datetime.now(tz=UTC)
        event = EconomicEvent(
            time=now + timedelta(minutes=10),  # 10 min away, blackout is 5
            name="CPI",
            impact="high",
        )
        state = _state(upcoming_events=[event])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_ignores_low_impact_events(self, guard):
        now = datetime.now(tz=UTC)
        event = EconomicEvent(
            time=now + timedelta(minutes=2),
            name="Factory Orders",
            impact="low",
        )
        state = _state(upcoming_events=[event])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_no_events(self, guard):
        state = _state(upcoming_events=[])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_blocks_post_event_via_in_blackout_flag(self, guard):
        """Post-event blackout: state.in_blackout=True blocks entries.

        The state engine sets in_blackout from EconomicCalendar.is_in_blackout()
        which includes both pre-event (5 min) and post-event (10 min) windows.
        """
        state = _state(in_blackout=True, upcoming_events=[])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is False
        assert "blackout" in result.reason

    def test_blocks_post_event_via_past_event_in_list(self, guard):
        """Post-event blackout: event that just passed (in upcoming_events list)."""
        now = datetime.now(tz=UTC)
        event = EconomicEvent(
            time=now - timedelta(minutes=3),  # 3 min ago, post-blackout is 10 min
            name="FOMC",
            impact="high",
        )
        state = _state(upcoming_events=[event])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is False
        assert "blackout" in result.reason

    def test_allows_after_post_event_window(self, guard):
        """After post-event blackout expires, entries allowed again."""
        now = datetime.now(tz=UTC)
        event = EconomicEvent(
            time=now - timedelta(minutes=15),  # 15 min ago, post-blackout is 10 min
            name="CPI",
            impact="high",
        )
        state = _state(upcoming_events=[event])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is True

    def test_post_blackout_custom_minutes(self):
        """Custom post-blackout window is respected."""
        guard = SessionRuleGuardrail(post_blackout_minutes=5)
        now = datetime.now(tz=UTC)
        event = EconomicEvent(
            time=now - timedelta(minutes=3),  # 3 min ago, post-blackout is 5 min
            name="NFP",
            impact="high",
        )
        state = _state(upcoming_events=[event])
        result = guard.check(_action(ActionType.ENTER), state)
        assert result.allowed is False

        # 7 minutes ago — beyond the 5 min post-blackout
        event2 = EconomicEvent(
            time=now - timedelta(minutes=7),
            name="NFP",
            impact="high",
        )
        state2 = _state(upcoming_events=[event2])
        result2 = guard.check(_action(ActionType.ENTER), state2)
        assert result2.allowed is True


# ── Passthrough ──────────────────────────────────────────────────────────────


class TestPassthrough:
    def test_scale_out_always_passes(self, guard):
        # Even during close, can scale out
        state = _state(session_phase=SessionPhase.CLOSE)
        result = guard.check(_action(ActionType.SCALE_OUT), state, daily_pnl=-500.0)
        assert result.allowed is True

    def test_flatten_always_passes(self, guard):
        state = _state(session_phase=SessionPhase.AFTER_HOURS)
        result = guard.check(_action(ActionType.FLATTEN), state)
        assert result.allowed is True

    def test_do_nothing_always_passes(self, guard):
        result = guard.check(_action(ActionType.DO_NOTHING), _state(), daily_pnl=-500.0)
        assert result.allowed is True
