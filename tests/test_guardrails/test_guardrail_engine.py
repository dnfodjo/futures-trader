"""Tests for the GuardrailEngine — full pipeline integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.core.events import EventBus
from src.core.types import (
    ActionType,
    EconomicEvent,
    LLMAction,
    MarketState,
    PositionState,
    SessionPhase,
    Side,
)
from src.guardrails.guardrail_engine import GuardrailEngine


def _action(
    action: ActionType = ActionType.ENTER,
    side: Side | None = Side.LONG,
    quantity: int | None = 1,
    stop_distance: float | None = 10.0,
    confidence: float = 0.8,
    reasoning: str = "test",
) -> LLMAction:
    return LLMAction(
        action=action,
        side=side,
        quantity=quantity,
        stop_distance=stop_distance,
        reasoning=reasoning,
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
    quantity: int = 2,
    avg_entry: float = 19850.0,
    adds_count: int = 0,
) -> PositionState:
    return PositionState(
        side=side, quantity=quantity, avg_entry=avg_entry, adds_count=adds_count,
    )


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def engine(bus):
    return GuardrailEngine(
        event_bus=bus,
        max_contracts=6,
        max_adds=3,
        min_stop_distance=3.0,
        max_stop_distance=25.0,
        min_confidence=0.3,
        min_entry_spacing_pts=5.0,
        max_consecutive_losers=4,
        daily_loss_limit=400.0,
        blackout_minutes=5,
    )


# ── Full Pipeline: Allowed ───────────────────────────────────────────────────


class TestAllowed:
    def test_valid_entry(self, engine):
        result = engine.check(
            _action(ActionType.ENTER, side=Side.LONG, quantity=2, stop_distance=10.0),
            _state(),
        )
        assert result.allowed is True

    def test_do_nothing_passes(self, engine):
        result = engine.check(_action(ActionType.DO_NOTHING), _state())
        assert result.allowed is True

    def test_flatten_always_passes(self, engine):
        result = engine.check(
            _action(ActionType.FLATTEN),
            _state(session_phase=SessionPhase.CLOSE),
            position=_position(),
            daily_pnl=-500.0,
            consecutive_losers=10,
        )
        assert result.allowed is True

    def test_move_stop_valid_long(self, engine):
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            new_stop_price=19840.0,  # below market (19850)
            reasoning="tighten stop",
        )
        result = engine.check(
            action,
            _state(last_price=19850.0),
            position=_position(side=Side.LONG),
        )
        assert result.allowed is True

    def test_move_stop_valid_short(self, engine):
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            new_stop_price=19860.0,  # above market (19850)
            reasoning="tighten stop",
        )
        result = engine.check(
            action,
            _state(last_price=19850.0),
            position=_position(side=Side.SHORT),
        )
        assert result.allowed is True

    def test_stop_trading_passes(self, engine):
        result = engine.check(_action(ActionType.STOP_TRADING), _state())
        assert result.allowed is True


# ── Full Pipeline: Blocked ───────────────────────────────────────────────────


class TestBlocked:
    def test_blocked_by_position_limit(self, engine):
        result = engine.check(
            _action(ActionType.ENTER, quantity=1),
            _state(),
            position=_position(),  # already in position
        )
        assert result.allowed is False
        assert "position_limit" in result.reason

    def test_blocked_by_session_rule(self, engine):
        result = engine.check(
            _action(ActionType.ENTER),
            _state(session_phase=SessionPhase.CLOSE),
        )
        assert result.allowed is False
        assert "session_rule" in result.reason

    def test_blocked_by_daily_loss(self, engine):
        result = engine.check(
            _action(ActionType.ENTER),
            _state(),
            daily_pnl=-450.0,
        )
        assert result.allowed is False
        assert "daily loss" in result.reason

    def test_blocked_by_consecutive_losers(self, engine):
        result = engine.check(
            _action(ActionType.ENTER),
            _state(),
            consecutive_losers=5,
        )
        assert result.allowed is False
        assert "consecutive losers" in result.reason

    def test_blocked_by_low_confidence(self, engine):
        result = engine.check(
            _action(ActionType.ENTER, confidence=0.1),
            _state(),
        )
        assert result.allowed is False
        assert "confidence" in result.reason

    def test_blocked_by_tight_stop(self, engine):
        result = engine.check(
            _action(ActionType.ENTER, stop_distance=1.0),
            _state(),
        )
        assert result.allowed is False
        assert "stop distance" in result.reason

    def test_blocked_by_wide_spread(self, engine):
        result = engine.check(
            _action(ActionType.ENTER),
            _state(bid=19845.0, ask=19855.0),  # 10pt spread
        )
        assert result.allowed is False
        assert "illiquid" in result.reason

    def test_blocked_by_news_blackout(self, engine):
        now = datetime.now(tz=UTC)
        event = EconomicEvent(time=now + timedelta(minutes=2), name="CPI", impact="high")
        result = engine.check(
            _action(ActionType.ENTER),
            _state(upcoming_events=[event]),
        )
        assert result.allowed is False
        assert "blackout" in result.reason


# ── MOVE_STOP Validation ────────────────────────────────────────────────────


class TestMoveStopValidation:
    def test_blocked_no_stop_price(self, engine):
        """MOVE_STOP without new_stop_price should be blocked."""
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            reasoning="move stop",
        )
        result = engine.check(action, _state(), position=_position())
        assert result.allowed is False
        assert "new_stop_price" in result.reason

    def test_blocked_no_position(self, engine):
        """MOVE_STOP without a position should be blocked."""
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            new_stop_price=19840.0,
            reasoning="move stop",
        )
        result = engine.check(action, _state(), position=None)
        assert result.allowed is False
        assert "without position" in result.reason

    def test_blocked_long_stop_above_market(self, engine):
        """LONG position with stop above market price should be blocked."""
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            new_stop_price=19860.0,  # above market (19850)
            reasoning="bad stop",
        )
        result = engine.check(
            action,
            _state(last_price=19850.0),
            position=_position(side=Side.LONG),
        )
        assert result.allowed is False
        assert "above/at market" in result.reason

    def test_blocked_short_stop_below_market(self, engine):
        """SHORT position with stop below market price should be blocked."""
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            new_stop_price=19840.0,  # below market (19850)
            reasoning="bad stop",
        )
        result = engine.check(
            action,
            _state(last_price=19850.0),
            position=_position(side=Side.SHORT),
        )
        assert result.allowed is False
        assert "below/at market" in result.reason

    def test_blocked_stop_at_market_price(self, engine):
        """Stop exactly at market price should be blocked (would fill immediately)."""
        action = LLMAction(
            action=ActionType.MOVE_STOP,
            new_stop_price=19850.0,
            reasoning="stop at market",
        )
        result = engine.check(
            action,
            _state(last_price=19850.0),
            position=_position(side=Side.LONG),
        )
        assert result.allowed is False


# ── Profit Preservation Override ─────────────────────────────────────────────


class TestProfitPreservation:
    def test_effective_max_overrides(self, engine):
        # Base max is 6, but profit preservation says 3
        result = engine.check(
            _action(ActionType.ENTER, quantity=5),
            _state(),
            effective_max_contracts=3,
        )
        assert result.allowed is True
        assert result.modified_quantity == 3

    def test_within_effective_max(self, engine):
        result = engine.check(
            _action(ActionType.ENTER, quantity=2),
            _state(),
            effective_max_contracts=3,
        )
        assert result.allowed is True
        assert result.modified_quantity is None


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats(self, engine):
        stats = engine.stats
        assert stats["checks_run"] == 0
        assert stats["checks_passed"] == 0
        assert stats["checks_blocked"] == 0

    def test_stats_after_checks(self, engine):
        engine.check(_action(ActionType.ENTER), _state())  # pass
        engine.check(_action(ActionType.DO_NOTHING), _state())  # pass
        engine.check(
            _action(ActionType.ENTER, confidence=0.1), _state(),
        )  # block

        stats = engine.stats
        assert stats["checks_run"] == 3
        assert stats["checks_passed"] == 2
        assert stats["checks_blocked"] == 1

    def test_block_reasons_tracked(self, engine):
        engine.check(_action(ActionType.ENTER, confidence=0.1), _state())
        engine.check(_action(ActionType.ENTER, confidence=0.05), _state())

        stats = engine.stats
        assert "risk_check" in stats["block_reasons"]
        assert stats["block_reasons"]["risk_check"] == 2

    def test_modified_count(self, engine):
        engine.check(
            _action(ActionType.ENTER, quantity=10),
            _state(),
        )  # clamped to 6

        stats = engine.stats
        assert stats["checks_modified"] == 1
