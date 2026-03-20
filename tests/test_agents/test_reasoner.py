"""Tests for the Reasoner — core reasoning loop."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import (
    ActionType,
    LLMAction,
    MarketState,
    PositionState,
    Regime,
    SessionPhase,
    Side,
)
from src.agents.llm_client import LLMCallFailed, LLMCostCapExceeded, LLMResponse
from src.agents.reasoner import Reasoner


def _make_state(**overrides) -> MarketState:
    """Create a MarketState for testing."""
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


def _make_llm_response(
    tool_input: dict | None = None,
    text: str = "",
    latency_ms: int = 500,
) -> LLMResponse:
    """Create a mock LLMResponse."""
    tool_calls = []
    if tool_input is not None:
        tool_calls = [
            {"id": "1", "name": "trading_decision", "input": tool_input}
        ]
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
        latency_ms=latency_ms,
        model="claude-sonnet-test",
    )


# ── Test: Model Selection ────────────────────────────────────────────────────


class TestModelSelection:
    def test_haiku_when_flat_midday(self) -> None:
        """Always returns sonnet — LLM is only called for pre-qualified setups."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(session_phase=SessionPhase.MIDDAY)
        assert reasoner._select_model(state) == "sonnet"

    def test_sonnet_when_flat_morning(self) -> None:
        """Uses Sonnet when flat during morning (best trading window)."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(session_phase=SessionPhase.MORNING)
        assert reasoner._select_model(state) == "sonnet"

    def test_sonnet_when_in_position(self) -> None:
        """Uses Sonnet when holding a position."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(
            position=PositionState(
                side=Side.LONG, quantity=2, avg_entry=19840.0
            )
        )
        assert reasoner._select_model(state) == "sonnet"

    def test_sonnet_during_open_drive(self) -> None:
        """Uses Sonnet during the open drive."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(session_phase=SessionPhase.OPEN_DRIVE)
        assert reasoner._select_model(state) == "sonnet"

    def test_sonnet_during_close(self) -> None:
        """Uses Sonnet during the close."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(session_phase=SessionPhase.CLOSE)
        assert reasoner._select_model(state) == "sonnet"

    def test_sonnet_in_blackout(self) -> None:
        """Uses Sonnet during blackout (careful decision needed)."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(in_blackout=True)
        assert reasoner._select_model(state) == "sonnet"


# ── Test: Response Parsing ───────────────────────────────────────────────────


class TestResponseParsing:
    def test_parse_do_nothing(self) -> None:
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        response = _make_llm_response(
            tool_input={
                "action": "DO_NOTHING",
                "reasoning": "No setup.",
                "confidence": 0.3,
            }
        )
        action = reasoner._parse_action(response, "haiku")
        assert action.action == ActionType.DO_NOTHING
        assert action.reasoning == "No setup."
        assert action.confidence == 0.3

    def test_parse_enter_long(self) -> None:
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        response = _make_llm_response(
            tool_input={
                "action": "ENTER",
                "side": "long",
                "quantity": 3,
                "stop_distance": 15.0,
                "reasoning": "Strong breakout.",
                "confidence": 0.8,
            }
        )
        action = reasoner._parse_action(response, "sonnet")
        assert action.action == ActionType.ENTER
        assert action.side == Side.LONG
        # quantity and stop_distance are not parsed from tool_input in current schema
        assert action.quantity is None
        assert action.stop_distance is None
        assert action.confidence == 0.8

    def test_parse_move_stop(self) -> None:
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        response = _make_llm_response(
            tool_input={
                "action": "MOVE_STOP",
                "new_stop_price": 19860.0,
                "reasoning": "Trailing.",
                "confidence": 0.7,
            }
        )
        action = reasoner._parse_action(response, "sonnet")
        assert action.action == ActionType.MOVE_STOP
        # new_stop_price is not parsed from tool_input in current schema
        assert action.new_stop_price is None

    def test_parse_no_tool_output(self) -> None:
        """Missing tool output returns DO_NOTHING."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        response = _make_llm_response(tool_input=None, text="No tool used")
        action = reasoner._parse_action(response, "sonnet")
        assert action.action == ActionType.DO_NOTHING
        assert reasoner.parse_error_count == 1

    def test_parse_invalid_action(self) -> None:
        """Invalid action string returns DO_NOTHING."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        response = _make_llm_response(
            tool_input={
                "action": "INVALID_ACTION",
                "reasoning": "test",
                "confidence": 0.5,
            }
        )
        action = reasoner._parse_action(response, "sonnet")
        assert action.action == ActionType.DO_NOTHING
        assert reasoner.parse_error_count == 1

    def test_confidence_clamped(self) -> None:
        """Confidence is clamped to [0, 1]."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        response = _make_llm_response(
            tool_input={
                "action": "DO_NOTHING",
                "reasoning": "test",
                "confidence": 1.5,
            }
        )
        action = reasoner._parse_action(response, "sonnet")
        assert action.confidence == 1.0


# ── Test: Full Decide Flow ───────────────────────────────────────────────────


class TestDecide:
    @pytest.mark.asyncio
    async def test_successful_decision(self) -> None:
        """Full decide flow returns an LLMAction."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "Waiting for setup.",
                    "confidence": 0.4,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm)
        state = _make_state()

        action = await reasoner.decide(state)
        assert action.action == ActionType.DO_NOTHING
        assert reasoner.decision_count == 1
        assert reasoner.do_nothing_count == 1

    @pytest.mark.asyncio
    async def test_cost_cap_returns_stop_trading(self) -> None:
        """Cost cap exceeded returns STOP_TRADING."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=LLMCostCapExceeded("cap hit"))

        reasoner = Reasoner(llm_client=llm)
        action = await reasoner.decide(_make_state())
        assert action.action == ActionType.STOP_TRADING

    @pytest.mark.asyncio
    async def test_call_failed_returns_do_nothing(self) -> None:
        """Call failure returns safe FLAT (via _safe_fallback)."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=LLMCallFailed("all retries failed"))

        reasoner = Reasoner(llm_client=llm)
        action = await reasoner.decide(_make_state())
        assert action.action == ActionType.FLAT
        assert reasoner.consecutive_parse_errors == 1

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_do_nothing(self) -> None:
        """Unexpected error returns safe FLAT (via _safe_fallback)."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=RuntimeError("unexpected"))

        reasoner = Reasoner(llm_client=llm)
        action = await reasoner.decide(_make_state())
        assert action.action == ActionType.FLAT

    @pytest.mark.asyncio
    async def test_game_plan_passed_to_system(self) -> None:
        """Game plan is included in system prompt."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "test",
                    "confidence": 0.5,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm)
        await reasoner.decide(
            _make_state(), game_plan="Bullish bias above VWAP"
        )

        call_args = llm.call.call_args
        system = call_args.kwargs["system"]
        # System should be a list of content blocks
        assert isinstance(system, list)
        # Should have game plan block
        assert any("Bullish bias" in str(block) for block in system)

    @pytest.mark.asyncio
    async def test_consecutive_errors_tracked(self) -> None:
        """Consecutive parse errors are tracked."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=LLMCallFailed("fail"))

        reasoner = Reasoner(llm_client=llm)
        await reasoner.decide(_make_state())
        assert reasoner.consecutive_parse_errors == 1

        await reasoner.decide(_make_state())
        assert reasoner.consecutive_parse_errors == 2

    @pytest.mark.asyncio
    async def test_consecutive_errors_reset_on_success(self) -> None:
        """Consecutive errors reset after successful parse."""
        llm = MagicMock()

        # First call fails
        llm.call = AsyncMock(side_effect=LLMCallFailed("fail"))
        reasoner = Reasoner(llm_client=llm)
        await reasoner.decide(_make_state())
        assert reasoner.consecutive_parse_errors == 1

        # Second call succeeds
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "ok",
                    "confidence": 0.5,
                }
            )
        )
        await reasoner.decide(_make_state())
        assert reasoner.consecutive_parse_errors == 0


# ── Test: Stats ──────────────────────────────────────────────────────────────


class TestReasonerStats:
    def test_initial_stats(self) -> None:
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        stats = reasoner.stats
        assert stats["decisions"] == 0
        assert stats["do_nothing"] == 0
        assert stats["parse_errors"] == 0
        assert stats["decision_memory_size"] == 0
        assert stats["level_memory_size"] == 0
        assert stats["has_postmortem_lessons"] is False


# ── Test: Adaptive Temperature ──────────────────────────────────────────────


class TestAdaptiveTemperature:
    """Verify that temperature adapts to decision context."""

    def test_routine_scan_uses_zero_temp(self) -> None:
        """No position, no setups → temperature 0.0 (deterministic DO_NOTHING)."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(session_phase=SessionPhase.MIDDAY)
        temp = reasoner._select_temperature(state, detected_setups="")
        assert temp == 0.0

    def test_in_position_uses_active_temp(self) -> None:
        """In position → temperature 0.0 (validation engine, always deterministic)."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(
            position=PositionState(side=Side.LONG, quantity=2, avg_entry=19840.0)
        )
        temp = reasoner._select_temperature(state, detected_setups="")
        assert temp == Reasoner.TEMP

    def test_setups_detected_uses_entry_temp(self) -> None:
        """Setups detected → temperature 0.0 (validation engine, always deterministic)."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state()
        temp = reasoner._select_temperature(
            state, detected_setups="VWAP pullback detected"
        )
        assert temp == Reasoner.TEMP

    def test_near_level_uses_entry_temp(self) -> None:
        """Near a key level → temperature 0.0 (validation engine, always deterministic)."""
        from src.core.types import KeyLevels

        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        state = _make_state(
            last_price=19870.0,
            levels=KeyLevels(prior_day_high=19872.0),
        )
        temp = reasoner._select_temperature(state, detected_setups="")
        assert temp == Reasoner.TEMP

    @pytest.mark.asyncio
    async def test_temperature_actually_passed_to_llm(self) -> None:
        """Verify adaptive temperature is passed to the LLM call."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "test",
                    "confidence": 0.3,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm)
        # Midday, flat, no setups → should be 0.0
        state = _make_state(session_phase=SessionPhase.MIDDAY)
        await reasoner.decide(state)

        call_args = llm.call.call_args
        assert call_args.kwargs["temperature"] == 0.0


# ── Test: Decision Memory ───────────────────────────────────────────────────


class TestDecisionMemory:
    """Verify that recent decisions are remembered and included in context."""

    @pytest.mark.asyncio
    async def test_decisions_recorded_in_memory(self) -> None:
        """Each decide() call records the decision in memory."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "No setup at VWAP.",
                    "confidence": 0.4,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm)

        await reasoner.decide(_make_state(last_price=19850.0))
        assert len(reasoner.recent_decisions) == 1
        assert reasoner.recent_decisions[0].action == "DO_NOTHING"
        assert reasoner.recent_decisions[0].price == 19850.0

    @pytest.mark.asyncio
    async def test_memory_bounded_by_size(self) -> None:
        """Decision memory is bounded (default 5)."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "test",
                    "confidence": 0.3,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm, decision_memory_size=3)

        for _ in range(5):
            await reasoner.decide(_make_state())

        assert len(reasoner.recent_decisions) == 3  # bounded

    @pytest.mark.asyncio
    async def test_memory_included_in_llm_context(self) -> None:
        """Decision memory appears in the extra context sent to LLM."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "No setup.",
                    "confidence": 0.4,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm)

        # First call creates a memory entry
        await reasoner.decide(_make_state())

        # Second call should include memory in context
        await reasoner.decide(_make_state())

        # Check that the second LLM call included "Recent Decisions" in system blocks
        call_args = llm.call.call_args
        system_blocks = call_args.kwargs["system"]
        system_text = " ".join(str(b) for b in system_blocks)
        assert "Recent Decisions" in system_text

    def test_clear_session_resets_memory(self) -> None:
        """clear_session() clears decision memory and counters."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)
        reasoner._decision_count = 10
        reasoner._do_nothing_count = 8
        reasoner._recent_decisions.append(
            __import__("src.agents.reasoner", fromlist=["DecisionMemory"]).DecisionMemory(
                timestamp=0, action="DO_NOTHING", confidence=0.5,
                reasoning="test", price=19850.0, model="sonnet",
            )
        )

        reasoner.clear_session()
        assert reasoner.decision_count == 0
        assert reasoner.do_nothing_count == 0
        assert len(reasoner.recent_decisions) == 0


# ── Test: Level Memory ──────────────────────────────────────────────────────


class TestLevelMemory:
    """Verify that level interactions are recorded and surfaced."""

    def test_record_level_interaction(self) -> None:
        """Can record a level interaction."""
        from src.agents.reasoner import LevelInteraction

        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        interaction = LevelInteraction(
            level_name="PDH",
            level_price=19870.0,
            visit_price=19868.5,
            timestamp=1000.0,
            action_taken="ENTER",
            outcome="entered_long",
            delta_at_visit=350.0,
        )
        reasoner.record_level_interaction(interaction)

        assert len(reasoner.level_interactions) == 1
        assert reasoner.level_interactions[0].level_name == "PDH"

    def test_level_memory_bounded(self) -> None:
        """Level memory is bounded (default 20)."""
        from src.agents.reasoner import LevelInteraction

        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm, level_memory_size=3)

        for i in range(5):
            reasoner.record_level_interaction(
                LevelInteraction(
                    level_name=f"L{i}",
                    level_price=19800.0 + i,
                    visit_price=19800.0 + i,
                    timestamp=float(i),
                    action_taken="ENTER",
                    outcome="bounced_up",
                )
            )

        assert len(reasoner.level_interactions) == 3

    def test_nearby_levels_included_in_context(self) -> None:
        """Level interactions near current price appear in enriched context."""
        from src.agents.reasoner import LevelInteraction
        import time as _time

        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        # Record a nearby interaction
        reasoner.record_level_interaction(
            LevelInteraction(
                level_name="PDH",
                level_price=19870.0,
                visit_price=19869.0,
                timestamp=_time.time() - 60,  # 1 minute ago
                action_taken="ENTER",
                outcome="bounced_down",
                delta_at_visit=-200.0,
            )
        )

        # Record a far-away interaction (should NOT appear)
        reasoner.record_level_interaction(
            LevelInteraction(
                level_name="PDL",
                level_price=19700.0,
                visit_price=19701.0,
                timestamp=_time.time() - 60,
                action_taken="FLATTEN",
                outcome="broke_through",
            )
        )

        # Build context with price near PDH
        state = _make_state(last_price=19865.0)
        context = reasoner._build_enriched_context(state, "")

        assert "PDH" in context
        assert "bounced_down" in context
        assert "PDL" not in context  # too far away

    def test_clear_level_memory(self) -> None:
        """clear_level_memory() clears all level interactions."""
        from src.agents.reasoner import LevelInteraction

        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        reasoner.record_level_interaction(
            LevelInteraction(
                level_name="VWAP", level_price=19850.0, visit_price=19850.0,
                timestamp=0.0, action_taken="ENTER", outcome="bounced_up",
            )
        )
        assert len(reasoner.level_interactions) == 1

        reasoner.clear_level_memory()
        assert len(reasoner.level_interactions) == 0


# ── Test: Postmortem Lessons ────────────────────────────────────────────────


class TestPostmortemLessons:
    """Verify postmortem lessons are stored and injected."""

    def test_set_postmortem_lessons(self) -> None:
        """Lessons can be set and appear in stats."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        reasoner.set_postmortem_lessons("Focus: VWAP pullbacks in morning.\nAvoid: midday chop.")
        assert reasoner.stats["has_postmortem_lessons"] is True

    def test_postmortem_included_in_context(self) -> None:
        """Postmortem lessons appear in enriched context."""
        llm = MagicMock()
        reasoner = Reasoner(llm_client=llm)

        reasoner.set_postmortem_lessons("Key lesson: Cut losers faster.\nFocus: Only A+ setups.")
        state = _make_state()
        context = reasoner._build_enriched_context(state, "")

        assert "Cut losers faster" in context
        assert "Lessons from Recent Sessions" in context

    @pytest.mark.asyncio
    async def test_postmortem_passed_to_llm(self) -> None:
        """Postmortem lessons appear in actual LLM call."""
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_llm_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "test",
                    "confidence": 0.3,
                }
            )
        )
        reasoner = Reasoner(llm_client=llm)
        reasoner.set_postmortem_lessons("Lesson: Avoid entries after 3pm")

        await reasoner.decide(_make_state())

        call_args = llm.call.call_args
        system_blocks = call_args.kwargs["system"]
        system_text = " ".join(str(b) for b in system_blocks)
        assert "Avoid entries after 3pm" in system_text
