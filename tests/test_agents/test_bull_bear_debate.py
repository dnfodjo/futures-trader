"""Tests for the BullBearDebate system."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.types import ActionType, MarketState, SessionPhase, Side
from src.agents.bull_bear_debate import BullBearDebate, DebateResult
from src.agents.llm_client import LLMCallFailed, LLMCostCapExceeded, LLMResponse


def _make_state() -> MarketState:
    return MarketState(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        last_price=19850.0,
        session_phase=SessionPhase.MORNING,
    )


def _make_response(
    text: str = "analysis",
    tool_input: dict | None = None,
    cost: float = 0.001,
    latency_ms: int = 300,
) -> LLMResponse:
    tool_calls = []
    if tool_input:
        tool_calls = [
            {"id": "1", "name": "trading_decision", "input": tool_input}
        ]
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        input_tokens=100,
        output_tokens=50,
        cost=cost,
        latency_ms=latency_ms,
        model="claude-sonnet-test",
    )


# ── Test: Successful Debate ──────────────────────────────────────────────────


class TestSuccessfulDebate:
    @pytest.mark.asyncio
    async def test_full_debate_do_nothing(self) -> None:
        """Debate producing DO_NOTHING."""
        llm = MagicMock()

        # Phase 1: bull/bear arguments
        bull_resp = _make_response(text="Price above VWAP, buyers in control")
        bear_resp = _make_response(text="Approaching resistance, VIX rising")

        # Phase 2: synthesis
        synthesis_resp = _make_response(
            tool_input={
                "action": "DO_NOTHING",
                "reasoning": "Both cases are moderate. Waiting.",
                "confidence": 0.4,
            }
        )

        call_count = 0

        async def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            # First 2 calls are bull/bear, 3rd is synthesis
            if call_count <= 2:
                if "bullish" in kwargs.get("system", "").lower():
                    return bull_resp
                return bear_resp
            return synthesis_resp

        llm.call = AsyncMock(side_effect=mock_call)

        debate = BullBearDebate(llm_client=llm)
        result = await debate.run(_make_state())

        assert isinstance(result, DebateResult)
        assert result.action.action == ActionType.DO_NOTHING
        assert result.bull_argument != ""
        assert result.bear_argument != ""
        assert result.total_latency_ms >= 0
        assert debate.debate_count == 1

    @pytest.mark.asyncio
    async def test_full_debate_enter_long(self) -> None:
        """Debate producing ENTER long."""
        llm = MagicMock()

        call_count = 0

        async def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_response(text="Strong case for direction")
            return _make_response(
                tool_input={
                    "action": "ENTER",
                    "side": "long",
                    "quantity": 3,
                    "stop_distance": 12.0,
                    "reasoning": "Bulls have stronger case.",
                    "confidence": 0.75,
                }
            )

        llm.call = AsyncMock(side_effect=mock_call)
        debate = BullBearDebate(llm_client=llm)
        result = await debate.run(_make_state())

        assert result.action.action == ActionType.ENTER
        assert result.action.side == Side.LONG
        assert result.action.quantity == 3


# ── Test: Error Handling ─────────────────────────────────────────────────────


class TestDebateErrors:
    @pytest.mark.asyncio
    async def test_argument_failure_still_synthesizes(self) -> None:
        """If bull/bear calls fail, synthesis still runs with error text."""
        llm = MagicMock()

        call_count = 0

        async def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("API error")
            return _make_response(
                tool_input={
                    "action": "DO_NOTHING",
                    "reasoning": "Insufficient data.",
                    "confidence": 0.2,
                }
            )

        llm.call = AsyncMock(side_effect=mock_call)
        debate = BullBearDebate(llm_client=llm)
        result = await debate.run(_make_state())

        assert result.action.action == ActionType.DO_NOTHING
        assert "unavailable" in result.bull_argument.lower()

    @pytest.mark.asyncio
    async def test_synthesis_failure_returns_do_nothing(self) -> None:
        """If synthesis fails, returns safe DO_NOTHING."""
        llm = MagicMock()

        call_count = 0

        async def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_response(text="test argument")
            raise LLMCallFailed("synthesis failed")

        llm.call = AsyncMock(side_effect=mock_call)
        debate = BullBearDebate(llm_client=llm)
        result = await debate.run(_make_state())

        assert result.action.action == ActionType.DO_NOTHING

    @pytest.mark.asyncio
    async def test_synthesis_no_tool_output(self) -> None:
        """Synthesis with no tool output returns DO_NOTHING."""
        llm = MagicMock()

        call_count = 0

        async def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_response(text="argument")
            return _make_response(text="I think we should go long", tool_input=None)

        llm.call = AsyncMock(side_effect=mock_call)
        debate = BullBearDebate(llm_client=llm)
        result = await debate.run(_make_state())

        assert result.action.action == ActionType.DO_NOTHING

    @pytest.mark.asyncio
    async def test_cost_cap_in_synthesis(self) -> None:
        """Cost cap in synthesis returns DO_NOTHING."""
        llm = MagicMock()

        call_count = 0

        async def mock_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _make_response(text="argument")
            raise LLMCostCapExceeded("cap hit")

        llm.call = AsyncMock(side_effect=mock_call)
        debate = BullBearDebate(llm_client=llm)
        result = await debate.run(_make_state())

        assert result.action.action == ActionType.DO_NOTHING


# ── Test: DebateResult ───────────────────────────────────────────────────────


class TestDebateResult:
    def test_to_dict(self) -> None:
        from src.core.types import LLMAction

        result = DebateResult(
            bull_argument="bull case",
            bear_argument="bear case",
            action=LLMAction(
                action=ActionType.ENTER,
                side=Side.LONG,
                quantity=2,
                reasoning="test",
                confidence=0.7,
            ),
            total_latency_ms=3000,
            total_cost=0.005,
        )
        d = result.to_dict()
        assert d["bull"] == "bull case"
        assert d["bear"] == "bear case"
        assert d["action"] == "ENTER"
        assert d["side"] == "long"
        assert d["confidence"] == 0.7


# ── Test: Stats ──────────────────────────────────────────────────────────────


class TestDebateStats:
    def test_initial_stats(self) -> None:
        llm = MagicMock()
        debate = BullBearDebate(llm_client=llm)
        stats = debate.stats
        assert stats["debates_run"] == 0
