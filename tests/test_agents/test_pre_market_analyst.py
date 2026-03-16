"""Tests for the PreMarketAnalyst."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from src.core.types import EconomicEvent
from src.agents.llm_client import LLMCallFailed, LLMResponse
from src.agents.pre_market_analyst import PreMarketAnalyst

ET = ZoneInfo("US/Eastern")


def _make_response(text: str = "Game plan text here", cost: float = 0.01) -> LLMResponse:
    return LLMResponse(
        text=text,
        input_tokens=200,
        output_tokens=150,
        cost=cost,
        latency_ms=2000,
        model="claude-sonnet-test",
    )


# ── Test: Successful Analysis ────────────────────────────────────────────────


class TestSuccessfulAnalysis:
    @pytest.mark.asyncio
    async def test_generates_game_plan(self) -> None:
        llm = MagicMock()
        llm.call = AsyncMock(
            return_value=_make_response(
                text="Bullish bias. Look for longs above 19830 VWAP."
            )
        )

        analyst = PreMarketAnalyst(llm_client=llm)
        plan = await analyst.analyze(
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            prior_day_close=19800.0,
            overnight_high=19860.0,
            overnight_low=19740.0,
            current_price=19820.0,
        )

        assert "Bullish" in plan
        assert analyst.last_game_plan == plan
        assert analyst.stats["has_game_plan"] is True

    @pytest.mark.asyncio
    async def test_with_events(self) -> None:
        llm = MagicMock()
        llm.call = AsyncMock(return_value=_make_response(text="CPI day — cautious."))

        events = [
            EconomicEvent(
                time=datetime(2025, 3, 14, 8, 30, tzinfo=ET),
                name="CPI m/m",
                impact="high",
                forecast="0.3%",
                prior="0.2%",
            )
        ]

        analyst = PreMarketAnalyst(llm_client=llm)
        plan = await analyst.analyze(
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            prior_day_close=19800.0,
            overnight_high=19860.0,
            overnight_low=19740.0,
            current_price=19820.0,
            events=events,
        )

        # Check that events were formatted in the call
        call_args = llm.call.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "CPI" in user_msg

    @pytest.mark.asyncio
    async def test_with_prior_day_summary(self) -> None:
        llm = MagicMock()
        llm.call = AsyncMock(return_value=_make_response(text="Continuation likely."))

        analyst = PreMarketAnalyst(llm_client=llm)
        await analyst.analyze(
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            prior_day_close=19800.0,
            overnight_high=19860.0,
            overnight_low=19740.0,
            current_price=19820.0,
            prior_day_summary="3W/1L, +$180, trending_up",
        )

        call_args = llm.call.call_args
        user_msg = call_args.kwargs["messages"][0]["content"]
        assert "trending_up" in user_msg


# ── Test: Gap Description ────────────────────────────────────────────────────


class TestGapDescription:
    def test_gap_up(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        desc = analyst._describe_gap(19800.0, 19850.0)
        assert "Gap UP" in desc

    def test_gap_down(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        desc = analyst._describe_gap(19800.0, 19750.0)
        assert "Gap DOWN" in desc

    def test_flat_open(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        desc = analyst._describe_gap(19800.0, 19802.0)
        assert "Flat" in desc

    def test_zero_price(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        desc = analyst._describe_gap(0.0, 19800.0)
        assert "Unknown" in desc


# ── Test: Event Formatting ───────────────────────────────────────────────────


class TestEventFormatting:
    def test_no_events(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        text = analyst._format_events([])
        assert "No high-impact" in text

    def test_with_events(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        events = [
            EconomicEvent(
                time=datetime(2025, 3, 14, 8, 30, tzinfo=ET),
                name="CPI m/m",
                impact="high",
                forecast="0.3%",
                prior="0.2%",
            )
        ]
        text = analyst._format_events(events)
        assert "CPI" in text
        assert "HIGH" in text
        assert "forecast" in text

    def test_event_without_forecast(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        events = [
            EconomicEvent(
                time=datetime(2025, 3, 14, 10, 0, tzinfo=ET),
                name="FOMC Decision",
                impact="high",
            )
        ]
        text = analyst._format_events(events)
        assert "FOMC" in text
        assert "forecast" not in text


# ── Test: Fallback Plan ──────────────────────────────────────────────────────


class TestFallbackPlan:
    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self) -> None:
        """Generates fallback plan when LLM fails."""
        llm = MagicMock()
        llm.call = AsyncMock(side_effect=LLMCallFailed("all retries failed"))

        analyst = PreMarketAnalyst(llm_client=llm)
        plan = await analyst.analyze(
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            prior_day_close=19800.0,
            overnight_high=19860.0,
            overnight_low=19740.0,
            current_price=19820.0,
        )

        assert "auto-generated" in plan
        assert "19900.00" in plan
        assert "Neutral" in plan

    def test_fallback_content(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        plan = analyst._fallback_plan(
            19900.0, 19700.0, 19800.0, 19860.0, 19740.0, 19820.0
        )
        assert "19900.00" in plan
        assert "19700.00" in plan
        assert "19860.00" in plan


# ── Test: Stats ──────────────────────────────────────────────────────────────


class TestAnalystStats:
    def test_initial_stats(self) -> None:
        analyst = PreMarketAnalyst(llm_client=MagicMock())
        stats = analyst.stats
        assert stats["has_game_plan"] is False
        assert stats["plan_length"] == 0
