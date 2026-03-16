"""Pre-market analyst — generates the daily game plan before market open.

Runs once at ~9:25 AM ET with overnight data, gap analysis, calendar,
and prior session summary. Produces a game plan that's included in the
system prompt context for the rest of the day.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from src.core.types import EconomicEvent
from src.agents.llm_client import LLMClient, LLMCallFailed, LLMCostCapExceeded
from src.agents.prompts.debate_prompts import (
    PRE_MARKET_SYSTEM,
    PRE_MARKET_USER_TEMPLATE,
)

logger = structlog.get_logger()


class PreMarketAnalyst:
    """Generates the daily game plan from overnight and prior session data.

    Usage:
        analyst = PreMarketAnalyst(llm_client=client)
        game_plan = await analyst.analyze(
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            prior_day_close=19800.0,
            overnight_high=19860.0,
            overnight_low=19740.0,
            current_price=19820.0,
            events=calendar.high_impact_today(),
            prior_day_summary="3W/1L, +$180, trending_up regime",
        )
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "sonnet",
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._last_game_plan: str = ""
        self._last_analysis_time: float = 0.0

    async def analyze(
        self,
        prior_day_high: float,
        prior_day_low: float,
        prior_day_close: float,
        overnight_high: float,
        overnight_low: float,
        current_price: float,
        events: list[EconomicEvent] | None = None,
        prior_day_summary: str = "",
    ) -> str:
        """Generate the daily game plan.

        Returns:
            Game plan text string (200-300 words).
        """
        start = time.monotonic()

        # Compute gap
        gap_description = self._describe_gap(prior_day_close, current_price)

        # Format calendar events
        calendar_text = self._format_events(events or [])

        # Build prompt
        user_msg = PRE_MARKET_USER_TEMPLATE.format(
            prior_day_high=f"{prior_day_high:.2f}",
            prior_day_low=f"{prior_day_low:.2f}",
            prior_day_close=f"{prior_day_close:.2f}",
            overnight_high=f"{overnight_high:.2f}",
            overnight_low=f"{overnight_low:.2f}",
            current_price=f"{current_price:.2f}",
            gap_description=gap_description,
            calendar_events=calendar_text,
            prior_day_summary=prior_day_summary or "No prior session data available.",
        )

        try:
            response = await self._llm.call(
                system=PRE_MARKET_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                model=self._model,
                max_tokens=600,
                temperature=0.4,
            )

            game_plan = response.text.strip()
            self._last_game_plan = game_plan
            self._last_analysis_time = time.monotonic() - start

            logger.info(
                "pre_market.analysis_complete",
                plan_length=len(game_plan),
                latency_ms=response.latency_ms,
                cost=response.cost,
            )

            return game_plan

        except (LLMCallFailed, LLMCostCapExceeded) as e:
            logger.error("pre_market.analysis_failed", error=str(e))
            return self._fallback_plan(
                prior_day_high, prior_day_low, prior_day_close,
                overnight_high, overnight_low, current_price,
            )

    def _describe_gap(self, prior_close: float, current: float) -> str:
        """Describe the gap from prior close to current price."""
        if prior_close == 0.0 or current == 0.0:
            return "Unknown (no data)"

        gap_points = current - prior_close
        gap_pct = (gap_points / prior_close) * 100

        if abs(gap_points) < 5:
            return f"Flat open ({gap_points:+.1f} pts, {gap_pct:+.2f}%)"
        elif gap_points > 0:
            return f"Gap UP {gap_points:+.1f} pts ({gap_pct:+.2f}%)"
        else:
            return f"Gap DOWN {gap_points:+.1f} pts ({gap_pct:+.2f}%)"

    def _format_events(self, events: list[EconomicEvent]) -> str:
        """Format economic events for the prompt."""
        if not events:
            return "No high-impact events scheduled today."

        lines = []
        for e in events:
            time_str = e.time.strftime("%I:%M %p ET")
            impact = e.impact.upper()
            line = f"- [{impact}] {time_str}: {e.name}"
            if e.forecast:
                line += f" (forecast: {e.forecast}"
                if e.prior:
                    line += f", prior: {e.prior}"
                line += ")"
            lines.append(line)

        return "\n".join(lines)

    def _fallback_plan(
        self,
        pdh: float,
        pdl: float,
        pdc: float,
        onh: float,
        onl: float,
        current: float,
    ) -> str:
        """Generate a minimal fallback game plan when LLM fails."""
        return (
            f"Game plan (auto-generated — LLM unavailable):\n"
            f"- Prior range: {pdl:.2f} - {pdh:.2f}, close {pdc:.2f}\n"
            f"- Overnight range: {onl:.2f} - {onh:.2f}\n"
            f"- Current: {current:.2f}\n"
            f"- Bias: Neutral (proceed with caution)\n"
            f"- Approach: React to levels, smaller size, tighter stops"
        )

    @property
    def last_game_plan(self) -> str:
        return self._last_game_plan

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "has_game_plan": bool(self._last_game_plan),
            "plan_length": len(self._last_game_plan),
            "analysis_time": round(self._last_analysis_time, 2),
        }
