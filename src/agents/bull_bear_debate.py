"""Bull/bear debate system for higher-quality entry decisions.

Before significant entries, we run a structured debate:
1. Bull Haiku call (parallel): strongest case for LONG
2. Bear Haiku call (parallel): strongest case for SHORT
3. Sonnet synthesis: given both cases + market state, decide

This forces the LLM to explicitly consider both sides before committing,
producing more robust decisions than a single-shot call.

Total latency: ~3-5 seconds (bull/bear parallel, then synthesis).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog

from src.core.types import ActionType, LLMAction, MarketState, Side
from src.agents.llm_client import LLMClient, LLMCallFailed, LLMCostCapExceeded, LLMResponse
from src.agents.prompts.debate_prompts import (
    BEAR_SYSTEM,
    BEAR_USER_TEMPLATE,
    BULL_SYSTEM,
    BULL_USER_TEMPLATE,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER_TEMPLATE,
)
from src.agents.prompts.system_prompt import TRADING_DECISION_TOOL

logger = structlog.get_logger()


class DebateResult:
    """Result of a bull/bear debate."""

    def __init__(
        self,
        bull_argument: str,
        bear_argument: str,
        action: LLMAction,
        total_latency_ms: int = 0,
        total_cost: float = 0.0,
    ) -> None:
        self.bull_argument = bull_argument
        self.bear_argument = bear_argument
        self.action = action
        self.total_latency_ms = total_latency_ms
        self.total_cost = total_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "bull": self.bull_argument,
            "bear": self.bear_argument,
            "action": self.action.action.value,
            "side": self.action.side.value if self.action.side else None,
            "reasoning": self.action.reasoning,
            "confidence": self.action.confidence,
            "latency_ms": self.total_latency_ms,
            "cost": self.total_cost,
        }


class BullBearDebate:
    """Runs a structured bull/bear debate for entry decisions.

    Usage:
        debate = BullBearDebate(llm_client=client)
        result = await debate.run(market_state)
        if result.action.action == ActionType.ENTER:
            # proceed with entry
    """

    def __init__(
        self,
        llm_client: LLMClient,
        debate_model: str = "haiku",
        synthesis_model: str = "sonnet",
    ) -> None:
        self._llm = llm_client
        self._debate_model = debate_model
        self._synthesis_model = synthesis_model
        self._debate_count: int = 0

    async def run(
        self,
        state: MarketState,
        detected_setups: str = "",
        price_action_narrative: str = "",
    ) -> DebateResult:
        """Run a full bull/bear debate and return the synthesized decision.

        Steps:
        1. Run bull and bear calls in parallel (Haiku)
        2. Synthesize with Sonnet using both arguments + market state + context
        3. Parse into LLMAction

        Args:
            state: Current MarketState snapshot.
            detected_setups: Pre-screened setup descriptions (from SetupDetector).
            price_action_narrative: Rich narrative text (from PriceActionAnalyzer).

        Returns:
            DebateResult with both arguments and the final action.
        """
        start = time.monotonic()
        state_json = json.dumps(state.to_llm_dict(), indent=2)

        # Enrich state JSON with pre-processed context for the debate
        enriched_json = state_json
        if detected_setups or price_action_narrative:
            enrichment_parts = [state_json]
            if detected_setups:
                enrichment_parts.append(f"\n## Detected Setups\n{detected_setups}")
            if price_action_narrative:
                enrichment_parts.append(f"\n## Price Action\n{price_action_narrative}")
            enriched_json = "\n".join(enrichment_parts)

        total_cost = 0.0

        # ── Phase 1: Parallel bull/bear calls ────────────────────────────
        bull_text, bear_text, phase1_cost = await self._run_arguments(enriched_json)
        total_cost += phase1_cost

        # ── Phase 2: Synthesis ───────────────────────────────────────────
        action, synthesis_cost = await self._run_synthesis(
            bull_text, bear_text, enriched_json
        )
        total_cost += synthesis_cost

        total_latency = int((time.monotonic() - start) * 1000)
        self._debate_count += 1

        logger.info(
            "debate.completed",
            action=action.action.value,
            side=action.side.value if action.side else None,
            confidence=action.confidence,
            latency_ms=total_latency,
            cost=round(total_cost, 4),
        )

        return DebateResult(
            bull_argument=bull_text,
            bear_argument=bear_text,
            action=action,
            total_latency_ms=total_latency,
            total_cost=total_cost,
        )

    async def _run_arguments(
        self, state_json: str
    ) -> tuple[str, str, float]:
        """Run bull and bear argument calls in parallel.

        Returns (bull_text, bear_text, total_cost).
        """
        bull_msg = BULL_USER_TEMPLATE.format(market_state_json=state_json)
        bear_msg = BEAR_USER_TEMPLATE.format(market_state_json=state_json)

        try:
            bull_task = self._llm.call(
                system=BULL_SYSTEM,
                messages=[{"role": "user", "content": bull_msg}],
                model=self._debate_model,
                max_tokens=300,
                temperature=0.5,
            )
            bear_task = self._llm.call(
                system=BEAR_SYSTEM,
                messages=[{"role": "user", "content": bear_msg}],
                model=self._debate_model,
                max_tokens=300,
                temperature=0.5,
            )

            bull_resp, bear_resp = await asyncio.gather(
                bull_task, bear_task, return_exceptions=True
            )

            bull_text = (
                bull_resp.text
                if isinstance(bull_resp, LLMResponse)
                else "Bull analysis unavailable due to error."
            )
            bear_text = (
                bear_resp.text
                if isinstance(bear_resp, LLMResponse)
                else "Bear analysis unavailable due to error."
            )

            cost = 0.0
            if isinstance(bull_resp, LLMResponse):
                cost += bull_resp.cost
            if isinstance(bear_resp, LLMResponse):
                cost += bear_resp.cost

            return bull_text, bear_text, cost

        except Exception as e:
            logger.warning("debate.argument_phase_error", error=str(e))
            return (
                "Bull analysis unavailable.",
                "Bear analysis unavailable.",
                0.0,
            )

    async def _run_synthesis(
        self,
        bull_text: str,
        bear_text: str,
        state_json: str,
    ) -> tuple[LLMAction, float]:
        """Run the synthesis call (Sonnet) to produce a final decision.

        Returns (action, cost).
        """
        user_msg = SYNTHESIS_USER_TEMPLATE.format(
            bull_argument=bull_text,
            bear_argument=bear_text,
            market_state_json=state_json,
        )

        try:
            response = await self._llm.call(
                system=SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                model=self._synthesis_model,
                tools=[TRADING_DECISION_TOOL],
                tool_choice={"type": "tool", "name": "trading_decision"},
                max_tokens=512,
                temperature=0.3,
            )

            action = self._parse_synthesis(response)
            return action, response.cost

        except (LLMCallFailed, LLMCostCapExceeded) as e:
            logger.error("debate.synthesis_failed", error=str(e))
            return (
                LLMAction(
                    action=ActionType.DO_NOTHING,
                    reasoning=f"Debate synthesis failed: {e}",
                    confidence=0.0,
                    model_used=self._synthesis_model,
                ),
                0.0,
            )

    def _parse_synthesis(self, response: LLMResponse) -> LLMAction:
        """Parse the synthesis response into an LLMAction."""
        tool_input = response.get_tool_input("trading_decision")

        if tool_input is None:
            logger.warning("debate.synthesis_no_tool_output")
            return LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning="Debate synthesis produced no structured output.",
                confidence=0.0,
                model_used=self._synthesis_model,
                latency_ms=response.latency_ms,
            )

        try:
            action_type = ActionType(tool_input.get("action", "DO_NOTHING"))

            side = None
            if "side" in tool_input and tool_input["side"]:
                side = Side(tool_input["side"])

            return LLMAction(
                action=action_type,
                side=side,
                quantity=tool_input.get("quantity"),
                stop_distance=tool_input.get("stop_distance"),
                new_stop_price=tool_input.get("new_stop_price"),
                reasoning=tool_input.get("reasoning", ""),
                confidence=max(0.0, min(1.0, float(tool_input.get("confidence", 0.5)))),
                model_used=self._synthesis_model,
                latency_ms=response.latency_ms,
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.warning("debate.synthesis_parse_error", error=str(e))
            return LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning=f"Debate parse error: {e}",
                confidence=0.0,
                model_used=self._synthesis_model,
                latency_ms=response.latency_ms,
            )

    @property
    def debate_count(self) -> int:
        return self._debate_count

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "debates_run": self._debate_count,
            "debate_model": self._debate_model,
            "synthesis_model": self._synthesis_model,
        }
