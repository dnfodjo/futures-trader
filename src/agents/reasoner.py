"""Core reasoning engine — the "brain" of the trading system.

Receives MarketState snapshots and produces LLMAction decisions.
Handles model selection (Haiku for routine, Sonnet for decisions),
response parsing, and error recovery.

This is the single decision-making pipeline:
1. Receive MarketState from StateEngine (via event bus)
2. Decide model tier (Haiku for "anything happening?" / Sonnet for entries/exits)
3. Build prompt with cached system prompt + dynamic market state
4. Call LLM with structured output (tool calling)
5. Parse response into LLMAction
6. Return action for validation and execution

Edge improvements:
- Adaptive temperature: 0.0 for routine checks, 0.2 for active decisions
- Decision memory: LLM sees its last 3 decisions for continuity
- Level memory: what happened last time price was near each key level
- Postmortem lessons: injected as extra context from nightly analysis
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from src.core.types import ActionType, LLMAction, MarketState, Side
from src.agents.llm_client import LLMClient, LLMCallFailed, LLMCostCapExceeded, LLMResponse
from src.agents.prompts.system_prompt import (
    TRADING_DECISION_TOOL,
    build_system_blocks,
)

logger = structlog.get_logger()


# ── Decision Memory ──────────────────────────────────────────────────────────


@dataclass
class DecisionMemory:
    """Record of a recent decision for context continuity."""

    timestamp: float
    action: str
    confidence: float
    reasoning: str
    price: float
    model: str


@dataclass
class LevelInteraction:
    """Record of what happened when price visited a key level."""

    level_name: str
    level_price: float
    visit_price: float
    timestamp: float
    action_taken: str
    outcome: str  # "bounced_up", "bounced_down", "broke_through", "absorbed", "no_action"
    delta_at_visit: float = 0.0


class Reasoner:
    """Core reasoning loop that converts MarketState into LLMAction.

    Enhanced with:
    - Adaptive temperature based on decision context
    - Rolling decision memory (last N decisions visible to LLM)
    - Level interaction memory (what happened at key levels)
    - Postmortem lesson injection

    Usage:
        reasoner = Reasoner(llm_client=client)
        action = await reasoner.decide(market_state)
    """

    # Always temperature 0.0 — the LLM is a validation engine, not a creative agent
    TEMP: float = 0.0

    def __init__(
        self,
        llm_client: LLMClient,
        confidence_threshold: float = 0.25,
        decision_memory_size: int = 5,
        level_memory_size: int = 20,
    ) -> None:
        self._llm = llm_client
        self._confidence_threshold = confidence_threshold
        self._decision_count: int = 0
        self._do_nothing_count: int = 0
        self._parse_error_count: int = 0
        self._consecutive_parse_errors: int = 0

        # Decision memory — LLM sees its last N decisions
        self._recent_decisions: deque[DecisionMemory] = deque(maxlen=decision_memory_size)

        # Level interaction memory — what happened at key levels
        self._level_interactions: deque[LevelInteraction] = deque(maxlen=level_memory_size)

        # Postmortem lessons — loaded from nightly analysis
        self._postmortem_lessons: str = ""

        # Last known state for tracking level visits
        self._last_price: float = 0.0

    # ── Public Methods ──────────────────────────────────────────────────────

    def set_postmortem_lessons(self, lessons: str) -> None:
        """Inject lessons from nightly postmortem analysis.

        Called by the orchestrator after loading the most recent postmortem.
        These lessons are included in every LLM call as extra context.
        """
        self._postmortem_lessons = lessons
        logger.info("reasoner.postmortem_lessons_loaded", length=len(lessons))

    def record_level_interaction(self, interaction: LevelInteraction) -> None:
        """Record what happened when price interacted with a key level.

        Called by the orchestrator after trade outcomes are known.
        """
        self._level_interactions.append(interaction)

    # ── Main Decision Method ─────────────────────────────────────────────────

    async def decide(
        self,
        state: MarketState,
        game_plan: str = "",
        extra_context: str = "",
        confluence_data: str = "",
        order_flow_data: str = "",
    ) -> LLMAction:
        """Validate a pre-qualified setup and assign confidence.

        The confluence engine has already scored the setup above the session
        minimum. The LLM synthesizes all data and gives final go/no-go.

        Args:
            state: Complete MarketState snapshot.
            game_plan: Pre-market game plan text.
            extra_context: Any additional context (decision memory, postmortem).
            confluence_data: JSON of confluence scoring breakdown.
            order_flow_data: JSON of order flow snapshot.

        Returns:
            LLMAction with the decision and reasoning.
        """
        # Always Sonnet — LLM only gets called when setup is pre-qualified
        model = "sonnet"

        # Build enriched extra context with decision memory + postmortem
        enriched_context = self._build_enriched_context(state, extra_context)

        # Build system prompt with caching
        system = build_system_blocks(
            game_plan=game_plan,
            extra_context=enriched_context,
            confluence_data=confluence_data,
            order_flow_data=order_flow_data,
        )

        # Build user message with market state
        state_json = json.dumps(state.to_llm_dict(), indent=2)
        messages = [
            {
                "role": "user",
                "content": f"Current market state:\n```json\n{state_json}\n```\n\nValidate this setup and provide your confidence score.",
            }
        ]

        try:
            response = await self._llm.call(
                system=system,
                messages=messages,
                model=model,
                tools=[TRADING_DECISION_TOOL],
                tool_choice={"type": "tool", "name": "trading_decision"},
                max_tokens=512,
                temperature=self.TEMP,
            )

            action = self._parse_action(response, model)
            self._decision_count += 1
            self._consecutive_parse_errors = 0

            if action.action in (ActionType.DO_NOTHING, ActionType.FLAT):
                self._do_nothing_count += 1

            # Record this decision in memory for future context
            self._recent_decisions.append(
                DecisionMemory(
                    timestamp=time.time(),
                    action=action.action.value,
                    confidence=action.confidence,
                    reasoning=action.reasoning[:120] if action.reasoning else "",
                    price=state.last_price,
                    model=model,
                )
            )
            self._last_price = state.last_price

            return action

        except LLMCostCapExceeded:
            logger.error("reasoner.cost_cap_exceeded")
            return LLMAction(
                action=ActionType.STOP_TRADING,
                reasoning="LLM daily cost cap exceeded. Stopping trading.",
                confidence=1.0,
                model_used=model,
            )

        except LLMCallFailed as e:
            logger.error("reasoner.call_failed", error=str(e))
            return self._safe_fallback(model, str(e))

        except Exception as e:
            logger.exception("reasoner.unexpected_error")
            return self._safe_fallback(model, str(e))

    # ── Temperature (fixed at 0.0 — validation engine) ────────────────────────

    def _select_temperature(self, state: MarketState, detected_setups: str = "") -> float:
        """Always 0.0 — the LLM is a validation engine, not a creative agent.
        """
        return self.TEMP

    # ── Context Enrichment ────────────────────────────────────────────────────

    def _build_enriched_context(self, state: MarketState, extra_context: str) -> str:
        """Build enriched context with decision memory, level memory, and postmortem lessons.

        This gives the LLM continuity between decisions and historical awareness.
        """
        parts: list[str] = []

        # Original extra context
        if extra_context:
            parts.append(extra_context)

        # Recent decision memory (last 3-5 decisions)
        if self._recent_decisions:
            memory_lines = ["### Recent Decisions (your last calls):"]
            for dm in self._recent_decisions:
                elapsed = time.time() - dm.timestamp
                if elapsed < 60:
                    time_ago = f"{elapsed:.0f}s ago"
                else:
                    time_ago = f"{elapsed / 60:.0f}m ago"
                memory_lines.append(
                    f"- [{time_ago}] {dm.action} @ {dm.price:.2f} "
                    f"(conf={dm.confidence:.0%}) — {dm.reasoning}"
                )
            parts.append("\n".join(memory_lines))

        # Level interaction memory (what happened at nearby levels)
        if self._level_interactions and state.last_price > 0:
            nearby = [
                li
                for li in self._level_interactions
                if abs(state.last_price - li.level_price) <= 20.0
            ]
            if nearby:
                level_lines = ["### Level History (what happened at nearby levels):"]
                for li in nearby[-5:]:  # Last 5 relevant interactions
                    elapsed = time.time() - li.timestamp
                    if elapsed < 3600:
                        time_ago = f"{elapsed / 60:.0f}m ago"
                    else:
                        time_ago = f"{elapsed / 3600:.1f}h ago"
                    level_lines.append(
                        f"- {li.level_name} ({li.level_price:.2f}) [{time_ago}]: "
                        f"Price was {li.visit_price:.2f}, delta={li.delta_at_visit:+.0f} → "
                        f"{li.outcome}. Action: {li.action_taken}"
                    )
                parts.append("\n".join(level_lines))

        # Postmortem lessons from previous sessions
        if self._postmortem_lessons:
            parts.append(f"### Lessons from Recent Sessions:\n{self._postmortem_lessons}")

        return "\n\n".join(parts) if parts else ""

    # ── Model Selection ──────────────────────────────────────────────────────

    def _select_model(self, state: MarketState) -> str:
        """Always Sonnet — LLM only gets called when setup is pre-qualified.

        The confluence engine gates LLM calls, so we use the best model
        every time. Cost is lower because calls are much less frequent.
        """
        return "sonnet"

    # ── Response Parsing ─────────────────────────────────────────────────────

    def _parse_action(self, response: LLMResponse, model: str) -> LLMAction:
        """Parse LLM response into an LLMAction.

        Uses tool calling for structured output. Falls back to DO_NOTHING
        if parsing fails.
        """
        tool_input = response.get_tool_input("trading_decision")

        if tool_input is None:
            self._parse_error_count += 1
            self._consecutive_parse_errors += 1
            logger.warning(
                "reasoner.no_tool_output",
                text_preview=response.text[:200] if response.text else "",
            )
            return LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning="Parse error: no structured output from LLM.",
                confidence=0.0,
                model_used=model,
                latency_ms=response.latency_ms,
            )

        try:
            action_str = tool_input.get("action", "FLAT")
            action_type = ActionType(action_str)

            # Derive side from action for new confluence-based actions
            side = None
            if action_type == ActionType.LONG:
                side = Side.LONG
            elif action_type == ActionType.SHORT:
                side = Side.SHORT
            elif "side" in tool_input and tool_input["side"]:
                side = Side(tool_input["side"])

            reasoning = tool_input.get("reasoning", "")
            confidence = float(tool_input.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            # New confluence fields
            primary_timeframe = tool_input.get("primary_timeframe")
            confluence_factors = tool_input.get("confluence_factors", [])
            order_flow_assessment = tool_input.get("order_flow_assessment", "")
            risk_flags = tool_input.get("risk_flags", [])

            return LLMAction(
                action=action_type,
                side=side,
                reasoning=reasoning,
                confidence=confidence,
                model_used=model,
                latency_ms=response.latency_ms,
                primary_timeframe=primary_timeframe,
                confluence_factors=confluence_factors,
                order_flow_assessment=order_flow_assessment,
                risk_flags=risk_flags,
            )

        except (ValueError, KeyError, TypeError) as e:
            self._parse_error_count += 1
            self._consecutive_parse_errors += 1
            logger.warning(
                "reasoner.parse_error",
                error=str(e),
                tool_input=tool_input,
            )
            return LLMAction(
                action=ActionType.DO_NOTHING,
                reasoning=f"Parse error: {e}",
                confidence=0.0,
                model_used=model,
                latency_ms=response.latency_ms,
            )

    def _safe_fallback(self, model: str, error: str) -> LLMAction:
        """Return a safe FLAT action on unrecoverable errors."""
        self._parse_error_count += 1
        self._consecutive_parse_errors += 1
        return LLMAction(
            action=ActionType.FLAT,
            reasoning=f"LLM error — defaulting to FLAT: {error}",
            confidence=0.0,
            model_used=model,
        )

    # ── Session Management ──────────────────────────────────────────────────

    def clear_session(self) -> None:
        """Clear session-specific state (call at start of each trading day).

        Preserves postmortem lessons and level memory across sessions,
        but clears decision memory and counters.
        """
        self._recent_decisions.clear()
        self._decision_count = 0
        self._do_nothing_count = 0
        self._parse_error_count = 0
        self._consecutive_parse_errors = 0
        self._last_price = 0.0
        logger.info("reasoner.session_cleared")

    def clear_level_memory(self) -> None:
        """Clear level interaction memory (e.g., new trading day with new levels)."""
        self._level_interactions.clear()
        logger.info("reasoner.level_memory_cleared")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def do_nothing_count(self) -> int:
        return self._do_nothing_count

    @property
    def parse_error_count(self) -> int:
        return self._parse_error_count

    @property
    def consecutive_parse_errors(self) -> int:
        return self._consecutive_parse_errors

    @property
    def recent_decisions(self) -> list[DecisionMemory]:
        """Return list of recent decisions for external inspection."""
        return list(self._recent_decisions)

    @property
    def level_interactions(self) -> list[LevelInteraction]:
        """Return list of level interactions for external inspection."""
        return list(self._level_interactions)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "decisions": self._decision_count,
            "do_nothing": self._do_nothing_count,
            "parse_errors": self._parse_error_count,
            "consecutive_parse_errors": self._consecutive_parse_errors,
            "confidence_threshold": self._confidence_threshold,
            "decision_memory_size": len(self._recent_decisions),
            "level_memory_size": len(self._level_interactions),
            "has_postmortem_lessons": bool(self._postmortem_lessons),
        }
