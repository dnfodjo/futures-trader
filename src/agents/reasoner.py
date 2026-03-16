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

    # Temperature settings for different contexts
    TEMP_ROUTINE: float = 0.0  # No position, no setups — deterministic DO_NOTHING
    TEMP_ACTIVE: float = 0.2  # In position or near levels — slight exploration
    TEMP_ENTRY: float = 0.15  # Entry decisions — low temp for consistency

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
        detected_setups: str = "",
        price_action_narrative: str = "",
    ) -> LLMAction:
        """Make a trading decision based on the current market state.

        Args:
            state: Complete MarketState snapshot.
            game_plan: Pre-market game plan text.
            extra_context: Any additional context (e.g., recent debate result).
            detected_setups: Pre-screened setup descriptions from SetupDetector.
            price_action_narrative: Rich text narrative from PriceActionAnalyzer.

        Returns:
            LLMAction with the decision and reasoning.
        """
        # Choose model based on context
        model = self._select_model(state)

        # Choose temperature based on context (adaptive)
        temperature = self._select_temperature(state, detected_setups)

        # Build enriched extra context with decision memory + level memory + postmortem
        enriched_context = self._build_enriched_context(state, extra_context)

        # Build system prompt with caching
        system = build_system_blocks(
            game_plan=game_plan,
            extra_context=enriched_context,
            detected_setups=detected_setups,
            price_action_narrative=price_action_narrative,
        )

        # Build user message with market state
        state_json = json.dumps(state.to_llm_dict(), indent=2)
        messages = [
            {
                "role": "user",
                "content": f"Current market state:\n```json\n{state_json}\n```\n\nWhat is your trading decision?",
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
                temperature=temperature,
            )

            action = self._parse_action(response, model)
            self._decision_count += 1
            self._consecutive_parse_errors = 0

            if action.action == ActionType.DO_NOTHING:
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

    # ── Adaptive Temperature ──────────────────────────────────────────────────

    def _select_temperature(self, state: MarketState, detected_setups: str) -> float:
        """Choose temperature based on decision context.

        Lower temperature = more deterministic, less creative.
        - Routine scans (no position, no setups): 0.0 — we want consistent DO_NOTHING
        - Active management (in position): 0.2 — need nuance in position management
        - Entry decisions (setups detected): 0.15 — want consistency but some flexibility
        """
        # In position — need nuanced management
        if state.position is not None:
            return self.TEMP_ACTIVE

        # Setups detected — potential entry
        if detected_setups:
            return self.TEMP_ENTRY

        # Near a key level — elevated attention
        if state.last_price > 0 and hasattr(state, "levels"):
            level_data = state.levels.model_dump()
            for val in level_data.values():
                if isinstance(val, (int, float)) and val > 0:
                    if abs(state.last_price - val) <= 5.0:
                        return self.TEMP_ENTRY

        # Routine scan — want deterministic DO_NOTHING
        return self.TEMP_ROUTINE

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
        """Choose model based on context.

        Model selection directly impacts decision quality vs cost:
        - Sonnet: Better at identifying subtle setups and managing positions.
          Used in all high-stakes situations.
        - Haiku: Fast and cheap for "is anything happening?" checks during
          low-activity periods when flat.

        Approximate cost: Sonnet ~$0.003/call, Haiku ~$0.0003/call (with caching).
        At 30s intervals over 23 hours: blended approach targets ~$3-5/day.
        """
        # Always Sonnet when in a position (managing risk is high-stakes)
        if state.position is not None:
            return "sonnet"

        # Sonnet during high-activity RTH phases (entries are most likely)
        if state.session_phase.value in ("open_drive", "close"):
            return "sonnet"

        # Sonnet during morning session (best RTH trading window)
        if state.session_phase.value == "morning":
            return "sonnet"

        # Sonnet during afternoon (institutional flow — real moves)
        if state.session_phase.value == "afternoon":
            return "sonnet"

        # Sonnet during London session (cleanest trends of the day)
        if state.session_phase.value == "london":
            return "sonnet"

        # Sonnet during pre-RTH (econ data drops at 8:30)
        if state.session_phase.value == "pre_rth":
            return "sonnet"

        # Sonnet in blackout (need careful "do nothing" decision)
        if state.in_blackout:
            return "sonnet"

        # Sonnet when at a decision point (near a key level)
        if state.last_price > 0:
            level_data = state.levels.model_dump()
            for val in level_data.values():
                if val > 0 and abs(state.last_price - val) <= 5.0:
                    return "sonnet"

        # Haiku for low-probability scan periods when flat and away from levels:
        # - Asian session (thin, mostly drift)
        # - Midday chop zone
        # - Post-RTH (thin, unwinding)
        return "haiku"

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
            action_str = tool_input.get("action", "DO_NOTHING")
            action_type = ActionType(action_str)

            # Parse optional fields
            side = None
            if "side" in tool_input and tool_input["side"]:
                side = Side(tool_input["side"])

            quantity = tool_input.get("quantity")
            stop_distance = tool_input.get("stop_distance")
            new_stop_price = tool_input.get("new_stop_price")
            reasoning = tool_input.get("reasoning", "")
            confidence = float(tool_input.get("confidence", 0.5))

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            setup_type = tool_input.get("setup_type")

            return LLMAction(
                action=action_type,
                side=side,
                quantity=quantity,
                stop_distance=stop_distance,
                new_stop_price=new_stop_price,
                reasoning=reasoning,
                confidence=confidence,
                setup_type=setup_type,
                model_used=model,
                latency_ms=response.latency_ms,
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
        """Return a safe DO_NOTHING action on unrecoverable errors."""
        self._parse_error_count += 1
        self._consecutive_parse_errors += 1
        return LLMAction(
            action=ActionType.DO_NOTHING,
            reasoning=f"LLM error — defaulting to DO_NOTHING: {error}",
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
