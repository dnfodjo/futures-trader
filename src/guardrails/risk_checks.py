"""Risk check guardrails — validate trade parameters.

Checks:
- Stop distance within acceptable range (not too tight, not too wide)
- LLM confidence above minimum threshold for entries
- Entry spacing: don't add too close to existing entry price
- No entries when bid-ask spread is unusually wide (illiquid)
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from src.core import clock
from src.core.types import ActionType, GuardrailResult, LLMAction, MarketState, PositionState

logger = structlog.get_logger()


class RiskCheckGuardrail:
    """Validates trade risk parameters before execution.

    Usage:
        guard = RiskCheckGuardrail(min_stop_distance=10.0)
        result = guard.check(action, state, position)
    """

    def __init__(
        self,
        min_stop_distance: float = 10.0,
        max_stop_distance: float = 25.0,
        max_stop_distance_eth: float = 12.0,
        min_confidence: float = 0.55,
        min_entry_spacing_pts: float = 8.0,
        max_spread_pts: float = 3.0,
    ) -> None:
        self._min_stop_distance = min_stop_distance
        self._max_stop_distance = max_stop_distance
        self._max_stop_distance_eth = max_stop_distance_eth
        self._min_confidence = min_confidence
        self._min_entry_spacing_pts = min_entry_spacing_pts
        self._max_spread_pts = max_spread_pts

    def check(
        self,
        action: LLMAction,
        state: MarketState,
        position: Optional[PositionState] = None,
    ) -> GuardrailResult:
        """Run risk checks on the proposed action.

        Args:
            action: The proposed action.
            state: Current market state.
            position: Current position (None if flat).

        Returns:
            GuardrailResult.
        """
        # Only validate entry-type actions
        if action.action not in (ActionType.ENTER, ActionType.ADD):
            return GuardrailResult(allowed=True)

        # 1. Confidence threshold
        if action.confidence < self._min_confidence:
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"risk_check: confidence {action.confidence:.2f} "
                    f"below minimum {self._min_confidence:.2f}"
                ),
            )

        # 2. Stop distance validation
        if action.stop_distance is not None:
            if action.stop_distance < self._min_stop_distance:
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"risk_check: stop distance {action.stop_distance:.1f}pts "
                        f"below minimum {self._min_stop_distance:.1f}pts"
                    ),
                )
            # Use tighter max stop during ETH sessions
            phase = clock.get_session_phase()
            effective_max = (
                self._max_stop_distance_eth
                if clock.is_eth(phase)
                else self._max_stop_distance
            )
            if action.stop_distance > effective_max:
                session_label = "ETH" if clock.is_eth(phase) else "RTH"
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"risk_check: stop distance {action.stop_distance:.1f}pts "
                        f"exceeds {session_label} maximum {effective_max:.1f}pts"
                    ),
                )

        # 3. Entry spacing (ADD only)
        if action.action == ActionType.ADD and position is not None:
            spacing = abs(state.last_price - position.avg_entry)
            if spacing < self._min_entry_spacing_pts:
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"risk_check: add too close to entry "
                        f"({spacing:.1f}pts < {self._min_entry_spacing_pts:.1f}pts min)"
                    ),
                )

        # 4. Spread check (illiquidity detection)
        spread = state.ask - state.bid
        if spread > self._max_spread_pts:
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"risk_check: spread {spread:.2f}pts exceeds "
                    f"max {self._max_spread_pts:.2f}pts (illiquid)"
                ),
            )

        # 5. Gate 1 enforcement: NO counter-trend entries
        # This is the #1 source of losses — the LLM ignores prompt instructions
        # to not trade against EMA alignment. Enforce it programmatically.
        if action.action == ActionType.ENTER and action.side is not None and state.emas:
            alignment = state.emas.get("alignment", "")
            if alignment in ("bullish", "bullish_partial") and action.side.value == "short":
                logger.warning(
                    "risk_check.gate1_violation",
                    ema_alignment=alignment,
                    attempted_side="short",
                    reasoning=action.reasoning[:100] if action.reasoning else "",
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"risk_check: Gate 1 violation — cannot SHORT with "
                        f"{alignment} EMA alignment. EMAs must be bearish or mixed."
                    ),
                )
            if alignment in ("bearish", "bearish_partial") and action.side.value == "long":
                logger.warning(
                    "risk_check.gate1_violation",
                    ema_alignment=alignment,
                    attempted_side="long",
                    reasoning=action.reasoning[:100] if action.reasoning else "",
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"risk_check: Gate 1 violation — cannot go LONG with "
                        f"{alignment} EMA alignment. EMAs must be bullish or mixed."
                    ),
                )

        return GuardrailResult(allowed=True)
