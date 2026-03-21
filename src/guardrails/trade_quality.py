"""Trade quality guardrails — mechanical pre-qualification before LLM entries.

These guardrails block obviously bad entries that the LLM consistently makes
despite prompt instructions. They are lightweight filters, not cooldowns.

Checks:
1. Session extreme proximity filter (don't short near session lows, etc.)
2. Pullback-to-EMA requirement (price must be near 21 EMA, not extended)

Note: RSI extreme filter was REMOVED — RSI is a mean-reversion indicator
that conflicts with trend-following ICT methodology. See orchestrator.py 4d2.
"""

from __future__ import annotations

from typing import Optional

import structlog

from src.core.types import (
    ActionType,
    GuardrailResult,
    LLMAction,
    MarketState,
    PositionState,
    Side,
)

logger = structlog.get_logger()

# Session extreme proximity — don't enter near session highs/lows
SESSION_EXTREME_BUFFER_PTS: float = 15.0

# Pullback-to-EMA — price must be within N points of the 21 EMA
EMA_PULLBACK_MAX_PTS: float = 20.0


class TradeQualityGuardrail:
    """Mechanical pre-qualification filters for entries.

    These are lightweight checks that block obviously bad entries:
    - Entering near session extremes (gets trapped)
    - Entering when price is extended far from the 21 EMA (chasing)

    Usage:
        guard = TradeQualityGuardrail()
        result = guard.check(action, state, position)
    """

    def __init__(
        self,
        session_extreme_buffer_pts: float = SESSION_EXTREME_BUFFER_PTS,
        ema_pullback_max_pts: float = EMA_PULLBACK_MAX_PTS,
    ) -> None:
        self._session_extreme_buffer_pts = session_extreme_buffer_pts
        self._ema_pullback_max_pts = ema_pullback_max_pts

    def check(
        self,
        action: LLMAction,
        state: MarketState,
        position: Optional[PositionState] = None,
    ) -> GuardrailResult:
        """Run trade quality checks on the proposed action."""
        # Only validate ENTER actions
        if action.action != ActionType.ENTER:
            return GuardrailResult(allowed=True)

        # ── 1. RSI extreme filter — REMOVED ────────────────────────────
        # RSI is a mean-reversion indicator that conflicts with the trend-following
        # ICT methodology. When MNQ trends strongly, RSI > 70 or < 30 persists
        # for extended periods — blocking entries during the strongest moves.
        # The EMA trend filter, direction-aware volume, and structure factor
        # provide far more accurate trend-vs-counter-trend filtering.
        # RSI is still logged by the orchestrator for post-session analysis.
        # See orchestrator.py 4d2 comment for full rationale.

        # ── 2. Session extreme proximity filter ──────────────────────────
        if (
            action.side is not None
            and hasattr(state, "levels")
            and state.levels.session_high > 0
            and state.levels.session_low > 0
            and state.levels.session_low < state.levels.session_high
        ):
            price = state.last_price
            dist_to_high = state.levels.session_high - price
            dist_to_low = price - state.levels.session_low

            if action.side == Side.LONG and dist_to_high < self._session_extreme_buffer_pts:
                logger.warning(
                    "trade_quality.session_extreme_blocked",
                    side="LONG",
                    price=price,
                    session_high=state.levels.session_high,
                    distance=dist_to_high,
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"trade_quality: session extreme — cannot LONG within "
                        f"{dist_to_high:.1f}pts of session high "
                        f"({state.levels.session_high:.2f})"
                    ),
                )
            if action.side == Side.SHORT and dist_to_low < self._session_extreme_buffer_pts:
                logger.warning(
                    "trade_quality.session_extreme_blocked",
                    side="SHORT",
                    price=price,
                    session_low=state.levels.session_low,
                    distance=dist_to_low,
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"trade_quality: session extreme — cannot SHORT within "
                        f"{dist_to_low:.1f}pts of session low "
                        f"({state.levels.session_low:.2f})"
                    ),
                )

        # ── 3. Pullback-to-EMA requirement ───────────────────────────────
        ema_21 = state.emas.get("ema_21") if state.emas else None
        if ema_21 is not None and ema_21 > 0:
            distance_from_ema = abs(state.last_price - ema_21)
            if distance_from_ema > self._ema_pullback_max_pts:
                logger.warning(
                    "trade_quality.ema_pullback_blocked",
                    price=state.last_price,
                    ema_21=ema_21,
                    distance=distance_from_ema,
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"trade_quality: price extended — {distance_from_ema:.1f}pts from "
                        f"21 EMA ({ema_21:.2f}), max {self._ema_pullback_max_pts:.0f}pts"
                    ),
                )

        return GuardrailResult(allowed=True)
