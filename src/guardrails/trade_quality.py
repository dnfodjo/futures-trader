"""Trade quality guardrails — mechanical pre-qualification before LLM entries.

These guardrails block obviously bad entries that the LLM consistently makes
despite prompt instructions. They are lightweight filters, not cooldowns.

Checks:
1. RSI extreme filter (don't long overbought, don't short oversold)
2. Session extreme proximity filter (don't short near session lows, etc.)
3. Pullback-to-EMA requirement (price must be near 21 EMA, not extended)
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

# RSI thresholds — block entries INTO extremes
RSI_OVERBOUGHT: float = 70.0  # block LONG when RSI > 70
RSI_OVERSOLD: float = 30.0    # block SHORT when RSI < 30

# Session extreme proximity — don't enter near session highs/lows
SESSION_EXTREME_BUFFER_PTS: float = 15.0

# Pullback-to-EMA — price must be within N points of the 21 EMA
EMA_PULLBACK_MAX_PTS: float = 20.0


class TradeQualityGuardrail:
    """Mechanical pre-qualification filters for entries.

    These are lightweight checks that block obviously bad entries:
    - Longing into overbought RSI
    - Shorting into oversold RSI
    - Entering near session extremes (gets trapped)
    - Entering when price is extended far from the 21 EMA (chasing)

    Usage:
        guard = TradeQualityGuardrail()
        result = guard.check(action, state, position)
    """

    def __init__(
        self,
        rsi_overbought: float = RSI_OVERBOUGHT,
        rsi_oversold: float = RSI_OVERSOLD,
        session_extreme_buffer_pts: float = SESSION_EXTREME_BUFFER_PTS,
        ema_pullback_max_pts: float = EMA_PULLBACK_MAX_PTS,
    ) -> None:
        self._rsi_overbought = rsi_overbought
        self._rsi_oversold = rsi_oversold
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

        # ── 1. RSI extreme filter ────────────────────────────────────────
        if state.rsi != 50.0 and action.side is not None:
            if action.side == Side.LONG and state.rsi > self._rsi_overbought:
                logger.warning(
                    "trade_quality.rsi_extreme_blocked",
                    side="LONG",
                    rsi=state.rsi,
                    threshold=self._rsi_overbought,
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"trade_quality: RSI extreme — cannot LONG with RSI "
                        f"{state.rsi:.1f} > {self._rsi_overbought:.0f} (overbought)"
                    ),
                )
            if action.side == Side.SHORT and state.rsi < self._rsi_oversold:
                logger.warning(
                    "trade_quality.rsi_extreme_blocked",
                    side="SHORT",
                    rsi=state.rsi,
                    threshold=self._rsi_oversold,
                )
                return GuardrailResult(
                    allowed=False,
                    reason=(
                        f"trade_quality: RSI extreme — cannot SHORT with RSI "
                        f"{state.rsi:.1f} < {self._rsi_oversold:.0f} (oversold)"
                    ),
                )

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
