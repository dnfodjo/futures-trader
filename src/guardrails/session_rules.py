"""Session rule guardrails — time-based and session state restrictions.

Checks:
- No new entries during economic news blackout periods
- No entries during close or after-hours session phases
- Daily loss limit not yet hit (redundant with SessionController but defense-in-depth)
- Consecutive loser streak not exceeded
- Midday size reduction (half contracts during low-volume period)
- Post-loss cooling: require higher confidence after consecutive losses
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Optional

import structlog

from src.core.types import (
    ActionType,
    GuardrailResult,
    LLMAction,
    MarketState,
    SessionPhase,
)

logger = structlog.get_logger()


class SessionRuleGuardrail:
    """Enforces session-based trading rules.

    Usage:
        guard = SessionRuleGuardrail(max_consecutive_losers=4)
        result = guard.check(action, state, daily_pnl, consecutive_losers)
    """

    # Phases where new entries are blocked
    BLOCKED_PHASES = frozenset({
        SessionPhase.CLOSE,
        SessionPhase.AFTER_HOURS,
    })

    def __init__(
        self,
        max_consecutive_losers: int = 4,
        daily_loss_limit: float = 400.0,
        blackout_minutes: int = 5,
        post_blackout_minutes: int = 10,
    ) -> None:
        self._max_consecutive_losers = max_consecutive_losers
        self._daily_loss_limit = daily_loss_limit
        self._blackout_minutes = blackout_minutes
        self._post_blackout_minutes = post_blackout_minutes

    def check(
        self,
        action: LLMAction,
        state: MarketState,
        daily_pnl: float = 0.0,
        consecutive_losers: int = 0,
    ) -> GuardrailResult:
        """Run session rule checks.

        Args:
            action: The proposed action.
            state: Current market state.
            daily_pnl: Today's net P&L.
            consecutive_losers: Current consecutive loser streak.

        Returns:
            GuardrailResult.
        """
        # Only block new entries (ENTER/ADD), not position management
        if action.action not in (ActionType.ENTER, ActionType.ADD):
            return GuardrailResult(allowed=True)

        # 1. Daily loss limit (defense-in-depth)
        if daily_pnl <= -self._daily_loss_limit:
            return GuardrailResult(
                allowed=False,
                reason=f"session_rule: daily loss limit hit (${daily_pnl:.2f})",
            )

        # 2. Consecutive losers
        if consecutive_losers >= self._max_consecutive_losers:
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"session_rule: {consecutive_losers} consecutive losers "
                    f"(max: {self._max_consecutive_losers})"
                ),
            )

        # 3. Session phase block
        if hasattr(state, "session_phase") and state.session_phase in self.BLOCKED_PHASES:
            return GuardrailResult(
                allowed=False,
                reason=f"session_rule: entries blocked during {state.session_phase.value}",
            )

        # 4. News blackout
        if self._in_blackout(state):
            return GuardrailResult(
                allowed=False,
                reason="session_rule: entries blocked during news blackout",
            )

        # 5. Post-loss cooling: after consecutive losses, require higher confidence
        if consecutive_losers >= 2 and action.confidence < 0.7:
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"session_rule: post-loss cooling — {consecutive_losers} consecutive "
                    f"losers requires confidence >= 0.70 (got {action.confidence:.2f})"
                ),
            )

        # 6. Midday size reduction — cap quantity at half during low-volume period
        if (
            action.action == ActionType.ENTER
            and hasattr(state, "session_phase")
            and state.session_phase == SessionPhase.MIDDAY
            and action.quantity is not None
            and action.quantity > 2
        ):
            reduced = max(1, action.quantity // 2)
            logger.info(
                "session_rule.midday_size_reduction",
                original=action.quantity,
                reduced=reduced,
            )
            return GuardrailResult(
                allowed=True,
                modified_quantity=reduced,
            )

        return GuardrailResult(allowed=True)

    def _in_blackout(self, state: MarketState) -> bool:
        """Check if we're within a news blackout window.

        Checks both:
        - Pre-event blackout: N minutes before a high-impact event
        - Post-event blackout: M minutes after a high-impact event

        Also respects the pre-computed `in_blackout` flag from the state engine
        (which is set by EconomicCalendar.is_in_blackout with both pre/post windows).
        """
        # Fast path: state engine already computed blackout (includes post-event)
        if hasattr(state, "in_blackout") and state.in_blackout:
            return True

        if not hasattr(state, "upcoming_events") or not state.upcoming_events:
            return False

        now = state.timestamp
        pre_window = timedelta(minutes=self._blackout_minutes)
        post_window = timedelta(minutes=self._post_blackout_minutes)

        for event in state.upcoming_events:
            if event.impact == "high":
                time_until = event.time - now
                # Pre-event: event is 0 to N minutes in the future
                if timedelta(0) <= time_until <= pre_window:
                    return True
                # Post-event: event is 0 to M minutes in the past
                time_since = now - event.time
                if timedelta(0) <= time_since <= post_window:
                    return True

        return False
