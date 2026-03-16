"""Guardrail engine — validates every LLM action before execution.

The guardrail engine is the safety layer between the LLM reasoner and
the order manager. Every action passes through here before execution.

Pipeline:
1. Position limits (max contracts, add limits)
2. Session rules (blackout, phase, daily loss, consecutive losers)
3. Risk checks (stop distance, confidence threshold, entry spacing)

If ANY check fails, the action is blocked and a GuardrailResult
is returned with allowed=False. Some checks may modify the action
(e.g., reduce quantity) rather than block entirely.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from src.core.events import EventBus
from src.core.types import (
    ActionType,
    Event,
    EventType,
    GuardrailResult,
    LLMAction,
    MarketState,
    PositionState,
    Side,
)
from src.guardrails.position_limits import PositionLimitGuardrail
from src.guardrails.risk_checks import RiskCheckGuardrail
from src.guardrails.session_rules import SessionRuleGuardrail

logger = structlog.get_logger()


class GuardrailEngine:
    """Runs all guardrail checks on an LLM action before execution.

    Usage:
        engine = GuardrailEngine(event_bus=bus)
        result = engine.check(action, state, position)
        if result.allowed:
            await order_manager.execute(action, position, last_price)
        else:
            log_blocked(result.reason)
    """

    def __init__(
        self,
        event_bus: EventBus,
        max_contracts: int = 6,
        max_adds: int = 3,
        min_stop_distance: float = 3.0,
        max_stop_distance: float = 25.0,
        min_confidence: float = 0.3,
        min_entry_spacing_pts: float = 5.0,
        max_consecutive_losers: int = 4,
        daily_loss_limit: float = 400.0,
        blackout_minutes: int = 5,
    ) -> None:
        self._bus = event_bus

        self._position_limits = PositionLimitGuardrail(
            max_contracts=max_contracts,
            max_adds=max_adds,
        )
        self._session_rules = SessionRuleGuardrail(
            max_consecutive_losers=max_consecutive_losers,
            daily_loss_limit=daily_loss_limit,
            blackout_minutes=blackout_minutes,
        )
        self._risk_checks = RiskCheckGuardrail(
            min_stop_distance=min_stop_distance,
            max_stop_distance=max_stop_distance,
            min_confidence=min_confidence,
            min_entry_spacing_pts=min_entry_spacing_pts,
        )

        # Stats
        self._checks_run: int = 0
        self._checks_passed: int = 0
        self._checks_blocked: int = 0
        self._checks_modified: int = 0
        self._block_reasons: dict[str, int] = {}

    def check(
        self,
        action: LLMAction,
        state: MarketState,
        position: Optional[PositionState] = None,
        daily_pnl: float = 0.0,
        consecutive_losers: int = 0,
        effective_max_contracts: Optional[int] = None,
    ) -> GuardrailResult:
        """Run all guardrail checks on an action.

        Args:
            action: The LLM's decided action.
            state: Current market state snapshot.
            position: Current position (None if flat).
            daily_pnl: Today's net P&L.
            consecutive_losers: Current consecutive losing streak.
            effective_max_contracts: Max contracts from profit preservation
                                     (overrides base max if provided).

        Returns:
            GuardrailResult with allowed=True/False and reason.
        """
        self._checks_run += 1

        # DO_NOTHING and STOP_TRADING always pass
        if action.action in (ActionType.DO_NOTHING, ActionType.STOP_TRADING):
            self._checks_passed += 1
            return GuardrailResult(allowed=True)

        # FLATTEN always allowed — we never block closing
        if action.action == ActionType.FLATTEN:
            self._checks_passed += 1
            return GuardrailResult(allowed=True)

        # MOVE_STOP — validate stop is on the correct side of the market
        if action.action == ActionType.MOVE_STOP:
            result = self._validate_move_stop(action, state, position)
            if not result.allowed:
                return self._record_block(result)
            self._checks_passed += 1
            return GuardrailResult(allowed=True)

        # ── Run pipeline for ENTER, ADD, SCALE_OUT ─────────────────────────

        max_contracts = effective_max_contracts if effective_max_contracts is not None else self._position_limits.max_contracts

        # Track modifications across pipeline stages
        modified_quantity: Optional[int] = None

        # 1. Position limits
        result = self._position_limits.check(
            action=action,
            position=position,
            max_contracts_override=max_contracts,
        )
        if not result.allowed:
            return self._record_block(result)
        if result.modified_quantity is not None:
            modified_quantity = result.modified_quantity

        # 2. Session rules
        result = self._session_rules.check(
            action=action,
            state=state,
            daily_pnl=daily_pnl,
            consecutive_losers=consecutive_losers,
        )
        if not result.allowed:
            return self._record_block(result)

        # 3. Risk checks
        result = self._risk_checks.check(
            action=action,
            state=state,
            position=position,
        )
        if not result.allowed:
            return self._record_block(result)

        # All checks passed
        self._checks_passed += 1

        # Return with any quantity modification from earlier stages
        if modified_quantity is not None:
            self._checks_modified += 1
            return GuardrailResult(allowed=True, modified_quantity=modified_quantity)

        return GuardrailResult(allowed=True)

    def _validate_move_stop(
        self,
        action: LLMAction,
        state: MarketState,
        position: Optional[PositionState],
    ) -> GuardrailResult:
        """Validate a MOVE_STOP action.

        Checks:
        - new_stop_price is provided
        - There is a position to protect
        - Stop is on correct side (below market for long, above for short)
        """
        if action.new_stop_price is None:
            return GuardrailResult(
                allowed=False,
                reason="guardrail: MOVE_STOP requires new_stop_price",
            )

        if position is None:
            return GuardrailResult(
                allowed=False,
                reason="guardrail: MOVE_STOP without position",
            )

        last_price = state.last_price

        # For LONG: stop must be below market price
        if position.side == Side.LONG and action.new_stop_price >= last_price:
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"guardrail: MOVE_STOP for LONG has stop "
                    f"({action.new_stop_price:.2f}) above/at market ({last_price:.2f})"
                ),
            )

        # For SHORT: stop must be above market price
        if position.side == Side.SHORT and action.new_stop_price <= last_price:
            return GuardrailResult(
                allowed=False,
                reason=(
                    f"guardrail: MOVE_STOP for SHORT has stop "
                    f"({action.new_stop_price:.2f}) below/at market ({last_price:.2f})"
                ),
            )

        return GuardrailResult(allowed=True)

    def _record_block(self, result: GuardrailResult) -> GuardrailResult:
        """Record a blocked action in stats and publish event."""
        self._checks_blocked += 1
        reason_key = result.reason.split(":")[0] if ":" in result.reason else result.reason
        self._block_reasons[reason_key] = self._block_reasons.get(reason_key, 0) + 1

        logger.warning(
            "guardrail.blocked",
            reason=result.reason,
            total_blocked=self._checks_blocked,
        )

        self._bus.publish_nowait(Event(
            type=EventType.GUARDRAIL_TRIGGERED,
            data={"reason": result.reason, "allowed": False},
        ))

        return result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "checks_run": self._checks_run,
            "checks_passed": self._checks_passed,
            "checks_blocked": self._checks_blocked,
            "checks_modified": self._checks_modified,
            "block_reasons": dict(self._block_reasons),
        }
