"""Position limit guardrails — enforce max contracts and add limits.

Checks:
- Total position would not exceed max contracts
- Number of adds does not exceed max adds per trade
- SCALE_OUT quantity does not exceed current position
- Quantity is clamped if it would exceed limits (modified, not blocked)
"""

from __future__ import annotations

from typing import Optional

import structlog

from src.core.types import ActionType, GuardrailResult, LLMAction, PositionState

logger = structlog.get_logger()


class PositionLimitGuardrail:
    """Enforces position size and add count limits.

    Usage:
        guard = PositionLimitGuardrail(max_contracts=6, max_adds=3)
        result = guard.check(action, position)
    """

    def __init__(
        self,
        max_contracts: int = 6,
        max_adds: int = 3,
    ) -> None:
        self._max_contracts = max_contracts
        self._max_adds = max_adds

    @property
    def max_contracts(self) -> int:
        return self._max_contracts

    def check(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        max_contracts_override: Optional[int] = None,
    ) -> GuardrailResult:
        """Check position limit rules.

        Args:
            action: The proposed action.
            position: Current position (None if flat).
            max_contracts_override: Override max contracts (e.g., from profit preservation).

        Returns:
            GuardrailResult — may modify quantity to fit within limits.
        """
        max_contracts = max_contracts_override if max_contracts_override is not None else self._max_contracts

        if action.action == ActionType.ENTER:
            return self._check_enter(action, position, max_contracts)
        elif action.action == ActionType.ADD:
            return self._check_add(action, position, max_contracts)
        elif action.action == ActionType.SCALE_OUT:
            return self._check_scale_out(action, position)

        return GuardrailResult(allowed=True)

    def _check_enter(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        max_contracts: int,
    ) -> GuardrailResult:
        """Validate entry action."""
        if position is not None:
            return GuardrailResult(
                allowed=False,
                reason="position_limit: cannot ENTER while already in position",
            )

        quantity = action.quantity or 1

        if quantity > max_contracts:
            # Clamp to max
            logger.info(
                "guardrail.entry_clamped",
                requested=quantity,
                max=max_contracts,
            )
            return GuardrailResult(
                allowed=True,
                modified_quantity=max_contracts,
            )

        return GuardrailResult(allowed=True)

    def _check_add(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        max_contracts: int,
    ) -> GuardrailResult:
        """Validate add-to-position action."""
        if position is None:
            return GuardrailResult(
                allowed=False,
                reason="position_limit: cannot ADD without a position",
            )

        # Check add count
        if position.adds_count >= self._max_adds:
            return GuardrailResult(
                allowed=False,
                reason=f"position_limit: max adds reached ({self._max_adds})",
            )

        quantity = action.quantity or 1
        new_total = position.quantity + quantity

        if new_total > max_contracts:
            # Clamp
            allowed_add = max_contracts - position.quantity
            if allowed_add <= 0:
                return GuardrailResult(
                    allowed=False,
                    reason=f"position_limit: already at max contracts ({max_contracts})",
                )
            logger.info(
                "guardrail.add_clamped",
                requested=quantity,
                allowed=allowed_add,
                current=position.quantity,
                max=max_contracts,
            )
            return GuardrailResult(
                allowed=True,
                modified_quantity=allowed_add,
            )

        return GuardrailResult(allowed=True)

    def _check_scale_out(
        self,
        action: LLMAction,
        position: Optional[PositionState],
    ) -> GuardrailResult:
        """Validate scale-out action."""
        if position is None:
            return GuardrailResult(
                allowed=False,
                reason="position_limit: cannot SCALE_OUT without a position",
            )

        quantity = action.quantity or 1
        if quantity > position.quantity:
            # Clamp to full position (becomes a flatten)
            return GuardrailResult(
                allowed=True,
                modified_quantity=position.quantity,
            )

        return GuardrailResult(allowed=True)
