"""Trail manager — client-side trailing stop logic.

Tradovate doesn't have server-side trailing stops for futures,
so we implement our own. The trail manager:

1. Watches price updates and tracks the high-water mark
2. Computes the new stop price as (peak - trail_distance)
3. Batches stop modifications (only moves every 3-5 points) to
   avoid rate limit exhaustion
4. Only moves stops in the favorable direction (never widens)

Trail activation:
- Trail begins after position reaches a minimum profit threshold
- Trail distance can be fixed or stepped (tighten at milestones)

Rate limit awareness:
- Tradovate allows ~50 requests/sec but stop mods are expensive
- We batch: only send modify when the new stop differs by >= batch_points
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.types import PositionState, Side

logger = structlog.get_logger()


class TrailManager:
    """Client-side trailing stop manager.

    Usage:
        trail = TrailManager(trail_distance=8.0, batch_points=3.0)
        trail.activate(position, current_price)

        # On each price update:
        new_stop = trail.update(current_price)
        if new_stop is not None:
            await order_manager.modify_stop(new_stop)
    """

    def __init__(
        self,
        trail_distance: float = 10.0,
        batch_points: float = 3.0,
        activation_profit_pts: float = 8.0,
        tighten_at_pts: float = 20.0,
        tighten_distance: float = 5.0,
    ) -> None:
        self._trail_distance = trail_distance
        self._batch_points = batch_points
        self._activation_profit_pts = activation_profit_pts
        self._tighten_at_pts = tighten_at_pts
        self._tighten_distance = tighten_distance

        # State
        self._is_active: bool = False
        self._side: Optional[Side] = None
        self._entry_price: float = 0.0
        self._peak_price: float = 0.0
        self._last_sent_stop: float = 0.0
        self._last_confirmed_stop: float = 0.0
        self._current_trail_stop: float = 0.0
        self._pending_stop: Optional[float] = None
        self._updates_sent: int = 0
        self._updates_skipped: int = 0
        self._modify_failures: int = 0
        self._activated_at: Optional[datetime] = None

    def activate(self, position: PositionState, current_price: float) -> None:
        """Start trailing for a position.

        Args:
            position: The current position state.
            current_price: The current market price.
        """
        self._side = position.side
        self._entry_price = position.avg_entry
        self._peak_price = current_price
        self._last_sent_stop = position.stop_price
        self._last_confirmed_stop = position.stop_price
        self._current_trail_stop = position.stop_price
        self._pending_stop = None
        self._is_active = True
        self._activated_at = datetime.now(tz=UTC)

        logger.info(
            "trail_manager.activated",
            side=self._side.value,
            entry=self._entry_price,
            initial_stop=position.stop_price,
            trail_distance=self._trail_distance,
        )

    def deactivate(self) -> None:
        """Stop trailing (position closed or trail disabled)."""
        self._is_active = False
        self._side = None
        self._peak_price = 0.0
        self._last_sent_stop = 0.0
        self._last_confirmed_stop = 0.0
        self._current_trail_stop = 0.0
        self._pending_stop = None

    def update(self, current_price: float) -> Optional[float]:
        """Process a price update and return new stop if one should be sent.

        Args:
            current_price: The latest market price.

        Returns:
            New stop price if a modification should be sent, None otherwise.
        """
        if not self._is_active or self._side is None:
            return None

        # Check activation threshold
        if not self._meets_activation_threshold(current_price):
            return None

        # Update peak
        if self._side == Side.LONG:
            if current_price > self._peak_price:
                self._peak_price = current_price
        else:
            if current_price < self._peak_price:
                self._peak_price = current_price

        # Compute ideal trail stop
        distance = self._effective_trail_distance(current_price)

        if self._side == Side.LONG:
            ideal_stop = round(self._peak_price - distance, 2)
            # Never move stop down (widen risk)
            if ideal_stop <= self._current_trail_stop:
                return None
            self._current_trail_stop = ideal_stop
        else:
            ideal_stop = round(self._peak_price + distance, 2)
            # Never move stop up (widen risk) for shorts
            if ideal_stop >= self._current_trail_stop and self._current_trail_stop > 0:
                return None
            self._current_trail_stop = ideal_stop

        # Batch: only send if moved enough from last sent stop
        if self._should_send(self._current_trail_stop):
            self._pending_stop = self._current_trail_stop
            self._last_sent_stop = self._current_trail_stop
            self._updates_sent += 1
            return self._current_trail_stop

        self._updates_skipped += 1
        return None

    def confirm_modify(self, success: bool) -> None:
        """Confirm whether the stop modify succeeded or failed.

        Must be called after every non-None return from update().
        On failure, rolls back last_sent_stop so the next update
        will re-attempt the modification.

        Args:
            success: True if the REST modify call succeeded.
        """
        if self._pending_stop is None:
            return

        if success:
            self._last_confirmed_stop = self._pending_stop
        else:
            # Roll back both sent and trail stop so the next update re-attempts
            self._last_sent_stop = self._last_confirmed_stop
            self._current_trail_stop = self._last_confirmed_stop
            self._modify_failures += 1
            logger.warning(
                "trail_manager.modify_failed",
                pending_stop=self._pending_stop,
                rolled_back_to=self._last_confirmed_stop,
            )

        self._pending_stop = None

    def _meets_activation_threshold(self, current_price: float) -> bool:
        """Check if position has enough profit to start trailing."""
        if self._side == Side.LONG:
            profit_pts = current_price - self._entry_price
        else:
            profit_pts = self._entry_price - current_price
        return profit_pts >= self._activation_profit_pts

    def _effective_trail_distance(self, current_price: float) -> float:
        """Get trail distance, potentially tightened at milestones."""
        if self._side == Side.LONG:
            profit_pts = current_price - self._entry_price
        else:
            profit_pts = self._entry_price - current_price

        if profit_pts >= self._tighten_at_pts:
            return self._tighten_distance
        return self._trail_distance

    def _should_send(self, new_stop: float) -> bool:
        """Check if the stop has moved enough to warrant a modify call."""
        if self._last_sent_stop == 0.0:
            return True
        diff = abs(new_stop - self._last_sent_stop)
        return diff >= self._batch_points

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def peak_price(self) -> float:
        return self._peak_price

    @property
    def current_trail_stop(self) -> float:
        return self._current_trail_stop

    @property
    def last_sent_stop(self) -> float:
        return self._last_sent_stop

    @property
    def trail_distance(self) -> float:
        return self._trail_distance

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "is_active": self._is_active,
            "side": self._side.value if self._side else None,
            "peak_price": self._peak_price,
            "current_trail_stop": self._current_trail_stop,
            "last_sent_stop": self._last_sent_stop,
            "last_confirmed_stop": self._last_confirmed_stop,
            "updates_sent": self._updates_sent,
            "updates_skipped": self._updates_skipped,
            "modify_failures": self._modify_failures,
            "trail_distance": self._trail_distance,
            "batch_points": self._batch_points,
        }
