"""Tick-level stop and take-profit monitor for QuantLynk execution.

Since QuantLynk only supports market orders (buy/sell/flatten), there are
no exchange-level stop or take-profit orders. This monitor checks every
incoming trade tick from Databento against stop and take-profit levels.

When a level is hit, it IMMEDIATELY sends a flatten signal via QuantLynk.

The monitor also handles trailing stop logic — updating the stop level
on every tick as price moves in our favor.

This replaces the slow software stop check that previously ran only during
the LLM decision cycle (every 10-30 seconds).

Usage:
    monitor = TickStopMonitor(flatten_fn=quantlynk.flatten)
    databento.on_trade(monitor.on_trade)  # wire to tick stream

    # When a position is opened:
    monitor.activate(
        side="short",
        entry_price=24970.0,
        stop_price=24982.0,      # initial stop
        take_profit_price=24940.0,  # optional TP
        trail_distance=12.0,     # trail distance in points
    )

    # Monitor auto-flattens when stop or TP is hit
    # After flatten, call:
    monitor.deactivate()
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Optional

import structlog

logger = structlog.get_logger()

FlattenFn = Callable[[float], Coroutine[Any, Any, Any]]


class TickStopMonitor:
    """Real-time tick-level stop/take-profit monitor.

    Fires on every trade tick from Databento. When stop or take-profit
    is hit, immediately sends flatten via the provided callback.
    """

    def __init__(
        self,
        flatten_fn: FlattenFn,
        target_symbol: str = "",
        trail_distance: float = 8.0,
        trail_activation_points: float = 4.0,
        min_stop_distance: float = 3.0,
        tighten_at_profit: float = 12.0,
        tightened_distance: float = 5.0,
    ) -> None:
        """Initialize the tick stop monitor.

        Args:
            flatten_fn: Async function to call for flatten (QuantLynk).
                        Signature: async def flatten(price: float) -> Any
            target_symbol: Symbol to monitor (e.g., "MNQM6"). Only ticks
                          matching the root (first 3 chars) are processed.
                          Empty string means process all (dangerous!).
            trail_distance: Points to trail behind best price.
            trail_activation_points: Minimum profit before trailing activates.
            min_stop_distance: Never trail stop closer than this to current price.
        """
        self._flatten_fn = flatten_fn
        self._target_symbol = target_symbol.upper() if target_symbol else ""
        self._trail_distance = trail_distance
        self._default_trail_distance = trail_distance
        self._trail_activation = trail_activation_points
        self._min_stop_distance = min_stop_distance
        self._tighten_at_profit = tighten_at_profit
        self._tightened_distance = tightened_distance

        # Position state
        self._active = False
        self._side: Optional[str] = None  # "long" or "short"
        self._entry_price: float = 0.0
        self._stop_price: float = 0.0
        self._take_profit_price: float = 0.0

        # Trail tracking
        self._best_price: float = 0.0  # best price seen since entry
        self._trail_active: bool = False

        # Prevent multiple flatten signals
        self._triggered: bool = False
        self._trigger_reason: str = ""
        self._trigger_price: float = 0.0

        # Stats
        self._ticks_processed: int = 0
        self._last_price: float = 0.0

    # ── Activation / Deactivation ──────────────────────────────────────────

    def activate(
        self,
        side: str,
        entry_price: float,
        stop_price: float,
        take_profit_price: float = 0.0,
        trail_distance: float = 0.0,
        atr: float = 0.0,
    ) -> None:
        """Activate monitoring for a new position.

        Args:
            side: "long" or "short"
            entry_price: Entry price of the position.
            stop_price: Initial stop-loss price.
            take_profit_price: Take-profit price (0 = no TP).
            trail_distance: Override trail distance (0 = use default or ATR-based).
            atr: Current ATR — if provided, trail distance = max(2*ATR, 5.0).
        """
        self._active = True
        self._side = side.lower()
        self._entry_price = entry_price
        self._stop_price = stop_price
        self._take_profit_price = take_profit_price
        self._best_price = entry_price
        self._trail_active = False
        self._triggered = False
        self._trigger_reason = ""
        self._ticks_processed = 0

        if trail_distance > 0:
            self._trail_distance = trail_distance
        elif atr > 0:
            # ATR-based trail: 2x ATR, clamped between 5 and 8 points
            self._trail_distance = round(max(5.0, min(8.0, atr * 2.0)), 1)
            self._tightened_distance = round(max(3.0, min(5.0, atr * 1.2)), 1)
        else:
            self._trail_distance = self._default_trail_distance

        logger.info(
            "tick_stop_monitor.activated",
            side=side,
            entry=entry_price,
            stop=stop_price,
            take_profit=take_profit_price,
            trail_distance=self._trail_distance,
        )

    def deactivate(self) -> None:
        """Deactivate monitoring (position closed)."""
        self._active = False
        self._side = None
        self._trail_active = False
        logger.info(
            "tick_stop_monitor.deactivated",
            ticks_processed=self._ticks_processed,
        )

    def update_stop(self, new_stop: float) -> None:
        """Manually update the stop price (e.g., from LLM decision)."""
        if self._active:
            old = self._stop_price
            self._stop_price = new_stop
            logger.info(
                "tick_stop_monitor.stop_updated",
                old_stop=old,
                new_stop=new_stop,
            )

    def update_take_profit(self, new_tp: float) -> None:
        """Manually update the take-profit price."""
        if self._active:
            self._take_profit_price = new_tp

    # ── Tick Processing ────────────────────────────────────────────────────

    async def on_trade(self, data: dict[str, Any]) -> None:
        """Process a trade tick — check stop and take-profit levels.

        This is registered as a Databento trade handler and fires
        on EVERY incoming trade tick. Symbol filtering ensures we only
        process ticks for our target instrument (e.g., MNQ, not ES).
        """
        if not self._active or self._triggered:
            return

        # Symbol filter: only process ticks for our target instrument.
        # Without this, ES ticks (~6700) would falsely trigger MNQ stops (~24800).
        if self._target_symbol:
            trade_symbol = (data.get("symbol") or "").upper()
            root = self._target_symbol[:3] if len(self._target_symbol) >= 3 else self._target_symbol
            if root not in trade_symbol:
                return

        price = data.get("price", 0.0)
        if price <= 0:
            return

        # Sanity check: reject prices wildly different from entry.
        # A tick at 6700 when entry is 24850 is clearly a different instrument
        # that bypassed the symbol filter. This prevents catastrophic false stops.
        if self._entry_price > 0:
            deviation = abs(price - self._entry_price) / self._entry_price
            if deviation > 0.10:  # >10% from entry → wrong instrument
                return

        self._ticks_processed += 1
        self._last_price = price

        # Update trail
        self._update_trail(price)

        # Check stop-loss
        if self._check_stop(price):
            self._triggered = True
            self._trigger_reason = "stop_hit"
            self._trigger_price = price
            logger.warning(
                "tick_stop_monitor.STOP_HIT",
                side=self._side,
                entry=self._entry_price,
                stop=self._stop_price,
                price=price,
                ticks_in_trade=self._ticks_processed,
                best_price=self._best_price,
            )
            await self._execute_flatten(price)
            return

        # Check take-profit
        if self._check_take_profit(price):
            self._triggered = True
            self._trigger_reason = "take_profit"
            self._trigger_price = price
            logger.info(
                "tick_stop_monitor.TAKE_PROFIT_HIT",
                side=self._side,
                entry=self._entry_price,
                take_profit=self._take_profit_price,
                price=price,
                ticks_in_trade=self._ticks_processed,
                best_price=self._best_price,
            )
            await self._execute_flatten(price)
            return

    def _check_stop(self, price: float) -> bool:
        """Check if stop-loss has been hit."""
        if self._stop_price <= 0:
            return False

        if self._side == "long":
            return price <= self._stop_price
        elif self._side == "short":
            return price >= self._stop_price

        return False

    def _check_take_profit(self, price: float) -> bool:
        """Check if take-profit has been hit."""
        if self._take_profit_price <= 0:
            return False

        if self._side == "long":
            return price >= self._take_profit_price
        elif self._side == "short":
            return price <= self._take_profit_price

        return False

    # ── Trailing Stop Logic ────────────────────────────────────────────────

    def _update_trail(self, price: float) -> None:
        """Update trailing stop based on current price.

        Trail activates once position has minimum profit. Then the stop
        follows price at trail_distance, only moving in our favor.

        Dynamic tightening: once profit exceeds tighten_at_profit (default 12pts),
        trail distance shrinks from 8 to 5 points to capture more profit.
        """
        if self._side == "long":
            # Track best (highest) price
            if price > self._best_price:
                self._best_price = price

            # Check if trail should activate
            profit = price - self._entry_price
            if profit >= self._trail_activation and not self._trail_active:
                self._trail_active = True
                logger.info(
                    "tick_stop_monitor.trail_activated",
                    side="long",
                    profit_points=profit,
                    best_price=self._best_price,
                )

            # Dynamic tightening based on profit
            best_profit = self._best_price - self._entry_price
            if best_profit >= self._tighten_at_profit:
                effective_trail = self._tightened_distance
            else:
                effective_trail = self._trail_distance

            # Update stop if trailing is active
            if self._trail_active:
                ideal_stop = self._best_price - effective_trail
                # Stop only moves UP for longs (never moves backward)
                if ideal_stop > self._stop_price:
                    self._stop_price = ideal_stop

        elif self._side == "short":
            # Track best (lowest) price
            if price < self._best_price:
                self._best_price = price

            # Check if trail should activate
            profit = self._entry_price - price
            if profit >= self._trail_activation and not self._trail_active:
                self._trail_active = True
                logger.info(
                    "tick_stop_monitor.trail_activated",
                    side="short",
                    profit_points=profit,
                    best_price=self._best_price,
                )

            # Dynamic tightening based on profit
            best_profit = self._entry_price - self._best_price
            if best_profit >= self._tighten_at_profit:
                effective_trail = self._tightened_distance
            else:
                effective_trail = self._trail_distance

            # Update stop if trailing is active
            if self._trail_active:
                ideal_stop = self._best_price + effective_trail
                # Stop only moves DOWN for shorts (never moves backward)
                if ideal_stop < self._stop_price:
                    self._stop_price = ideal_stop

    # ── Flatten Execution ──────────────────────────────────────────────────

    async def _execute_flatten(self, price: float) -> None:
        """Send flatten signal via QuantLynk."""
        try:
            result = await self._flatten_fn(price)
            logger.info(
                "tick_stop_monitor.flatten_sent",
                price=price,
                reason=self._trigger_reason,
                result=str(result)[:100] if result else "ok",
            )
        except Exception as e:
            logger.error(
                "tick_stop_monitor.flatten_failed",
                error=str(e),
                price=price,
                reason=self._trigger_reason,
            )
            # Even if flatten fails, we stay triggered to prevent
            # sending multiple signals. The orchestrator will handle
            # the failed flatten via its own stop check.

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_reason(self) -> str:
        return self._trigger_reason

    @property
    def current_stop(self) -> float:
        return self._stop_price

    @property
    def current_take_profit(self) -> float:
        return self._take_profit_price

    @property
    def best_price(self) -> float:
        return self._best_price

    @property
    def trail_active(self) -> bool:
        return self._trail_active

    @property
    def side(self) -> Optional[str]:
        return self._side

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "active": self._active,
            "side": self._side,
            "entry_price": self._entry_price,
            "stop_price": self._stop_price,
            "take_profit_price": self._take_profit_price,
            "best_price": self._best_price,
            "trail_active": self._trail_active,
            "trail_distance": self._trail_distance,
            "triggered": self._triggered,
            "trigger_reason": self._trigger_reason,
            "trigger_price": self._trigger_price,
            "ticks_processed": self._ticks_processed,
            "last_price": self._last_price,
        }
