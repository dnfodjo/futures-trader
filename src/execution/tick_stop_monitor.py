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
import json
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import structlog

logger = structlog.get_logger()

_PERSIST_PATH = Path("data/tick_stop_state.json")

FlattenFn = Callable[[float], Coroutine[Any, Any, Any]]
PartialFn = Callable[..., Coroutine[Any, Any, Any]]


class TickStopMonitor:
    """Real-time tick-level stop/take-profit monitor.

    Fires on every trade tick from Databento. When stop or take-profit
    is hit, immediately sends flatten via the provided callback.
    """

    def __init__(
        self,
        flatten_fn: FlattenFn,
        target_symbol: str = "",
        trail_distance: float = 15.0,       # was 12 — wider trail prevents rapid exits
        trail_activation_points: float = 20.0,  # was 10 — don't trail until solid profit
        min_stop_distance: float = 5.0,      # was 4 — never trail closer than 5pts
        tighten_at_profit: float = 25.0,     # was 20 — only tighten at big profit
        tightened_distance: float = 6.0,     # was 5 — still give room at high profit
        mid_tighten_at_profit: float = 15.0, # was 12 — intermediate tightening later
        mid_tightened_distance: float = 8.0, # was 7 — wider mid-tier trail
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
            mid_tighten_at_profit: Profit threshold for mid-tier tightening.
            mid_tightened_distance: Trail distance at mid-tier (between default and tight).
        """
        self._flatten_fn = flatten_fn
        self._target_symbol = target_symbol.upper() if target_symbol else ""
        self._trail_distance = trail_distance
        self._default_trail_distance = trail_distance
        self._trail_activation = trail_activation_points
        self._min_stop_distance = min_stop_distance
        self._tighten_at_profit = tighten_at_profit
        self._tightened_distance = tightened_distance
        self._mid_tighten_at_profit = mid_tighten_at_profit
        self._mid_tightened_distance = mid_tightened_distance

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

        # Grace period — don't check stops for N seconds after activation.
        # Prevents instant stop-outs from stale entry prices or transient ticks.
        self._grace_period_sec: float = 3.0
        self._activation_time: float = 0.0

        # Minimum hold time — don't allow trail-based or partial exits
        # within this period.  Hard stop (initial stop loss) always triggers.
        self._min_hold_sec: float = 60.0
        self._initial_stop_price: float = 0.0

        # Partial profit taking
        self._partial_taken: bool = False
        self._partial_target: float = 0.0
        self._partial_fn: Optional[PartialFn] = None
        self._partial_quantity: int = 1
        self._breakeven_offset: float = 1.0

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
        is_eth: bool = False,
        eth_trail_distance: float = 5.0,
        eth_trail_activation: float = 5.0,
        eth_mid_tighten_at_profit: float = 4.0,
        eth_mid_tightened_distance: float = 4.0,
        eth_tighten_at_profit: float = 8.0,
        eth_tightened_distance: float = 3.0,
        partial_target_points: float = 0.0,
        partial_fn: Optional[PartialFn] = None,
        partial_quantity: int = 1,
        breakeven_offset: float = 1.0,
    ) -> None:
        """Activate monitoring for a new position.

        Args:
            side: "long" or "short"
            entry_price: Entry price of the position.
            stop_price: Initial stop-loss price.
            take_profit_price: Take-profit price (0 = no TP).
            trail_distance: Override trail distance (0 = use default or ATR-based).
            atr: Current ATR — if provided, trail distance = max(2*ATR, 5.0).
            is_eth: True during Extended Trading Hours (Asian/London) — uses
                    tighter trail params since ranges are smaller.
            eth_trail_distance: Trail distance during ETH (default 5.0).
            eth_trail_activation: Min profit before trail activates in ETH (default 2.0).
            eth_mid_tighten_at_profit: Mid-tier profit threshold in ETH.
            eth_mid_tightened_distance: Mid-tier trail distance in ETH.
            eth_tighten_at_profit: Tight-tier profit threshold in ETH.
            eth_tightened_distance: Tight-tier trail distance in ETH.
        """
        self._active = True
        self._side = side.lower()
        self._entry_price = entry_price
        self._stop_price = stop_price
        self._initial_stop_price = stop_price  # remember hard stop for min-hold bypass
        self._take_profit_price = take_profit_price
        self._best_price = entry_price
        self._trail_active = False
        self._triggered = False
        self._trigger_reason = ""
        self._ticks_processed = 0
        self._activation_time = time.monotonic()

        # Partial profit taking
        self._partial_taken = False
        self._partial_fn = partial_fn
        self._partial_quantity = partial_quantity
        self._breakeven_offset = breakeven_offset
        if partial_target_points > 0 and partial_fn is not None:
            if self._side == "long":
                self._partial_target = entry_price + partial_target_points
            else:
                self._partial_target = entry_price - partial_target_points
        else:
            self._partial_target = 0.0

        # ETH session: override trail params with tighter values
        if is_eth:
            self._trail_activation = eth_trail_activation
            self._mid_tighten_at_profit = eth_mid_tighten_at_profit
            self._mid_tightened_distance = eth_mid_tightened_distance
            self._tighten_at_profit = eth_tighten_at_profit
            self._tightened_distance = eth_tightened_distance

            if trail_distance > 0:
                self._trail_distance = trail_distance
            elif atr > 0:
                # ETH ATR-based: wider clamps to survive normal pullbacks
                self._trail_distance = round(max(6.0, min(10.0, atr * 2.5)), 1)
                self._mid_tightened_distance = round(max(4.0, min(7.0, atr * 1.8)), 1)
                self._tightened_distance = round(max(3.0, min(5.0, atr * 1.3)), 1)
            else:
                self._trail_distance = eth_trail_distance
        else:
            # RTH: restore default activation threshold
            self._trail_activation = self._trail_activation  # keep constructor default

            if trail_distance > 0:
                self._trail_distance = trail_distance
            elif atr > 0:
                # ATR-based trail: 1.5x ATR, clamped between 8 and 10 points
                # Tighter cap locks in profit while giving room for pullbacks.
                # At 10pt activation + 10pt trail: stop at breakeven when trail starts.
                # At 10pt activation + 8pt trail: 2pts locked in at trail start.
                self._trail_distance = round(max(8.0, min(10.0, atr * 1.5)), 1)
                self._mid_tightened_distance = round(max(5.0, min(7.0, atr * 1.2)), 1)
                self._tightened_distance = round(max(4.0, min(5.0, atr * 0.8)), 1)
            else:
                self._trail_distance = self._default_trail_distance

        logger.info(
            "tick_stop_monitor.activated",
            side=side,
            entry=entry_price,
            stop=stop_price,
            take_profit=take_profit_price,
            trail_distance=self._trail_distance,
            trail_activation=self._trail_activation,
            is_eth=is_eth,
        )
        self.persist_to_disk()

    def deactivate(self) -> None:
        """Deactivate monitoring (position closed)."""
        self._active = False
        self._side = None
        self._trail_active = False
        # Reset partial state
        self._partial_taken = False
        self._partial_target = 0.0
        self._partial_fn = None
        self._partial_quantity = 1
        self._breakeven_offset = 1.0
        logger.info(
            "tick_stop_monitor.deactivated",
            ticks_processed=self._ticks_processed,
        )
        self.clear_persisted_state()

    def update_stop(self, new_stop: float, *, force: bool = False) -> None:
        """Manually update the stop price.

        Protection rule: never degrade an existing trail.  For longs the
        new stop must be *higher* (or equal) than the current stop; for
        shorts it must be *lower* (or equal).  Pass ``force=True`` to
        bypass this check (used only for the initial stop placement).
        """
        if not self._active:
            return

        old = self._stop_price

        # Protection: never move the stop to a worse level
        if not force and old > 0:
            if self._side == "long" and new_stop < old:
                logger.info(
                    "tick_stop_monitor.stop_degradation_blocked",
                    side="long",
                    current_stop=old,
                    requested_stop=new_stop,
                    msg="Kept superior existing stop (long: new < current)",
                )
                return
            if self._side == "short" and new_stop > old:
                logger.info(
                    "tick_stop_monitor.stop_degradation_blocked",
                    side="short",
                    current_stop=old,
                    requested_stop=new_stop,
                    msg="Kept superior existing stop (short: new > current)",
                )
                return

        self._stop_price = new_stop
        logger.info(
            "tick_stop_monitor.stop_updated",
            old_stop=old,
            new_stop=new_stop,
        )
        self.persist_to_disk()

    def update_take_profit(self, new_tp: float) -> None:
        """Manually update the take-profit price."""
        if self._active:
            self._take_profit_price = new_tp
            self.persist_to_disk()

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

        # Grace period — skip stop checks for first N seconds after activation.
        # Prevents instant stop-outs from stale entry prices or price slippage
        # between the LLM decision and order execution.
        elapsed = time.monotonic() - self._activation_time
        in_grace = elapsed < self._grace_period_sec

        # Update trail (always, even during grace period / min hold)
        self._update_trail(price)

        # Min-hold check: within the first N seconds, only allow hard-stop exits.
        # Trail stops, partials, and take-profits are suppressed to prevent
        # rapid exits (trades lasting 2-41 seconds).
        in_min_hold = elapsed < self._min_hold_sec

        # Check hard stop FIRST — always triggers regardless of min hold.
        # A "hard stop" is when price breaches the *initial* stop level.
        if not in_grace and self._check_hard_stop(price):
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
                hard_stop=True,
            )
            await self._execute_flatten(price)
            return

        # Everything below is suppressed during min-hold period
        if in_min_hold:
            return

        # Check partial profit (skip during grace period)
        if (
            not in_grace
            and not self._partial_taken
            and self._partial_target > 0
            and self._partial_fn is not None
            and self._check_partial(price)
        ):
            self._partial_taken = True
            await self._execute_partial(price)

        # Check trail-based stop-loss (skip during grace period)
        if not in_grace and self._check_stop(price):
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

        # Check take-profit (skip during grace period)
        if not in_grace and self._check_take_profit(price):
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

    def _check_hard_stop(self, price: float) -> bool:
        """Check if the *initial* hard stop has been hit.

        This uses ``_initial_stop_price`` (the stop set at entry) so it
        always triggers even during the min-hold period.  The trailing
        stop (``_stop_price``) may have moved; this method ignores that.
        """
        if self._initial_stop_price <= 0:
            return False
        if self._side == "long":
            return price <= self._initial_stop_price
        elif self._side == "short":
            return price >= self._initial_stop_price
        return False

    def _check_stop(self, price: float) -> bool:
        """Check if stop-loss has been hit (trail or initial)."""
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

    # ── Partial Profit Logic ────────────────────────────────────────────────

    def _check_partial(self, price: float) -> bool:
        """Check if price has reached the partial profit target."""
        if self._side == "long":
            return price >= self._partial_target
        elif self._side == "short":
            return price <= self._partial_target
        return False

    async def _execute_partial(self, price: float) -> None:
        """Execute partial close and move stop to breakeven."""
        try:
            await self._partial_fn(
                side=self._side,
                quantity=self._partial_quantity,
                price=price,
            )

            # Move stop to breakeven + offset
            if self._side == "long":
                new_stop = self._entry_price + self._breakeven_offset
            else:
                new_stop = self._entry_price - self._breakeven_offset

            self.update_stop(new_stop, force=True)

            logger.info(
                "tick_stop_monitor.PARTIAL_TAKEN",
                side=self._side,
                entry=self._entry_price,
                target=self._partial_target,
                price=price,
                quantity=self._partial_quantity,
                new_stop=new_stop,
            )
            self.persist_to_disk()

        except Exception as e:
            logger.error(
                "tick_stop_monitor.partial_failed",
                error=str(e),
                price=price,
                side=self._side,
            )

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

            # Dynamic tightening based on profit (3 tiers)
            best_profit = self._best_price - self._entry_price
            if best_profit >= self._tighten_at_profit:
                effective_trail = self._tightened_distance
            elif best_profit >= self._mid_tighten_at_profit:
                effective_trail = self._mid_tightened_distance
            else:
                effective_trail = self._trail_distance

            # Update stop if trailing is active
            if self._trail_active:
                ideal_stop = self._best_price - effective_trail
                # Stop only moves UP for longs (never moves backward)
                if ideal_stop > self._stop_price:
                    self._stop_price = ideal_stop
                    self.persist_to_disk()

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

            # Dynamic tightening based on profit (3 tiers)
            best_profit = self._entry_price - self._best_price
            if best_profit >= self._tighten_at_profit:
                effective_trail = self._tightened_distance
            elif best_profit >= self._mid_tighten_at_profit:
                effective_trail = self._mid_tightened_distance
            else:
                effective_trail = self._trail_distance

            # Update stop if trailing is active
            if self._trail_active:
                ideal_stop = self._best_price + effective_trail
                # Stop only moves DOWN for shorts (never moves backward)
                if ideal_stop < self._stop_price:
                    self._stop_price = ideal_stop
                    self.persist_to_disk()

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

    # ── State Persistence ─────────────────────────────────────────────────

    def save_state(self) -> dict:
        """Serialize current monitor state for persistence."""
        return {
            "active": self._active,
            "side": self._side,
            "entry_price": self._entry_price,
            "stop_price": self._stop_price,
            "take_profit_price": self._take_profit_price,
            "best_price": self._best_price,
            "trail_active": self._trail_active,
            "trail_distance": self._trail_distance,
            "trail_activation": self._trail_activation,
            "mid_tighten_at_profit": self._mid_tighten_at_profit,
            "mid_tightened_distance": self._mid_tightened_distance,
            "tighten_at_profit": self._tighten_at_profit,
            "tightened_distance": self._tightened_distance,
            "partial_taken": self._partial_taken,
            "partial_target": self._partial_target,
            "partial_quantity": self._partial_quantity,
            "breakeven_offset": self._breakeven_offset,
        }

    def load_state(self, data: dict) -> None:
        """Restore monitor state from persisted data.

        Call this on startup if broker reconciliation shows an open position.
        """
        if not data or not data.get("active"):
            return
        self._active = True
        self._side = data.get("side")
        self._entry_price = data.get("entry_price", 0.0)
        self._stop_price = data.get("stop_price", 0.0)
        self._take_profit_price = data.get("take_profit_price", 0.0)
        self._best_price = data.get("best_price", 0.0)
        self._trail_active = data.get("trail_active", False)
        self._trail_distance = data.get("trail_distance", self._default_trail_distance)
        self._trail_activation = data.get("trail_activation", self._trail_activation)
        self._mid_tighten_at_profit = data.get("mid_tighten_at_profit", self._mid_tighten_at_profit)
        self._mid_tightened_distance = data.get("mid_tightened_distance", self._mid_tightened_distance)
        self._tighten_at_profit = data.get("tighten_at_profit", self._tighten_at_profit)
        self._tightened_distance = data.get("tightened_distance", self._tightened_distance)
        self._partial_taken = data.get("partial_taken", False)
        self._partial_target = data.get("partial_target", 0.0)
        self._partial_quantity = data.get("partial_quantity", 1)
        self._breakeven_offset = data.get("breakeven_offset", 1.0)
        self._triggered = False
        self._trigger_reason = ""
        self._activation_time = time.monotonic()
        logger.info(
            "tick_stop_monitor.state_restored",
            side=self._side,
            entry=self._entry_price,
            stop=self._stop_price,
            best=self._best_price,
            trail_active=self._trail_active,
        )

    def persist_to_disk(self) -> None:
        """Write current state to disk for crash recovery."""
        try:
            _PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PERSIST_PATH.write_text(json.dumps(self.save_state(), indent=2))
        except Exception:
            logger.warning("tick_stop_monitor.persist_failed", exc_info=True)

    def load_from_disk(self) -> bool:
        """Load state from disk if file exists. Returns True if loaded."""
        try:
            if _PERSIST_PATH.exists():
                data = json.loads(_PERSIST_PATH.read_text())
                self.load_state(data)
                return self._active
        except Exception:
            logger.warning("tick_stop_monitor.load_failed", exc_info=True)
        return False

    def clear_persisted_state(self) -> None:
        """Remove persisted state file (call after position is closed)."""
        try:
            if _PERSIST_PATH.exists():
                _PERSIST_PATH.unlink()
        except Exception:
            pass
