"""QuantLynk order manager — bridges LLM actions to QuantLynk webhook calls.

Drop-in replacement for OrderManager when using QuantLynk for execution
instead of direct Tradovate API. Same execute() interface, same event
publishing, same simulation mode support.

Key differences from Tradovate OrderManager:
- No bracket orders (QuantLynk sends market orders only)
- Stops are managed locally by our TrailManager + Databento price feed
  (we monitor price and send flatten when stop is hit)
- No order ID tracking (QuantLynk is fire-and-forget via webhook)
- Flatten sends a "flatten" action that closes all positions on the account
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import structlog

from src.core.events import EventBus
from src.core.types import (
    ActionType,
    Event,
    EventType,
    LLMAction,
    PositionState,
    Side,
)
from src.execution.quantlynk_client import QuantLynkClient

# Avoid circular import — PositionTracker imported at type-check time only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.execution.position_tracker import PositionTracker

logger = structlog.get_logger()


class QuantLynkOrderManager:
    """Translates LLM actions into QuantLynk webhook calls.

    Same interface as OrderManager so the orchestrator doesn't care
    which execution backend is wired in.

    Usage:
        om = QuantLynkOrderManager(client=ql_client, event_bus=bus)
        result = await om.execute(action, position, last_price)
    """

    def __init__(
        self,
        client: QuantLynkClient,
        event_bus: EventBus,
        symbol: str = "MNQM6",
        default_stop_distance: float = 12.0,
        point_value: float = 2.0,
        position_tracker: Optional[Any] = None,
    ) -> None:
        self._client = client
        self._bus = event_bus
        self._symbol = symbol
        self._default_stop_distance = default_stop_distance
        self._point_value = point_value
        self._position_tracker = position_tracker

        # Internal position tracking (since we don't get fills back from QuantLynk)
        self._current_stop_price: Optional[float] = None

        # Simulation mode (no live orders — circuit breaker override)
        self._simulation_mode: bool = False

        # Stats
        self._orders_placed: int = 0
        self._orders_filled: int = 0  # assumed filled on successful webhook
        self._orders_rejected: int = 0

    def set_simulation_mode(self, enabled: bool) -> None:
        """Enable/disable simulation mode (no live orders sent)."""
        self._simulation_mode = enabled
        logger.info("quantlynk_om.simulation_mode", enabled=enabled)

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    async def execute(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
        key_levels: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        """Execute an LLM action via QuantLynk webhook.

        Args:
            action: The LLM's decided action.
            position: Current position state (None if flat).
            last_price: Current market price.
            key_levels: Optional key levels (used for stop placement).

        Returns:
            Dict with execution result details.
        """
        start = asyncio.get_event_loop().time()

        # Simulation mode: log but don't execute entry/add orders
        if self._simulation_mode and action.action in (
            ActionType.ENTER, ActionType.ADD
        ):
            logger.info(
                "quantlynk_om.sim_mode_blocked",
                action=action.action.value,
                side=action.side.value if action.side else None,
                quantity=action.quantity,
            )
            return {
                "status": "simulated",
                "action": action.action.value,
                "message": "Simulation mode — order not sent",
            }

        try:
            if action.action == ActionType.ENTER:
                result = await self._execute_enter(action, position, last_price)
            elif action.action == ActionType.ADD:
                result = await self._execute_add(action, position, last_price)
            elif action.action == ActionType.SCALE_OUT:
                result = await self._execute_scale_out(action, position, last_price)
            elif action.action == ActionType.MOVE_STOP:
                result = await self._execute_move_stop(action, position)
            elif action.action == ActionType.FLATTEN:
                result = await self._execute_flatten(last_price)
            elif action.action == ActionType.STOP_TRADING:
                result = await self._execute_flatten(last_price)
                result["action"] = "STOP_TRADING"
            elif action.action == ActionType.DO_NOTHING:
                result = {"status": "no_action", "action": "DO_NOTHING"}
            else:
                logger.warning("quantlynk_om.unknown_action", action=action.action)
                result = {"status": "unknown_action", "action": str(action.action)}

            elapsed = int((asyncio.get_event_loop().time() - start) * 1000)
            result["latency_ms"] = elapsed
            return result

        except Exception as e:
            self._orders_rejected += 1
            logger.error(
                "quantlynk_om.error",
                error=str(e),
                action=action.action.value,
            )
            self._bus.publish_nowait(Event(
                type=EventType.ORDER_REJECTED,
                data={"action": action.action.value, "error": str(e)},
            ))
            return {"status": "error", "error": str(e)}

    # ── Position Tracker Sync ────────────────────────────────────────────────

    async def _sync_fill(self, action: str, qty: int, price: float) -> None:
        """Notify the PositionTracker of an assumed fill.

        Since QuantLynk is fire-and-forget (market orders assumed filled),
        we immediately update the position tracker as if the fill occurred.
        """
        if self._position_tracker is None:
            return

        fill_data = {
            "action": action,  # "Buy" or "Sell"
            "qty": qty,
            "price": price,
        }
        try:
            await self._position_tracker.on_fill(fill_data)
        except Exception:
            logger.warning("quantlynk_om.fill_sync_failed", exc_info=True)

    # ── ENTER ──────────────────────────────────────────────────────────────────

    async def _execute_enter(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
    ) -> dict[str, Any]:
        """Enter a new position via QuantLynk."""
        if position is not None:
            logger.warning("quantlynk_om.enter_while_in_position")
            return {"status": "skipped", "reason": "already_in_position"}

        side = action.side
        if side is None:
            return {"status": "skipped", "reason": "no_side_specified"}

        quantity = action.quantity or 1

        # Send buy or sell to QuantLynk
        if side == Side.LONG:
            result = await self._client.buy(quantity=quantity, price=last_price)
        else:
            result = await self._client.sell(quantity=quantity, price=last_price)

        self._orders_placed += 1
        self._orders_filled += 1  # assume filled (market order)

        # Calculate stop price (managed locally, not by QuantLynk)
        stop_distance = action.stop_distance or self._default_stop_distance
        if side == Side.LONG:
            self._current_stop_price = round(last_price - stop_distance, 2)
        else:
            self._current_stop_price = round(last_price + stop_distance, 2)

        # Sync fill to position tracker
        fill_action = "Buy" if side == Side.LONG else "Sell"
        await self._sync_fill(fill_action, quantity, last_price)

        logger.info(
            "quantlynk_om.enter",
            side=side.value,
            qty=quantity,
            price=last_price,
            stop=self._current_stop_price,
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={
                "action": "ENTER",
                "side": side.value,
                "quantity": quantity,
                "price": last_price,
                "stop_price": self._current_stop_price,
            },
        ))

        return {
            "status": "placed",
            "action": "ENTER",
            "side": side.value,
            "quantity": quantity,
            "price": last_price,
            "stop_price": self._current_stop_price,
        }

    # ── ADD ────────────────────────────────────────────────────────────────────

    async def _execute_add(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
    ) -> dict[str, Any]:
        """Add to an existing position via QuantLynk."""
        if position is None:
            return {"status": "skipped", "reason": "no_position_to_add_to"}

        quantity = action.quantity or 1

        if position.side == Side.LONG:
            result = await self._client.buy(quantity=quantity, price=last_price)
        else:
            result = await self._client.sell(quantity=quantity, price=last_price)

        self._orders_placed += 1
        self._orders_filled += 1

        # Sync fill to position tracker (same side = add)
        fill_action = "Buy" if position.side == Side.LONG else "Sell"
        await self._sync_fill(fill_action, quantity, last_price)

        logger.info(
            "quantlynk_om.add",
            side=position.side.value,
            qty=quantity,
            price=last_price,
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={
                "action": "ADD",
                "side": position.side.value,
                "quantity": quantity,
                "price": last_price,
            },
        ))

        return {
            "status": "placed",
            "action": "ADD",
            "side": position.side.value,
            "quantity": quantity,
            "price": last_price,
        }

    # ── SCALE_OUT ──────────────────────────────────────────────────────────────

    async def _execute_scale_out(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
    ) -> dict[str, Any]:
        """Scale out (partial exit) via QuantLynk."""
        if position is None:
            return {"status": "skipped", "reason": "no_position"}

        quantity = action.quantity or 1

        # Opposite side to close partial
        if position.side == Side.LONG:
            result = await self._client.sell(quantity=quantity, price=last_price)
        else:
            result = await self._client.buy(quantity=quantity, price=last_price)

        self._orders_placed += 1
        self._orders_filled += 1

        # Sync fill to position tracker (opposite side = reduce/close)
        fill_action = "Sell" if position.side == Side.LONG else "Buy"
        await self._sync_fill(fill_action, quantity, last_price)

        logger.info(
            "quantlynk_om.scale_out",
            qty=quantity,
            price=last_price,
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={
                "action": "SCALE_OUT",
                "quantity": quantity,
                "price": last_price,
            },
        ))

        return {
            "status": "placed",
            "action": "SCALE_OUT",
            "quantity": quantity,
            "price": last_price,
        }

    # ── MOVE_STOP ──────────────────────────────────────────────────────────────

    async def _execute_move_stop(
        self,
        action: LLMAction,
        position: Optional[PositionState],
    ) -> dict[str, Any]:
        """Move stop price — managed locally, no webhook needed.

        QuantLynk doesn't support native stop orders. Our TrailManager
        monitors price via Databento and triggers a flatten webhook
        when the stop level is hit.
        """
        if position is None:
            return {"status": "skipped", "reason": "no_position"}

        new_stop = action.new_stop_price
        if new_stop is None or new_stop <= 0:
            return {"status": "skipped", "reason": "no_stop_price_specified"}

        old_stop = self._current_stop_price
        self._current_stop_price = new_stop

        logger.info(
            "quantlynk_om.move_stop",
            old_stop=old_stop,
            new_stop=new_stop,
        )

        return {
            "status": "updated",
            "action": "MOVE_STOP",
            "old_stop": old_stop,
            "new_stop": new_stop,
        }

    # ── FLATTEN ────────────────────────────────────────────────────────────────

    async def _execute_flatten(self, last_price: float) -> dict[str, Any]:
        """Flatten all positions via QuantLynk."""
        # Get position info BEFORE flatten for fill sync
        pos = self._position_tracker.position if self._position_tracker else None

        result = await self._client.flatten(price=last_price)

        self._orders_placed += 1
        self._orders_filled += 1
        self._current_stop_price = None

        # Sync fill to position tracker (opposite side closes position)
        if pos is not None:
            fill_action = "Sell" if pos.side == Side.LONG else "Buy"
            await self._sync_fill(fill_action, pos.quantity, last_price)

        logger.info("quantlynk_om.flatten", price=last_price)

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={"action": "FLATTEN", "price": last_price},
        ))

        return {
            "status": "placed",
            "action": "FLATTEN",
            "price": last_price,
        }

    # ── Stop Check (called by TrailManager or orchestrator) ──────────────────

    def check_stop_hit(self, current_price: float, side: Side) -> bool:
        """Check if current price has hit our locally-managed stop.

        Returns True if the stop has been triggered.
        Called every tick/state update by the orchestrator or TrailManager.
        """
        if self._current_stop_price is None:
            return False

        if side == Side.LONG:
            return current_price <= self._current_stop_price
        else:
            return current_price >= self._current_stop_price

    @property
    def current_stop_price(self) -> Optional[float]:
        return self._current_stop_price

    @current_stop_price.setter
    def current_stop_price(self, price: Optional[float]) -> None:
        self._current_stop_price = price

    # ── Stats ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_rejected": self._orders_rejected,
            "current_stop": self._current_stop_price,
            "simulation_mode": self._simulation_mode,
            "quantlynk_stats": self._client.stats,
        }
