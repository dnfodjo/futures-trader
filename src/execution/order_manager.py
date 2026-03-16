"""Order manager — bridges LLM actions to Tradovate order calls.

Translates ActionType decisions from the reasoner into concrete
Tradovate REST API calls. Handles:
- ENTER: place bracket order (market entry + stop)
- ADD: place additional bracket at current price
- SCALE_OUT: cancel partial stop, place market exit for partial qty
- MOVE_STOP: modify existing stop order price
- FLATTEN: cancel all working orders, liquidate position
- STOP_TRADING: flatten then signal session end

All orders include isAutomated=true per CME Rule 536-B (handled by TradovateREST).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.events import EventBus
from src.core.exceptions import (
    KillSwitchTriggered,
    OrderModifyFailedError,
    OrderRejectedError,
    TradovateConnectionError,
)
from src.core.types import (
    ActionType,
    Event,
    EventType,
    LLMAction,
    OrderState,
    PositionState,
    Side,
)
from src.execution.tradovate_rest import TradovateREST

logger = structlog.get_logger()

# ── Stop Hunt Avoidance ────────────────────────────────────────────────────

# Round number multiples to avoid placing stops at (MNQ moves in 0.25 increments)
_ROUND_NUMBER_MULTIPLES = (50.0, 100.0, 25.0)
_STOP_HUNT_OFFSET = 2.75  # points to offset away from obvious levels


def _avoid_stop_hunt(
    stop_price: float,
    side: Side,
    key_levels: list[float] | None = None,
) -> float:
    """Adjust stop price to avoid obvious levels where stop hunts cluster.

    - Avoids round numbers (multiples of 25, 50, 100)
    - Avoids key structural levels (PDH, PDL, session H/L, etc.)
    - Offsets by ~2.75 points away from the level (past it for a long,
      before it for a short, so the stop is harder to hunt)

    Args:
        stop_price: Initial calculated stop price.
        side: LONG or SHORT — determines offset direction.
        key_levels: Optional list of important price levels to avoid.

    Returns:
        Adjusted stop price rounded to nearest 0.25.
    """
    adjusted = stop_price

    # Check round numbers
    for mult in _ROUND_NUMBER_MULTIPLES:
        remainder = stop_price % mult
        if remainder < 1.0 or (mult - remainder) < 1.0:
            # Too close to a round number — offset
            if side == Side.LONG:
                adjusted = stop_price - _STOP_HUNT_OFFSET
            else:
                adjusted = stop_price + _STOP_HUNT_OFFSET
            break

    # Check key levels
    if key_levels:
        for level in key_levels:
            if abs(adjusted - level) < 2.0:  # within 2 points of a key level
                if side == Side.LONG:
                    adjusted = level - _STOP_HUNT_OFFSET
                else:
                    adjusted = level + _STOP_HUNT_OFFSET
                break

    # Round to nearest 0.25 (MNQ tick size)
    adjusted = round(adjusted * 4) / 4
    return adjusted


class OrderManager:
    """Translates LLM actions into Tradovate orders.

    Usage:
        om = OrderManager(rest=rest, event_bus=bus, account_id=12345)
        result = await om.execute(action, position, last_price)
    """

    def __init__(
        self,
        rest: TradovateREST,
        event_bus: EventBus,
        account_id: int,
        symbol: str = "MNQM6",
        default_stop_distance: float = 10.0,
        point_value: float = 2.0,
        max_retries: int = 2,
        retry_delay_sec: float = 0.5,
    ) -> None:
        self._rest = rest
        self._bus = event_bus
        self._account_id = account_id
        self._symbol = symbol
        self._default_stop_distance = default_stop_distance
        self._point_value = point_value
        self._max_retries = max_retries
        self._retry_delay_sec = retry_delay_sec

        # Order tracking
        self._working_orders: dict[int, OrderState] = {}
        self._stop_order_id: Optional[int] = None  # primary stop
        self._entry_order_id: Optional[int] = None

        # Simulation mode (no live orders — circuit breaker override)
        self._simulation_mode: bool = False

        # Stats
        self._orders_placed: int = 0
        self._orders_filled: int = 0
        self._orders_rejected: int = 0
        self._orders_modified: int = 0
        self._retries_attempted: int = 0
        self._total_latency_ms: int = 0

    async def _retry_call(self, coro_factory: Any) -> Any:
        """Retry a REST call on transient connection errors.

        Args:
            coro_factory: A zero-arg callable that returns a new coroutine
                          each time (e.g., lambda: self._rest.place_market_order(...)).

        Returns:
            The result from the successful call.

        Raises:
            The original exception if all retries fail.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1 + self._max_retries):
            try:
                return await coro_factory()
            except TradovateConnectionError as e:
                last_error = e
                if attempt < self._max_retries:
                    self._retries_attempted += 1
                    delay = self._retry_delay_sec * (2 ** attempt)
                    logger.warning(
                        "order_manager.retry",
                        attempt=attempt + 1,
                        max_retries=self._max_retries,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
        raise last_error  # type: ignore[misc]

    def set_simulation_mode(self, enabled: bool) -> None:
        """Enable/disable simulation mode (no live orders sent).

        When enabled, execute() logs what it WOULD do but doesn't place
        real orders. Used by circuit breakers after 3+ consecutive red days.
        """
        self._simulation_mode = enabled
        logger.info("order_manager.simulation_mode", enabled=enabled)

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    async def execute(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
        key_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Execute an LLM action by placing appropriate orders.

        Args:
            action: The LLM's decided action.
            position: Current position state (None if flat).
            last_price: Current market price for stop calculations.
            key_levels: Optional list of key price levels for stop hunt avoidance.

        Returns:
            Dict with execution result details.
        """
        start = asyncio.get_event_loop().time()

        # Simulation mode: log but don't execute entry/add orders
        if self._simulation_mode and action.action in (
            ActionType.ENTER, ActionType.ADD
        ):
            logger.info(
                "order_manager.sim_mode_blocked",
                action=action.action.value,
                side=action.side.value if action.side else None,
                quantity=action.quantity,
                reasoning=action.reasoning[:60] if action.reasoning else "",
            )
            return {
                "status": "simulated",
                "action": action.action.value,
                "message": "Simulation mode — order not sent",
            }

        try:
            if action.action == ActionType.ENTER:
                result = await self._execute_enter(action, position, last_price, key_levels)
            elif action.action == ActionType.ADD:
                result = await self._execute_add(action, position, last_price, key_levels)
            elif action.action == ActionType.SCALE_OUT:
                result = await self._execute_scale_out(action, position)
            elif action.action == ActionType.MOVE_STOP:
                result = await self._execute_move_stop(action, position)
            elif action.action == ActionType.FLATTEN:
                result = await self._execute_flatten(action)
            elif action.action == ActionType.STOP_TRADING:
                result = await self._execute_stop_trading(action)
            elif action.action == ActionType.DO_NOTHING:
                result = {"status": "no_action", "action": "DO_NOTHING"}
            else:
                logger.warning("order_manager.unknown_action", action=action.action)
                result = {"status": "unknown_action", "action": str(action.action)}

            elapsed = int((asyncio.get_event_loop().time() - start) * 1000)
            self._total_latency_ms += elapsed
            result["latency_ms"] = elapsed

            return result

        except OrderRejectedError as e:
            self._orders_rejected += 1
            logger.error("order_manager.rejected", error=str(e), action=action.action.value)
            self._bus.publish_nowait(Event(
                type=EventType.ORDER_REJECTED,
                data={"action": action.action.value, "error": str(e)},
            ))
            return {"status": "rejected", "error": str(e)}

        except KillSwitchTriggered:
            raise  # never swallow kill switch

        except Exception as e:
            logger.error("order_manager.error", error=str(e), action=action.action.value)
            return {"status": "error", "error": str(e)}

    # ── ENTER ──────────────────────────────────────────────────────────────────

    async def _execute_enter(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
        key_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Place a bracket order for a new entry."""
        if position is not None:
            logger.warning("order_manager.enter_while_in_position")
            return {"status": "skipped", "reason": "already_in_position"}

        side = action.side
        if side is None:
            return {"status": "skipped", "reason": "no_side_specified"}

        quantity = action.quantity or 1
        trado_action = "Buy" if side == Side.LONG else "Sell"
        stop_distance = action.stop_distance or self._default_stop_distance

        if side == Side.LONG:
            raw_stop = round(last_price - stop_distance, 2)
        else:
            raw_stop = round(last_price + stop_distance, 2)

        # Apply stop hunt avoidance
        stop_price = _avoid_stop_hunt(raw_stop, side, key_levels)

        result = await self._retry_call(lambda: self._rest.place_bracket_order(
            symbol=self._symbol,
            action=trado_action,
            quantity=quantity,
            stop_price=stop_price,
        ))

        order_id = result.get("orderId", 0)
        self._entry_order_id = order_id
        self._orders_placed += 1

        # Track the stop order if bracket returned it
        oso_order_id = result.get("osoOrderId")
        if oso_order_id:
            self._stop_order_id = oso_order_id

        logger.info(
            "order_manager.enter",
            side=side.value,
            qty=quantity,
            stop=stop_price,
            order_id=order_id,
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={
                "action": "ENTER",
                "side": side.value,
                "quantity": quantity,
                "stop_price": stop_price,
                "order_id": order_id,
            },
        ))

        return {
            "status": "placed",
            "action": "ENTER",
            "side": side.value,
            "quantity": quantity,
            "stop_price": stop_price,
            "order_id": order_id,
        }

    # ── ADD ────────────────────────────────────────────────────────────────────

    async def _execute_add(
        self,
        action: LLMAction,
        position: Optional[PositionState],
        last_price: float,
        key_levels: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add to an existing position."""
        if position is None:
            return {"status": "skipped", "reason": "no_position_to_add_to"}

        quantity = action.quantity or 1
        trado_action = "Buy" if position.side == Side.LONG else "Sell"
        stop_distance = action.stop_distance or self._default_stop_distance

        if position.side == Side.LONG:
            raw_stop = round(last_price - stop_distance, 2)
        else:
            raw_stop = round(last_price + stop_distance, 2)

        # Apply stop hunt avoidance
        stop_price = _avoid_stop_hunt(raw_stop, position.side, key_levels)

        result = await self._retry_call(lambda: self._rest.place_bracket_order(
            symbol=self._symbol,
            action=trado_action,
            quantity=quantity,
            stop_price=stop_price,
        ))

        order_id = result.get("orderId", 0)
        self._orders_placed += 1

        logger.info(
            "order_manager.add",
            side=position.side.value,
            qty=quantity,
            stop=stop_price,
            order_id=order_id,
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={
                "action": "ADD",
                "side": position.side.value,
                "quantity": quantity,
                "stop_price": stop_price,
                "order_id": order_id,
            },
        ))

        return {
            "status": "placed",
            "action": "ADD",
            "side": position.side.value,
            "quantity": quantity,
            "stop_price": stop_price,
            "order_id": order_id,
        }

    # ── SCALE_OUT ──────────────────────────────────────────────────────────────

    async def _execute_scale_out(
        self,
        action: LLMAction,
        position: Optional[PositionState],
    ) -> dict[str, Any]:
        """Scale out of a portion of the position."""
        if position is None:
            return {"status": "skipped", "reason": "no_position"}

        quantity = action.quantity or 1
        if quantity >= position.quantity:
            return await self._execute_flatten(action)

        # Place market order on the opposite side to reduce position
        close_action = "Sell" if position.side == Side.LONG else "Buy"

        result = await self._retry_call(lambda: self._rest.place_market_order(
            symbol=self._symbol,
            action=close_action,
            quantity=quantity,
        ))

        order_id = result.get("orderId", 0)
        self._orders_placed += 1

        logger.info(
            "order_manager.scale_out",
            qty=quantity,
            remaining=position.quantity - quantity,
            order_id=order_id,
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_PLACED,
            data={
                "action": "SCALE_OUT",
                "quantity": quantity,
                "order_id": order_id,
            },
        ))

        return {
            "status": "placed",
            "action": "SCALE_OUT",
            "quantity": quantity,
            "remaining": position.quantity - quantity,
            "order_id": order_id,
        }

    # ── MOVE_STOP ──────────────────────────────────────────────────────────────

    async def _execute_move_stop(
        self,
        action: LLMAction,
        position: Optional[PositionState],
    ) -> dict[str, Any]:
        """Move the stop price on the existing stop order."""
        if position is None:
            return {"status": "skipped", "reason": "no_position"}

        new_stop = action.new_stop_price
        if new_stop is None:
            return {"status": "skipped", "reason": "no_stop_price_specified"}

        if self._stop_order_id is None:
            return {"status": "skipped", "reason": "no_stop_order_tracked"}

        try:
            result = await self._retry_call(lambda: self._rest.modify_order(
                order_id=self._stop_order_id,
                quantity=position.quantity,
                order_type="Stop",
                stop_price=new_stop,
            ))
            self._orders_modified += 1

            logger.info(
                "order_manager.move_stop",
                old_stop=position.stop_price,
                new_stop=new_stop,
                order_id=self._stop_order_id,
            )

            self._bus.publish_nowait(Event(
                type=EventType.ORDER_MODIFIED,
                data={
                    "action": "MOVE_STOP",
                    "old_stop": position.stop_price,
                    "new_stop": new_stop,
                    "order_id": self._stop_order_id,
                },
            ))

            return {
                "status": "modified",
                "action": "MOVE_STOP",
                "old_stop": position.stop_price,
                "new_stop": new_stop,
                "order_id": self._stop_order_id,
            }

        except OrderModifyFailedError as e:
            logger.warning("order_manager.move_stop_failed", error=str(e))
            return {"status": "failed", "action": "MOVE_STOP", "error": str(e)}

    # ── FLATTEN ────────────────────────────────────────────────────────────────

    async def _execute_flatten(self, action: LLMAction) -> dict[str, Any]:
        """Flatten the entire position immediately."""
        result = await self._retry_call(
            lambda: self._rest.liquidate_position(self._account_id)
        )

        self._stop_order_id = None
        self._entry_order_id = None
        self._working_orders.clear()
        self._orders_placed += 1

        logger.info("order_manager.flatten", reason=action.reasoning[:80])

        self._bus.publish_nowait(Event(
            type=EventType.POSITION_CHANGED,
            data={"action": "FLATTEN", "reason": action.reasoning[:120]},
        ))

        return {
            "status": "flattened",
            "action": "FLATTEN",
            "reason": action.reasoning[:120],
        }

    # ── STOP_TRADING ───────────────────────────────────────────────────────────

    async def _execute_stop_trading(self, action: LLMAction) -> dict[str, Any]:
        """Flatten and signal end of trading for the day."""
        flatten_result = await self._execute_flatten(action)

        self._bus.publish_nowait(Event(
            type=EventType.DAILY_LIMIT_HIT,
            data={"reason": action.reasoning[:120]},
        ))

        return {
            "status": "stopped",
            "action": "STOP_TRADING",
            "flatten_result": flatten_result,
            "reason": action.reasoning[:120],
        }

    # ── Order Event Handlers ───────────────────────────────────────────────────

    def on_fill(self, data: dict[str, Any]) -> None:
        """Handle a fill event from Tradovate WS."""
        order_id = data.get("orderId", 0)
        self._orders_filled += 1

        logger.info(
            "order_manager.fill",
            order_id=order_id,
            qty=data.get("qty"),
            price=data.get("price"),
        )

        self._bus.publish_nowait(Event(
            type=EventType.ORDER_FILLED,
            data=data,
        ))

    def on_order_update(self, data: dict[str, Any]) -> None:
        """Handle an order update event from Tradovate WS."""
        order_id = data.get("id", 0)
        status = data.get("ordStatus", "")

        if status == "Cancelled":
            self._working_orders.pop(order_id, None)
            if order_id == self._stop_order_id:
                self._stop_order_id = None
        elif status == "Rejected":
            self._orders_rejected += 1
            self._working_orders.pop(order_id, None)

    def update_stop_order_id(self, order_id: int) -> None:
        """Manually set the tracked stop order ID."""
        self._stop_order_id = order_id

    def clear_tracking(self) -> None:
        """Clear all order tracking state (for session reset)."""
        self._working_orders.clear()
        self._stop_order_id = None
        self._entry_order_id = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def stop_order_id(self) -> Optional[int]:
        return self._stop_order_id

    @property
    def entry_order_id(self) -> Optional[int]:
        return self._entry_order_id

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_rejected": self._orders_rejected,
            "orders_modified": self._orders_modified,
            "retries_attempted": self._retries_attempted,
            "total_latency_ms": self._total_latency_ms,
            "working_orders": len(self._working_orders),
            "has_stop": self._stop_order_id is not None,
        }
