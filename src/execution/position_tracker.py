"""Position tracker — real-time position state with WS + REST reconciliation.

Maintains the authoritative PositionState by:
1. Processing fill events from the Tradovate WebSocket (fast path)
2. Periodically reconciling with REST position/list (slow path, correctness)
3. Using an asyncio.Lock to prevent concurrent state mutations

The tracker is the single source of truth for "what position do we hold."
Every other component reads from here instead of querying Tradovate directly.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Optional

import structlog

from src.core.events import EventBus
from src.core.types import Event, EventType, PositionState, Side, TradeRecord

logger = structlog.get_logger()


class PositionTracker:
    """Tracks real-time position state with WS updates and REST reconciliation.

    Usage:
        tracker = PositionTracker(event_bus=bus, symbol="MNQM6")
        tracker.on_fill(fill_data)       # called from WS handler
        await tracker.reconcile(rest)     # periodic REST check
        pos = tracker.position            # current state or None
    """

    def __init__(
        self,
        event_bus: EventBus,
        symbol: str = "MNQM6",
        point_value: float = 2.0,
    ) -> None:
        self._bus = event_bus
        self._symbol = symbol
        self._point_value = point_value

        # Position state
        self._position: Optional[PositionState] = None
        self._lock = asyncio.Lock()

        # Trade history for the current session
        self._completed_trades: list[TradeRecord] = []

        # Stats
        self._fills_processed: int = 0
        self._reconciliations: int = 0
        self._mismatches: int = 0
        self._last_reconcile: Optional[datetime] = None

    async def on_fill(self, fill_data: dict[str, Any]) -> None:
        """Process a fill event from the Tradovate WebSocket.

        This is the fast path — updates position in real-time as fills arrive.

        Args:
            fill_data: Fill event from Tradovate WS, containing at minimum:
                - action: "Buy" or "Sell"
                - qty: number of contracts filled
                - price: fill price
                - orderId: the order that was filled
        """
        async with self._lock:
            action = fill_data.get("action", "")
            qty = fill_data.get("qty", 0)
            price = fill_data.get("price", 0.0)

            if not action or qty <= 0 or price <= 0:
                logger.warning("position_tracker.invalid_fill", data=fill_data)
                return

            fill_side = Side.LONG if action == "Buy" else Side.SHORT
            self._fills_processed += 1

            if self._position is None:
                # Opening a new position
                self._position = PositionState(
                    side=fill_side,
                    quantity=qty,
                    avg_entry=price,
                    entry_time=datetime.now(tz=UTC),
                )

                logger.info(
                    "position_tracker.opened",
                    side=fill_side.value,
                    qty=qty,
                    entry=price,
                )

            elif fill_side == self._position.side:
                # Adding to position — compute new weighted average
                total_qty = self._position.quantity + qty
                new_avg = (
                    (self._position.avg_entry * self._position.quantity + price * qty)
                    / total_qty
                )
                self._position = self._position.model_copy(update={
                    "quantity": total_qty,
                    "avg_entry": round(new_avg, 4),
                    "adds_count": self._position.adds_count + 1,
                })

                logger.info(
                    "position_tracker.added",
                    qty=qty,
                    total_qty=total_qty,
                    new_avg=round(new_avg, 4),
                )

            else:
                # Reducing or closing position
                remaining = self._position.quantity - qty

                if remaining <= 0:
                    # Position closed
                    pnl = self._compute_pnl(
                        self._position.side,
                        self._position.avg_entry,
                        price,
                        self._position.quantity,
                    )

                    trade = TradeRecord(
                        timestamp_entry=self._position.entry_time,
                        timestamp_exit=datetime.now(tz=UTC),
                        side=self._position.side,
                        entry_quantity=self._position.quantity,
                        exit_quantity=self._position.quantity,
                        entry_price=self._position.avg_entry,
                        exit_price=price,
                        stop_price=self._position.stop_price,
                        pnl=round(pnl, 2),
                        max_favorable_excursion=self._position.max_favorable,
                        max_adverse_excursion=self._position.max_adverse,
                        adds=self._position.adds_count,
                    )
                    self._completed_trades.append(trade)

                    logger.info(
                        "position_tracker.closed",
                        pnl=round(pnl, 2),
                        entry=self._position.avg_entry,
                        exit=price,
                    )

                    self._position = None

                    self._bus.publish_nowait(Event(
                        type=EventType.POSITION_CHANGED,
                        data={
                            "action": "closed",
                            "pnl": round(pnl, 2),
                            "trade_id": trade.id,
                        },
                    ))
                    return

                else:
                    # Partial close
                    pnl_partial = self._compute_pnl(
                        self._position.side,
                        self._position.avg_entry,
                        price,
                        qty,
                    )

                    self._position = self._position.model_copy(update={
                        "quantity": remaining,
                    })

                    logger.info(
                        "position_tracker.partial_close",
                        closed_qty=qty,
                        remaining=remaining,
                        partial_pnl=round(pnl_partial, 2),
                    )

            self._bus.publish_nowait(Event(
                type=EventType.POSITION_CHANGED,
                data={
                    "side": self._position.side.value if self._position else None,
                    "quantity": self._position.quantity if self._position else 0,
                },
            ))

    def update_unrealized(self, current_price: float) -> None:
        """Update unrealized P&L and excursion tracking.

        Called on each price tick to keep position state current.

        Args:
            current_price: Latest market price.
        """
        if self._position is None:
            return

        pnl = self._compute_pnl(
            self._position.side,
            self._position.avg_entry,
            current_price,
            self._position.quantity,
        )

        updates: dict[str, Any] = {"unrealized_pnl": round(pnl, 2)}

        if pnl > self._position.max_favorable:
            updates["max_favorable"] = round(pnl, 2)
        if pnl < -abs(self._position.max_adverse):
            updates["max_adverse"] = round(abs(pnl), 2)

        # Update hold time
        elapsed = (datetime.now(tz=UTC) - self._position.entry_time).total_seconds()
        updates["time_in_trade_sec"] = int(elapsed)

        self._position = self._position.model_copy(update=updates)

    def update_stop_price(self, stop_price: float) -> None:
        """Update the stop price on the tracked position.

        Args:
            stop_price: New stop price.
        """
        if self._position is not None:
            self._position = self._position.model_copy(
                update={"stop_price": stop_price}
            )

    async def reconcile(self, positions_data: list[dict[str, Any]]) -> None:
        """Reconcile internal state with REST position data.

        Args:
            positions_data: Result from TradovateREST.get_positions().
        """
        async with self._lock:
            self._reconciliations += 1
            self._last_reconcile = datetime.now(tz=UTC)

            # Find our symbol in the positions
            our_pos = None
            for p in positions_data:
                contract_name = p.get("contractName", "")
                if self._symbol in contract_name:
                    our_pos = p
                    break

            net_pos = our_pos.get("netPos", 0) if our_pos else 0
            net_price = our_pos.get("netPrice", 0.0) if our_pos else 0.0

            if net_pos == 0 and self._position is not None:
                # We think we have a position but REST says flat
                logger.warning(
                    "position_tracker.reconcile_mismatch",
                    internal="has_position",
                    rest="flat",
                )
                self._mismatches += 1
                self._position = None

            elif net_pos != 0 and self._position is None:
                # REST shows a position but we don't track one
                side = Side.LONG if net_pos > 0 else Side.SHORT
                self._position = PositionState(
                    side=side,
                    quantity=abs(net_pos),
                    avg_entry=net_price,
                    entry_time=datetime.now(tz=UTC),
                )
                self._mismatches += 1
                logger.warning(
                    "position_tracker.reconcile_mismatch",
                    internal="flat",
                    rest=f"{side.value} x{abs(net_pos)}",
                )

            elif net_pos != 0 and self._position is not None:
                # Both agree we have a position — check quantities match
                rest_qty = abs(net_pos)
                if rest_qty != self._position.quantity:
                    logger.warning(
                        "position_tracker.reconcile_qty_mismatch",
                        internal_qty=self._position.quantity,
                        rest_qty=rest_qty,
                    )
                    self._mismatches += 1
                    self._position = self._position.model_copy(
                        update={"quantity": rest_qty}
                    )

            logger.debug(
                "position_tracker.reconciled",
                reconciliation_count=self._reconciliations,
                mismatches=self._mismatches,
            )

    def _compute_pnl(
        self,
        side: Side,
        entry: float,
        exit_price: float,
        quantity: int,
    ) -> float:
        """Compute P&L in dollars for a given trade."""
        if side == Side.LONG:
            points = exit_price - entry
        else:
            points = entry - exit_price
        return points * quantity * self._point_value

    # ── Session Management ─────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset for a new trading session."""
        self._position = None
        self._completed_trades.clear()
        self._fills_processed = 0
        self._reconciliations = 0
        self._mismatches = 0
        self._last_reconcile = None

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def position(self) -> Optional[PositionState]:
        return self._position

    @property
    def is_flat(self) -> bool:
        return self._position is None

    @property
    def in_position(self) -> bool:
        return self._position is not None

    @property
    def completed_trades(self) -> list[TradeRecord]:
        return self._completed_trades

    @property
    def last_trade(self) -> Optional[TradeRecord]:
        return self._completed_trades[-1] if self._completed_trades else None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "in_position": self.in_position,
            "side": self._position.side.value if self._position else None,
            "quantity": self._position.quantity if self._position else 0,
            "unrealized_pnl": (
                self._position.unrealized_pnl if self._position else 0.0
            ),
            "fills_processed": self._fills_processed,
            "reconciliations": self._reconciliations,
            "mismatches": self._mismatches,
            "completed_trades": len(self._completed_trades),
        }
