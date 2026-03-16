"""Alert manager — routes trading events to Telegram notifications.

Subscribes to the EventBus and dispatches formatted messages through
the TelegramClient. Handles:

- Trade entries and exits → formatted trade notifications
- Guardrail blocks → formatted block alerts
- Kill switch activations → priority alerts (bypass throttle)
- System events (connection loss, daily limit) → system alerts
- Periodic heartbeat → health check every N minutes
- End-of-day summary → session report

The alert manager is the bridge between the event bus and Telegram.
Business logic lives in formatters; routing logic lives here.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import structlog

from src.core.events import EventBus
from src.core.types import (
    ActionType,
    Event,
    EventType,
    LLMAction,
    PositionState,
    SessionSummary,
    TradeRecord,
)
from src.notifications.formatters import (
    format_guardrail_block,
    format_heartbeat,
    format_risk_alert,
    format_session_summary,
    format_shutdown,
    format_startup,
    format_system_alert,
    format_trade_entry,
    format_trade_exit,
)
from src.notifications.telegram_client import TelegramClient

logger = structlog.get_logger()


class AlertManager:
    """Routes trading system events to Telegram notifications.

    Usage:
        manager = AlertManager(telegram=client, event_bus=bus)
        manager.start()   # Subscribe to events
        # ... system runs ...
        await manager.send_heartbeat(daily_pnl, position, ...)
        await manager.send_session_summary(summary)
        manager.stop()     # Unsubscribe
    """

    def __init__(
        self,
        telegram: TelegramClient,
        event_bus: EventBus,
        heartbeat_interval_min: int = 30,
        alert_on_guardrails: bool = True,
    ) -> None:
        self._telegram = telegram
        self._bus = event_bus
        self._heartbeat_interval_min = heartbeat_interval_min
        self._alert_on_guardrails = alert_on_guardrails
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._started: bool = False
        self._start_time: float = time.monotonic()

        # Stats
        self._alerts_sent: int = 0
        self._alerts_failed: int = 0

    def start(self) -> None:
        """Subscribe to event bus topics."""
        self._bus.subscribe(EventType.ORDER_FILLED, self._on_order_filled)
        self._bus.subscribe(EventType.POSITION_CHANGED, self._on_position_changed)
        self._bus.subscribe(EventType.GUARDRAIL_TRIGGERED, self._on_guardrail)
        self._bus.subscribe(EventType.KILL_SWITCH_ACTIVATED, self._on_kill_switch)
        self._bus.subscribe(EventType.CONNECTION_LOST, self._on_connection_lost)
        self._bus.subscribe(EventType.CONNECTION_RESTORED, self._on_connection_restored)
        self._bus.subscribe(EventType.DAILY_LIMIT_HIT, self._on_daily_limit)
        self._bus.subscribe(EventType.PROFIT_PRESERVATION, self._on_profit_preservation)
        self._started = True
        logger.info("alert_manager.started")

    def stop(self) -> None:
        """Unsubscribe from event bus and cancel heartbeat."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        self._started = False
        logger.info("alert_manager.stopped")

    # ── Direct Send Methods ───────────────────────────────────────────────────

    async def send_startup(
        self,
        mode: str,
        max_contracts: int,
        daily_loss_limit: float,
        symbol: str,
    ) -> None:
        """Send system startup notification."""
        text = format_startup(mode, max_contracts, daily_loss_limit, symbol)
        await self._send(text, priority=True)

    async def send_shutdown(self, reason: str, daily_pnl: float) -> None:
        """Send system shutdown notification."""
        text = format_shutdown(reason, daily_pnl)
        await self._send(text, priority=True)

    async def send_trade_entry(
        self,
        action: LLMAction,
        fill_price: float,
        position_qty: int,
        daily_pnl: float = 0.0,
    ) -> None:
        """Send trade entry notification."""
        text = format_trade_entry(action, fill_price, position_qty, daily_pnl)
        await self._send(text)

    async def send_trade_exit(
        self,
        trade: TradeRecord,
        daily_pnl: float,
        winners: int,
        losers: int,
    ) -> None:
        """Send trade exit notification."""
        text = format_trade_exit(trade, daily_pnl, winners, losers)
        await self._send(text)

    async def send_heartbeat(
        self,
        daily_pnl: float,
        position: Optional[PositionState],
        trades_today: int,
        winners: int,
        losers: int,
    ) -> None:
        """Send a periodic heartbeat notification."""
        uptime_min = (time.monotonic() - self._start_time) / 60.0
        text = format_heartbeat(
            daily_pnl=daily_pnl,
            position=position,
            system_uptime_min=uptime_min,
            trades_today=trades_today,
            winners=winners,
            losers=losers,
        )
        await self._send(text)

    async def send_session_summary(self, summary: SessionSummary) -> None:
        """Send end-of-day session summary."""
        text = format_session_summary(summary)
        await self._send(text, priority=True)

    async def send_risk_alert(
        self,
        alert_type: str,
        current_value: float,
        limit_value: float,
        details: str = "",
    ) -> None:
        """Send a risk alert notification."""
        text = format_risk_alert(alert_type, current_value, limit_value, details)
        await self._send(text, priority=True)

    async def send_system_alert(self, message: str) -> None:
        """Send a system-level alert (circuit breaker, kill switch, etc.)."""
        text = f"🚨 SYSTEM ALERT\n{message}"
        await self._send(text, priority=True)

    # ── Heartbeat Loop ────────────────────────────────────────────────────────

    def start_heartbeat(
        self,
        get_state_fn: Any,
    ) -> None:
        """Start the periodic heartbeat loop.

        Args:
            get_state_fn: Async callable that returns dict with keys:
                daily_pnl, position, trades_today, winners, losers
        """
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(get_state_fn)
        )

    async def _heartbeat_loop(self, get_state_fn: Any) -> None:
        """Periodic heartbeat sender."""
        interval_sec = self._heartbeat_interval_min * 60
        try:
            while True:
                await asyncio.sleep(interval_sec)
                try:
                    state = await get_state_fn()
                    await self.send_heartbeat(
                        daily_pnl=state.get("daily_pnl", 0.0),
                        position=state.get("position"),
                        trades_today=state.get("trades_today", 0),
                        winners=state.get("winners", 0),
                        losers=state.get("losers", 0),
                    )
                except Exception:
                    logger.exception("alert_manager.heartbeat_error")
        except asyncio.CancelledError:
            pass

    # ── Event Handlers ────────────────────────────────────────────────────────

    async def _on_order_filled(self, event: Event) -> None:
        """Handle order fill events — route to entry or exit formatter."""
        data = event.data
        # Fill events with trade entry info
        if data.get("action_type") in ("ENTER", "ADD"):
            action_data = data.get("action")
            if action_data and isinstance(action_data, LLMAction):
                await self.send_trade_entry(
                    action=action_data,
                    fill_price=data.get("fill_price", 0.0),
                    position_qty=data.get("position_qty", 0),
                    daily_pnl=data.get("daily_pnl", 0.0),
                )

    async def _on_position_changed(self, event: Event) -> None:
        """Handle position change events — send exit notifications."""
        data = event.data
        trade = data.get("closed_trade")
        if trade and isinstance(trade, TradeRecord):
            await self.send_trade_exit(
                trade=trade,
                daily_pnl=data.get("daily_pnl", 0.0),
                winners=data.get("winners", 0),
                losers=data.get("losers", 0),
            )

    async def _on_guardrail(self, event: Event) -> None:
        """Handle guardrail trigger events."""
        if not self._alert_on_guardrails:
            return

        data = event.data
        text = format_guardrail_block(
            reason=data.get("reason", "unknown"),
            action_type=data.get("action_type", "UNKNOWN"),
        )
        await self._send(text)

    async def _on_kill_switch(self, event: Event) -> None:
        """Handle kill switch — ALWAYS priority."""
        data = event.data
        text = format_system_alert(
            alert_type="KILL SWITCH",
            message=data.get("reason", "Kill switch activated — flattened all"),
            severity="critical",
        )
        await self._send(text, priority=True)

    async def _on_connection_lost(self, event: Event) -> None:
        """Handle connection loss — priority alert."""
        text = format_system_alert(
            alert_type="CONNECTION LOST",
            message=event.data.get("details", "WebSocket connection lost"),
            severity="critical",
        )
        await self._send(text, priority=True)

    async def _on_connection_restored(self, event: Event) -> None:
        """Handle connection restored — info alert."""
        downtime = event.data.get("downtime_sec", 0)
        text = format_system_alert(
            alert_type="CONNECTION RESTORED",
            message=f"Reconnected after {downtime:.0f}s downtime",
            severity="info",
        )
        await self._send(text)

    async def _on_daily_limit(self, event: Event) -> None:
        """Handle daily loss limit hit — priority alert."""
        data = event.data
        await self.send_risk_alert(
            alert_type="Daily loss limit hit",
            current_value=data.get("daily_pnl", 0.0),
            limit_value=data.get("limit", 0.0),
            details="System shutting down for the day",
        )

    async def _on_profit_preservation(self, event: Event) -> None:
        """Handle profit preservation tier change."""
        data = event.data
        text = format_system_alert(
            alert_type="PROFIT PRESERVATION",
            message=(
                f"Tier {data.get('tier', '?')} activated at P&L {data.get('daily_pnl', 0.0):+.2f}\n"
                f"Max contracts reduced to {data.get('new_max', '?')}"
            ),
            severity="info",
        )
        await self._send(text)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _send(self, text: str, priority: bool = False) -> None:
        """Send through TelegramClient and track stats."""
        try:
            result = await self._telegram.send(text, priority=priority)
            if result.get("ok"):
                self._alerts_sent += 1
            else:
                self._alerts_failed += 1
                logger.warning(
                    "alert_manager.send_failed",
                    error=result.get("error"),
                )
        except Exception:
            self._alerts_failed += 1
            logger.exception("alert_manager.send_exception")

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "started": self._started,
            "alerts_sent": self._alerts_sent,
            "alerts_failed": self._alerts_failed,
            "uptime_min": round((time.monotonic() - self._start_time) / 60.0, 1),
            "telegram_stats": self._telegram.stats,
        }
