"""Async event bus for internal communication between system components.

Simple publish/subscribe using asyncio.Queue. Components publish events,
subscribers receive them asynchronously. Includes event history for
debugging and replay.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Callable, Coroutine

import structlog

from src.core.types import Event, EventType

logger = structlog.get_logger()

Subscriber = Callable[[Event], Coroutine]

# Default max events to retain in history
_DEFAULT_HISTORY_SIZE = 1000


class EventBus:
    """Async pub/sub event bus with history.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.TICK_RECEIVED, my_handler)
        await bus.publish(Event(type=EventType.TICK_RECEIVED, data={...}))
    """

    def __init__(self, history_size: int = _DEFAULT_HISTORY_SIZE) -> None:
        self._subscribers: dict[EventType, list[Subscriber]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False
        self._history: deque[Event] = deque(maxlen=history_size)
        self._event_counts: dict[EventType, int] = defaultdict(int)

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Register a handler for an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug("event_bus.subscribe", event_type=event_type.value, handler=handler.__name__)

    def unsubscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Remove a handler for an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        await self._queue.put(event)

    def publish_nowait(self, event: Event) -> None:
        """Publish an event without waiting (fire and forget)."""
        self._queue.put_nowait(event)

    async def run(self) -> None:
        """Process events from the queue. Run as an asyncio task."""
        self._running = True
        logger.info("event_bus.started")

        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Track in history and counts
            self._history.append(event)
            self._event_counts[event.type] += 1

            handlers = self._subscribers.get(event.type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception:
                    logger.exception(
                        "event_bus.handler_error",
                        event_type=event.type.value,
                        handler=handler.__name__,
                    )

            self._queue.task_done()

        logger.info("event_bus.stopped")

    async def stop(self) -> None:
        """Stop the event bus gracefully."""
        self._running = False
        # Drain remaining events
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    @property
    def pending_count(self) -> int:
        """Number of events waiting to be processed."""
        return self._queue.qsize()

    @property
    def subscriber_count(self) -> int:
        """Total number of registered subscribers across all event types."""
        return sum(len(handlers) for handlers in self._subscribers.values())

    def subscribers_for(self, event_type: EventType) -> int:
        """Number of subscribers for a specific event type."""
        return len(self._subscribers.get(event_type, []))

    @property
    def history(self) -> list[Event]:
        """Recent event history (most recent last)."""
        return list(self._history)

    def get_event_count(self, event_type: EventType) -> int:
        """Total number of events processed for a given type."""
        return self._event_counts.get(event_type, 0)

    def clear_history(self) -> None:
        """Clear event history and counts."""
        self._history.clear()
        self._event_counts.clear()
