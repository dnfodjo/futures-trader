"""Tests for the async event bus."""

import asyncio

import pytest

from src.core.events import EventBus
from src.core.types import Event, EventType


@pytest.fixture
def bus():
    return EventBus()


async def test_publish_and_subscribe(bus: EventBus):
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.ORDER_FILLED, handler)

    # Start the bus in background
    task = asyncio.create_task(bus.run())

    event = Event(type=EventType.ORDER_FILLED, data={"price": 19850.0})
    await bus.publish(event)

    # Give the bus time to process
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert len(received) == 1
    assert received[0].data["price"] == 19850.0


async def test_multiple_subscribers(bus: EventBus):
    results_a = []
    results_b = []

    async def handler_a(event: Event):
        results_a.append(event)

    async def handler_b(event: Event):
        results_b.append(event)

    bus.subscribe(EventType.TICK_RECEIVED, handler_a)
    bus.subscribe(EventType.TICK_RECEIVED, handler_b)

    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED, data={"price": 100}))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert len(results_a) == 1
    assert len(results_b) == 1


async def test_handler_error_doesnt_crash_bus(bus: EventBus):
    received = []

    async def bad_handler(event: Event):
        raise ValueError("oops")

    async def good_handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.TICK_RECEIVED, bad_handler)
    bus.subscribe(EventType.TICK_RECEIVED, good_handler)

    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    # Good handler still received the event despite bad handler crashing
    assert len(received) == 1


async def test_unsubscribe(bus: EventBus):
    received = []

    async def handler(event: Event):
        received.append(event)

    bus.subscribe(EventType.TICK_RECEIVED, handler)
    bus.unsubscribe(EventType.TICK_RECEIVED, handler)

    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert len(received) == 0


async def test_different_event_types_routed_correctly(bus: EventBus):
    tick_events = []
    fill_events = []

    async def tick_handler(event: Event):
        tick_events.append(event)

    async def fill_handler(event: Event):
        fill_events.append(event)

    bus.subscribe(EventType.TICK_RECEIVED, tick_handler)
    bus.subscribe(EventType.ORDER_FILLED, fill_handler)

    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await bus.publish(Event(type=EventType.ORDER_FILLED))
    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert len(tick_events) == 2
    assert len(fill_events) == 1


# ── New tests for event history and subscriber tracking ──


async def test_event_history(bus: EventBus):
    """Processed events should be recorded in history."""
    async def noop(event: Event):
        pass

    bus.subscribe(EventType.TICK_RECEIVED, noop)
    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED, data={"a": 1}))
    await bus.publish(Event(type=EventType.TICK_RECEIVED, data={"a": 2}))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    history = bus.history
    assert len(history) == 2
    assert history[0].data["a"] == 1
    assert history[1].data["a"] == 2


async def test_event_history_max_size():
    """History should respect max size limit."""
    bus = EventBus(history_size=3)

    async def noop(event: Event):
        pass

    bus.subscribe(EventType.TICK_RECEIVED, noop)
    task = asyncio.create_task(bus.run())

    for i in range(5):
        await bus.publish(Event(type=EventType.TICK_RECEIVED, data={"i": i}))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    history = bus.history
    assert len(history) == 3
    # Should have the last 3 events (indices 2, 3, 4)
    assert history[0].data["i"] == 2
    assert history[2].data["i"] == 4


async def test_event_counts(bus: EventBus):
    """Event counts should track totals per event type."""
    async def noop(event: Event):
        pass

    bus.subscribe(EventType.TICK_RECEIVED, noop)
    bus.subscribe(EventType.ORDER_FILLED, noop)
    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await bus.publish(Event(type=EventType.ORDER_FILLED))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert bus.get_event_count(EventType.TICK_RECEIVED) == 2
    assert bus.get_event_count(EventType.ORDER_FILLED) == 1
    assert bus.get_event_count(EventType.ORDER_CANCELLED) == 0


def test_subscriber_count(bus: EventBus):
    """subscriber_count should track total registrations."""
    async def handler_a(event: Event):
        pass

    async def handler_b(event: Event):
        pass

    assert bus.subscriber_count == 0

    bus.subscribe(EventType.TICK_RECEIVED, handler_a)
    assert bus.subscriber_count == 1

    bus.subscribe(EventType.TICK_RECEIVED, handler_b)
    assert bus.subscriber_count == 2

    bus.subscribe(EventType.ORDER_FILLED, handler_a)
    assert bus.subscriber_count == 3


def test_subscribers_for(bus: EventBus):
    """subscribers_for should return count for a specific event type."""
    async def handler(event: Event):
        pass

    assert bus.subscribers_for(EventType.TICK_RECEIVED) == 0

    bus.subscribe(EventType.TICK_RECEIVED, handler)
    assert bus.subscribers_for(EventType.TICK_RECEIVED) == 1
    assert bus.subscribers_for(EventType.ORDER_FILLED) == 0


async def test_clear_history(bus: EventBus):
    """clear_history should reset both history and counts."""
    async def noop(event: Event):
        pass

    bus.subscribe(EventType.TICK_RECEIVED, noop)
    task = asyncio.create_task(bus.run())

    await bus.publish(Event(type=EventType.TICK_RECEIVED))
    await asyncio.sleep(0.1)
    await bus.stop()
    task.cancel()

    assert len(bus.history) == 1
    assert bus.get_event_count(EventType.TICK_RECEIVED) == 1

    bus.clear_history()
    assert len(bus.history) == 0
    assert bus.get_event_count(EventType.TICK_RECEIVED) == 0


def test_publish_nowait(bus: EventBus):
    """publish_nowait should add to queue without awaiting."""
    bus.publish_nowait(Event(type=EventType.TICK_RECEIVED))
    assert bus.pending_count == 1
    bus.publish_nowait(Event(type=EventType.ORDER_FILLED))
    assert bus.pending_count == 2
