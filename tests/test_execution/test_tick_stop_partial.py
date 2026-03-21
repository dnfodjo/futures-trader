"""Tests for partial profit taking in TickStopMonitor.

TDD: These tests define the expected behavior for partial profit taking.
Uses realistic MNQ prices (~20000) to pass the 10% deviation sanity check.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.tick_stop_monitor import TickStopMonitor


# Use realistic MNQ entry price so ticks pass the 10% deviation sanity check
ENTRY = 20000.0


def _make_monitor(**kwargs) -> TickStopMonitor:
    """Create a TickStopMonitor with a mock flatten function."""
    flatten_fn = kwargs.pop("flatten_fn", AsyncMock())
    monitor = TickStopMonitor(
        flatten_fn=flatten_fn,
        target_symbol="MNQM6",
        trail_distance=12.0,
        trail_activation_points=10.0,
        **kwargs,
    )
    return monitor


def _tick(price: float, symbol: str = "MNQM6") -> dict:
    """Create a mock trade tick."""
    return {"price": price, "symbol": symbol}


class TestPartialProfitLong:
    """Test partial profit taking for long positions."""

    @pytest.mark.asyncio
    async def test_partial_fires_at_exact_target(self):
        """Entry at ENTRY, target +15pts. Partial fires when price reaches target."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        target = ENTRY + 15.0  # 20015.0

        # Ticks below target -- no partial
        for p in [ENTRY + 5.0, ENTRY + 10.0, ENTRY + 14.99]:
            await monitor.on_trade(_tick(p))
            partial_fn.assert_not_called()

        # Tick at target -- partial fires
        await monitor.on_trade(_tick(target))
        partial_fn.assert_called_once_with(side="long", quantity=1, price=target)

    @pytest.mark.asyncio
    async def test_stop_moves_to_breakeven_after_partial(self):
        """After partial fires, stop should move to entry + breakeven_offset."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        await monitor.on_trade(_tick(ENTRY + 15.0))
        # Stop should be at entry + offset = ENTRY + 1.0
        assert monitor.current_stop == ENTRY + 1.0

    @pytest.mark.asyncio
    async def test_trail_continues_after_partial(self):
        """Trail should keep working after partial is taken."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Take partial at target
        await monitor.on_trade(_tick(ENTRY + 15.0))
        assert partial_fn.called

        # Stop after partial should be breakeven
        stop_after_partial = monitor.current_stop
        assert stop_after_partial == ENTRY + 1.0

        # Feed ticks that push price higher -- trail should move stop up
        # Trail activation is at +10pts. We're at +15 profit, trail is active.
        await monitor.on_trade(_tick(ENTRY + 25.0))  # big profit
        # Stop should have moved up from breakeven
        assert monitor.current_stop > stop_after_partial


class TestPartialProfitShort:
    """Test partial profit taking for short positions."""

    @pytest.mark.asyncio
    async def test_partial_fires_at_exact_target_short(self):
        """Entry at ENTRY, target +15pts down = ENTRY-15. Partial fires at target."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="short",
            entry_price=ENTRY,
            stop_price=ENTRY + 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        target = ENTRY - 15.0  # 19985.0

        # Ticks above target -- no partial
        for p in [ENTRY - 5.0, ENTRY - 10.0, ENTRY - 14.99]:
            await monitor.on_trade(_tick(p))
            partial_fn.assert_not_called()

        # Tick at target -- partial fires
        await monitor.on_trade(_tick(target))
        partial_fn.assert_called_once_with(side="short", quantity=1, price=target)

    @pytest.mark.asyncio
    async def test_stop_moves_to_breakeven_after_partial_short(self):
        """After partial fires on short, stop should move to entry - breakeven_offset."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="short",
            entry_price=ENTRY,
            stop_price=ENTRY + 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        await monitor.on_trade(_tick(ENTRY - 15.0))
        # Stop should be at entry - offset = ENTRY - 1.0
        assert monitor.current_stop == ENTRY - 1.0


class TestPartialEdgeCases:
    """Edge cases for partial profit taking."""

    @pytest.mark.asyncio
    async def test_no_double_partial(self):
        """After partial fires, more ticks above target should NOT fire again."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Fire partial
        await monitor.on_trade(_tick(ENTRY + 15.0))
        assert partial_fn.call_count == 1

        # More ticks above target -- should NOT fire again
        await monitor.on_trade(_tick(ENTRY + 16.0))
        await monitor.on_trade(_tick(ENTRY + 20.0))
        await monitor.on_trade(_tick(ENTRY + 30.0))
        assert partial_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_partial_skipped_during_grace_period(self):
        """Within grace period, partial should not fire even if price hits target."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        # Default 3s grace period -- do NOT set to 0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Immediately send tick at target -- should be skipped (grace period)
        await monitor.on_trade(_tick(ENTRY + 15.0))
        partial_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_partial_when_target_is_zero(self):
        """When partial_target_points=0, no partial should fire."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=0.0,
            partial_fn=partial_fn,
        )

        await monitor.on_trade(_tick(ENTRY + 15.0))
        await monitor.on_trade(_tick(ENTRY + 20.0))
        partial_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_partial_when_no_callback(self):
        """When partial_fn is None, no partial should fire."""
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=None,
        )

        # Should not crash
        await monitor.on_trade(_tick(ENTRY + 15.0))
        await monitor.on_trade(_tick(ENTRY + 20.0))

    @pytest.mark.asyncio
    async def test_partial_callback_receives_correct_args(self):
        """Verify side, quantity, price are passed correctly to the callback."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 15.0,
            partial_target_points=20.0,
            partial_fn=partial_fn,
            partial_quantity=2,
            breakeven_offset=2.0,
        )

        target = ENTRY + 20.0
        await monitor.on_trade(_tick(target))
        partial_fn.assert_called_once_with(side="long", quantity=2, price=target)


class TestPartialStatePersistence:
    """Test that partial state persists to and restores from disk."""

    @pytest.mark.asyncio
    async def test_state_persistence_includes_partial_fields(self):
        """save_state() should include partial_taken, partial_target, partial_quantity."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Take partial
        await monitor.on_trade(_tick(ENTRY + 15.0))

        state = monitor.save_state()
        assert state["partial_taken"] is True
        assert state["partial_target"] == ENTRY + 15.0
        assert state["partial_quantity"] == 1

    @pytest.mark.asyncio
    async def test_state_restore_preserves_partial_taken(self):
        """After restore, partial_taken=True should prevent re-fire."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Take partial
        await monitor.on_trade(_tick(ENTRY + 15.0))
        assert partial_fn.call_count == 1

        # Save and restore state
        state = monitor.save_state()

        # Create a new monitor and restore
        monitor2 = _make_monitor()
        monitor2._grace_period_sec = 0.0
        partial_fn2 = AsyncMock()
        monitor2.load_state(state)
        # Set partial_fn on the restored monitor (callbacks aren't serializable)
        monitor2._partial_fn = partial_fn2

        # Feed tick above target -- should NOT fire again
        await monitor2.on_trade(_tick(ENTRY + 20.0))
        partial_fn2.assert_not_called()


class TestPartialDeactivation:
    """Test that deactivate resets partial fields."""

    def test_deactivate_resets_partial_fields(self):
        """After deactivate(), all partial fields should be reset."""
        monitor = _make_monitor()
        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 12.0,
            partial_target_points=15.0,
            partial_fn=AsyncMock(),
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        monitor.deactivate()

        # Verify partial state is reset
        assert monitor._partial_taken is False
        assert monitor._partial_target == 0.0
        assert monitor._partial_fn is None
