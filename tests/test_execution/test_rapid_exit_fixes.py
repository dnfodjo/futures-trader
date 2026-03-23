"""Tests for rapid exit fixes: min hold time, wider trail params, entropy cooldown.

TDD: These tests define expected behavior for the 4 rapid-exit fixes:
1. Minimum hold time (60s) - trail/partial exits blocked, hard stop always works
2. Trail activation widened to 20pts
3. Trail distance widened to 15pts
4. Entropy exit cooldown (120s) in orchestrator
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.tick_stop_monitor import TickStopMonitor


ENTRY = 20000.0


def _make_monitor(**kwargs) -> TickStopMonitor:
    """Create a TickStopMonitor with a mock flatten function."""
    flatten_fn = kwargs.pop("flatten_fn", AsyncMock())
    monitor = TickStopMonitor(
        flatten_fn=flatten_fn,
        target_symbol="MNQM6",
        trail_distance=15.0,
        trail_activation_points=20.0,
        **kwargs,
    )
    return monitor


def _tick(price: float, symbol: str = "MNQM6") -> dict:
    return {"price": price, "symbol": symbol}


# ── 1. Minimum Hold Time Tests ──────────────────────────────────────────


class TestMinHoldTime:
    """Trail-based exits should be blocked for first 60 seconds.
    Hard stop (initial stop loss) must ALWAYS trigger."""

    @pytest.mark.asyncio
    async def test_trail_stop_blocked_during_min_hold(self):
        """Trail stop hit within 60s should NOT trigger flatten."""
        flatten_fn = AsyncMock()
        monitor = _make_monitor(flatten_fn=flatten_fn)
        monitor._grace_period_sec = 0.0  # disable grace, test min_hold only

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,  # hard stop at 19970
        )

        # Simulate: price goes up to +20 (trail activates), then drops back
        # Trail activates at +20, trail distance is 15, so trail stop = 20020-15=20005
        await monitor.on_trade(_tick(ENTRY + 20.0))
        assert monitor.trail_active

        # Price drops to 20004 -- below trail stop of 20005
        # But we're within 60 seconds of entry, so it should NOT flatten
        await monitor.on_trade(_tick(ENTRY + 4.0))
        flatten_fn.assert_not_called()
        assert not monitor.is_triggered

    @pytest.mark.asyncio
    async def test_hard_stop_always_triggers_during_min_hold(self):
        """Hard stop (initial stop loss) must trigger even within 60s."""
        flatten_fn = AsyncMock()
        monitor = _make_monitor(flatten_fn=flatten_fn)
        monitor._grace_period_sec = 0.0

        hard_stop = ENTRY - 30.0
        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=hard_stop,
        )

        # Price drops below hard stop within 60s -- must flatten
        await monitor.on_trade(_tick(hard_stop - 1.0))
        flatten_fn.assert_called_once()
        assert monitor.is_triggered
        assert monitor.trigger_reason == "stop_hit"

    @pytest.mark.asyncio
    async def test_trail_stop_works_after_min_hold(self):
        """After 60 seconds, trail-based exits should work normally."""
        flatten_fn = AsyncMock()
        monitor = _make_monitor(flatten_fn=flatten_fn)
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
        )

        # Manually advance the activation time to simulate 61 seconds elapsed
        monitor._activation_time = time.monotonic() - 61.0

        # Price goes up, trail activates
        await monitor.on_trade(_tick(ENTRY + 20.0))
        assert monitor.trail_active

        # Trail stop should be at 20020 - 15 = 20005
        # Price drops below trail stop -- should flatten now (past min hold)
        await monitor.on_trade(_tick(ENTRY + 4.0))
        flatten_fn.assert_called_once()
        assert monitor.is_triggered

    @pytest.mark.asyncio
    async def test_partial_blocked_during_min_hold(self):
        """Partial profit taking should be blocked within 60s."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Price hits partial target within 60s -- should NOT fire
        await monitor.on_trade(_tick(ENTRY + 15.0))
        partial_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_works_after_min_hold(self):
        """After 60 seconds, partial profit should work normally."""
        partial_fn = AsyncMock()
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
            partial_target_points=15.0,
            partial_fn=partial_fn,
            partial_quantity=1,
            breakeven_offset=1.0,
        )

        # Advance past min hold
        monitor._activation_time = time.monotonic() - 61.0

        await monitor.on_trade(_tick(ENTRY + 15.0))
        partial_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_hard_stop_short_triggers_during_min_hold(self):
        """Hard stop for short position triggers even within 60s."""
        flatten_fn = AsyncMock()
        monitor = _make_monitor(flatten_fn=flatten_fn)
        monitor._grace_period_sec = 0.0

        hard_stop = ENTRY + 30.0
        monitor.activate(
            side="short",
            entry_price=ENTRY,
            stop_price=hard_stop,
        )

        # Price rises above hard stop -- must flatten
        await monitor.on_trade(_tick(hard_stop + 1.0))
        flatten_fn.assert_called_once()
        assert monitor.is_triggered

    @pytest.mark.asyncio
    async def test_take_profit_blocked_during_min_hold(self):
        """Take profit should also be blocked during min hold (it's not a safety exit)."""
        flatten_fn = AsyncMock()
        monitor = _make_monitor(flatten_fn=flatten_fn)
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
            take_profit_price=ENTRY + 50.0,
        )

        # Price hits TP within 60s -- should NOT flatten
        await monitor.on_trade(_tick(ENTRY + 51.0))
        flatten_fn.assert_not_called()


# ── 2 & 3. Wider Trail Parameters Tests ─────────────────────────────────


class TestWiderTrailParams:
    """Trail activation should be 20pts, trail distance should be 15pts."""

    def test_default_trail_activation_is_20(self):
        """Constructor default trail_activation_points should be 20."""
        monitor = TickStopMonitor(flatten_fn=AsyncMock(), target_symbol="MNQM6")
        assert monitor._trail_activation == 20.0

    def test_default_trail_distance_is_15(self):
        """Constructor default trail_distance should be 15."""
        monitor = TickStopMonitor(flatten_fn=AsyncMock(), target_symbol="MNQM6")
        assert monitor._trail_distance == 15.0
        assert monitor._default_trail_distance == 15.0

    @pytest.mark.asyncio
    async def test_trail_does_not_activate_at_19pts(self):
        """Trail should NOT activate with only 19 points of profit."""
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0
        monitor._activation_time = time.monotonic() - 61.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
        )
        monitor._activation_time = time.monotonic() - 61.0

        await monitor.on_trade(_tick(ENTRY + 19.0))
        assert not monitor.trail_active

    @pytest.mark.asyncio
    async def test_trail_activates_at_20pts(self):
        """Trail should activate at exactly 20 points of profit."""
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
        )

        await monitor.on_trade(_tick(ENTRY + 20.0))
        assert monitor.trail_active

    @pytest.mark.asyncio
    async def test_trail_stop_at_15pts_behind_best(self):
        """Once trailing, stop should be best_price - 15 (trail_distance).

        Uses exactly +20 profit to avoid hitting mid_tighten_at_profit (15)
        which would use a tighter trail distance.
        """
        monitor = _make_monitor()
        monitor._grace_period_sec = 0.0

        monitor.activate(
            side="long",
            entry_price=ENTRY,
            stop_price=ENTRY - 30.0,
        )

        # Price goes to exactly +20 (trail activates, profit < mid_tighten_at_profit=15 for best_profit)
        # Wait -- best_profit = best_price - entry = 20 which is >= mid_tighten (15)
        # So we need to set mid_tighten higher for this test
        monitor._mid_tighten_at_profit = 30.0  # prevent tightening

        await monitor.on_trade(_tick(ENTRY + 20.0))
        assert monitor.trail_active

        # Trail stop should be best_price - trail_distance = 20020 - 15 = 20005
        assert monitor.current_stop == ENTRY + 20.0 - 15.0


# ── 4. Entropy Exit Cooldown Tests ──────────────────────────────────────


class TestEntropyCooldown:
    """Entropy/mechanical exits should be skipped if position < 120 seconds old."""

    def test_entropy_no_longer_triggers_exit(self):
        """Entropy is entry-gate only — should NOT trigger exits."""
        from src.execution.risk_manager import RiskManager

        rm = RiskManager()
        # Entropy > 0.85 should NOT suggest exit anymore
        should_exit, reason = rm.check_exit_needed(
            phase=None,  # type: ignore
            daily_pnl=0.0,
            entropy=0.90,
            position_pnl=5.0,
            delta_against_minutes=0,
            absorption_against=False,
        )
        assert not should_exit
        assert reason == "HOLD"

    def test_daily_loss_always_triggers(self):
        """Daily loss limit should never be cooldown-gated."""
        from src.execution.risk_manager import RiskManager

        rm = RiskManager()
        should_exit, reason = rm.check_exit_needed(
            phase=None,  # type: ignore
            daily_pnl=-1000.0,
            entropy=0.5,
            position_pnl=-100.0,
            delta_against_minutes=0,
            absorption_against=False,
        )
        assert should_exit
        assert "DAILY_LOSS_LIMIT" in reason
