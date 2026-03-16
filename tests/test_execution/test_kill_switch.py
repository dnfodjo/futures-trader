"""Tests for the KillSwitch — emergency flatten conditions."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.core.events import EventBus
from src.core.types import EventType
from src.execution.kill_switch import KillSwitch


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def kill_switch(event_bus):
    return KillSwitch(
        event_bus=event_bus,
        flash_crash_points=50.0,
        flash_crash_seconds=60.0,
        connection_timeout_sec=30.0,
        llm_failure_threshold=3,
        daily_loss_limit=400.0,
    )


# ── Flash Crash Detection ────────────────────────────────────────────────────


class TestFlashCrash:
    def test_no_crash_normal_movement(self, kill_switch):
        base = 1000.0
        for i in range(10):
            result = kill_switch.check_flash_crash(19850.0 + i * 2, base + i)
        assert result is False
        assert kill_switch.is_triggered is False

    def test_crash_detected_large_move(self, kill_switch):
        base = 1000.0
        kill_switch.check_flash_crash(19850.0, base)
        result = kill_switch.check_flash_crash(19800.0, base + 30)  # 50pt drop in 30s

        assert result is True
        assert kill_switch.is_triggered is True
        assert "Flash crash" in kill_switch.trigger_reason

    def test_crash_prunes_old_entries(self, kill_switch):
        base = 1000.0
        kill_switch.check_flash_crash(19800.0, base)  # old price

        # 70 seconds later — outside window
        result = kill_switch.check_flash_crash(19850.0, base + 70)

        assert result is False  # old entry pruned, only 1 entry remains

    def test_crash_exactly_at_threshold(self, kill_switch):
        base = 1000.0
        kill_switch.check_flash_crash(19850.0, base)
        result = kill_switch.check_flash_crash(19800.0, base + 59)  # exactly 50pts

        assert result is True

    def test_crash_just_below_threshold(self, kill_switch):
        base = 1000.0
        kill_switch.check_flash_crash(19850.0, base)
        result = kill_switch.check_flash_crash(19801.0, base + 30)  # 49pts

        assert result is False


# ── Connection Monitoring ────────────────────────────────────────────────────


class TestConnection:
    def test_no_timeout_when_flat(self, kill_switch):
        old_time = datetime.now(tz=UTC) - timedelta(seconds=60)
        result = kill_switch.check_connection(old_time, in_position=False)

        assert result is False

    def test_timeout_while_in_position(self, kill_switch):
        old_time = datetime.now(tz=UTC) - timedelta(seconds=35)
        result = kill_switch.check_connection(old_time, in_position=True)

        assert result is True
        assert "Connection lost" in kill_switch.trigger_reason

    def test_no_timeout_recent_data(self, kill_switch):
        recent_time = datetime.now(tz=UTC) - timedelta(seconds=10)
        result = kill_switch.check_connection(recent_time, in_position=True)

        assert result is False

    def test_exactly_at_timeout_threshold(self, kill_switch):
        # 30 seconds exactly — should trigger (>=)
        boundary_time = datetime.now(tz=UTC) - timedelta(seconds=30)
        result = kill_switch.check_connection(boundary_time, in_position=True)

        assert result is True


# ── LLM Failure Monitoring ───────────────────────────────────────────────────


class TestLLMFailures:
    def test_no_trigger_below_threshold(self, kill_switch):
        result = kill_switch.check_llm_failures(2, in_position=True)
        assert result is False

    def test_trigger_at_threshold(self, kill_switch):
        result = kill_switch.check_llm_failures(3, in_position=True)
        assert result is True
        assert "LLM API failed" in kill_switch.trigger_reason

    def test_no_trigger_when_flat(self, kill_switch):
        result = kill_switch.check_llm_failures(5, in_position=False)
        assert result is False


# ── Daily Loss ───────────────────────────────────────────────────────────────


class TestDailyLoss:
    def test_no_trigger_above_limit(self, kill_switch):
        result = kill_switch.check_daily_loss(-200.0)
        assert result is False

    def test_trigger_at_limit(self, kill_switch):
        result = kill_switch.check_daily_loss(-400.0)
        assert result is True
        assert "Daily loss limit" in kill_switch.trigger_reason

    def test_trigger_below_limit(self, kill_switch):
        result = kill_switch.check_daily_loss(-500.0)
        assert result is True


# ── Manual Trigger ───────────────────────────────────────────────────────────


class TestManualTrigger:
    def test_manual_trigger(self, kill_switch):
        kill_switch.trigger_manual("Operator intervention")
        assert kill_switch.is_triggered is True
        assert kill_switch.trigger_reason == "Operator intervention"
        assert kill_switch.trigger_time is not None


# ── Execute Flatten ──────────────────────────────────────────────────────────


class TestExecuteFlatten:
    @pytest.mark.asyncio
    async def test_flatten_async(self, event_bus):
        flatten_fn = AsyncMock(return_value={"status": "ok"})
        ks = KillSwitch(event_bus=event_bus, flatten_fn=flatten_fn)
        ks.trigger_manual("test")

        result = await ks.execute_flatten()
        assert result["status"] == "flattened"
        flatten_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_flatten_sync_fn(self, event_bus):
        flatten_fn = lambda: {"status": "ok"}
        ks = KillSwitch(event_bus=event_bus, flatten_fn=flatten_fn)

        result = await ks.execute_flatten()
        assert result["status"] == "flattened"

    @pytest.mark.asyncio
    async def test_flatten_no_fn(self, event_bus):
        ks = KillSwitch(event_bus=event_bus, flatten_fn=None)
        result = await ks.execute_flatten()
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_flatten_exception(self, event_bus):
        flatten_fn = AsyncMock(side_effect=Exception("Connection failed"))
        ks = KillSwitch(event_bus=event_bus, flatten_fn=flatten_fn)

        result = await ks.execute_flatten()
        assert result["status"] == "flatten_failed"

    @pytest.mark.asyncio
    async def test_flatten_timeout(self, event_bus):
        async def slow_flatten():
            await asyncio.sleep(10)  # hangs for 10s
            return {"status": "ok"}

        ks = KillSwitch(event_bus=event_bus, flatten_fn=slow_flatten)

        # With 0.1s timeout, should hit timeout before 10s sleep completes
        result = await ks.execute_flatten(timeout_sec=0.1)
        assert result["status"] == "timeout"
        assert "0.1s" in result["error"]

    @pytest.mark.asyncio
    async def test_flatten_within_timeout(self, event_bus):
        async def fast_flatten():
            await asyncio.sleep(0.01)
            return {"status": "ok"}

        ks = KillSwitch(event_bus=event_bus, flatten_fn=fast_flatten)

        result = await ks.execute_flatten(timeout_sec=5.0)
        assert result["status"] == "flattened"


# ── Reset ────────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset(self, kill_switch):
        kill_switch.trigger_manual("test")
        assert kill_switch.is_triggered is True

        kill_switch.reset()
        assert kill_switch.is_triggered is False
        assert kill_switch.trigger_reason == ""
        assert kill_switch.trigger_time is None


# ── Already Triggered ────────────────────────────────────────────────────────


class TestAlreadyTriggered:
    def test_all_checks_return_true_when_triggered(self, kill_switch):
        kill_switch.trigger_manual("already done")

        assert kill_switch.check_flash_crash(19850.0, 1000.0) is True
        assert kill_switch.check_connection(
            datetime.now(tz=UTC), in_position=True
        ) is True
        assert kill_switch.check_llm_failures(0, in_position=True) is True
        assert kill_switch.check_daily_loss(0.0) is True


# ── Stats ────────────────────────────────────────────────────────────────────


class TestKillSwitchStats:
    def test_initial_stats(self, kill_switch):
        stats = kill_switch.stats
        assert stats["is_triggered"] is False
        assert stats["trigger_count"] == 0
        assert stats["thresholds"]["flash_crash_points"] == 50.0
        assert stats["thresholds"]["daily_loss_limit"] == 400.0

    def test_stats_after_trigger(self, kill_switch):
        kill_switch.trigger_manual("test")
        stats = kill_switch.stats
        assert stats["is_triggered"] is True
        assert stats["trigger_count"] == 1
        assert stats["trigger_reason"] == "test"
