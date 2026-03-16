"""Tests for the PositionTracker — real-time position state management."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.core.events import EventBus
from src.core.types import PositionState, Side
from src.execution.position_tracker import PositionTracker


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def tracker(event_bus):
    return PositionTracker(event_bus=event_bus, symbol="MNQM6", point_value=2.0)


# ── Opening Positions ────────────────────────────────────────────────────────


class TestOpenPosition:
    @pytest.mark.asyncio
    async def test_open_long(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})

        assert tracker.in_position is True
        assert tracker.position.side == Side.LONG
        assert tracker.position.quantity == 2
        assert tracker.position.avg_entry == 19850.0

    @pytest.mark.asyncio
    async def test_open_short(self, tracker):
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19860.0})

        assert tracker.position.side == Side.SHORT
        assert tracker.position.quantity == 1
        assert tracker.position.avg_entry == 19860.0

    @pytest.mark.asyncio
    async def test_invalid_fill_ignored(self, tracker):
        await tracker.on_fill({"action": "", "qty": 0, "price": 0.0})
        assert tracker.is_flat is True

    @pytest.mark.asyncio
    async def test_starts_flat(self, tracker):
        assert tracker.is_flat is True
        assert tracker.position is None


# ── Adding to Positions ──────────────────────────────────────────────────────


class TestAddPosition:
    @pytest.mark.asyncio
    async def test_add_to_long(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19860.0})

        assert tracker.position.quantity == 3
        # Weighted avg: (19850*2 + 19860*1) / 3 = 19853.3333
        assert abs(tracker.position.avg_entry - 19853.3333) < 0.01
        assert tracker.position.adds_count == 1

    @pytest.mark.asyncio
    async def test_add_to_short(self, tracker):
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19860.0})
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19850.0})

        assert tracker.position.quantity == 2
        assert tracker.position.avg_entry == 19855.0  # (19860+19850)/2


# ── Closing Positions ────────────────────────────────────────────────────────


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_long_winner(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})
        await tracker.on_fill({"action": "Sell", "qty": 2, "price": 19860.0})

        assert tracker.is_flat is True
        assert len(tracker.completed_trades) == 1

        trade = tracker.last_trade
        assert trade.side == Side.LONG
        assert trade.entry_price == 19850.0
        assert trade.exit_price == 19860.0
        # P&L: (19860 - 19850) * 2 * 2.0 = 40.0
        assert trade.pnl == 40.0

    @pytest.mark.asyncio
    async def test_close_long_loser(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19840.0})

        trade = tracker.last_trade
        # P&L: (19840 - 19850) * 1 * 2.0 = -20.0
        assert trade.pnl == -20.0

    @pytest.mark.asyncio
    async def test_close_short_winner(self, tracker):
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19860.0})
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})

        trade = tracker.last_trade
        # P&L: (19860 - 19850) * 1 * 2.0 = 20.0
        assert trade.pnl == 20.0

    @pytest.mark.asyncio
    async def test_close_short_loser(self, tracker):
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19850.0})
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19860.0})

        trade = tracker.last_trade
        # P&L: (19850 - 19860) * 1 * 2.0 = -20.0
        assert trade.pnl == -20.0


# ── Partial Close ────────────────────────────────────────────────────────────


class TestPartialClose:
    @pytest.mark.asyncio
    async def test_partial_close(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 3, "price": 19850.0})
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19860.0})

        assert tracker.in_position is True
        assert tracker.position.quantity == 2
        assert len(tracker.completed_trades) == 0  # partial, not completed

    @pytest.mark.asyncio
    async def test_partial_then_full_close(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 3, "price": 19850.0})
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19860.0})
        await tracker.on_fill({"action": "Sell", "qty": 2, "price": 19855.0})

        assert tracker.is_flat is True
        assert len(tracker.completed_trades) == 1


# ── Unrealized P&L Updates ───────────────────────────────────────────────────


class TestUnrealizedPnl:
    @pytest.mark.asyncio
    async def test_update_unrealized_long(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})
        tracker.update_unrealized(19860.0)

        # (19860 - 19850) * 2 * 2.0 = 40.0
        assert tracker.position.unrealized_pnl == 40.0

    @pytest.mark.asyncio
    async def test_update_unrealized_short(self, tracker):
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19860.0})
        tracker.update_unrealized(19850.0)

        # (19860 - 19850) * 1 * 2.0 = 20.0
        assert tracker.position.unrealized_pnl == 20.0

    @pytest.mark.asyncio
    async def test_max_favorable_excursion(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})
        tracker.update_unrealized(19870.0)  # +40
        tracker.update_unrealized(19855.0)  # +10

        assert tracker.position.max_favorable == 40.0

    @pytest.mark.asyncio
    async def test_max_adverse_excursion(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})
        tracker.update_unrealized(19840.0)  # -20

        assert tracker.position.max_adverse == 20.0

    @pytest.mark.asyncio
    async def test_max_adverse_excursion_short(self, tracker):
        """Short position: adverse is when price goes UP."""
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19850.0})
        tracker.update_unrealized(19860.0)  # price up = adverse for short
        tracker.update_unrealized(19855.0)  # still adverse but less

        # unrealized at 19860: (19850 - 19860) * 1 * 2.0 = -20 → adverse = 20
        assert tracker.position.max_adverse == 20.0

    @pytest.mark.asyncio
    async def test_max_favorable_excursion_short(self, tracker):
        """Short position: favorable is when price goes DOWN."""
        await tracker.on_fill({"action": "Sell", "qty": 1, "price": 19850.0})
        tracker.update_unrealized(19840.0)  # price down = favorable for short
        tracker.update_unrealized(19845.0)  # less favorable

        # unrealized at 19840: (19850 - 19840) * 1 * 2.0 = 20 → favorable = 20
        assert tracker.position.max_favorable == 20.0

    def test_update_no_position(self, tracker):
        # Should not raise
        tracker.update_unrealized(19860.0)


# ── Stop Price Updates ───────────────────────────────────────────────────────


class TestStopPrice:
    @pytest.mark.asyncio
    async def test_update_stop_price(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})
        tracker.update_stop_price(19845.0)

        assert tracker.position.stop_price == 19845.0

    def test_update_stop_no_position(self, tracker):
        tracker.update_stop_price(19845.0)  # should not raise


# ── Reconciliation ───────────────────────────────────────────────────────────


class TestReconciliation:
    @pytest.mark.asyncio
    async def test_reconcile_matches(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})

        await tracker.reconcile([{
            "contractName": "MNQM6",
            "netPos": 2,
            "netPrice": 19850.0,
        }])

        assert tracker.stats["mismatches"] == 0

    @pytest.mark.asyncio
    async def test_reconcile_rest_flat_but_internal_has_position(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})

        await tracker.reconcile([{
            "contractName": "MNQM6",
            "netPos": 0,
            "netPrice": 0.0,
        }])

        assert tracker.is_flat is True
        assert tracker.stats["mismatches"] == 1

    @pytest.mark.asyncio
    async def test_reconcile_rest_has_position_but_internal_flat(self, tracker):
        await tracker.reconcile([{
            "contractName": "MNQM6",
            "netPos": 3,
            "netPrice": 19855.0,
        }])

        assert tracker.in_position is True
        assert tracker.position.quantity == 3
        assert tracker.stats["mismatches"] == 1

    @pytest.mark.asyncio
    async def test_reconcile_qty_mismatch(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})

        await tracker.reconcile([{
            "contractName": "MNQM6",
            "netPos": 3,  # REST says 3, we think 2
            "netPrice": 19850.0,
        }])

        assert tracker.position.quantity == 3  # REST wins
        assert tracker.stats["mismatches"] == 1

    @pytest.mark.asyncio
    async def test_reconcile_no_matching_symbol(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})

        # Different symbol in REST data
        await tracker.reconcile([{
            "contractName": "ESM6",
            "netPos": 1,
            "netPrice": 5500.0,
        }])

        # No match for MNQM6, netPos=0, should clear our position
        assert tracker.is_flat is True
        assert tracker.stats["mismatches"] == 1

    @pytest.mark.asyncio
    async def test_reconcile_empty_data(self, tracker):
        await tracker.reconcile([])
        assert tracker.is_flat is True


# ── Reset ────────────────────────────────────────────────────────────────────


class TestReset:
    @pytest.mark.asyncio
    async def test_reset(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 1, "price": 19850.0})
        tracker.reset()

        assert tracker.is_flat is True
        assert len(tracker.completed_trades) == 0
        assert tracker.stats["fills_processed"] == 0


# ── Stats ────────────────────────────────────────────────────────────────────


class TestStats:
    def test_initial_stats(self, tracker):
        stats = tracker.stats
        assert stats["in_position"] is False
        assert stats["fills_processed"] == 0
        assert stats["completed_trades"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_trades(self, tracker):
        await tracker.on_fill({"action": "Buy", "qty": 2, "price": 19850.0})
        await tracker.on_fill({"action": "Sell", "qty": 2, "price": 19860.0})

        stats = tracker.stats
        assert stats["fills_processed"] == 2
        assert stats["completed_trades"] == 1
        assert stats["in_position"] is False
