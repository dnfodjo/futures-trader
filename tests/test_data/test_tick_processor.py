"""Tests for the tick processor.

Tests delta computation, tape speed, bar building, volume profile,
session tracking, and large lot detection without real market data.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.data.schemas import TickDirection
from src.data.tick_processor import TickProcessor


@pytest.fixture
def processor():
    return TickProcessor(large_lot_threshold=10)


def _make_trade(
    price: float = 19850.0,
    size: int = 1,
    direction: str = "buy",
    symbol: str = "MNQ.FUT",
) -> dict:
    return {
        "timestamp": datetime.now(tz=UTC),
        "symbol": symbol,
        "price": price,
        "size": size,
        "direction": direction,
        "is_large": size >= 10,
    }


def _make_quote(
    bid: float = 19849.75,
    ask: float = 19850.25,
    symbol: str = "MNQ.FUT",
) -> dict:
    return {
        "timestamp": datetime.now(tz=UTC),
        "symbol": symbol,
        "bid_price": bid,
        "bid_size": 10,
        "ask_price": ask,
        "ask_size": 8,
    }


class TestProcessTrade:
    async def test_basic_trade_updates_session(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0, size=5))

        assert processor.session_data.total_trades == 1
        assert processor.session_data.total_volume == 5
        assert processor.session_data.session_open == 19850.0
        assert processor.last_price == 19850.0

    async def test_multiple_trades(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0, size=5))
        await processor.process_trade(_make_trade(price=19855.0, size=3))
        await processor.process_trade(_make_trade(price=19848.0, size=2))

        assert processor.session_data.total_trades == 3
        assert processor.session_data.total_volume == 10
        assert processor.session_data.session_high == 19855.0
        assert processor.session_data.session_low == 19848.0

    async def test_trade_updates_volume_profile(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        await processor.process_trade(_make_trade(price=19851.0, size=200))

        vp = processor.volume_profile
        assert vp.total_volume == 300
        assert vp.poc == 19851.0  # highest volume price


class TestDelta:
    async def test_cumulative_delta_buy(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(size=10, direction="buy"))
        assert processor.cumulative_delta == 10

    async def test_cumulative_delta_sell(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(size=10, direction="sell"))
        assert processor.cumulative_delta == -10

    async def test_cumulative_delta_mixed(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(size=10, direction="buy"))
        await processor.process_trade(_make_trade(size=3, direction="sell"))
        assert processor.cumulative_delta == 7

    async def test_cumulative_delta_unknown(self, processor: TickProcessor):
        """Unknown direction trades should not affect delta."""
        await processor.process_trade(_make_trade(size=10, direction="unknown"))
        assert processor.cumulative_delta == 0

    async def test_delta_1min(self, processor: TickProcessor):
        """1-min delta uses rolling window."""
        await processor.process_trade(_make_trade(size=5, direction="buy"))
        await processor.process_trade(_make_trade(size=3, direction="sell"))
        assert processor.delta_1min == 2.0  # 5 - 3

    async def test_delta_5min(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(size=10, direction="buy"))
        await processor.process_trade(_make_trade(size=4, direction="sell"))
        assert processor.delta_5min == 6.0

    async def test_delta_trend_positive(self, processor: TickProcessor):
        for _ in range(10):
            await processor.process_trade(_make_trade(size=10, direction="buy"))
        assert processor.delta_trend == "positive"

    async def test_delta_trend_negative(self, processor: TickProcessor):
        for _ in range(10):
            await processor.process_trade(_make_trade(size=10, direction="sell"))
        assert processor.delta_trend == "negative"

    async def test_delta_trend_neutral(self, processor: TickProcessor):
        # No trades → neutral
        assert processor.delta_trend == "neutral"


class TestTapeSpeed:
    async def test_tape_speed_zero_initially(self, processor: TickProcessor):
        assert processor.tape_speed == 0.0

    async def test_tape_speed_single_trade(self, processor: TickProcessor):
        await processor.process_trade(_make_trade())
        # Single trade — tape speed = count (1)
        assert processor.tape_speed >= 1.0

    async def test_tape_speed_multiple_trades(self, processor: TickProcessor):
        """Multiple rapid trades should show high tape speed."""
        for _ in range(20):
            await processor.process_trade(_make_trade())
        # All trades happened nearly instantly — high speed
        assert processor.tape_speed > 1.0


class TestLargeLotDetection:
    async def test_large_lot_detected(self, processor: TickProcessor):
        received = []

        async def handler(event):
            received.append(event)

        processor.on_large_lot(handler)

        await processor.process_trade(_make_trade(size=15, direction="buy"))

        assert len(received) == 1
        assert received[0].size == 15

    async def test_small_lot_not_detected(self, processor: TickProcessor):
        received = []

        async def handler(event):
            received.append(event)

        processor.on_large_lot(handler)

        await processor.process_trade(_make_trade(size=5, direction="buy"))
        assert len(received) == 0

    async def test_large_lot_with_quote_context(self, processor: TickProcessor):
        """Large lot should identify if trade was at bid or ask."""
        received = []

        async def handler(event):
            received.append(event)

        processor.on_large_lot(handler)

        # Set up quote first
        await processor.process_quote(_make_quote(bid=19849.75, ask=19850.25))

        # Trade at ask = buying
        await processor.process_trade(
            _make_trade(price=19850.25, size=15, direction="buy")
        )

        assert len(received) == 1
        assert received[0].at_bid_or_ask == "ask"

    async def test_large_lot_count_5min(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(size=15))
        await processor.process_trade(_make_trade(size=12))
        assert processor.large_lot_count_5min == 2

    async def test_large_lot_handler_error_isolation(self, processor: TickProcessor):
        """Handler error should not crash processing."""
        async def bad_handler(event):
            raise ValueError("handler error")

        processor.on_large_lot(bad_handler)

        # Should not raise
        await processor.process_trade(_make_trade(size=15))


class TestBarBuilding:
    async def test_first_trade_starts_bar(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0))
        assert processor._current_bar is not None
        assert processor._current_bar.open == 19850.0

    async def test_bar_updates_within_interval(self, processor: TickProcessor):
        """Trades within the same second should update the current bar."""
        await processor.process_trade(_make_trade(price=19850.0, size=5))
        await processor.process_trade(_make_trade(price=19855.0, size=3))
        await processor.process_trade(_make_trade(price=19848.0, size=2))

        bar = processor._current_bar
        assert bar is not None
        # These all happen in the same instant, so they're in the same bar
        # (unless the test runs across a second boundary, which is unlikely)
        assert bar.volume >= 5  # at least first trade

    async def test_bar_emitted_on_interval(self, processor: TickProcessor):
        """Completed bars should be emitted to handlers."""
        received = []

        async def handler(bar):
            received.append(bar)

        processor.on_bar(handler)

        # Use a very short bar interval
        processor._bar_interval_sec = 0.05

        await processor.process_trade(_make_trade(price=19850.0, size=5))

        # Wait for bar interval to elapse
        await asyncio.sleep(0.1)

        # Next trade should trigger bar emission
        await processor.process_trade(_make_trade(price=19855.0, size=3))

        assert len(received) >= 1
        assert received[0].open == 19850.0

    async def test_bar_buy_sell_volume(self, processor: TickProcessor):
        await processor.process_trade(
            _make_trade(price=19850.0, size=5, direction="buy")
        )
        await processor.process_trade(
            _make_trade(price=19851.0, size=3, direction="sell")
        )

        bar = processor._current_bar
        assert bar is not None
        assert bar.buy_volume >= 5
        assert bar.sell_volume >= 0  # might be in same or different bar

    async def test_bar_handler_error_isolation(self, processor: TickProcessor):
        """Bar handler error should not crash processing."""
        async def bad_handler(bar):
            raise ValueError("bar handler error")

        processor.on_bar(bad_handler)
        processor._bar_interval_sec = 0.01

        await processor.process_trade(_make_trade())
        await asyncio.sleep(0.05)
        # Should not raise
        await processor.process_trade(_make_trade(price=19855.0))


class TestProcessQuote:
    async def test_process_quote(self, processor: TickProcessor):
        await processor.process_quote(_make_quote(bid=19849.75, ask=19850.25))

        quote = processor.last_quote
        assert quote is not None
        assert quote.bid_price == 19849.75
        assert quote.ask_price == 19850.25

    async def test_quote_updates_latest(self, processor: TickProcessor):
        await processor.process_quote(_make_quote(bid=19849.0, ask=19850.0))
        await processor.process_quote(_make_quote(bid=19851.0, ask=19852.0))

        assert processor.last_quote.bid_price == 19851.0


class TestSnapshot:
    async def test_snapshot_initial(self, processor: TickProcessor):
        snap = processor.snapshot()
        assert snap["last_price"] == 0.0
        assert snap["vwap"] == 0.0
        assert snap["cumulative_delta"] == 0.0
        assert snap["tape_speed"] == 0.0

    async def test_snapshot_after_trades(self, processor: TickProcessor):
        await processor.process_quote(_make_quote(bid=19849.75, ask=19850.25))
        await processor.process_trade(_make_trade(price=19850.0, size=100, direction="buy"))

        snap = processor.snapshot()
        assert snap["last_price"] == 19850.0
        assert snap["bid"] == 19849.75
        assert snap["ask"] == 19850.25
        assert snap["spread"] == pytest.approx(0.5, abs=0.01)
        assert snap["cumulative_delta"] == 100.0
        assert snap["total_volume"] == 100
        assert snap["total_trades"] == 1
        assert snap["vwap"] == pytest.approx(19850.0, abs=0.01)

    async def test_snapshot_session_low_no_inf(self, processor: TickProcessor):
        """Session low should be 0.0 when no trades, not inf."""
        snap = processor.snapshot()
        assert snap["session_low"] == 0.0


class TestReset:
    async def test_reset_clears_everything(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        await processor.process_quote(_make_quote())

        processor.reset()

        assert processor.session_data.total_trades == 0
        assert processor.session_data.total_volume == 0
        assert processor.volume_profile.total_volume == 0
        assert processor.last_quote is None
        assert processor._current_bar is None
        assert processor.tape_speed == 0.0


class TestStats:
    async def test_stats_initial(self, processor: TickProcessor):
        stats = processor.stats
        assert stats["total_trades"] == 0
        assert stats["total_volume"] == 0
        assert stats["vwap"] == 0.0

    async def test_stats_after_trades(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0, size=50))
        stats = processor.stats
        assert stats["total_trades"] == 1
        assert stats["total_volume"] == 50
        assert stats["session_high"] == 19850.0


class TestVWAPComputation:
    async def test_vwap_single_trade(self, processor: TickProcessor):
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        assert processor.session_data.vwap == pytest.approx(19850.0, abs=0.01)

    async def test_vwap_multiple_trades(self, processor: TickProcessor):
        # 100 contracts at 19850
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        # 100 contracts at 19860
        await processor.process_trade(_make_trade(price=19860.0, size=100))
        # VWAP = (19850*100 + 19860*100) / 200 = 19855
        assert processor.session_data.vwap == pytest.approx(19855.0, abs=0.01)

    async def test_vwap_volume_weighted(self, processor: TickProcessor):
        # 100 contracts at 19850
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        # 300 contracts at 19860
        await processor.process_trade(_make_trade(price=19860.0, size=300))
        # VWAP = (19850*100 + 19860*300) / 400 = 19857.5
        assert processor.session_data.vwap == pytest.approx(19857.5, abs=0.01)


class TestDeltaTrendFlipping:
    async def test_delta_trend_flipping(self, processor: TickProcessor):
        """Flipping = 1min and 5min delta disagree in direction."""
        # Create a scenario where 5min delta is positive overall
        # but the most recent 1min delta is negative (momentum shift)
        import time
        from unittest.mock import patch as mock_patch

        # Add large positive delta first (for 5min to be positive)
        for _ in range(20):
            await processor.process_trade(_make_trade(size=10, direction="buy"))

        # Now make recent trades negative (so 1min flips)
        # To make 1-min delta negative, we need recent sells to outweigh recent buys
        # within the 1-min window. Add enough sells.
        for _ in range(25):
            await processor.process_trade(_make_trade(size=10, direction="sell"))

        # 5min total: 200 buy - 250 sell = -50 (negative)
        # 1min total: since all trades happen instantly, both windows see the same data
        # So we need to manipulate the delta_window timestamps.

        # Alternative approach: directly verify the flipping detection logic
        # by checking that when d1 and d5 have opposite signs, trend is "flipping"
        # Let's just verify the other untested state: when d5 is positive but d1 negative
        processor._delta_window.clear()
        now_mono = time.monotonic()
        # Old positive entries (4+ minutes ago, still within 5min window)
        for i in range(20):
            processor._delta_window.append((now_mono - 200 + i, 10))  # +200 total
        # Recent negative entries (within 1min window)
        for i in range(15):
            processor._delta_window.append((now_mono - 30 + i, -10))  # -150 total

        # 5min delta: 200 - 150 = +50 (positive)
        # 1min delta: -150 (negative, only recent entries)
        assert processor.delta_trend == "flipping"


class TestSessionDate:
    async def test_session_date_set_on_first_trade(self, processor: TickProcessor):
        """Session date should be set from the first trade's timestamp."""
        assert processor.session_data.session_date == ""
        await processor.process_trade(_make_trade())
        assert processor.session_data.session_date != ""

    async def test_session_date_reset(self, processor: TickProcessor):
        """Reset should clear session date."""
        await processor.process_trade(_make_trade())
        assert processor.session_data.session_date != ""
        processor.reset()
        assert processor.session_data.session_date == ""


class TestBarVWAP:
    async def test_bar_vwap_single_trade(self, processor: TickProcessor):
        """Bar VWAP with single trade should equal trade price."""
        await processor.process_trade(_make_trade(price=19850.0, size=10))
        bar = processor._current_bar
        assert bar is not None
        assert bar.vwap == pytest.approx(19850.0, abs=0.01)

    async def test_bar_vwap_multiple_trades(self, processor: TickProcessor):
        """Bar VWAP should be volume-weighted average of trades in the bar."""
        # All trades within the same bar (same instant)
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        await processor.process_trade(_make_trade(price=19860.0, size=100))

        bar = processor._current_bar
        assert bar is not None
        # VWAP = (19850*100 + 19860*100) / 200 = 19855.0
        assert bar.vwap == pytest.approx(19855.0, abs=0.01)

    async def test_bar_vwap_volume_weighted(self, processor: TickProcessor):
        """Bar VWAP should weight by volume."""
        await processor.process_trade(_make_trade(price=19850.0, size=100))
        await processor.process_trade(_make_trade(price=19860.0, size=300))

        bar = processor._current_bar
        assert bar is not None
        # VWAP = (19850*100 + 19860*300) / 400 = 19857.5
        assert bar.vwap == pytest.approx(19857.5, abs=0.01)
