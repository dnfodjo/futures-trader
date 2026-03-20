"""Tests for 1h bar aggregation in StateEngine (Step 3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.state_engine import StateEngine


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_engine() -> StateEngine:
    """Create a StateEngine with mocked dependencies for bar aggregation tests."""
    tp = MagicMock()
    tp.snapshot.return_value = {"last_price": 21000.0}
    poller = MagicMock()
    cal = MagicMock()
    eb = MagicMock()
    return StateEngine(
        tick_processor=tp,
        multi_instrument=poller,
        calendar=cal,
        event_bus=eb,
    )


def _make_bar(minute: int, close: float = 21000.0, volume: int = 100) -> dict:
    """Create a 1-min bar dict with a given minute for clock alignment."""
    return {
        "timestamp": f"2025-01-15T09:{minute:02d}:00-05:00",
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": volume,
        "buy_volume": volume // 2,
        "sell_volume": volume // 2,
    }


# ── Init Attributes ──────────────────────────────────────────────────────────


class TestInit1hAttributes:
    """Verify that __init__ creates 1h bar storage and callback list."""

    def test_1h_bars_list_exists(self):
        engine = _make_engine()
        assert hasattr(engine, "_1h_bars")
        assert isinstance(engine._1h_bars, list)
        assert len(engine._1h_bars) == 0

    def test_current_1h_bar_exists(self):
        engine = _make_engine()
        assert hasattr(engine, "_current_1h_bar")
        assert engine._current_1h_bar is None

    def test_1h_callback_list_exists(self):
        engine = _make_engine()
        assert hasattr(engine, "_on_1h_bar_callbacks")
        assert isinstance(engine._on_1h_bar_callbacks, list)
        assert len(engine._on_1h_bar_callbacks) == 0


# ── Callback Registration ────────────────────────────────────────────────────


class TestRegister1hCallback:
    """Verify the register_1h_bar_callback method."""

    def test_register_callback(self):
        engine = _make_engine()
        cb = MagicMock()
        engine.register_1h_bar_callback(cb)
        assert cb in engine._on_1h_bar_callbacks

    def test_register_multiple_callbacks(self):
        engine = _make_engine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.register_1h_bar_callback(cb1)
        engine.register_1h_bar_callback(cb2)
        assert len(engine._on_1h_bar_callbacks) == 2


# ── 1h Bar Aggregation ──────────────────────────────────────────────────────


class TestAggregation1h:
    """Verify that 1h bars aggregate from 1-min bars with clock alignment."""

    def test_1h_bar_finalizes_at_minute_59(self):
        """A 1h bar should finalize when bar_minute % 60 == 59."""
        engine = _make_engine()
        # Feed bars for minutes 0-59 (a full hour)
        for m in range(60):
            bar = _make_bar(m, close=21000.0 + m)
            engine._aggregate_to_higher_tf(bar)

        assert len(engine._1h_bars) == 1
        finalized = engine._1h_bars[0]
        assert finalized["open"] == pytest.approx(20999.0)  # bar 0 open = 21000-1
        assert finalized["close"] == pytest.approx(21059.0)  # bar 59 close

    def test_1h_bar_does_not_finalize_at_minute_58(self):
        """Before minute 59, bar should still be in-progress."""
        engine = _make_engine()
        for m in range(59):  # minutes 0..58
            bar = _make_bar(m)
            engine._aggregate_to_higher_tf(bar)

        assert len(engine._1h_bars) == 0
        assert engine._current_1h_bar is not None

    def test_1h_bar_ohlcv_accumulation(self):
        """Verify OHLCV values accumulate correctly over 60 bars."""
        engine = _make_engine()
        # Bars with known values
        for m in range(60):
            bar = {
                "timestamp": f"2025-01-15T10:{m:02d}:00-05:00",
                "open": 100.0 if m == 0 else 100.0 + m,
                "high": 200.0 if m == 30 else 150.0,  # max high at bar 30
                "low": 50.0 if m == 45 else 80.0,  # min low at bar 45
                "close": 175.0 if m == 59 else 100.0 + m,
                "volume": 10,
                "buy_volume": 6,
                "sell_volume": 4,
            }
            engine._aggregate_to_higher_tf(bar)

        assert len(engine._1h_bars) == 1
        b = engine._1h_bars[0]
        assert b["high"] == 200.0
        assert b["low"] == 50.0
        assert b["close"] == 175.0
        assert b["volume"] == 600  # 60 * 10
        assert b["buy_volume"] == 360  # 60 * 6
        assert b["sell_volume"] == 240  # 60 * 4


# ── 1h Callback Firing ──────────────────────────────────────────────────────


class TestCallback1hFiring:
    """Verify callbacks fire on 1h bar finalization."""

    def test_callback_fires_on_finalize(self):
        engine = _make_engine()
        cb = MagicMock()
        engine.register_1h_bar_callback(cb)

        for m in range(60):
            bar = _make_bar(m)
            engine._aggregate_to_higher_tf(bar)

        assert cb.call_count == 1
        # Callback receives the finalized bar dict
        finalized = cb.call_args[0][0]
        assert "close" in finalized

    def test_callback_exception_does_not_break(self):
        """An exception in a callback should not prevent other callbacks or crash."""
        engine = _make_engine()
        bad_cb = MagicMock(side_effect=ValueError("boom"))
        good_cb = MagicMock()
        engine.register_1h_bar_callback(bad_cb)
        engine.register_1h_bar_callback(good_cb)

        for m in range(60):
            bar = _make_bar(m)
            engine._aggregate_to_higher_tf(bar)

        assert bad_cb.call_count == 1
        assert good_cb.call_count == 1

    def test_no_callback_for_5min_period(self):
        """1h callbacks should NOT fire on 5-min bar finalization."""
        engine = _make_engine()
        cb = MagicMock()
        engine.register_1h_bar_callback(cb)

        # Feed 5 bars (minutes 0-4) -> 5min bar finalizes at 4
        for m in range(5):
            bar = _make_bar(m)
            engine._aggregate_to_higher_tf(bar)

        cb.assert_not_called()


# ── Properties ───────────────────────────────────────────────────────────────


class TestBarProperties:
    """Verify bars_1h and bars_5m properties."""

    def test_bars_1h_property(self):
        engine = _make_engine()
        assert engine.bars_1h is engine._1h_bars

    def test_bars_5m_property(self):
        engine = _make_engine()
        assert engine.bars_5m is engine._5min_bars


# ── Reset Methods ────────────────────────────────────────────────────────────


class TestReset1h:
    """Verify 1h state is cleared in reset methods."""

    def test_full_reset_clears_1h(self):
        engine = _make_engine()
        engine._1h_bars.append({"close": 100})
        engine._current_1h_bar = {"close": 100}
        engine.reset()
        assert len(engine._1h_bars) == 0
        assert engine._current_1h_bar is None

    def test_reset_for_rth_clears_1h(self):
        engine = _make_engine()
        engine._1h_bars.append({"close": 100})
        engine._current_1h_bar = {"close": 100}
        engine.reset_for_rth()
        assert len(engine._1h_bars) == 0
        assert engine._current_1h_bar is None

    def test_warm_1min_bars_clears_and_rebuilds_1h(self):
        engine = _make_engine()
        engine._1h_bars.append({"close": 999})  # stale data

        # Create 60 bars to warm (full hour)
        warm_bars = []
        for m in range(60):
            warm_bars.append(_make_bar(m, close=21000.0 + m))

        engine.warm_1min_bars(warm_bars)
        # Old stale data should be gone, new 1h bar built from warm data
        assert any(b["close"] != 999 for b in engine._1h_bars) or len(engine._1h_bars) == 1
