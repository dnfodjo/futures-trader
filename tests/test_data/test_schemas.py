"""Tests for market data schemas."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from src.data.schemas import (
    LargeLotEvent,
    OHLCVBar,
    RVOLBaseline,
    RawQuote,
    RawTrade,
    SessionData,
    TickDirection,
    VolumeProfile,
)


class TestRawTrade:
    def test_create(self):
        t = RawTrade(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            price=19850.0,
            size=5,
            direction=TickDirection.BUY,
        )
        assert t.price == 19850.0
        assert t.size == 5
        assert t.direction == TickDirection.BUY
        assert t.is_large is False

    def test_large_lot_flag(self):
        t = RawTrade(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            price=19850.0,
            size=15,
            is_large=True,
        )
        assert t.is_large is True


class TestRawQuote:
    def test_spread(self):
        q = RawQuote(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            bid_price=19849.75,
            bid_size=10,
            ask_price=19850.25,
            ask_size=8,
        )
        assert q.spread == pytest.approx(0.5, abs=0.01)

    def test_mid_price(self):
        q = RawQuote(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            bid_price=19849.75,
            bid_size=10,
            ask_price=19850.25,
            ask_size=8,
        )
        assert q.mid_price == pytest.approx(19850.0, abs=0.01)


class TestOHLCVBar:
    def test_delta(self):
        bar = OHLCVBar(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            open=19850.0,
            high=19855.0,
            low=19848.0,
            close=19853.0,
            volume=100,
            buy_volume=65,
            sell_volume=35,
        )
        assert bar.delta == 30  # 65 - 35

    def test_is_up(self):
        bar = OHLCVBar(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            open=19850.0,
            high=19855.0,
            low=19848.0,
            close=19853.0,
        )
        assert bar.is_up is True

    def test_is_down(self):
        bar = OHLCVBar(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            open=19855.0,
            high=19856.0,
            low=19848.0,
            close=19849.0,
        )
        assert bar.is_up is False


class TestVolumeProfile:
    def test_add_volume(self):
        vp = VolumeProfile()
        vp.add_volume(19850.0, 100)
        vp.add_volume(19850.25, 50)
        assert vp.total_volume == 150

    def test_poc_single_bucket(self):
        vp = VolumeProfile()
        vp.add_volume(19850.0, 100)
        assert vp.poc == 19850.0

    def test_poc_multiple_buckets(self):
        vp = VolumeProfile()
        vp.add_volume(19850.0, 100)
        vp.add_volume(19851.0, 200)
        vp.add_volume(19852.0, 50)
        assert vp.poc == 19851.0

    def test_poc_empty(self):
        vp = VolumeProfile()
        assert vp.poc == 0.0

    def test_value_area(self):
        vp = VolumeProfile()
        # Create a profile centered at 19850
        vp.add_volume(19848.0, 10)
        vp.add_volume(19849.0, 30)
        vp.add_volume(19850.0, 100)  # POC
        vp.add_volume(19851.0, 30)
        vp.add_volume(19852.0, 10)

        va_low, va_high = vp.value_area(pct=0.70)
        # POC = 19850, 70% of 180 = 126
        # Should expand to include 19849 and 19851
        assert va_low <= 19850.0
        assert va_high >= 19850.0

    def test_value_area_empty(self):
        vp = VolumeProfile()
        assert vp.value_area() == (0.0, 0.0)

    def test_reset(self):
        vp = VolumeProfile()
        vp.add_volume(19850.0, 100)
        vp.reset()
        assert vp.total_volume == 0
        assert vp.poc == 0.0

    def test_bucket_snapping(self):
        """Prices should snap to 0.25 bucket boundaries."""
        vp = VolumeProfile(bucket_size=0.25)
        vp.add_volume(19850.10, 10)  # snaps to 19850.0
        vp.add_volume(19850.30, 20)  # snaps to 19850.25
        assert 19850.0 in vp.buckets
        assert 19850.25 in vp.buckets


class TestSessionData:
    def test_initial_state(self):
        sd = SessionData()
        assert sd.vwap == 0.0
        assert sd.total_volume == 0

    def test_update_from_trade(self):
        sd = SessionData()
        trade = RawTrade(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQ.FUT",
            price=19850.0,
            size=10,
            direction=TickDirection.BUY,
        )
        sd.update_from_trade(trade)

        assert sd.session_open == 19850.0
        assert sd.session_high == 19850.0
        assert sd.session_low == 19850.0
        assert sd.session_close == 19850.0
        assert sd.total_volume == 10
        assert sd.total_trades == 1
        assert sd.cumulative_delta == 10  # buy

    def test_vwap_calculation(self):
        sd = SessionData()
        # Trade 1: 100 @ 19850
        sd.update_from_trade(
            RawTrade(
                timestamp=datetime.now(tz=UTC),
                symbol="MNQ.FUT",
                price=19850.0,
                size=100,
                direction=TickDirection.BUY,
            )
        )
        # Trade 2: 100 @ 19860
        sd.update_from_trade(
            RawTrade(
                timestamp=datetime.now(tz=UTC),
                symbol="MNQ.FUT",
                price=19860.0,
                size=100,
                direction=TickDirection.SELL,
            )
        )
        # VWAP = (19850*100 + 19860*100) / 200 = 19855.0
        assert sd.vwap == pytest.approx(19855.0, abs=0.01)

    def test_delta_buy_sell(self):
        sd = SessionData()
        sd.update_from_trade(
            RawTrade(
                timestamp=datetime.now(tz=UTC),
                symbol="MNQ.FUT",
                price=19850.0,
                size=10,
                direction=TickDirection.BUY,
            )
        )
        sd.update_from_trade(
            RawTrade(
                timestamp=datetime.now(tz=UTC),
                symbol="MNQ.FUT",
                price=19850.0,
                size=3,
                direction=TickDirection.SELL,
            )
        )
        assert sd.cumulative_delta == 7  # 10 - 3

    def test_session_high_low(self):
        sd = SessionData()
        for price in [19850.0, 19860.0, 19840.0, 19855.0]:
            sd.update_from_trade(
                RawTrade(
                    timestamp=datetime.now(tz=UTC),
                    symbol="MNQ.FUT",
                    price=price,
                    size=1,
                )
            )
        assert sd.session_high == 19860.0
        assert sd.session_low == 19840.0
        assert sd.session_open == 19850.0  # first trade
        assert sd.session_close == 19855.0  # last trade

    def test_large_lot_tracking(self):
        sd = SessionData()
        sd.update_from_trade(
            RawTrade(
                timestamp=datetime.now(tz=UTC),
                symbol="MNQ.FUT",
                price=19850.0,
                size=15,
                is_large=True,
            )
        )
        assert sd.large_lot_count == 1

    def test_reset(self):
        sd = SessionData()
        sd.update_from_trade(
            RawTrade(
                timestamp=datetime.now(tz=UTC),
                symbol="MNQ.FUT",
                price=19850.0,
                size=10,
                direction=TickDirection.BUY,
            )
        )
        sd.reset()
        assert sd.total_volume == 0
        assert sd.session_open == 0.0
        assert sd.cumulative_delta == 0.0


class TestRVOLBaseline:
    def test_get_expected_volume(self):
        baseline = RVOLBaseline(
            volume_by_time={"09:35": 5000, "10:00": 15000}
        )
        assert baseline.get_expected_volume("09:35") == 5000
        assert baseline.get_expected_volume("10:00") == 15000

    def test_snap_to_5min_bucket(self):
        baseline = RVOLBaseline(
            volume_by_time={"09:35": 5000}
        )
        # 09:37 should snap to 09:35
        assert baseline.get_expected_volume("09:37") == 5000

    def test_missing_bucket(self):
        baseline = RVOLBaseline()
        assert baseline.get_expected_volume("09:35") == 0

    def test_compute_rvol(self):
        baseline = RVOLBaseline(
            volume_by_time={"09:35": 5000}
        )
        # Current volume is 6500 — RVOL = 1.3
        assert baseline.compute_rvol(6500, "09:35") == pytest.approx(1.3, abs=0.01)

    def test_compute_rvol_no_baseline(self):
        baseline = RVOLBaseline()
        # No baseline data, RVOL defaults to 1.0
        assert baseline.compute_rvol(1000, "09:35") == 1.0


class TestRVOLBaselineIO:
    def test_load_from_file(self, tmp_path):
        data = {"volume_by_time": {"09:30": 1200, "09:35": 3500, "10:00": 8000}}
        filepath = tmp_path / "rvol_baseline.json"
        filepath.write_text(json.dumps(data))

        baseline = RVOLBaseline.load_from_file(filepath)
        assert baseline.get_expected_volume("09:30") == 1200
        assert baseline.get_expected_volume("09:35") == 3500
        assert baseline.get_expected_volume("10:00") == 8000

    def test_load_nonexistent_file(self):
        baseline = RVOLBaseline.load_from_file("/nonexistent/path.json")
        assert baseline.volume_by_time == {}

    def test_load_invalid_json(self, tmp_path):
        filepath = tmp_path / "bad.json"
        filepath.write_text("not valid json{{{")
        baseline = RVOLBaseline.load_from_file(filepath)
        assert baseline.volume_by_time == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        baseline = RVOLBaseline(
            volume_by_time={"09:30": 1200, "09:35": 3500}
        )
        filepath = tmp_path / "rvol_baseline.json"
        baseline.save_to_file(filepath)

        loaded = RVOLBaseline.load_from_file(filepath)
        assert loaded.volume_by_time == baseline.volume_by_time

    def test_save_creates_parent_dirs(self, tmp_path):
        baseline = RVOLBaseline(volume_by_time={"09:30": 100})
        filepath = tmp_path / "sub" / "dir" / "rvol.json"
        baseline.save_to_file(filepath)
        assert filepath.exists()

    def test_load_with_invalid_values(self, tmp_path):
        """Should skip entries with non-numeric values."""
        data = {"volume_by_time": {"09:30": 1200, "09:35": "not_a_number", "10:00": 8000}}
        filepath = tmp_path / "rvol.json"
        filepath.write_text(json.dumps(data))

        baseline = RVOLBaseline.load_from_file(filepath)
        assert baseline.get_expected_volume("09:30") == 1200
        assert baseline.get_expected_volume("09:35") == 0  # skipped
        assert baseline.get_expected_volume("10:00") == 8000


class TestSessionDataReset:
    def test_reset_clears_session_date(self):
        sd = SessionData()
        sd.session_date = "2025-03-14"
        sd.reset()
        assert sd.session_date == ""


class TestTickDirection:
    def test_enum_values(self):
        assert TickDirection.BUY == "buy"
        assert TickDirection.SELL == "sell"
        assert TickDirection.UNKNOWN == "unknown"
