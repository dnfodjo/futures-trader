"""Tests for the Databento client.

Tests the handler registration, record parsing, dispatching, and
connection state management without real Databento connections.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import DatabentoConfig, TradingConfig
from src.core.exceptions import DatabentoConnectionError
from src.data.databento_client import DatabentoClient


@pytest.fixture
def db_config():
    return DatabentoConfig(api_key="test-key-123")


@pytest.fixture
def trading_config():
    return TradingConfig(symbol="MNQM6")


@pytest.fixture
def client(db_config, trading_config):
    return DatabentoClient(db_config, trading_config)


class TestInitialization:
    def test_initial_state(self, client: DatabentoClient):
        assert client.is_connected is False
        assert client.is_streaming is False
        assert client.last_tick_time is None
        assert client.seconds_since_last_tick == float("inf")

    def test_initial_stats(self, client: DatabentoClient):
        stats = client.stats
        assert stats["connected"] is False
        assert stats["streaming"] is False
        assert stats["trades_received"] == 0
        assert stats["quotes_received"] == 0
        assert stats["errors"] == 0

    def test_custom_symbols(self, db_config, trading_config):
        client = DatabentoClient(
            db_config, trading_config, symbols=["MNQ.FUT"]
        )
        assert client._symbols == ["MNQ.FUT"]


class TestHandlerRegistration:
    def test_register_trade_handler(self, client: DatabentoClient):
        async def handler(data):
            pass

        client.on_trade(handler)
        assert len(client._trade_handlers) == 1

    def test_register_quote_handler(self, client: DatabentoClient):
        async def handler(data):
            pass

        client.on_quote(handler)
        assert len(client._quote_handlers) == 1

    def test_register_error_handler(self, client: DatabentoClient):
        async def handler(data):
            pass

        client.on_error(handler)
        assert len(client._error_handlers) == 1

    def test_multiple_handlers(self, client: DatabentoClient):
        async def h1(data):
            pass

        async def h2(data):
            pass

        client.on_trade(h1)
        client.on_trade(h2)
        assert len(client._trade_handlers) == 2


class TestConnect:
    async def test_connect_no_api_key(self, trading_config):
        cfg = DatabentoConfig(api_key="")
        client = DatabentoClient(cfg, trading_config)

        with pytest.raises(DatabentoConnectionError, match="API key not configured"):
            await client.connect()

    async def test_connect_missing_package(self, client: DatabentoClient):
        """Should raise if databento package is not installed."""
        with patch.dict("sys.modules", {"databento": None}):
            with pytest.raises(DatabentoConnectionError):
                await client.connect()


class TestRecordParsing:
    def test_parse_trade(self, client: DatabentoClient):
        """Should parse a mock trade record."""
        record = MagicMock()
        record.price = 19850.0  # regular float format
        record.size = 5
        record.side = "A"  # ask = buyer initiated
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        result = client._parse_trade(record)
        assert result is not None
        assert result["type"] == "trade"
        assert result["price"] == 19850.0
        assert result["size"] == 5
        assert result["direction"] == "buy"

    def test_parse_trade_sell(self, client: DatabentoClient):
        record = MagicMock()
        record.price = 19850.0
        record.size = 3
        record.side = "B"  # bid = seller initiated
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        result = client._parse_trade(record)
        assert result is not None
        assert result["direction"] == "sell"

    def test_parse_trade_large_lot(self, client: DatabentoClient):
        record = MagicMock()
        record.price = 19850.0
        record.size = 15  # >= 10 threshold
        record.side = "A"
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        result = client._parse_trade(record)
        assert result is not None
        assert result["is_large"] is True

    def test_parse_trade_fixed_point_price(self, client: DatabentoClient):
        """Databento uses 1e-9 fixed-point prices."""
        record = MagicMock()
        record.price = 19850_000_000_000  # 1e-9 format
        record.size = 1
        record.side = "A"
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        result = client._parse_trade(record)
        assert result is not None
        assert result["price"] == pytest.approx(19850.0, rel=1e-6)

    def test_parse_quote(self, client: DatabentoClient):
        """Should parse a mock MBP-1 quote record."""
        record = MagicMock()
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        # Flat attribute format
        record.levels = None
        record.bid_px_00 = 19849.75
        record.bid_sz_00 = 10
        record.ask_px_00 = 19850.25
        record.ask_sz_00 = 8

        result = client._parse_quote(record)
        assert result is not None
        assert result["type"] == "quote"
        assert result["bid_price"] == 19849.75
        assert result["ask_price"] == 19850.25

    def test_parse_quote_with_levels(self, client: DatabentoClient):
        """Should parse MBP-1 with levels array."""
        level = MagicMock()
        level.bid_px = 19849.75
        level.bid_sz = 10
        level.ask_px = 19850.25
        level.ask_sz = 8

        record = MagicMock()
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)
        record.levels = [level]

        result = client._parse_quote(record)
        assert result is not None
        assert result["bid_price"] == 19849.75
        assert result["ask_price"] == 19850.25

    def test_parse_timestamp_nanoseconds(self, client: DatabentoClient):
        """Should handle nanosecond Unix timestamps."""
        record = MagicMock()
        # A nanosecond timestamp
        record.ts_event = 1710400000_000_000_000  # ~2024-03-14

        ts = client._parse_timestamp(record)
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_parse_timestamp_datetime(self, client: DatabentoClient):
        """Should pass through datetime objects."""
        now = datetime.now(tz=UTC)
        record = MagicMock()
        record.ts_event = now

        ts = client._parse_timestamp(record)
        assert ts == now


class TestDispatch:
    async def test_dispatch_trade_record(self, client: DatabentoClient):
        """Trade records should dispatch to trade handlers."""
        received = []

        async def handler(data):
            received.append(data)

        client.on_trade(handler)

        # Create a mock trade record
        record = MagicMock()
        type(record).__name__ = "TradeMsg"
        record.price = 19850.0
        record.size = 5
        record.side = "A"
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        await client._dispatch_record(record)

        assert len(received) == 1
        assert received[0]["type"] == "trade"
        assert client._trades_received == 1

    async def test_dispatch_quote_record(self, client: DatabentoClient):
        """MBP1 records should dispatch to quote handlers."""
        received = []

        async def handler(data):
            received.append(data)

        client.on_quote(handler)

        record = MagicMock()
        type(record).__name__ = "MBP1Msg"
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)
        record.levels = None
        record.bid_px_00 = 19849.75
        record.bid_sz_00 = 10
        record.ask_px_00 = 19850.25
        record.ask_sz_00 = 8

        await client._dispatch_record(record)

        assert len(received) == 1
        assert received[0]["type"] == "quote"
        assert client._quotes_received == 1

    async def test_handler_error_isolation(self, client: DatabentoClient):
        """Handler errors should not crash the dispatch."""
        results = []

        async def bad_handler(data):
            raise ValueError("handler crash")

        async def good_handler(data):
            results.append(data)

        client.on_trade(bad_handler)
        client.on_trade(good_handler)

        record = MagicMock()
        type(record).__name__ = "TradeMsg"
        record.price = 19850.0
        record.size = 1
        record.side = "A"
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        # Should not raise
        await client._dispatch_record(record)
        assert len(results) == 1


class TestStreamNotConnected:
    async def test_stream_not_connected(self, client: DatabentoClient):
        """Should raise if stream() called without connect()."""
        with pytest.raises(DatabentoConnectionError, match="Not connected"):
            await client.stream()


class TestClose:
    async def test_close_when_not_connected(self, client: DatabentoClient):
        """Should be safe to close without connecting."""
        await client.close()
        assert client.is_connected is False

    async def test_close_clears_client(self, client: DatabentoClient):
        """Close should clear the internal client reference."""
        client._client = MagicMock()
        client._connected = True

        await client.close()
        assert client._client is None
        assert client.is_connected is False


class TestResetStats:
    def test_reset_stats(self, client: DatabentoClient):
        client._trades_received = 100
        client._quotes_received = 200
        client._errors = 5

        client.reset_stats()

        assert client.stats["trades_received"] == 0
        assert client.stats["quotes_received"] == 0
        assert client.stats["errors"] == 0


class TestSymbolResolution:
    def test_resolve_symbol_from_attribute(self, client: DatabentoClient):
        record = MagicMock()
        record.symbol = "MNQ.FUT"
        assert client._resolve_symbol(record) == "MNQ.FUT"

    def test_resolve_symbol_strips_null_bytes(self, client: DatabentoClient):
        record = MagicMock()
        record.symbol = "MNQ.FUT\x00\x00"
        assert client._resolve_symbol(record) == "MNQ.FUT"

    def test_resolve_symbol_fallback_to_instrument_id(self, client: DatabentoClient):
        record = MagicMock(spec=[])
        record.instrument_id = 42
        assert "42" in client._resolve_symbol(record)

    def test_resolve_symbol_unknown(self, client: DatabentoClient):
        record = MagicMock(spec=[])
        assert client._resolve_symbol(record) == "unknown"


class TestResolveFrontMonth:
    """Tests for resolve_front_month() and update_symbol()."""

    @patch("src.data.databento_client.db", create=True)
    def test_resolve_front_month_success(self, mock_db):
        """Should resolve MNQ.c.0 continuous to front-month raw symbol."""
        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client

        # Mock the definition record returned by timeseries.get_range
        mock_record = MagicMock()
        mock_record.raw_symbol = "MNQM6"
        mock_client.timeseries.get_range.return_value = [mock_record]

        with patch.dict("sys.modules", {"databento": mock_db}):
            result = DatabentoClient.resolve_front_month(api_key="test-key")

        assert result == "MNQM6"
        mock_client.timeseries.get_range.assert_called_once()

    @patch("src.data.databento_client.db", create=True)
    def test_resolve_front_month_strips_null_bytes(self, mock_db):
        """Should strip null bytes from raw_symbol."""
        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client
        mock_record = MagicMock()
        mock_record.raw_symbol = "MNQM6\x00\x00\x00"
        mock_client.timeseries.get_range.return_value = [mock_record]

        with patch.dict("sys.modules", {"databento": mock_db}):
            result = DatabentoClient.resolve_front_month(api_key="test-key")

        assert result == "MNQM6"

    @patch("src.data.databento_client.db", create=True)
    def test_resolve_front_month_no_records(self, mock_db):
        """Should return None when no definition records returned."""
        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client
        mock_client.timeseries.get_range.return_value = []

        with patch.dict("sys.modules", {"databento": mock_db}):
            result = DatabentoClient.resolve_front_month(api_key="test-key")

        assert result is None

    @patch("src.data.databento_client.db", create=True)
    def test_resolve_front_month_api_failure(self, mock_db):
        """Should return None on API error."""
        mock_client = MagicMock()
        mock_db.Historical.return_value = mock_client
        mock_client.timeseries.get_range.side_effect = Exception("API down")

        with patch.dict("sys.modules", {"databento": mock_db}):
            result = DatabentoClient.resolve_front_month(api_key="test-key")

        assert result is None

    def test_resolve_front_month_no_databento(self):
        """Should return None if databento module not installed."""
        with patch.dict("sys.modules", {"databento": None}):
            result = DatabentoClient.resolve_front_month(api_key="test-key")
        assert result is None

    def test_update_symbol_changes_config(self, db_config, trading_config):
        """Should update trading config and subscription list."""
        trading_config.symbol = "MNQM6"
        client = DatabentoClient(db_config, trading_config)
        # Ensure the old symbol is in the subscription list
        assert "MNQM6" in client._symbols

        changed = client.update_symbol("MNQU6")
        assert changed is True
        assert trading_config.symbol == "MNQU6"
        assert "MNQU6" in client._symbols
        assert "MNQM6" not in client._symbols

    def test_update_symbol_same_symbol_no_change(self, db_config, trading_config):
        """Should return False when symbol hasn't changed."""
        trading_config.symbol = "MNQM6"
        client = DatabentoClient(db_config, trading_config)
        changed = client.update_symbol("MNQM6")
        assert changed is False
        assert trading_config.symbol == "MNQM6"


class TestIsForwardRoll:
    """Tests for is_forward_roll() contract comparison."""

    def test_forward_roll_h_to_m(self):
        assert DatabentoClient.is_forward_roll("MNQH6", "MNQM6") is True

    def test_forward_roll_m_to_u(self):
        assert DatabentoClient.is_forward_roll("MNQM6", "MNQU6") is True

    def test_forward_roll_z_to_h_next_year(self):
        assert DatabentoClient.is_forward_roll("MNQZ5", "MNQH6") is True

    def test_backward_roll_m_to_h(self):
        assert DatabentoClient.is_forward_roll("MNQM6", "MNQH6") is False

    def test_same_contract(self):
        assert DatabentoClient.is_forward_roll("MNQM6", "MNQM6") is False

    def test_backward_roll_h6_to_z5(self):
        assert DatabentoClient.is_forward_roll("MNQH6", "MNQZ5") is False

    def test_invalid_symbol_format(self):
        assert DatabentoClient.is_forward_roll("MNQ", "MNQM6") is False


class TestConfigurableThreshold:
    def test_custom_large_lot_threshold(self, db_config, trading_config):
        """Should use custom large lot threshold."""
        client = DatabentoClient(db_config, trading_config, large_lot_threshold=5)
        assert client._large_lot_threshold == 5

    def test_default_large_lot_threshold(self, client: DatabentoClient):
        """Default threshold should be 10."""
        assert client._large_lot_threshold == 10

    def test_custom_threshold_affects_parsing(self, db_config, trading_config):
        """Trades should use custom threshold for is_large classification."""
        client = DatabentoClient(db_config, trading_config, large_lot_threshold=3)

        record = MagicMock()
        record.price = 19850.0
        record.size = 5  # >= 3 (custom threshold) but < 10 (default)
        record.side = "A"
        record.symbol = "MNQ.FUT"
        record.ts_event = datetime.now(tz=UTC)

        result = client._parse_trade(record)
        assert result is not None
        assert result["is_large"] is True


class _FailingAsyncIter:
    """Async iterator that raises on iteration."""
    def __init__(self, error: Exception):
        self._error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._error


class _StopAsyncIter:
    """Async iterator that stops the client then raises StopAsyncIteration."""
    def __init__(self, client: DatabentoClient):
        self._client = client

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._client._running = False
        raise StopAsyncIteration


class TestReconnect:
    async def test_reconnect_on_stream_failure(self, client: DatabentoClient):
        """Should attempt reconnect on stream failure."""
        client._connected = True
        client._client = _FailingAsyncIter(ConnectionError("connection lost"))

        reconnect_count = 0

        async def mock_reconnect():
            nonlocal reconnect_count
            reconnect_count += 1
            client._connected = True
            if reconnect_count >= 2:
                # After 2 reconnects, provide a client that stops
                client._client = _StopAsyncIter(client)
            else:
                client._client = _FailingAsyncIter(ConnectionError("still failing"))

        with patch.object(client, "_reconnect", side_effect=mock_reconnect):
            await client.stream(max_reconnects=5)

        assert reconnect_count >= 2
        assert client._reconnect_count >= 2

    async def test_max_reconnects_exceeded(self, client: DatabentoClient):
        """Should raise after max reconnect attempts."""
        client._connected = True
        client._client = _FailingAsyncIter(ConnectionError("permanent failure"))

        with patch.object(client, "_reconnect", new_callable=AsyncMock):
            with pytest.raises(DatabentoConnectionError, match="reconnect attempts"):
                await client.stream(max_reconnects=2)


# ── Historical Data Tests ──────────────────────────────────────────────────


class TestHistoricalFetch:
    @pytest.mark.asyncio
    async def test_fetch_historical_requires_dates(self):
        """Should raise ValueError without start/end dates."""
        with pytest.raises(ValueError, match="start and end dates"):
            await DatabentoClient.fetch_historical(
                api_key="test-key",
                start="",
                end="",
            )

    @pytest.mark.asyncio
    async def test_fetch_historical_requires_start(self):
        """Should raise ValueError without start date."""
        with pytest.raises(ValueError, match="start and end dates"):
            await DatabentoClient.fetch_historical(
                api_key="test-key",
                start="",
                end="2026-03-14",
            )

    @pytest.mark.asyncio
    async def test_fetch_historical_parses_trades(self):
        """Should parse TradeMsg records from historical data."""
        import sys

        class TradeMsg:
            pass

        mock_record = TradeMsg()
        mock_record.price = 19850.0
        mock_record.size = 5
        mock_record.side = "A"  # buyer initiated
        mock_record.ts_event = int(datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC).timestamp() * 1e9)

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = [mock_record]

        # Create a mock databento module
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            records = await DatabentoClient.fetch_historical(
                api_key="test-key",
                start="2026-03-14",
                end="2026-03-15",
            )

        assert len(records) == 1
        assert records[0]["type"] == "trade"
        assert records[0]["price"] == 19850.0
        assert records[0]["size"] == 5
        assert records[0]["direction"] == "buy"

    @pytest.mark.asyncio
    async def test_fetch_historical_empty_returns_empty(self):
        """Should return empty list when no records found."""
        import sys

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = []

        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            records = await DatabentoClient.fetch_historical(
                api_key="test-key",
                start="2026-03-14",
                end="2026-03-15",
            )

        assert records == []


# ── RVOL Baseline Tests ────────────────────────────────────────────────────


class TestRVOLBaseline:
    @pytest.mark.asyncio
    async def test_compute_rvol_baseline_calls_fetch(self):
        """Should call fetch_historical to get trade data."""
        # Mock the fetch_historical method
        with patch.object(
            DatabentoClient, "fetch_historical",
            new_callable=AsyncMock,
            return_value=[
                {
                    "type": "trade",
                    "timestamp": datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),  # 9:30 ET
                    "price": 19850.0,
                    "size": 10,
                    "direction": "buy",
                },
                {
                    "type": "trade",
                    "timestamp": datetime(2026, 3, 14, 14, 31, 0, tzinfo=UTC),  # 9:31 ET
                    "price": 19851.0,
                    "size": 15,
                    "direction": "sell",
                },
            ],
        ) as mock_fetch:
            baseline = await DatabentoClient.compute_rvol_baseline(
                api_key="test-key",
                lookback_days=1,
            )

            mock_fetch.assert_called_once()
            assert isinstance(baseline, dict)

    @pytest.mark.asyncio
    async def test_compute_rvol_baseline_buckets(self):
        """Should group trades into 5-minute time buckets."""
        trades = []
        # Create trades at 9:30, 9:31, 9:35 ET (14:30, 14:31, 14:35 UTC)
        for minute in [30, 31, 35]:
            trades.append({
                "type": "trade",
                "timestamp": datetime(2026, 3, 14, 14, minute, 0, tzinfo=UTC),
                "price": 19850.0,
                "size": 10,
                "direction": "buy",
            })

        with patch.object(
            DatabentoClient, "fetch_historical",
            new_callable=AsyncMock,
            return_value=trades,
        ):
            baseline = await DatabentoClient.compute_rvol_baseline(
                api_key="test-key",
                lookback_days=1,
            )

            # 09:30 bucket should have trades from 9:30 and 9:31
            # 09:35 bucket should have trade from 9:35
            assert isinstance(baseline, dict)
            # Baseline will have HH:MM keys
            for key in baseline:
                assert ":" in key  # Format check

    @pytest.mark.asyncio
    async def test_compute_rvol_filters_outside_rth(self):
        """Should exclude trades outside RTH (9:30-16:00 ET)."""
        trades = [
            {
                "type": "trade",
                "timestamp": datetime(2026, 3, 14, 13, 0, 0, tzinfo=UTC),  # 8:00 ET — before RTH
                "price": 19850.0,
                "size": 100,
                "direction": "buy",
            },
            {
                "type": "trade",
                "timestamp": datetime(2026, 3, 14, 14, 30, 0, tzinfo=UTC),  # 9:30 ET — RTH
                "price": 19850.0,
                "size": 10,
                "direction": "buy",
            },
        ]

        with patch.object(
            DatabentoClient, "fetch_historical",
            new_callable=AsyncMock,
            return_value=trades,
        ):
            baseline = await DatabentoClient.compute_rvol_baseline(
                api_key="test-key",
                lookback_days=1,
            )

            # Only the 9:30 trade should be included
            total_volume = sum(baseline.values())
            assert total_volume == 10  # Not 110


# ── Aggregate Bars Tests ──────────────────────────────────────────────────


class TestAggregateBars:
    """Tests for DatabentoClient._aggregate_bars() static helper."""

    def test_aggregate_4_bars_into_1(self):
        """Should aggregate 4 bars into 1 with correct OHLCV logic."""
        bars = [
            {"timestamp": datetime(2026, 3, 14, 10, 0, tzinfo=UTC), "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 1000},
            {"timestamp": datetime(2026, 3, 14, 11, 0, tzinfo=UTC), "open": 105.0, "high": 115.0, "low": 100.0, "close": 112.0, "volume": 1200},
            {"timestamp": datetime(2026, 3, 14, 12, 0, tzinfo=UTC), "open": 112.0, "high": 120.0, "low": 108.0, "close": 118.0, "volume": 800},
            {"timestamp": datetime(2026, 3, 14, 13, 0, tzinfo=UTC), "open": 118.0, "high": 125.0, "low": 110.0, "close": 122.0, "volume": 1500},
        ]
        result = DatabentoClient._aggregate_bars(bars, 4)
        assert len(result) == 1
        agg = result[0]
        assert agg["timestamp"] == datetime(2026, 3, 14, 10, 0, tzinfo=UTC)  # first timestamp
        assert agg["open"] == 100.0  # first open
        assert agg["high"] == 125.0  # max high
        assert agg["low"] == 95.0  # min low
        assert agg["close"] == 122.0  # last close
        assert agg["volume"] == 4500  # sum volume

    def test_aggregate_discards_incomplete_chunk(self):
        """Should discard the final incomplete chunk."""
        bars = [
            {"timestamp": datetime(2026, 3, 14, 10, 0, tzinfo=UTC), "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 1000},
            {"timestamp": datetime(2026, 3, 14, 11, 0, tzinfo=UTC), "open": 105.0, "high": 115.0, "low": 100.0, "close": 112.0, "volume": 1200},
            {"timestamp": datetime(2026, 3, 14, 12, 0, tzinfo=UTC), "open": 112.0, "high": 120.0, "low": 108.0, "close": 118.0, "volume": 800},
            {"timestamp": datetime(2026, 3, 14, 13, 0, tzinfo=UTC), "open": 118.0, "high": 125.0, "low": 110.0, "close": 122.0, "volume": 1500},
            # 5th bar — incomplete chunk of 4, should be discarded
            {"timestamp": datetime(2026, 3, 14, 14, 0, tzinfo=UTC), "open": 122.0, "high": 130.0, "low": 120.0, "close": 128.0, "volume": 900},
        ]
        result = DatabentoClient._aggregate_bars(bars, 4)
        assert len(result) == 1  # Only 1 complete chunk of 4

    def test_aggregate_multiple_chunks(self):
        """Should handle multiple complete chunks."""
        bars = [
            {"timestamp": datetime(2026, 3, i + 10, 10, 0, tzinfo=UTC), "open": float(100 + i), "high": float(110 + i), "low": float(90 + i), "close": float(105 + i), "volume": 1000}
            for i in range(10)
        ]
        result = DatabentoClient._aggregate_bars(bars, 5)
        assert len(result) == 2  # 10 bars / 5 = 2 chunks

    def test_aggregate_empty_input(self):
        """Should return empty list for empty input."""
        result = DatabentoClient._aggregate_bars([], 4)
        assert result == []

    def test_aggregate_fewer_than_period(self):
        """Should return empty if bars < period."""
        bars = [
            {"timestamp": datetime(2026, 3, 14, 10, 0, tzinfo=UTC), "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 1000},
        ]
        result = DatabentoClient._aggregate_bars(bars, 4)
        assert result == []

    def test_aggregate_weekly_from_daily(self):
        """Should aggregate 5 daily bars into 1 weekly bar."""
        bars = [
            {"timestamp": datetime(2026, 3, 16 + i, 0, 0, tzinfo=UTC), "open": float(19800 + i * 10), "high": float(19850 + i * 10), "low": float(19780 + i * 10), "close": float(19840 + i * 10), "volume": 50000 + i * 1000}
            for i in range(5)
        ]
        result = DatabentoClient._aggregate_bars(bars, 5)
        assert len(result) == 1
        assert result[0]["open"] == 19800.0  # first open
        assert result[0]["high"] == 19890.0  # max of 19850..19890
        assert result[0]["low"] == 19780.0  # min of 19780..19820
        assert result[0]["close"] == 19880.0  # last close
        assert result[0]["volume"] == 260000  # 50000+51000+52000+53000+54000


# ── HTF Bar Fetch Tests ───────────────────────────────────────────────────


class TestFetchHTFBars:
    """Tests for DatabentoClient.fetch_htf_bars() static async method."""

    def _make_ohlcv_record(self, open_p, high_p, low_p, close_p, volume, ts_ns):
        """Create a mock OHLCV record matching Databento OhlcvMsg format."""
        record = MagicMock()
        type(record).__name__ = "Ohlcv1HMsg"
        record.open = int(open_p * 1e9)
        record.high = int(high_p * 1e9)
        record.low = int(low_p * 1e9)
        record.close = int(close_p * 1e9)
        record.volume = volume
        record.ts_event = ts_ns
        return record

    def _make_daily_record(self, open_p, high_p, low_p, close_p, volume, ts_ns):
        """Create a mock daily OHLCV record."""
        record = MagicMock()
        type(record).__name__ = "Ohlcv1DMsg"
        record.open = int(open_p * 1e9)
        record.high = int(high_p * 1e9)
        record.low = int(low_p * 1e9)
        record.close = int(close_p * 1e9)
        record.volume = volume
        record.ts_event = ts_ns
        return record

    @pytest.mark.asyncio
    async def test_fetch_1h_bars_returns_correct_format(self):
        """Should fetch 1h bars and return list of dicts with correct keys."""
        import sys

        ts_ns = int(datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC).timestamp() * 1e9)
        mock_record = self._make_ohlcv_record(19850.0, 19870.0, 19830.0, 19860.0, 5000, ts_ns)

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = [mock_record]

        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            bars = await DatabentoClient.fetch_htf_bars(
                api_key="test-key",
                timeframe="1h",
                lookback_days=60,
            )

        assert len(bars) == 1
        bar = bars[0]
        assert "timestamp" in bar
        assert "open" in bar
        assert "high" in bar
        assert "low" in bar
        assert "close" in bar
        assert "volume" in bar
        assert bar["open"] == pytest.approx(19850.0, rel=1e-6)
        assert bar["high"] == pytest.approx(19870.0, rel=1e-6)
        assert bar["low"] == pytest.approx(19830.0, rel=1e-6)
        assert bar["close"] == pytest.approx(19860.0, rel=1e-6)
        assert bar["volume"] == 5000

    @pytest.mark.asyncio
    async def test_fetch_1h_uses_correct_schema(self):
        """1h timeframe should use ohlcv-1h schema."""
        import sys

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = []
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            await DatabentoClient.fetch_htf_bars(api_key="test-key", timeframe="1h")

        call_kwargs = mock_historical.timeseries.get_range.call_args
        assert call_kwargs.kwargs.get("schema") == "ohlcv-1h" or call_kwargs[1].get("schema") == "ohlcv-1h"

    @pytest.mark.asyncio
    async def test_fetch_1d_uses_correct_schema(self):
        """1d timeframe should use ohlcv-1d schema."""
        import sys

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = []
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            await DatabentoClient.fetch_htf_bars(api_key="test-key", timeframe="1d")

        call_kwargs = mock_historical.timeseries.get_range.call_args
        assert call_kwargs.kwargs.get("schema") == "ohlcv-1d" or call_kwargs[1].get("schema") == "ohlcv-1d"

    @pytest.mark.asyncio
    async def test_fetch_4h_fetches_1h_and_aggregates(self):
        """4h should fetch 1h bars then aggregate by 4."""
        import sys

        ts_base = int(datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC).timestamp() * 1e9)
        records = []
        for i in range(8):  # 8 hours = 2 x 4h bars
            ts_ns = ts_base + i * 3600 * int(1e9)
            records.append(self._make_ohlcv_record(
                19850.0 + i, 19870.0 + i, 19830.0 + i, 19860.0 + i, 5000, ts_ns
            ))

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = records
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            bars = await DatabentoClient.fetch_htf_bars(
                api_key="test-key",
                timeframe="4h",
                lookback_days=60,
            )

        # Should use ohlcv-1h schema
        call_kwargs = mock_historical.timeseries.get_range.call_args
        schema_used = call_kwargs.kwargs.get("schema") or call_kwargs[1].get("schema")
        assert schema_used == "ohlcv-1h"

        # Should aggregate: 8 bars / 4 = 2 aggregated bars
        assert len(bars) == 2

    @pytest.mark.asyncio
    async def test_fetch_1w_fetches_daily_and_aggregates(self):
        """1w should fetch daily bars then aggregate by 5."""
        import sys

        ts_base = int(datetime(2026, 3, 10, 0, 0, 0, tzinfo=UTC).timestamp() * 1e9)
        records = []
        for i in range(10):  # 10 days = 2 weeks
            ts_ns = ts_base + i * 86400 * int(1e9)
            records.append(self._make_daily_record(
                19850.0 + i, 19870.0 + i, 19830.0 + i, 19860.0 + i, 50000, ts_ns
            ))

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = records
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            bars = await DatabentoClient.fetch_htf_bars(
                api_key="test-key",
                timeframe="1w",
                lookback_days=730,
            )

        # Should use ohlcv-1d schema
        call_kwargs = mock_historical.timeseries.get_range.call_args
        schema_used = call_kwargs.kwargs.get("schema") or call_kwargs[1].get("schema")
        assert schema_used == "ohlcv-1d"

        # Should aggregate: 10 bars / 5 = 2 weekly bars
        assert len(bars) == 2

    @pytest.mark.asyncio
    async def test_fetch_htf_bars_error_returns_empty(self):
        """Should return empty list on API failure (graceful degradation)."""
        import sys

        mock_db = MagicMock()
        mock_db.Historical.side_effect = Exception("API error")

        with patch.dict(sys.modules, {"databento": mock_db}):
            bars = await DatabentoClient.fetch_htf_bars(
                api_key="test-key",
                timeframe="1h",
            )

        assert bars == []

    @pytest.mark.asyncio
    async def test_fetch_htf_bars_fixed_point_prices(self):
        """Should handle Databento fixed-point prices (/ 1e9)."""
        import sys

        ts_ns = int(datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC).timestamp() * 1e9)
        record = MagicMock()
        type(record).__name__ = "OhlcvMsg"
        record.open = 19850_000_000_000  # fixed-point 1e9
        record.high = 19870_000_000_000
        record.low = 19830_000_000_000
        record.close = 19860_000_000_000
        record.volume = 5000
        record.ts_event = ts_ns

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = [record]
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            bars = await DatabentoClient.fetch_htf_bars(
                api_key="test-key",
                timeframe="1h",
            )

        assert len(bars) == 1
        assert bars[0]["open"] == pytest.approx(19850.0, rel=1e-6)
        assert bars[0]["close"] == pytest.approx(19860.0, rel=1e-6)

    @pytest.mark.asyncio
    async def test_fetch_htf_bars_timestamp_is_datetime(self):
        """Timestamp should be a timezone-aware datetime."""
        import sys

        ts_ns = int(datetime(2026, 3, 14, 10, 0, 0, tzinfo=UTC).timestamp() * 1e9)
        mock_record = self._make_ohlcv_record(19850.0, 19870.0, 19830.0, 19860.0, 5000, ts_ns)

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = [mock_record]
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            bars = await DatabentoClient.fetch_htf_bars(
                api_key="test-key",
                timeframe="1h",
            )

        assert isinstance(bars[0]["timestamp"], datetime)
        assert bars[0]["timestamp"].tzinfo is not None

    @pytest.mark.asyncio
    async def test_fetch_htf_bars_uses_parent_stype(self):
        """Should use stype_in='parent' matching fetch_historical pattern."""
        import sys

        mock_historical = MagicMock()
        mock_historical.timeseries.get_range.return_value = []
        mock_db = MagicMock()
        mock_db.Historical.return_value = mock_historical

        with patch.dict(sys.modules, {"databento": mock_db}):
            await DatabentoClient.fetch_htf_bars(api_key="test-key", timeframe="1h")

        call_kwargs = mock_historical.timeseries.get_range.call_args
        stype = call_kwargs.kwargs.get("stype_in") or call_kwargs[1].get("stype_in")
        assert stype == "parent"
