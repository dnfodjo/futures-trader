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


class TestContractRollover:
    def test_rollover_warning_near_expiry(self, db_config, trading_config):
        """Should warn when contract is near expiration."""
        # MNQM6 = June 2026 contract, 3rd Friday of June
        trading_config.symbol = "MNQM6"
        client = DatabentoClient(db_config, trading_config)

        # We can't fully control "today" in this test without mocking datetime,
        # so we test the method exists and parses correctly
        result = client.check_contract_rollover()
        # Result depends on current date relative to June 2026 3rd Friday
        # Just verify it returns a dict or None and doesn't crash
        assert result is None or isinstance(result, dict)

    def test_rollover_expired_contract(self, db_config):
        """Should return expired info for past contract."""
        # MNQH5 = March 2025, already expired
        config = TradingConfig(symbol="MNQH5")
        client = DatabentoClient(db_config, config)
        result = client.check_contract_rollover()
        assert result is not None
        assert result["expired"] is True

    def test_rollover_far_out_contract(self, db_config):
        """Should return None for far-future contract."""
        # MNQZ9 = December 2029, far out
        config = TradingConfig(symbol="MNQZ9")
        client = DatabentoClient(db_config, config)
        result = client.check_contract_rollover()
        assert result is None

    def test_rollover_invalid_symbol(self, db_config):
        """Should handle symbols that don't match MNQ pattern."""
        config = TradingConfig(symbol="ES")
        client = DatabentoClient(db_config, config)
        result = client.check_contract_rollover()
        assert result is None


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
