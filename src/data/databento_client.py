"""Databento Live streaming client for MNQ and ES market data.

Connects to Databento's real-time feed via their Python SDK.
Subscribes to L1 quotes (MBP-1) and trades for MNQ (primary)
and ES (cross-market context). Publishes events on the event bus.

Databento's `MNQ.FUT` parent symbol auto-rolls to the front-month
contract, so we don't need to track expiration here. However,
Tradovate requires the specific contract symbol (e.g., MNQM6) —
that mapping is handled by the config layer.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Coroutine, Optional

import structlog

from src.core.config import DatabentoConfig, TradingConfig
from src.core.exceptions import DatabentoConnectionError

logger = structlog.get_logger()

# MNQ quarterly expiration months: H=Mar, M=Jun, U=Sep, Z=Dec
_EXPIRATION_MONTHS: dict[str, int] = {"H": 3, "M": 6, "U": 9, "Z": 12}
# Expiration is the 3rd Friday of the contract month
_ROLLOVER_WARNING_DAYS = 14

# Type alias for data callbacks
DataCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class DatabentoClient:
    """Async wrapper around the Databento Live client.

    Streams MNQ and ES data via Databento's Live API. Converts
    raw Databento records to our internal schema dicts and dispatches
    to registered callbacks.

    Usage:
        client = DatabentoClient(db_config, trading_config)
        client.on_trade(my_trade_handler)
        client.on_quote(my_quote_handler)
        await client.connect()
        await client.stream()  # runs until close() is called
    """

    def __init__(
        self,
        db_config: DatabentoConfig,
        trading_config: TradingConfig,
        symbols: list[str] | None = None,
        large_lot_threshold: int = 10,
    ) -> None:
        self._config = db_config
        self._trading_config = trading_config
        # Use specific contract symbol for the trading instrument to avoid
        # rollover ambiguity (e.g., MNQM6 instead of MNQ.FUT which may stream
        # both expiring and front-month contracts during rollover week).
        # ES.FUT is fine as a parent — we only use it for cross-market context.
        default_symbols = [trading_config.symbol, "ES.FUT"]
        self._symbols = symbols or default_symbols
        self._large_lot_thresh = large_lot_threshold

        # Callbacks
        self._trade_handlers: list[DataCallback] = []
        self._quote_handlers: list[DataCallback] = []
        self._depth_handlers: list[DataCallback] = []
        self._error_handlers: list[DataCallback] = []

        # State
        self._client: Any = None  # databento.Live instance
        self._connected = False
        self._running = False
        self._last_tick_time: datetime | None = None
        self._reconnect_count = 0

        # Instrument ID → symbol name mapping
        # Databento live stream uses instrument_ids; we resolve to symbol names
        # using price heuristics (MNQ > 10000, ES < 10000) on first encounter.
        self._instrument_map: dict[int, str] = {}

        # Stats
        self._trades_received = 0
        self._quotes_received = 0
        self._errors = 0

        # mbp-10 authorization flag — set to False if subscription is rejected
        self._mbp10_authorized = True

    # ── Handler Registration ──────────────────────────────────────────────

    def on_trade(self, handler: DataCallback) -> None:
        """Register a callback for trade (time & sales) events."""
        self._trade_handlers.append(handler)

    def on_quote(self, handler: DataCallback) -> None:
        """Register a callback for L1 quote (BBO) events."""
        self._quote_handlers.append(handler)

    def on_depth(self, handler: DataCallback) -> None:
        """Register a callback for 10-level order book depth (mbp-10) events."""
        self._depth_handlers.append(handler)

    def on_error(self, handler: DataCallback) -> None:
        """Register a callback for connection/data errors."""
        self._error_handlers.append(handler)

    # ── Contract Rollover Check ────────────────────────────────────────────

    def check_contract_rollover(self) -> dict[str, Any] | None:
        """Check if the configured contract is near expiration.

        MNQ expires quarterly on the 3rd Friday:
        H=March, M=June, U=September, Z=December.

        Returns a dict with warning info, or None if not near expiration.
        """
        symbol = self._trading_config.symbol  # e.g., "MNQM6"

        # Parse month code and year digit from symbol
        # Format: MNQ + month_code + year_digit (e.g., MNQM6 = June 2026)
        if len(symbol) < 5:
            return None

        month_code = symbol[-2].upper()
        year_digit = symbol[-1]

        if month_code not in _EXPIRATION_MONTHS:
            return None

        try:
            exp_month = _EXPIRATION_MONTHS[month_code]
            # Year digit: assume 202X
            exp_year = 2020 + int(year_digit)

            # Find 3rd Friday of the expiration month
            # Start from the 1st of the month and find the third Friday
            first_day = datetime(exp_year, exp_month, 1, tzinfo=UTC)
            # Find first Friday
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            third_friday = first_friday + timedelta(weeks=2)

            today = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            days_until_expiry = (third_friday - today).days

            if days_until_expiry <= _ROLLOVER_WARNING_DAYS:
                warning = {
                    "symbol": symbol,
                    "expiration_date": third_friday.strftime("%Y-%m-%d"),
                    "days_until_expiry": days_until_expiry,
                    "expired": days_until_expiry <= 0,
                }
                if days_until_expiry <= 0:
                    logger.error(
                        "databento.contract_expired",
                        **warning,
                    )
                else:
                    logger.warning(
                        "databento.contract_expiring_soon",
                        **warning,
                    )
                return warning
        except (ValueError, IndexError):
            logger.debug("databento.rollover_check_failed", symbol=symbol)

        return None

    # ── Connection ────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Initialize Databento Live client and subscribe to data.

        This creates the subscription but does NOT start streaming.
        Call stream() to begin receiving data.
        """
        # Check contract rollover before connecting
        self.check_contract_rollover()

        if not self._config.api_key:
            raise DatabentoConnectionError("Databento API key not configured")

        try:
            import databento as db

            self._client = db.Live(key=self._config.api_key)

            # Separate subscriptions: specific contract for trading instrument,
            # parent symbol for cross-market (ES).
            # This prevents rollover ambiguity during expiration week.
            trading_symbol = self._trading_config.symbol  # e.g., "MNQM6"
            cross_market_symbols = [s for s in self._symbols if s != trading_symbol]

            # Trading instrument: use raw_symbol for exact contract match
            self._client.subscribe(
                dataset=self._config.dataset,
                schema="mbp-1",
                stype_in="raw_symbol",
                symbols=[trading_symbol],
            )
            self._client.subscribe(
                dataset=self._config.dataset,
                schema="trades",
                stype_in="raw_symbol",
                symbols=[trading_symbol],
            )

            # 10-level order book depth for order flow analysis (optional)
            if self._depth_handlers and self._mbp10_authorized:
                self._client.subscribe(
                    dataset=self._config.dataset,
                    schema="mbp-10",
                    stype_in="raw_symbol",
                    symbols=[trading_symbol],
                )
                logger.info("databento.mbp10_subscribed", symbol=trading_symbol)

            # Cross-market instruments (ES): use parent for auto-roll
            if cross_market_symbols:
                self._client.subscribe(
                    dataset=self._config.dataset,
                    schema="mbp-1",
                    stype_in="parent",
                    symbols=cross_market_symbols,
                )
                self._client.subscribe(
                    dataset=self._config.dataset,
                    schema="trades",
                    stype_in="parent",
                    symbols=cross_market_symbols,
                )

            self._connected = True
            logger.info(
                "databento.connected",
                symbols=self._symbols,
                dataset=self._config.dataset,
            )

        except ImportError:
            raise DatabentoConnectionError(
                "databento package not installed. Run: pip install databento"
            )
        except Exception as e:
            self._connected = False
            raise DatabentoConnectionError(f"Failed to connect to Databento: {e}") from e

    async def stream(self, max_reconnects: int = 5) -> None:
        """Stream data from Databento with automatic reconnection.

        Reconnects with exponential backoff on stream failures.
        Gives up after `max_reconnects` consecutive failures.

        Args:
            max_reconnects: Maximum consecutive reconnect attempts before raising.
        """
        if not self._connected or self._client is None:
            raise DatabentoConnectionError("Not connected. Call connect() first.")

        self._running = True
        consecutive_failures = 0
        logger.info("databento.streaming_started")

        try:
            while self._running:
                try:
                    # Databento's Live client provides an async iterator
                    async for record in self._client:
                        if not self._running:
                            return

                        self._last_tick_time = datetime.now(tz=UTC)
                        consecutive_failures = 0  # reset on successful data

                        try:
                            await self._dispatch_record(record)
                        except Exception:
                            self._errors += 1
                            logger.exception("databento.dispatch_error")

                    # Stream ended (server closed connection or end-of-day)
                    if not self._running:
                        return
                    # If still supposed to be running, treat as disconnect and reconnect
                    logger.warning("databento.stream_ended_unexpectedly")
                    consecutive_failures += 1
                    self._reconnect_count += 1
                    delay = min(2 ** (consecutive_failures - 1), 30)
                    await asyncio.sleep(delay)
                    try:
                        await self._reconnect()
                    except Exception as re:
                        logger.error("databento.reconnect_after_stream_end_failed", error=str(re))
                    continue

                except asyncio.CancelledError:
                    logger.info("databento.stream_cancelled")
                    raise
                except Exception as e:
                    self._errors += 1
                    consecutive_failures += 1
                    self._reconnect_count += 1

                    # If mbp-10 is not authorized, disable it and reconnect without it
                    err_str = str(e).lower()
                    if ("not authorized" in err_str and "mbp" in err_str) or \
                       ("subscription" in err_str and ("mbp-10" in err_str or "schema" in err_str)):
                        self._mbp10_authorized = False
                        logger.warning(
                            "databento.mbp10_not_authorized",
                            msg="Disabling mbp-10 depth subscription — account not authorized. "
                                "Order flow will use trade data only (no DOM imbalance).",
                        )
                        consecutive_failures = 0  # Don't count as a real failure
                        try:
                            await self._reconnect()
                        except Exception as re:
                            logger.error("databento.reconnect_failed", error=str(re))
                        continue

                    if consecutive_failures > max_reconnects:
                        logger.error(
                            "databento.max_reconnects_exceeded",
                            attempts=consecutive_failures,
                            error=str(e),
                        )
                        await self._notify_error({
                            "error": str(e),
                            "type": "max_reconnects_exceeded",
                            "attempts": consecutive_failures,
                        })
                        raise DatabentoConnectionError(
                            f"Stream failed after {consecutive_failures} reconnect attempts: {e}"
                        ) from e

                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
                    delay = min(2 ** (consecutive_failures - 1), 30)
                    logger.warning(
                        "databento.reconnecting",
                        attempt=consecutive_failures,
                        delay_sec=delay,
                        error=str(e),
                    )
                    await self._notify_error({
                        "error": str(e),
                        "type": "stream_error",
                        "reconnecting": True,
                        "attempt": consecutive_failures,
                    })

                    await asyncio.sleep(delay)

                    # Try to reconnect
                    try:
                        await self._reconnect()
                    except Exception as re:
                        logger.error("databento.reconnect_failed", error=str(re))
                        continue

        finally:
            self._running = False
            logger.info("databento.streaming_stopped")

    async def _reconnect(self) -> None:
        """Reconnect to Databento after a stream failure."""
        # Close the old client
        if self._client is not None:
            try:
                if hasattr(self._client, "close"):
                    self._client.close()
                elif hasattr(self._client, "stop"):
                    self._client.stop()
            except Exception:
                pass
            self._client = None

        self._connected = False

        # Reconnect
        await self.connect()
        logger.info("databento.reconnected", total_reconnects=self._reconnect_count)

    async def _dispatch_record(self, record: Any) -> None:
        """Convert a Databento record to our internal format and dispatch.

        Databento records come as typed objects with attributes like
        .ts_event, .price, .size, .action, etc. We convert to dicts
        for handler consumption.
        """
        record_type = type(record).__name__

        if record_type == "TradeMsg":
            trade_data = self._parse_trade(record)
            if trade_data:
                self._trades_received += 1
                await self._dispatch_to_handlers(self._trade_handlers, trade_data)

        elif record_type in ("MBP1Msg", "Mbp1Msg"):
            quote_data = self._parse_quote(record)
            if quote_data:
                self._quotes_received += 1
                await self._dispatch_to_handlers(self._quote_handlers, quote_data)

        elif record_type in ("MBP10Msg", "Mbp10Msg"):
            depth_data = self._parse_depth(record)
            if depth_data:
                await self._dispatch_to_handlers(self._depth_handlers, depth_data)

        # Other record types (SystemMsg, ErrorMsg, etc.) are logged but not dispatched

    def _parse_trade(self, record: Any) -> dict[str, Any] | None:
        """Parse a Databento TradeMsg into our internal dict format."""
        try:
            # Databento prices are in fixed-point (1e-9 precision)
            price = record.price / 1e9 if record.price > 1e6 else record.price
            size = record.size

            # Classify trade direction using side field
            # Databento side: 'A' = ask (buyer initiated), 'B' = bid (seller initiated)
            side = getattr(record, "side", None)
            if side == "A" or side == ord("A"):
                direction = "buy"
            elif side == "B" or side == ord("B"):
                direction = "sell"
            else:
                direction = "unknown"

            # Get symbol from instrument_id mapping or raw symbol
            symbol = self._resolve_symbol(record, price=price)

            # Timestamp from Databento (nanosecond Unix timestamp)
            ts = self._parse_timestamp(record)

            return {
                "type": "trade",
                "timestamp": ts,
                "symbol": symbol,
                "price": price,
                "size": size,
                "direction": direction,
                "is_large": size >= self._large_lot_threshold,
            }
        except Exception:
            logger.exception("databento.parse_trade_error")
            return None

    def _parse_quote(self, record: Any) -> dict[str, Any] | None:
        """Parse a Databento MBP1Msg into our internal dict format."""
        try:
            # Databento MBP-1 has levels[0] for top of book
            levels = record.levels if hasattr(record, "levels") else None

            if levels and len(levels) > 0:
                bid_price = levels[0].bid_px / 1e9 if levels[0].bid_px > 1e6 else levels[0].bid_px
                bid_size = levels[0].bid_sz
                ask_price = levels[0].ask_px / 1e9 if levels[0].ask_px > 1e6 else levels[0].ask_px
                ask_size = levels[0].ask_sz
            else:
                # Fallback for flat record structure
                bid_price = getattr(record, "bid_px_00", 0) or 0
                ask_price = getattr(record, "ask_px_00", 0) or 0
                bid_size = getattr(record, "bid_sz_00", 0) or 0
                ask_size = getattr(record, "ask_sz_00", 0) or 0

                if bid_price > 1e6:
                    bid_price /= 1e9
                if ask_price > 1e6:
                    ask_price /= 1e9

            symbol = self._resolve_symbol(record, price=bid_price)
            ts = self._parse_timestamp(record)

            return {
                "type": "quote",
                "timestamp": ts,
                "symbol": symbol,
                "bid_price": bid_price,
                "bid_size": bid_size,
                "ask_price": ask_price,
                "ask_size": ask_size,
            }
        except Exception:
            logger.exception("databento.parse_quote_error")
            return None

    def _parse_depth(self, record: Any) -> dict[str, Any] | None:
        """Parse a Databento MBP10Msg into our internal depth format.

        Extracts 10 levels of bid/ask price and size for DOM analysis.
        """
        try:
            levels_data = []
            raw_levels = record.levels if hasattr(record, "levels") else None

            if raw_levels and len(raw_levels) >= 10:
                for i in range(10):
                    lvl = raw_levels[i]
                    bid_px = lvl.bid_px / 1e9 if lvl.bid_px > 1e6 else lvl.bid_px
                    ask_px = lvl.ask_px / 1e9 if lvl.ask_px > 1e6 else lvl.ask_px
                    levels_data.append({
                        "bid_price": bid_px,
                        "bid_size": lvl.bid_sz,
                        "ask_price": ask_px,
                        "ask_size": lvl.ask_sz,
                    })
            else:
                # Fallback: try flat field names (bid_px_00..bid_px_09)
                for i in range(10):
                    bp = getattr(record, f"bid_px_{i:02d}", 0) or 0
                    bs = getattr(record, f"bid_sz_{i:02d}", 0) or 0
                    ap = getattr(record, f"ask_px_{i:02d}", 0) or 0
                    az = getattr(record, f"ask_sz_{i:02d}", 0) or 0
                    if bp > 1e6:
                        bp /= 1e9
                    if ap > 1e6:
                        ap /= 1e9
                    levels_data.append({
                        "bid_price": bp,
                        "bid_size": bs,
                        "ask_price": ap,
                        "ask_size": az,
                    })

            if not levels_data:
                return None

            ts = self._parse_timestamp(record)

            return {
                "type": "depth",
                "timestamp": ts,
                "levels": levels_data,
            }
        except Exception:
            logger.exception("databento.parse_depth_error")
            return None

    def _resolve_symbol(self, record: Any, price: float = 0.0) -> str:
        """Resolve the symbol name from a Databento record.

        Databento live stream records have instrument_id but may not
        populate the symbol field. We build an instrument_id → symbol
        mapping using price heuristics on first encounter:
        - MNQ trades at ~15000-30000
        - ES trades at ~3000-7000

        Args:
            record: Databento record object.
            price: Parsed price (after 1e-9 conversion) for price-based
                   instrument identification on first encounter.
        """
        # Check instrument_id mapping first (fastest path)
        iid = getattr(record, "instrument_id", None)
        if iid is not None and iid in self._instrument_map:
            return self._instrument_map[iid]

        # Try the publisher-assigned symbol
        if hasattr(record, "symbol"):
            sym = str(record.symbol).strip().rstrip("\x00")
            if sym and sym != "unknown":
                # Cache the mapping for future lookups
                if iid is not None:
                    self._instrument_map[iid] = sym
                    logger.info(
                        "databento.instrument_mapped",
                        instrument_id=iid,
                        symbol=sym,
                        source="record.symbol",
                    )
                return sym

        # Price-based heuristic for instrument identification
        if iid is not None and price > 0:
            if price > 10000:
                # MNQ range (~15000-30000)
                mapped = self._trading_config.symbol  # e.g., "MNQM6"
            elif price > 1000:
                # ES range (~3000-7000)
                mapped = "ES"
            else:
                mapped = f"instrument_{iid}"

            self._instrument_map[iid] = mapped
            logger.info(
                "databento.instrument_mapped",
                instrument_id=iid,
                symbol=mapped,
                price=price,
                source="price_heuristic",
            )
            return mapped

        # Absolute fallback
        if iid is not None:
            return f"instrument_{iid}"
        return "unknown"

    def _parse_timestamp(self, record: Any) -> datetime:
        """Parse Databento's nanosecond timestamp to datetime."""
        ts_event = getattr(record, "ts_event", None)
        if ts_event is not None:
            if isinstance(ts_event, (int, float)):
                # Nanosecond Unix timestamp
                return datetime.fromtimestamp(ts_event / 1e9, tz=UTC)
            if isinstance(ts_event, datetime):
                return ts_event
        return datetime.now(tz=UTC)

    @property
    def _large_lot_threshold(self) -> int:
        """Threshold for large lot detection (default: 10 contracts for MNQ)."""
        return self._large_lot_thresh

    async def _dispatch_to_handlers(
        self, handlers: list[DataCallback], data: dict[str, Any]
    ) -> None:
        """Dispatch data to all registered handlers with error isolation."""
        for handler in handlers:
            try:
                await handler(data)
            except Exception:
                logger.exception(
                    "databento.handler_error",
                    handler=handler.__name__,
                    data_type=data.get("type"),
                )

    async def _notify_error(self, data: dict[str, Any]) -> None:
        """Notify all error handlers."""
        await self._dispatch_to_handlers(self._error_handlers, data)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Stop streaming and close the Databento connection."""
        self._running = False

        if self._client is not None:
            try:
                # Databento's Live client may have a close/stop method
                if hasattr(self._client, "close"):
                    self._client.close()
                elif hasattr(self._client, "stop"):
                    self._client.stop()
            except Exception:
                logger.exception("databento.close_error")
            finally:
                self._client = None

        self._connected = False
        logger.info("databento.closed")

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_streaming(self) -> bool:
        return self._running

    @property
    def last_tick_time(self) -> datetime | None:
        return self._last_tick_time

    @property
    def seconds_since_last_tick(self) -> float:
        """Seconds since the last tick was received. Inf if no ticks yet."""
        if self._last_tick_time is None:
            return float("inf")
        return (datetime.now(tz=UTC) - self._last_tick_time).total_seconds()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "streaming": self._running,
            "trades_received": self._trades_received,
            "quotes_received": self._quotes_received,
            "errors": self._errors,
            "reconnect_count": self._reconnect_count,
            "seconds_since_last_tick": self.seconds_since_last_tick,
        }

    def reset_stats(self) -> None:
        """Reset counters (e.g., for a new session)."""
        self._trades_received = 0
        self._quotes_received = 0
        self._errors = 0

    # ── Historical Data for Backtesting ────────────────────────────────────

    @staticmethod
    async def fetch_historical(
        api_key: str,
        symbol: str = "MNQ.FUT",
        start: str = "",
        end: str = "",
        schema: str = "trades",
        dataset: str = "GLBX.MDP3",
    ) -> list[dict[str, Any]]:
        """Fetch historical market data from Databento for backtesting.

        Downloads historical tick data (trades or quotes) and returns
        as a list of dicts in the same format as live streaming callbacks.

        Args:
            api_key: Databento API key.
            symbol: Symbol to fetch (default MNQ.FUT auto-rolls).
            start: Start date in ISO format (e.g., "2026-03-10").
            end: End date in ISO format (e.g., "2026-03-14").
            schema: Data schema — "trades" or "mbp-1".
            dataset: Databento dataset (default CME).

        Returns:
            List of dicts matching the live data callback format.
        """
        if not start or not end:
            raise ValueError("start and end dates are required for historical fetch")

        try:
            import databento as db
        except ImportError:
            raise DatabentoConnectionError(
                "databento package required. Run: pip install databento"
            )

        client = db.Historical(key=api_key)
        records: list[dict[str, Any]] = []

        data = client.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            stype_in="parent",
            symbols=[symbol],
            start=start,
            end=end,
        )

        for record in data:
            record_type = type(record).__name__

            if record_type == "TradeMsg":
                price = record.price / 1e9 if record.price > 1e6 else record.price
                size = record.size
                side = getattr(record, "side", None)
                if side == "A" or side == ord("A"):
                    direction = "buy"
                elif side == "B" or side == ord("B"):
                    direction = "sell"
                else:
                    direction = "unknown"

                ts_event = getattr(record, "ts_event", None)
                ts = (
                    datetime.fromtimestamp(ts_event / 1e9, tz=UTC)
                    if isinstance(ts_event, (int, float))
                    else datetime.now(tz=UTC)
                )

                records.append({
                    "type": "trade",
                    "timestamp": ts,
                    "symbol": symbol,
                    "price": price,
                    "size": size,
                    "direction": direction,
                    "is_large": size >= 10,
                })

            elif record_type in ("MBP1Msg", "Mbp1Msg"):
                levels = record.levels if hasattr(record, "levels") else None
                if levels and len(levels) > 0:
                    bid_price = levels[0].bid_px / 1e9 if levels[0].bid_px > 1e6 else levels[0].bid_px
                    bid_size = levels[0].bid_sz
                    ask_price = levels[0].ask_px / 1e9 if levels[0].ask_px > 1e6 else levels[0].ask_px
                    ask_size = levels[0].ask_sz
                else:
                    continue

                ts_event = getattr(record, "ts_event", None)
                ts = (
                    datetime.fromtimestamp(ts_event / 1e9, tz=UTC)
                    if isinstance(ts_event, (int, float))
                    else datetime.now(tz=UTC)
                )

                records.append({
                    "type": "quote",
                    "timestamp": ts,
                    "symbol": symbol,
                    "bid_price": bid_price,
                    "bid_size": bid_size,
                    "ask_price": ask_price,
                    "ask_size": ask_size,
                })

        logger.info(
            "databento.historical_fetched",
            symbol=symbol,
            schema=schema,
            start=start,
            end=end,
            records=len(records),
        )
        return records

    @staticmethod
    async def compute_rvol_baseline(
        api_key: str,
        symbol: str = "MNQ.FUT",
        lookback_days: int = 20,
        bucket_minutes: int = 5,
        dataset: str = "GLBX.MDP3",
    ) -> dict[str, float]:
        """Compute RVOL baseline from historical data.

        Downloads the last N trading days of trade data and computes
        average volume by time-of-day bucket. The result is a dict
        mapping "HH:MM" strings to average volume for that bucket.

        Args:
            api_key: Databento API key.
            symbol: Symbol to fetch.
            lookback_days: Number of trading days to average over.
            bucket_minutes: Width of each time bucket in minutes.
            dataset: Databento dataset.

        Returns:
            Dict mapping "HH:MM" to average volume per bucket.
            Example: {"09:30": 1542.3, "09:35": 1201.8, ...}
        """
        from collections import defaultdict
        from zoneinfo import ZoneInfo

        et = ZoneInfo("America/New_York")

        # Calculate date range (add weekends buffer)
        end_date = datetime.now(tz=UTC)
        start_date = end_date - timedelta(days=int(lookback_days * 1.5 + 5))

        trades = await DatabentoClient.fetch_historical(
            api_key=api_key,
            symbol=symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            schema="trades",
            dataset=dataset,
        )

        # Bucket volume by time-of-day
        bucket_volumes: dict[str, list[int]] = defaultdict(list)
        day_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for trade in trades:
            ts = trade["timestamp"]
            ts_et = ts.astimezone(et)

            # Only RTH: 9:30 AM - 4:00 PM ET
            if ts_et.hour < 9 or (ts_et.hour == 9 and ts_et.minute < 30):
                continue
            if ts_et.hour >= 16:
                continue

            # Round to bucket
            bucket_min = (ts_et.minute // bucket_minutes) * bucket_minutes
            bucket_key = f"{ts_et.hour:02d}:{bucket_min:02d}"
            day_key = ts_et.strftime("%Y-%m-%d")

            day_buckets[day_key][bucket_key] += trade["size"]

        # Average across days
        for day_key, buckets in day_buckets.items():
            for bucket_key, volume in buckets.items():
                bucket_volumes[bucket_key].append(volume)

        baseline: dict[str, float] = {}
        for bucket_key, volumes in sorted(bucket_volumes.items()):
            baseline[bucket_key] = sum(volumes) / len(volumes) if volumes else 0.0

        logger.info(
            "databento.rvol_baseline_computed",
            trading_days=len(day_buckets),
            buckets=len(baseline),
            lookback_days=lookback_days,
        )
        return baseline
