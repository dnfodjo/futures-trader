"""Tick processor — converts raw trades/quotes into computed market metrics.

Takes raw trade and quote dicts from DatabentoClient and computes:
- Cumulative delta (buy vol - sell vol)
- Rolling delta (1-min, 5-min windows)
- Delta trend classification
- Large lot detection (>= 10 contracts per trade)
- Tape speed (trades per second over rolling 10s window)
- 1-second OHLCV bars with per-bar delta
- Session running VWAP
- Volume profile (POC + Value Area)

All computation is pure Python / in-memory. No LLM calls.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any, Callable, Coroutine

import structlog

from src.data.schemas import (
    LargeLotEvent,
    OHLCVBar,
    RawQuote,
    RawTrade,
    SessionData,
    TickDirection,
    VolumeProfile,
)

logger = structlog.get_logger()

BarCallback = Callable[[OHLCVBar], Coroutine[Any, Any, None]]
LargeLotCallback = Callable[[LargeLotEvent], Coroutine[Any, Any, None]]


class TickProcessor:
    """Processes raw ticks into structured market metrics.

    Receives trade/quote dicts from DatabentoClient and computes
    delta, tape speed, OHLCV bars, volume profile, and session data.

    Usage:
        processor = TickProcessor(large_lot_threshold=10)
        processor.on_bar(my_bar_handler)  # get 1-second bars
        await processor.process_trade(trade_dict)
        await processor.process_quote(quote_dict)
        snapshot = processor.snapshot()  # current metrics
    """

    def __init__(
        self,
        target_symbol: str = "",
        large_lot_threshold: int = 10,
        bar_interval_sec: float = 1.0,
        tape_speed_window_sec: float = 10.0,
        delta_1min_window: int = 60,
        delta_5min_window: int = 300,
    ) -> None:
        # Symbol filter: only process trades matching this symbol.
        # Empty string means process all trades (backward compat).
        self._target_symbol = target_symbol.upper() if target_symbol else ""
        self._symbol_filter_logged = False  # Log first filtered symbol for debugging
        self._symbol_accepted_logged = False  # Log first accepted symbol
        self._large_lot_threshold = large_lot_threshold
        self._bar_interval_sec = bar_interval_sec
        self._tape_speed_window = tape_speed_window_sec

        # Current bar being built
        self._current_bar: OHLCVBar | None = None
        self._bar_start_time: float = 0.0
        self._bar_cum_pv: float = 0.0  # cumulative price*volume for bar VWAP

        # Trade timestamps for tape speed (monotonic timestamps)
        self._trade_times: deque[float] = deque()

        # Delta tracking — store (timestamp_mono, signed_size) tuples
        self._delta_window: deque[tuple[float, int]] = deque()
        self._delta_1min_window = delta_1min_window
        self._delta_5min_window = delta_5min_window

        # Large lot events (recent, for metrics)
        self._large_lots_5min: deque[tuple[float, LargeLotEvent]] = deque()

        # Session accumulators
        self._session = SessionData()
        self._volume_profile = VolumeProfile()

        # Latest quote
        self._last_quote: RawQuote | None = None

        # Callbacks
        self._bar_handlers: list[BarCallback] = []
        self._large_lot_handlers: list[LargeLotCallback] = []

        # Lock for thread safety (multiple concurrent trade events)
        self._lock = asyncio.Lock()

    # ── Handler Registration ──────────────────────────────────────────────

    def on_bar(self, handler: BarCallback) -> None:
        """Register callback for completed OHLCV bars."""
        self._bar_handlers.append(handler)

    def on_large_lot(self, handler: LargeLotCallback) -> None:
        """Register callback for large lot detections."""
        self._large_lot_handlers.append(handler)

    # ── Processing ────────────────────────────────────────────────────────

    async def process_trade(self, data: dict[str, Any]) -> None:
        """Process a raw trade dict from DatabentoClient.

        Expected keys: timestamp, symbol, price, size, direction, is_large
        """
        # Symbol filter: skip trades from other instruments (e.g., ES when
        # tracking MNQ).  This prevents cross-instrument VWAP pollution.
        # Use root symbol (first 3 chars, e.g., "MNQ" from "MNQM6") for
        # matching since Databento may send different contract variants.
        if self._target_symbol:
            trade_symbol = (data.get("symbol") or "").upper()
            root = self._target_symbol[:3] if len(self._target_symbol) >= 3 else self._target_symbol
            if root not in trade_symbol:
                if not self._symbol_filter_logged:
                    logger.info(
                        "tick_processor.symbol_filtered",
                        trade_symbol=trade_symbol,
                        target=self._target_symbol,
                        root=root,
                    )
                    self._symbol_filter_logged = True
                return

        # Log first accepted trade for debugging
        if self._target_symbol and not self._symbol_accepted_logged:
            trade_symbol = (data.get("symbol") or "").upper()
            logger.info(
                "tick_processor.symbol_accepted",
                trade_symbol=trade_symbol,
                target=self._target_symbol,
                price=data.get("price"),
            )
            self._symbol_accepted_logged = True

        async with self._lock:
            now_mono = time.monotonic()

            # Parse direction
            dir_str = data.get("direction", "unknown")
            direction = TickDirection(dir_str) if dir_str in TickDirection.__members__.values() else TickDirection.UNKNOWN

            trade = RawTrade(
                timestamp=data["timestamp"],
                symbol=data["symbol"],
                price=data["price"],
                size=data["size"],
                direction=direction,
                is_large=data.get("is_large", data["size"] >= self._large_lot_threshold),
            )

            # Set session date on first trade
            if not self._session.session_date:
                self._session.session_date = trade.timestamp.strftime("%Y-%m-%d")

            # Update session data
            self._session.update_from_trade(trade)
            if self._session.total_trades <= 3:
                logger.info(
                    "tick_processor.trade_processed",
                    price=trade.price,
                    session_close=self._session.session_close,
                    total_trades=self._session.total_trades,
                    session_id=id(self._session),
                )

            # Update volume profile
            self._volume_profile.add_volume(trade.price, trade.size)

            # Delta tracking
            signed_size = trade.size if direction == TickDirection.BUY else (
                -trade.size if direction == TickDirection.SELL else 0
            )
            self._delta_window.append((now_mono, signed_size))

            # Tape speed tracking
            self._trade_times.append(now_mono)

            # OHLCV bar update
            await self._update_bar(trade, now_mono)

            # Large lot detection
            if trade.is_large:
                at_boa = "unknown"
                if self._last_quote:
                    if trade.price <= self._last_quote.bid_price:
                        at_boa = "bid"
                    elif trade.price >= self._last_quote.ask_price:
                        at_boa = "ask"
                    else:
                        at_boa = "between"

                event = LargeLotEvent(
                    timestamp=trade.timestamp,
                    symbol=trade.symbol,
                    price=trade.price,
                    size=trade.size,
                    direction=direction,
                    at_bid_or_ask=at_boa,
                )
                self._large_lots_5min.append((now_mono, event))

                for handler in self._large_lot_handlers:
                    try:
                        await handler(event)
                    except Exception:
                        logger.exception("tick_processor.large_lot_handler_error")

            # Prune old data
            self._prune_windows(now_mono)

    async def process_quote(self, data: dict[str, Any]) -> None:
        """Process a raw quote dict from DatabentoClient.

        Expected keys: timestamp, symbol, bid_price, bid_size, ask_price, ask_size
        """
        async with self._lock:
            self._last_quote = RawQuote(
                timestamp=data["timestamp"],
                symbol=data["symbol"],
                bid_price=data["bid_price"],
                bid_size=data["bid_size"],
                ask_price=data["ask_price"],
                ask_size=data["ask_size"],
            )

    # ── Bar Building ──────────────────────────────────────────────────────

    async def _update_bar(self, trade: RawTrade, now_mono: float) -> None:
        """Update the current 1-second bar or emit a completed bar."""
        if self._current_bar is None:
            # Start a new bar
            self._current_bar = OHLCVBar(
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.size,
                trade_count=1,
                vwap=trade.price,  # single trade → VWAP = price
                buy_volume=trade.size if trade.direction == TickDirection.BUY else 0,
                sell_volume=trade.size if trade.direction == TickDirection.SELL else 0,
            )
            self._bar_cum_pv = trade.price * trade.size
            self._bar_start_time = now_mono
            return

        # Check if we need to emit the current bar and start a new one
        elapsed = now_mono - self._bar_start_time
        if elapsed >= self._bar_interval_sec:
            # Emit completed bar (VWAP already correct from updates)
            completed_bar = self._current_bar

            for handler in self._bar_handlers:
                try:
                    await handler(completed_bar)
                except Exception:
                    logger.exception("tick_processor.bar_handler_error")

            # Start new bar
            self._current_bar = OHLCVBar(
                timestamp=trade.timestamp,
                symbol=trade.symbol,
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.size,
                trade_count=1,
                vwap=trade.price,
                buy_volume=trade.size if trade.direction == TickDirection.BUY else 0,
                sell_volume=trade.size if trade.direction == TickDirection.SELL else 0,
            )
            self._bar_cum_pv = trade.price * trade.size
            self._bar_start_time = now_mono
        else:
            # Update current bar
            bar = self._current_bar
            bar.high = max(bar.high, trade.price)
            bar.low = min(bar.low, trade.price)
            bar.close = trade.price
            bar.volume += trade.size
            bar.trade_count += 1
            if trade.direction == TickDirection.BUY:
                bar.buy_volume += trade.size
            elif trade.direction == TickDirection.SELL:
                bar.sell_volume += trade.size
            # Update bar VWAP: cumulative(price * volume) / cumulative(volume)
            self._bar_cum_pv += trade.price * trade.size
            bar.vwap = self._bar_cum_pv / bar.volume

    # ── Windowed Metrics ──────────────────────────────────────────────────

    def _prune_windows(self, now_mono: float) -> None:
        """Remove expired entries from rolling windows."""
        # Tape speed: keep last N seconds
        cutoff_tape = now_mono - self._tape_speed_window
        while self._trade_times and self._trade_times[0] < cutoff_tape:
            self._trade_times.popleft()

        # Delta: keep last 5 minutes (300 seconds)
        cutoff_delta = now_mono - self._delta_5min_window
        while self._delta_window and self._delta_window[0][0] < cutoff_delta:
            self._delta_window.popleft()

        # Large lots: keep last 5 minutes
        cutoff_ll = now_mono - 300
        while self._large_lots_5min and self._large_lots_5min[0][0] < cutoff_ll:
            self._large_lots_5min.popleft()

    @property
    def tape_speed(self) -> float:
        """Trades per second over the rolling window."""
        if not self._trade_times:
            return 0.0
        count = len(self._trade_times)
        if count < 2:
            return float(count)
        window = self._trade_times[-1] - self._trade_times[0]
        if window <= 0:
            return float(count)
        return count / window

    @property
    def delta_1min(self) -> float:
        """Cumulative delta over the last 1 minute."""
        if not self._delta_window:
            return 0.0
        now = time.monotonic()
        cutoff = now - self._delta_1min_window
        return float(sum(s for t, s in self._delta_window if t >= cutoff))

    @property
    def delta_5min(self) -> float:
        """Cumulative delta over the last 5 minutes."""
        return float(sum(s for _, s in self._delta_window))

    @property
    def cumulative_delta(self) -> float:
        """Session cumulative delta."""
        return self._session.cumulative_delta

    @property
    def delta_trend(self) -> str:
        """Classify the delta trend.

        - "positive": 1-min and 5-min delta both positive
        - "negative": 1-min and 5-min delta both negative
        - "flipping": 1-min and 5-min disagree (momentum shift)
        - "neutral": near zero
        """
        d1 = self.delta_1min
        d5 = self.delta_5min

        # Threshold for "near zero" — less than 5 contracts delta
        threshold = 5.0

        if abs(d1) < threshold and abs(d5) < threshold:
            return "neutral"

        if d1 > 0 and d5 > 0:
            return "positive"
        if d1 < 0 and d5 < 0:
            return "negative"

        return "flipping"

    @property
    def large_lot_count_5min(self) -> int:
        """Number of large lot prints in the last 5 minutes."""
        return len(self._large_lots_5min)

    # ── Session & Volume Profile ──────────────────────────────────────────

    @property
    def session_data(self) -> SessionData:
        """Current session accumulated data."""
        return self._session

    @property
    def volume_profile(self) -> VolumeProfile:
        """Current session volume profile."""
        return self._volume_profile

    @property
    def last_price(self) -> float:
        """Last trade price."""
        return self._session.session_close

    @property
    def last_quote(self) -> RawQuote | None:
        """Most recent quote."""
        return self._last_quote

    # ── Snapshot (for MarketState assembly) ────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of all computed metrics.

        Used by the state engine to build the full MarketState.
        """
        vp = self._volume_profile
        va_low, va_high = vp.value_area()

        if self._session.session_close > 0 and self._session.total_trades > 0:
            pass  # Normal: live ticks flowing
        else:
            logger.info(
                "tick_processor.snapshot_price_debug",
                session_close=self._session.session_close,
                total_trades=self._session.total_trades,
                session_id=id(self._session),
            )

        return {
            # Price
            "last_price": self._session.session_close,
            "bid": self._last_quote.bid_price if self._last_quote else 0.0,
            "ask": self._last_quote.ask_price if self._last_quote else 0.0,
            "spread": self._last_quote.spread if self._last_quote else 0.0,
            # Session levels
            "session_open": self._session.session_open,
            "session_high": self._session.session_high,
            "session_low": self._session.session_low if self._session.session_low != float("inf") else 0.0,
            "vwap": self._session.vwap,
            "poc": vp.poc,
            "value_area_high": va_high,
            "value_area_low": va_low,
            # Order flow
            "cumulative_delta": self._session.cumulative_delta,
            "delta_1min": self.delta_1min,
            "delta_5min": self.delta_5min,
            "delta_trend": self.delta_trend,
            "tape_speed": round(self.tape_speed, 2),
            "large_lot_count_5min": self.large_lot_count_5min,
            # Volume
            "total_volume": self._session.total_volume,
            "total_trades": self._session.total_trades,
        }

    # ── Reset ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all accumulators for a new session."""
        self._current_bar = None
        self._bar_start_time = 0.0
        self._bar_cum_pv = 0.0
        self._trade_times.clear()
        self._delta_window.clear()
        self._large_lots_5min.clear()
        self._session.reset()
        self._volume_profile.reset()
        self._last_quote = None

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_trades": self._session.total_trades,
            "total_volume": self._session.total_volume,
            "session_high": self._session.session_high,
            "session_low": self._session.session_low if self._session.session_low != float("inf") else 0.0,
            "vwap": self._session.vwap,
            "cumulative_delta": self._session.cumulative_delta,
            "tape_speed": round(self.tape_speed, 2),
            "large_lots_5min": self.large_lot_count_5min,
            "poc": self._volume_profile.poc,
        }
