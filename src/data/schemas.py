"""Pydantic models for raw market data records.

These are the intermediate data structures between raw Databento records
and our domain types (MarketState, OrderFlowData, etc.). They capture
tick-level data before aggregation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class TickDirection(str, Enum):
    """Classify a trade relative to the bid/ask at time of execution."""

    BUY = "buy"  # trade at or above ask → aggressive buyer
    SELL = "sell"  # trade at or below bid → aggressive seller
    UNKNOWN = "unknown"  # midpoint or unclear


class RawTrade(BaseModel):
    """A single trade (time & sales) from Databento."""

    timestamp: datetime
    symbol: str
    price: float
    size: int  # number of contracts
    direction: TickDirection = TickDirection.UNKNOWN
    is_large: bool = False  # True if size >= large lot threshold


class RawQuote(BaseModel):
    """Top-of-book L1 quote from Databento (MBP-1)."""

    timestamp: datetime
    symbol: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int

    @property
    def spread(self) -> float:
        """Bid-ask spread in price units."""
        return self.ask_price - self.bid_price

    @property
    def mid_price(self) -> float:
        """Midpoint price."""
        return (self.bid_price + self.ask_price) / 2.0


class OHLCVBar(BaseModel):
    """Aggregated OHLCV bar (typically 1-second resolution)."""

    timestamp: datetime  # bar open time
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    trade_count: int = 0
    buy_volume: int = 0
    sell_volume: int = 0
    vwap: float = 0.0  # VWAP within this bar

    @property
    def delta(self) -> int:
        """Order flow delta for this bar (buy vol - sell vol)."""
        return self.buy_volume - self.sell_volume

    @property
    def is_up(self) -> bool:
        return self.close >= self.open


class LargeLotEvent(BaseModel):
    """A large lot print detected in the tape.

    MNQ large lot threshold: >= 10 contracts in a single trade.
    """

    timestamp: datetime
    symbol: str
    price: float
    size: int
    direction: TickDirection
    at_bid_or_ask: str = ""  # "bid", "ask", "between"


class VolumeProfile(BaseModel):
    """Session volume profile for computing POC and Value Area.

    Accumulates volume in price buckets (0.25-point resolution for MNQ).
    """

    bucket_size: float = 0.25
    buckets: dict[float, int] = Field(default_factory=dict)
    total_volume: int = 0

    def add_volume(self, price: float, volume: int) -> None:
        """Add volume at a given price, snapped to bucket size."""
        bucket = round(price / self.bucket_size) * self.bucket_size
        self.buckets[bucket] = self.buckets.get(bucket, 0) + volume
        self.total_volume += volume

    @property
    def poc(self) -> float:
        """Point of Control — price with highest volume."""
        if not self.buckets:
            return 0.0
        return max(self.buckets, key=self.buckets.get)  # type: ignore[arg-type]

    def value_area(self, pct: float = 0.70) -> tuple[float, float]:
        """Value Area — range containing `pct` of total volume centered on POC.

        Returns (value_area_low, value_area_high).
        """
        if not self.buckets:
            return (0.0, 0.0)

        poc_price = self.poc
        target_volume = int(self.total_volume * pct)

        # Start from POC and expand outward
        sorted_prices = sorted(self.buckets.keys())
        poc_idx = sorted_prices.index(poc_price)

        low_idx = poc_idx
        high_idx = poc_idx
        accumulated = self.buckets[poc_price]

        while accumulated < target_volume:
            can_go_lower = low_idx > 0
            can_go_higher = high_idx < len(sorted_prices) - 1

            if not can_go_lower and not can_go_higher:
                break

            lower_vol = self.buckets[sorted_prices[low_idx - 1]] if can_go_lower else -1
            higher_vol = self.buckets[sorted_prices[high_idx + 1]] if can_go_higher else -1

            if lower_vol >= higher_vol and can_go_lower:
                low_idx -= 1
                accumulated += self.buckets[sorted_prices[low_idx]]
            elif can_go_higher:
                high_idx += 1
                accumulated += self.buckets[sorted_prices[high_idx]]
            else:
                low_idx -= 1
                accumulated += self.buckets[sorted_prices[low_idx]]

        return (sorted_prices[low_idx], sorted_prices[high_idx])

    def reset(self) -> None:
        """Clear all accumulated volume data."""
        self.buckets.clear()
        self.total_volume = 0


class SessionData(BaseModel):
    """Accumulated session-level data updated on every tick.

    Tracks running VWAP, session high/low, volume, and delta.
    """

    session_date: str = ""
    session_open: float = 0.0
    session_high: float = 0.0
    session_low: float = float("inf")
    session_close: float = 0.0

    # Running VWAP = cumulative(price * volume) / cumulative(volume)
    cum_price_volume: float = 0.0
    cum_volume: int = 0

    # Order flow
    cumulative_delta: float = 0.0  # running buy_vol - sell_vol

    # Volume
    total_volume: int = 0
    total_trades: int = 0
    large_lot_count: int = 0

    @property
    def vwap(self) -> float:
        """Session VWAP."""
        if self.cum_volume == 0:
            return 0.0
        return self.cum_price_volume / self.cum_volume

    def update_from_trade(self, trade: RawTrade) -> None:
        """Update session data from a raw trade."""
        price = trade.price
        size = trade.size

        # Price tracking
        if self.session_open == 0.0:
            self.session_open = price
        self.session_close = price
        if price > self.session_high:
            self.session_high = price
        if price < self.session_low:
            self.session_low = price

        # VWAP
        self.cum_price_volume += price * size
        self.cum_volume += size

        # Volume
        self.total_volume += size
        self.total_trades += 1

        # Delta
        if trade.direction == TickDirection.BUY:
            self.cumulative_delta += size
        elif trade.direction == TickDirection.SELL:
            self.cumulative_delta -= size

        # Large lots
        if trade.is_large:
            self.large_lot_count += 1

    def reset(self) -> None:
        """Reset for a new session."""
        self.session_date = ""
        self.session_open = 0.0
        self.session_high = 0.0
        self.session_low = float("inf")
        self.session_close = 0.0
        self.cum_price_volume = 0.0
        self.cum_volume = 0
        self.cumulative_delta = 0.0
        self.total_volume = 0
        self.total_trades = 0
        self.large_lot_count = 0


class RVOLBaseline(BaseModel):
    """Relative Volume baseline — average volume by time-of-day.

    Pre-computed from 20 days of historical data. Each bucket is a
    5-minute window (e.g., "09:35" covers 09:35:00 to 09:39:59).
    """

    # Maps "HH:MM" -> average cumulative volume at that time
    volume_by_time: dict[str, int] = Field(default_factory=dict)

    def get_expected_volume(self, time_str: str) -> int:
        """Get expected cumulative volume at a given time.

        Args:
            time_str: "HH:MM" format, snapped to nearest 5-min bucket.
        """
        # Snap to 5-minute bucket
        parts = time_str.split(":")
        hour = int(parts[0])
        minute = (int(parts[1]) // 5) * 5
        bucket = f"{hour:02d}:{minute:02d}"
        return self.volume_by_time.get(bucket, 0)

    def compute_rvol(self, current_volume: int, time_str: str) -> float:
        """Compute RVOL = current volume / expected volume at this time.

        Returns 1.0 if no baseline data available.
        """
        expected = self.get_expected_volume(time_str)
        if expected == 0:
            return 1.0
        return current_volume / expected

    @classmethod
    def load_from_file(cls, filepath: str | Path) -> RVOLBaseline:
        """Load RVOL baseline from a pre-computed JSON file.

        Expected format:
        {
            "volume_by_time": {
                "09:30": 1200,
                "09:35": 3500,
                ...
            }
        }

        Returns a default (empty) baseline if file doesn't exist or is invalid.
        """
        path = Path(filepath)
        if not path.exists():
            logger.debug("rvol_baseline.file_not_found", path=str(path))
            return cls()

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("rvol_baseline.file_parse_error", error=str(e))
            return cls()

        volume_by_time = data.get("volume_by_time", {})
        # Convert keys to strings and values to ints
        cleaned: dict[str, int] = {}
        for k, v in volume_by_time.items():
            try:
                cleaned[str(k)] = int(v)
            except (ValueError, TypeError):
                continue

        logger.info("rvol_baseline.loaded", buckets=len(cleaned), path=str(path))
        return cls(volume_by_time=cleaned)

    def save_to_file(self, filepath: str | Path) -> None:
        """Save RVOL baseline to a JSON file for future use."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {"volume_by_time": self.volume_by_time}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("rvol_baseline.saved", buckets=len(self.volume_by_time), path=str(path))
