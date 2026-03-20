"""Order flow analysis engine for smart money MNQ futures trading.

Consumes Databento L2 (mbp-10) depth updates and tick-level trades to produce:
  - DOM imbalance & bid/ask stacking detection
  - Shannon entropy of price changes (trending vs choppy regime)
  - VPIN (Volume-Synchronized Probability of Informed Trading)
  - Absorption detection (large volume absorbed with minimal price movement)
"""

from __future__ import annotations

import math
from collections import deque
from datetime import UTC, datetime, timedelta

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MNQ_TICK = 0.25  # Minimum tick size for Micro E-mini Nasdaq

_ABSORPTION_WINDOW_SECONDS = 30
_ABSORPTION_MAX_TRADES = 500
_ABSORPTION_VOLUME_THRESHOLD = 500
_ABSORPTION_PRICE_RANGE_FACTOR = 0.50  # price range must be < 50% of avg 1m bar
_ABSORPTION_SIDE_RATIO = 2.0

_DEFAULT_AVG_1MIN_RANGE = 5.0  # fallback if no bars observed yet (MNQ points)


class OrderFlowEngine:
    """Real-time order flow analytics from L2 depth and trade data.

    Usage::

        engine = OrderFlowEngine()

        # On each mbp-10 update:
        engine.update_depth(levels)

        # On each trade:
        engine.update_trade(price, size, direction, timestamp)

        # Query current state:
        state = engine.snapshot()
    """

    def __init__(
        self,
        vpin_bucket_size: int = 500,
        entropy_window: int = 50,
    ) -> None:
        # -- DOM state --
        self._dom_imbalance: float = 1.0
        self._dom_stacking: str = "neutral"
        self._dom_data_available: bool = False  # True after first update_depth() call

        # -- Entropy state --
        self._entropy_window = entropy_window
        self._price_changes: deque[str] = deque(maxlen=entropy_window)
        self._last_trade_price: float | None = None

        # -- VPIN state --
        self._vpin_bucket_size = vpin_bucket_size
        self._current_bucket_buy: int = 0
        self._current_bucket_sell: int = 0
        self._current_bucket_volume: int = 0
        self._finalized_buckets: deque[float] = deque(maxlen=50)

        # -- Absorption state --
        self._recent_trades: deque[dict] = deque(maxlen=_ABSORPTION_MAX_TRADES)
        self._absorption_detected: bool = False
        self._absorption_side: str = ""

        # -- 1-min bar range estimator (EMA) --
        self._avg_1min_range: float = _DEFAULT_AVG_1MIN_RANGE
        self._bar_high: float | None = None
        self._bar_low: float | None = None
        self._bar_start: datetime | None = None

    # ------------------------------------------------------------------
    # DOM Imbalance
    # ------------------------------------------------------------------
    def update_depth(self, levels: list[dict]) -> None:
        """Process a 10-level mbp depth snapshot.

        Each element in *levels* should contain:
        ``{"bid_price": float, "bid_size": int, "ask_price": float, "ask_size": int}``
        """
        if not levels:
            return

        self._dom_data_available = True
        total_bid = 0
        total_ask = 0
        per_level_ratios: list[float] = []

        for lvl in levels:
            bid_sz = lvl.get("bid_size", 0)
            ask_sz = lvl.get("ask_size", 0)
            total_bid += bid_sz
            total_ask += ask_sz

            if ask_sz > 0:
                per_level_ratios.append(bid_sz / ask_sz)
            elif bid_sz > 0:
                per_level_ratios.append(float("inf"))
            else:
                per_level_ratios.append(1.0)

        self._dom_imbalance = total_bid / total_ask if total_ask > 0 else (
            float("inf") if total_bid > 0 else 1.0
        )

        # Stacking detection: 3+ consecutive levels with >2x imbalance
        self._dom_stacking = self._detect_stacking(per_level_ratios)

    def update_bbo(self, bid_size: int, ask_size: int) -> None:
        """Update DOM imbalance from BBO (mbp-1) as a fallback when mbp-10 is unavailable.

        This gives a single-level imbalance ratio, which is less granular than
        10-level depth but still provides directional pressure information.
        Stacking detection is not possible with only 1 level, so it stays 'neutral'.
        """
        if self._dom_data_available:
            # mbp-10 is providing full depth — don't override with BBO
            return
        if bid_size <= 0 and ask_size <= 0:
            return

        self._dom_imbalance = bid_size / ask_size if ask_size > 0 else (
            float("inf") if bid_size > 0 else 1.0
        )
        # Mark as available — even though it's BBO-only, it's real market data
        # The LLM context will show dom_data_available=True but stacking stays 'neutral'
        # since we can't detect stacking from a single level.

    @staticmethod
    def _detect_stacking(ratios: list[float]) -> str:
        """Return 'bid_stacking', 'ask_stacking', or 'neutral'."""
        bid_streak = 0
        ask_streak = 0
        max_bid_streak = 0
        max_ask_streak = 0

        for r in ratios:
            # Bid stacking: bid_sz > 2x ask_sz
            if r > 2.0:
                bid_streak += 1
                max_bid_streak = max(max_bid_streak, bid_streak)
            else:
                bid_streak = 0

            # Ask stacking: ask_sz > 2x bid_sz  (ratio < 0.5)
            if r < 0.5:
                ask_streak += 1
                max_ask_streak = max(max_ask_streak, ask_streak)
            else:
                ask_streak = 0

        if max_bid_streak >= 3:
            return "bid_stacking"
        if max_ask_streak >= 3:
            return "ask_stacking"
        return "neutral"

    # ------------------------------------------------------------------
    # Trade processing (entropy, VPIN, absorption)
    # ------------------------------------------------------------------
    def update_trade(
        self,
        price: float,
        size: int,
        direction: str,
        timestamp: datetime,
    ) -> None:
        """Process a single trade tick.

        Parameters
        ----------
        price : float
            Trade price.
        size : int
            Number of contracts.
        direction : str
            ``'buy'``, ``'sell'``, or ``'unknown'``.
        timestamp : datetime
            Exchange timestamp of the trade.
        """
        # -- 1) Entropy: classify price change --
        if self._last_trade_price is not None:
            delta = price - self._last_trade_price
            if abs(delta) <= MNQ_TICK:
                self._price_changes.append("flat")
            elif delta > 0:
                self._price_changes.append("up")
            else:
                self._price_changes.append("down")
        self._last_trade_price = price

        # -- 2) VPIN: accumulate into volume-sync bucket --
        self._accumulate_vpin(size, direction)

        # -- 3) Absorption: store trade and re-evaluate --
        self._recent_trades.append({
            "price": price,
            "size": size,
            "direction": direction,
            "timestamp": timestamp,
        })
        self._evaluate_absorption(timestamp)

        # -- 4) Update 1-min bar range estimator --
        self._update_bar_range(price, timestamp)

    # ------------------------------------------------------------------
    # Shannon Entropy
    # ------------------------------------------------------------------
    def _compute_entropy(self) -> float:
        """Normalized Shannon entropy over recent price change categories."""
        n = len(self._price_changes)
        if n < 5:
            return 0.5  # insufficient data, assume neutral

        counts = {"up": 0, "down": 0, "flat": 0}
        for c in self._price_changes:
            counts[c] += 1

        entropy = 0.0
        for cat in ("up", "down", "flat"):
            p = counts[cat] / n
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by log2(3) so result is 0-1
        max_entropy = math.log2(3)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------
    def _accumulate_vpin(self, size: int, direction: str) -> None:
        """Add trade volume to the current VPIN bucket, finalizing when full."""
        remaining = size

        while remaining > 0:
            space = self._vpin_bucket_size - self._current_bucket_volume
            fill = min(remaining, space)

            if direction == "buy":
                self._current_bucket_buy += fill
            elif direction == "sell":
                self._current_bucket_sell += fill
            else:
                # Unknown: split 50/50
                self._current_bucket_buy += fill // 2
                self._current_bucket_sell += fill - fill // 2

            self._current_bucket_volume += fill
            remaining -= fill

            if self._current_bucket_volume >= self._vpin_bucket_size:
                imbalance = (
                    abs(self._current_bucket_buy - self._current_bucket_sell)
                    / self._vpin_bucket_size
                )
                self._finalized_buckets.append(imbalance)
                self._current_bucket_buy = 0
                self._current_bucket_sell = 0
                self._current_bucket_volume = 0

    def _compute_vpin(self) -> float:
        """Mean imbalance across finalized VPIN buckets."""
        if not self._finalized_buckets:
            return 0.0
        return sum(self._finalized_buckets) / len(self._finalized_buckets)

    # ------------------------------------------------------------------
    # Absorption Detection
    # ------------------------------------------------------------------
    def _evaluate_absorption(self, now: datetime) -> None:
        """Check for absorption in the recent trade window."""
        # Trim trades older than the time window
        cutoff = now - timedelta(seconds=_ABSORPTION_WINDOW_SECONDS)
        while self._recent_trades and self._recent_trades[0]["timestamp"] < cutoff:
            self._recent_trades.popleft()

        if len(self._recent_trades) < 10:
            self._absorption_detected = False
            self._absorption_side = ""
            return

        total_volume = 0
        buy_volume = 0
        sell_volume = 0
        high = -math.inf
        low = math.inf

        for t in self._recent_trades:
            total_volume += t["size"]
            if t["direction"] == "buy":
                buy_volume += t["size"]
            elif t["direction"] == "sell":
                sell_volume += t["size"]
            high = max(high, t["price"])
            low = min(low, t["price"])

        price_range = high - low

        # Check criteria
        if total_volume < _ABSORPTION_VOLUME_THRESHOLD:
            self._absorption_detected = False
            self._absorption_side = ""
            return

        if price_range >= self._avg_1min_range * _ABSORPTION_PRICE_RANGE_FACTOR:
            self._absorption_detected = False
            self._absorption_side = ""
            return

        # Check side dominance
        if buy_volume >= sell_volume * _ABSORPTION_SIDE_RATIO:
            # Large buy volume, price barely moved = absorbing sell pressure
            self._absorption_detected = True
            self._absorption_side = "bid"
            logger.info(
                "absorption_detected",
                side="bid",
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                price_range=price_range,
            )
        elif sell_volume >= buy_volume * _ABSORPTION_SIDE_RATIO:
            # Large sell volume, price barely moved = absorbing buy pressure
            self._absorption_detected = True
            self._absorption_side = "ask"
            logger.info(
                "absorption_detected",
                side="ask",
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                price_range=price_range,
            )
        else:
            self._absorption_detected = False
            self._absorption_side = ""

    # ------------------------------------------------------------------
    # 1-Minute Bar Range Estimator (for absorption baseline)
    # ------------------------------------------------------------------
    def _update_bar_range(self, price: float, timestamp: datetime) -> None:
        """Maintain a running EMA of 1-minute bar ranges."""
        if self._bar_start is None:
            self._bar_start = timestamp
            self._bar_high = price
            self._bar_low = price
            return

        self._bar_high = max(self._bar_high, price)  # type: ignore[arg-type]
        self._bar_low = min(self._bar_low, price)  # type: ignore[arg-type]

        if timestamp - self._bar_start >= timedelta(minutes=1):
            bar_range = self._bar_high - self._bar_low  # type: ignore[operator]
            if bar_range > 0:
                # EMA with alpha = 0.1 for smooth adaptation
                alpha = 0.1
                if self._avg_1min_range == _DEFAULT_AVG_1MIN_RANGE:
                    self._avg_1min_range = bar_range  # seed with first real bar
                else:
                    self._avg_1min_range = (
                        alpha * bar_range + (1 - alpha) * self._avg_1min_range
                    )

            # Reset bar
            self._bar_start = timestamp
            self._bar_high = price
            self._bar_low = price

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return current order flow state as a dictionary.

        Keys
        ----
        dom_imbalance : float
            Ratio of total bid size to total ask size across 10 levels.
            >1.3 bullish, <0.7 bearish.
        dom_stacking : str
            ``'bid_stacking'``, ``'ask_stacking'``, or ``'neutral'``.
        entropy : float
            Normalized Shannon entropy (0-1). <0.4 trending, >0.7 choppy, >0.85 hard block.
        vpin : float
            Volume-Synchronized Probability of Informed Trading. >0.6 significant.
        absorption_detected : bool
            Whether absorption is currently detected.
        absorption_side : str
            ``'bid'`` (bullish), ``'ask'`` (bearish), or ``''``.
        """
        return {
            "dom_imbalance": round(self._dom_imbalance, 4),
            "dom_stacking": self._dom_stacking,
            "entropy": round(self._compute_entropy(), 4),
            "vpin": round(self._compute_vpin(), 4),
            "absorption_detected": self._absorption_detected,
            "absorption_side": self._absorption_side,
            "dom_data_available": self._dom_data_available,
        }
