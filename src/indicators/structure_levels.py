"""HTF Structure Level Engine — S/R Zones from 1h/4h/Daily/Weekly fractals.

Computes volume-confirmed fractal support/resistance zones from higher-timeframe
OHLCV bars and provides proximity checks for the confluence scoring engine.

Usage:
    manager = StructureLevelManager()

    # At startup: compute levels from historical bars
    manager.compute_levels(bars_1h, "1h")
    manager.compute_levels(bars_daily, "D")

    # On each scoring cycle:
    result = manager.check_proximity(price, "long", bars_1m, bars_5m)
    #  => {"bounce_score": 1, "bos_score": 0, "blocked": False, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Proximity multipliers per timeframe (fraction of level's own ATR)
PROXIMITY_MULT: dict[str, float] = {
    "1h": 0.5,
    "4h": 0.75,
    "D": 1.0,
    "W": 1.5,
}

# Anti-signal multiplier (uses daily_atr for D and W)
ANTI_SIGNAL_MULT: float = 1.0

# Volume MA period for fractal confirmation (matches Pine Script TF1_VolMA1Input)
VOLUME_MA_PERIOD: int = 6

# ATR period
ATR_PERIOD: int = 14

# BOS volume threshold
BOS_VOLUME_MULT: float = 1.5

# Max ages for level pruning (days)
MAX_AGE_DAYS: dict[str, int] = {
    "1h": 5,
    "4h": 20,
    "D": 180,
    "W": 99999,
}

# Broken level max age (days)
BROKEN_MAX_AGE_DAYS: int = 2

# Timeframe priority ordering (higher index = higher priority)
TF_PRIORITY: dict[str, int] = {
    "1h": 0,
    "4h": 1,
    "D": 2,
    "W": 3,
}

# Anti-signal timeframes (only D and W)
ANTI_SIGNAL_TIMEFRAMES: set[str] = {"D", "W"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StructureLevel:
    """A single S/R level with metadata."""

    price: float  # The key price level (fractal high or low)
    zone_high: float  # Upper bound of zone
    zone_low: float  # Lower bound of zone
    level_type: str  # "support" or "resistance"
    timeframe: str  # "1h", "4h", "D", "W"
    timeframe_atr: float  # ATR(14) of THIS level's timeframe
    volume_confirmed: bool  # Was volume above MA at fractal?
    created_at: datetime
    last_tested: datetime | None = None
    touch_count: int = 0
    broken: bool = False
    broken_at: datetime | None = None  # When the level was broken (for BOS recency)
    flipped: bool = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_atr(bars: list[dict], period: int = ATR_PERIOD) -> float:
    """Compute ATR(period) from OHLCV bars.

    Uses the standard True Range formula:
      TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    """
    if len(bars) < 2:
        return 0.0

    true_ranges: list[float] = []
    for i in range(1, len(bars)):
        high = bars[i]["high"]
        low = bars[i]["low"]
        prev_close = bars[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    if not true_ranges:
        return 0.0

    # Simple average of last `period` true ranges
    recent = true_ranges[-period:] if len(true_ranges) >= period else true_ranges
    return sum(recent) / len(recent)


def _compute_volume_sma(volumes: list[float], period: int = VOLUME_MA_PERIOD) -> float:
    """Compute SMA of volumes over the given period."""
    if not volumes or len(volumes) < period:
        if not volumes:
            return 0.0
        return sum(volumes) / len(volumes)
    window = volumes[-period:]
    return sum(window) / len(window)


# ---------------------------------------------------------------------------
# Structure Level Manager
# ---------------------------------------------------------------------------


class StructureLevelManager:
    """Manages HTF S/R levels across timeframes."""

    def __init__(self) -> None:
        self._levels: list[StructureLevel] = []
        self.daily_atr: float = 0.0  # Stored for anti-signal checks
        # Rolling buffer of live 1h bars for intraday level recomputation
        self._live_1h_bars: list[dict] = []

    @property
    def levels(self) -> list[StructureLevel]:
        """Read-only access to all current levels."""
        return list(self._levels)

    # ------------------------------------------------------------------
    # compute_levels
    # ------------------------------------------------------------------

    def compute_levels(
        self, bars: list[dict], timeframe: str
    ) -> list[StructureLevel]:
        """Compute fractal S/R zones from OHLCV bars.

        Uses volume-confirmed fractal logic adapted from Pine Script:
          Fractal Up (resistance):
            high[i-3] > high[i-4] > high[i-5]
            high[i-2] < high[i-3]
            high[i-1] < high[i-2]
            volume[i-3] > volume_MA[i-3]
          Fractal Down (support): mirror with lows.

        Zone boundaries:
          Resistance: (high[i-3], max(open[i-3], close[i-3]))
          Support:    (low[i-3], min(open[i-3], close[i-3]))

        Args:
            bars: OHLCV bars (oldest first). Each bar has keys:
                  open, high, low, close, volume.
            timeframe: "1h", "4h", "D", or "W".

        Returns:
            List of newly detected StructureLevel objects (also stored internally).
        """
        if len(bars) < 6:
            return []

        # Clear existing levels for this timeframe before recomputing
        # (prevents duplicates on restart/reconnect)
        self._levels = [lv for lv in self._levels if lv.timeframe != timeframe]

        # Compute ATR for this timeframe
        atr = _compute_atr(bars, ATR_PERIOD)
        if atr <= 0:
            atr = 1.0  # Fallback to avoid zero-division

        # Store daily ATR for anti-signal checks
        if timeframe == "D":
            self.daily_atr = atr

        # Build volume array for SMA calculation
        volumes = [b["volume"] for b in bars]

        new_levels: list[StructureLevel] = []

        # Pine Script indexing: at bar i, we look back to i-5..i-1 with pivot at i-3.
        # In 0-based array terms: iterate from index 5 to len-1.
        # At index i: pivot is at i-3, left side is i-4 and i-5, right side is i-2 and i-1.
        for i in range(5, len(bars)):
            pivot_idx = i - 3
            bar_pivot = bars[pivot_idx]

            # Use pivot bar's timestamp for created_at (not wall-clock)
            # so pruning is based on actual bar age, not computation time.
            pivot_ts = bar_pivot.get("timestamp")
            if pivot_ts is not None and isinstance(pivot_ts, datetime):
                pivot_created_at = pivot_ts if pivot_ts.tzinfo else pivot_ts.replace(tzinfo=UTC)
            elif pivot_ts is not None and isinstance(pivot_ts, str):
                try:
                    pivot_created_at = datetime.fromisoformat(pivot_ts)
                    if pivot_created_at.tzinfo is None:
                        pivot_created_at = pivot_created_at.replace(tzinfo=UTC)
                except (ValueError, TypeError):
                    pivot_created_at = datetime.now(tz=UTC)
            else:
                pivot_created_at = datetime.now(tz=UTC)

            # Volume MA at pivot: SMA(6) of volumes ending at pivot_idx
            vol_start = max(0, pivot_idx - VOLUME_MA_PERIOD + 1)
            vol_window = volumes[vol_start : pivot_idx + 1]
            vol_ma = sum(vol_window) / len(vol_window) if vol_window else 0.0

            volume_confirmed = bar_pivot["volume"] > vol_ma

            # -- Fractal Up (Resistance) --
            # high[i-3] > high[i-4] > high[i-5]
            # high[i-2] < high[i-3]
            # high[i-1] < high[i-2]
            h_pivot = bars[pivot_idx]["high"]
            h_left1 = bars[i - 4]["high"]
            h_left2 = bars[i - 5]["high"]
            h_right1 = bars[i - 2]["high"]
            h_right2 = bars[i - 1]["high"]

            if (
                h_pivot > h_left1
                and h_left1 > h_left2
                and h_right1 < h_pivot
                and h_right2 < h_right1
                and volume_confirmed
            ):
                zone_high = h_pivot
                zone_low = max(bar_pivot["open"], bar_pivot["close"])
                level = StructureLevel(
                    price=h_pivot,
                    zone_high=zone_high,
                    zone_low=zone_low,
                    level_type="resistance",
                    timeframe=timeframe,
                    timeframe_atr=atr,
                    volume_confirmed=True,
                    created_at=pivot_created_at,
                )
                new_levels.append(level)
                self._levels.append(level)

            # -- Fractal Down (Support) --
            # low[i-3] < low[i-4] < low[i-5]
            # low[i-2] > low[i-3]
            # low[i-1] > low[i-2]
            l_pivot = bars[pivot_idx]["low"]
            l_left1 = bars[i - 4]["low"]
            l_left2 = bars[i - 5]["low"]
            l_right1 = bars[i - 2]["low"]
            l_right2 = bars[i - 1]["low"]

            if (
                l_pivot < l_left1
                and l_left1 < l_left2
                and l_right1 > l_pivot
                and l_right2 > l_right1
                and volume_confirmed
            ):
                zone_low = l_pivot
                zone_high = min(bar_pivot["open"], bar_pivot["close"])
                level = StructureLevel(
                    price=l_pivot,
                    zone_high=zone_high,
                    zone_low=zone_low,
                    level_type="support",
                    timeframe=timeframe,
                    timeframe_atr=atr,
                    volume_confirmed=True,
                    created_at=pivot_created_at,
                )
                new_levels.append(level)
                self._levels.append(level)

        logger.info(
            "structure_levels.computed",
            timeframe=timeframe,
            new_levels=len(new_levels),
            total_levels=len(self._levels),
            atr=round(atr, 2),
        )
        return new_levels

    # ------------------------------------------------------------------
    # check_proximity
    # ------------------------------------------------------------------

    def check_proximity(
        self,
        price: float,
        side: str,
        bars_1m: list[dict],
        bars_5m: list[dict],
        *,
        current_volume: float = 0.0,
        avg_volume: float = 0.0,
    ) -> dict:
        """Check if price is near any HTF level.

        ATOMIC: Calls _update_breaks() first to ensure all level states are
        current before scoring.

        SIDE-AWARE: Returns the best-scoring level FOR the given side.
        Filters levels that support the trade direction, picks highest TF.

        Args:
            price: Current price.
            side: "long" or "short".
            bars_1m: Recent 1-min bars for bounce confirmation.
            bars_5m: Recent 5-min bars for bounce confirmation.
            current_volume: Current bar volume (for BOS detection).
            avg_volume: Average volume (for BOS detection).

        Returns:
            dict with bounce_score, bos_score, blocked, block_reason,
            nearest_level, detail.
        """
        empty_result: dict = {
            "bounce_score": 0,
            "bos_score": 0,
            "blocked": False,
            "block_reason": "",
            "nearest_level": None,
            "detail": "no HTF levels",
        }

        if not self._levels:
            return empty_result

        # ATOMIC: update breaks before scoring
        # Use candle CLOSE (not tick price) for BOS detection — wicks must not count.
        # MNQ can wick 15-20 pts through resistance on a spike and close back inside.
        close_price = bars_1m[-1]["close"] if bars_1m else price
        self._update_breaks(close_price, current_volume, avg_volume)

        # Filter to unbroken levels only for proximity
        active_levels = [lv for lv in self._levels if not lv.broken]
        if not active_levels:
            return empty_result

        # --- Anti-signal: block only when APPROACHING a zone edge (D/W) ---
        # The block only fires when price is near the edge it would hit:
        #   - Long approaching resistance from below → block near zone_low
        #   - Short approaching support from above → block near zone_high
        # Breakouts (price above resistance top / below support bottom)
        # are GOOD — that's BOS, don't block.
        # Deep inside a wide zone = no block (let confluence decide).
        blocked = False
        block_reason = ""
        EDGE_BUFFER = 15.0  # points from the approaching edge

        for lv in active_levels:
            if lv.timeframe not in ANTI_SIGNAL_TIMEFRAMES:
                continue

            # Long vs resistance: block only when approaching from below
            # (price near zone_low). If price > zone_high, that's a breakout.
            if side == "long" and lv.level_type == "resistance":
                if price < lv.zone_high and abs(price - lv.zone_low) <= EDGE_BUFFER:
                    blocked = True
                    block_reason = (
                        f"long blocked: approaching {lv.timeframe} resistance at "
                        f"{lv.zone_low:.0f}-{lv.zone_high:.0f}"
                    )
                    break

            # Short vs support: block only when approaching from above
            # (price near zone_high). If price < zone_low, that's a breakdown.
            if side == "short" and lv.level_type == "support":
                if price > lv.zone_low and abs(price - lv.zone_high) <= EDGE_BUFFER:
                    blocked = True
                    block_reason = (
                        f"short blocked: approaching {lv.timeframe} support at "
                        f"{lv.zone_low:.0f}-{lv.zone_high:.0f}"
                    )
                    break

        # --- Side-aware filtering ---
        # For bounce: long looks at support, short looks at resistance
        if side == "long":
            side_levels = [lv for lv in active_levels if lv.level_type == "support"]
        else:
            side_levels = [lv for lv in active_levels if lv.level_type == "resistance"]

        if not side_levels:
            result = dict(empty_result)
            result["blocked"] = blocked
            result["block_reason"] = block_reason
            result["detail"] = "no levels for side"
            return result

        # --- Find nearby levels within proximity threshold ---
        nearby: list[tuple[StructureLevel, float]] = []
        for lv in side_levels:
            threshold = PROXIMITY_MULT.get(lv.timeframe, 1.0) * lv.timeframe_atr
            dist = self._distance_to_zone(price, lv)
            if dist <= threshold:
                nearby.append((lv, dist))

        if not nearby:
            # Log nearest levels for debugging (find closest even if outside threshold)
            if side_levels:
                closest = min(side_levels, key=lambda lv: self._distance_to_zone(price, lv))
                closest_dist = self._distance_to_zone(price, closest)
                closest_thresh = PROXIMITY_MULT.get(closest.timeframe, 1.0) * closest.timeframe_atr
                logger.debug(
                    "structure_levels.nearest_level",
                    side=side,
                    price=round(price, 2),
                    nearest_price=round(closest.price, 2),
                    nearest_zone=f"{closest.zone_low:.0f}-{closest.zone_high:.0f}",
                    nearest_tf=closest.timeframe,
                    nearest_type=closest.level_type,
                    distance=round(closest_dist, 2),
                    threshold=round(closest_thresh, 2),
                    gap=round(closest_dist - closest_thresh, 2),
                )
            # Still return anti-signal info even if no nearby side levels
            result = dict(empty_result)
            result["blocked"] = blocked
            result["block_reason"] = block_reason
            result["detail"] = "no nearby levels for side"
            return result

        # Pick best: highest timeframe priority, then closest distance
        nearby.sort(
            key=lambda pair: (-TF_PRIORITY.get(pair[0].timeframe, 0), pair[1])
        )
        best_level, best_dist = nearby[0]

        # Update touch tracking
        best_level.last_tested = datetime.now(tz=UTC)
        best_level.touch_count += 1

        # --- Bounce detection (multi-bar confirmation) ---
        bounce_score = 0
        bounce_confirmed = self._check_bounce_confirmation(
            best_level, side, bars_1m, bars_5m
        )
        if bounce_confirmed:
            bounce_score = 1

        # --- BOS detection ---
        # Only count RECENT breaks (within last 2 minutes) to avoid stale BOS
        # scores persisting all session after a single break event.
        # Track the broken level to return its timeframe for TF-weighted scoring.
        bos_score = 0
        bos_level: StructureLevel | None = None
        bos_cutoff = datetime.now(tz=UTC) - timedelta(minutes=2)
        for lv in self._levels:
            if (
                lv.broken
                and not lv.flipped
                and lv.broken_at is not None
                and lv.broken_at > bos_cutoff
            ):
                # Check if this break aligns with side
                if side == "long" and lv.level_type == "resistance":
                    # Prefer highest-TF break
                    if bos_level is None or TF_PRIORITY.get(lv.timeframe, 0) > TF_PRIORITY.get(bos_level.timeframe, 0):
                        bos_score = 1
                        bos_level = lv
                elif side == "short" and lv.level_type == "support":
                    if bos_level is None or TF_PRIORITY.get(lv.timeframe, 0) > TF_PRIORITY.get(bos_level.timeframe, 0):
                        bos_score = 1
                        bos_level = lv

        detail_parts: list[str] = []
        if bounce_score:
            detail_parts.append(
                f"bounce at {best_level.timeframe} "
                f"{best_level.level_type} {best_level.zone_low:.0f}-{best_level.zone_high:.0f}"
            )
        if bos_score and bos_level:
            detail_parts.append(
                f"BOS through {bos_level.timeframe} {bos_level.level_type} "
                f"at {bos_level.zone_low:.0f}-{bos_level.zone_high:.0f}"
            )
        if blocked:
            detail_parts.append(block_reason)

        return {
            "bounce_score": bounce_score,
            "bos_score": bos_score,
            "bos_tf": bos_level.timeframe if bos_level else None,
            "blocked": blocked,
            "block_reason": block_reason,
            "nearest_level": best_level,
            "detail": ", ".join(detail_parts) if detail_parts else "near level, no signal",
        }

    # ------------------------------------------------------------------
    # _update_breaks
    # ------------------------------------------------------------------

    def _update_breaks(
        self, price: float, volume: float, avg_volume: float
    ) -> None:
        """Update broken/flipped state for all levels.

        IMPORTANT: `price` should be a candle CLOSE, not a tick price.
        Wicks through zones must not trigger BOS.

        For each unbroken level:
          - Resistance: if close > zone_high AND volume > 1.5 * avg_volume
            => mark broken, create flipped support level
          - Support: if close < zone_low AND volume > 1.5 * avg_volume
            => mark broken, create flipped resistance level
        """
        if volume <= 0 or avg_volume <= 0:
            logger.debug(
                "structure_levels.update_breaks_skipped",
                reason="volume or avg_volume <= 0",
                volume=volume,
                avg_volume=avg_volume,
            )
            return

        volume_confirmed = volume > BOS_VOLUME_MULT * avg_volume

        new_flipped: list[StructureLevel] = []

        for lv in self._levels:
            if lv.broken:
                continue

            if (
                lv.level_type == "resistance"
                and price > lv.zone_high
                and volume_confirmed
            ):
                lv.broken = True
                lv.broken_at = datetime.now(tz=UTC)
                # Create flipped support
                flipped = StructureLevel(
                    price=lv.price,
                    zone_high=lv.zone_high,
                    zone_low=lv.zone_low,
                    level_type="support",
                    timeframe=lv.timeframe,
                    timeframe_atr=lv.timeframe_atr,
                    volume_confirmed=lv.volume_confirmed,
                    created_at=datetime.now(tz=UTC),
                    flipped=True,
                )
                new_flipped.append(flipped)
                logger.info(
                    "structure_levels.break",
                    level_type="resistance",
                    timeframe=lv.timeframe,
                    price=lv.price,
                    flipped_to="support",
                )

            elif (
                lv.level_type == "support"
                and price < lv.zone_low
                and volume_confirmed
            ):
                lv.broken = True
                lv.broken_at = datetime.now(tz=UTC)
                # Create flipped resistance
                flipped = StructureLevel(
                    price=lv.price,
                    zone_high=lv.zone_high,
                    zone_low=lv.zone_low,
                    level_type="resistance",
                    timeframe=lv.timeframe,
                    timeframe_atr=lv.timeframe_atr,
                    volume_confirmed=lv.volume_confirmed,
                    created_at=datetime.now(tz=UTC),
                    flipped=True,
                )
                new_flipped.append(flipped)
                logger.info(
                    "structure_levels.break",
                    level_type="support",
                    timeframe=lv.timeframe,
                    price=lv.price,
                    flipped_to="resistance",
                )

        self._levels.extend(new_flipped)

    # ------------------------------------------------------------------
    # update_on_bar_close
    # ------------------------------------------------------------------

    def update_on_bar_close(self, bar: dict, timeframe: str) -> None:
        """Update levels when a new HTF bar closes. Also prunes stale levels.

        For 1h bars: accumulates into a rolling buffer and recomputes 1h levels
        once we have enough bars (>= 6) for fractal detection. This ensures
        intraday structure forming on the 1h chart is visible to the engine,
        not frozen at startup.

        For 4h/D/W: only prunes (these update too slowly for live recomputation).
        """
        self._prune_stale_levels()

        if timeframe == "1h":
            self._live_1h_bars.append(bar)
            # Keep at most 60 bars (~2.5 trading days)
            if len(self._live_1h_bars) > 60:
                self._live_1h_bars = self._live_1h_bars[-60:]
            # Recompute 1h levels from live buffer once we have enough bars
            if len(self._live_1h_bars) >= 6:
                self.compute_levels(self._live_1h_bars, "1h")

        logger.debug(
            "structure_levels.bar_close",
            timeframe=timeframe,
            active_levels=len([lv for lv in self._levels if not lv.broken]),
            live_1h_bars=len(self._live_1h_bars),
        )

    # ------------------------------------------------------------------
    # _prune_stale_levels
    # ------------------------------------------------------------------

    def _prune_stale_levels(self) -> None:
        """Remove levels older than max_age per timeframe.

        Max ages: 1h=5 days, 4h=20 days, D=180 days, W=99999 days (never).
        Also removes broken levels older than 2 days.
        """
        now = datetime.now(tz=UTC)
        kept: list[StructureLevel] = []

        for lv in self._levels:
            age = now - lv.created_at
            age_days = age.total_seconds() / 86400

            # Broken levels: remove if older than 2 days
            if lv.broken and age_days > BROKEN_MAX_AGE_DAYS:
                continue

            # Per-TF max age
            max_age = MAX_AGE_DAYS.get(lv.timeframe, 180)
            if age_days > max_age:
                continue

            kept.append(lv)

        pruned = len(self._levels) - len(kept)
        if pruned > 0:
            logger.info("structure_levels.pruned", count=pruned)
        self._levels = kept

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_to_zone(price: float, level: StructureLevel) -> float:
        """Compute distance from price to the nearest edge of a level's zone.

        Returns 0 if price is inside the zone.
        """
        if level.zone_low <= price <= level.zone_high:
            return 0.0
        if price < level.zone_low:
            return level.zone_low - price
        return price - level.zone_high

    @staticmethod
    def _check_bounce_confirmation(
        level: StructureLevel,
        side: str,
        bars_1m: list[dict],
        bars_5m: list[dict],
    ) -> bool:
        """Check for multi-bar rejection at a level.

        Requires:
          - At least 2 consecutive 1-min bars showing wick rejection, OR
          - A completed 5-min bar showing wick rejection with close inside zone.

        A "wick rejection" at support means: low extends below zone, close inside.
        A "wick rejection" at resistance means: high extends above zone, close inside.
        """
        zone_high = level.zone_high
        zone_low = level.zone_low

        # Check 2 consecutive 1-min bars (last 10 only — bounce must be recent)
        if len(bars_1m) >= 2:
            consecutive_rejections = 0
            for bar in bars_1m[-10:]:
                if _is_rejection(bar, zone_high, zone_low, side):
                    consecutive_rejections += 1
                    if consecutive_rejections >= 2:
                        return True
                else:
                    consecutive_rejections = 0

        # Check last 5-min bar (only if recent — within last 10 minutes)
        if bars_5m:
            last_5m = bars_5m[-1]
            ts = last_5m.get("timestamp")
            is_recent = True
            if ts is not None and hasattr(ts, "timestamp"):
                age_sec = (datetime.now(tz=UTC) - ts).total_seconds()
                is_recent = age_sec < 600  # 10 minutes
            if is_recent and _is_rejection(last_5m, zone_high, zone_low, side):
                return True

        return False


def _is_rejection(
    bar: dict, zone_high: float, zone_low: float, side: str
) -> bool:
    """Check if a bar shows wick rejection at a zone.

    For support (long side): wick below zone_low, close >= zone_low.
    For resistance (short side): wick above zone_high, close <= zone_high.
    """
    if side == "long":
        # Support rejection: low dips below zone, close stays inside or above
        return bar["low"] < zone_low and bar["close"] >= zone_low
    else:
        # Resistance rejection: high pokes above zone, close stays inside or below
        return bar["high"] > zone_high and bar["close"] <= zone_high
