"""7-Factor Confluence Scoring Engine for MNQ Smart Money Strategy.

Scores potential entries on a 0-9 scale across:
  1. Market Speed   (FILTER - not scored, but can block)
  2. Trend          (1 point - multi-TF EMA alignment)
  3. Order Block    (2 points - ICT order block tap)
  4. Candle Pattern (1 point - engulfing / rejection / strong body)
  5. Liquidity Sweep(1 point - swept swing pivots or session levels)
  6. Volume         (1 point - above-average volume)
  7. Structure      (up to 3 points - HTF S/R bounce or BOS, gated by core>=2)

Usage:
    engine = ConfluenceEngine()

    # On each new 1-min bar:
    engine.update(bars_1m, atr)

    # When evaluating a signal:
    result = engine.score(
        side="long",
        last_price=21450.25,
        bars_1m=bars,
        atr=4.5,
        multi_tf_emas={"5m": {"ema_9": ..., "ema_50": ...}, ...},
        session_levels={"asian_high": ..., "asian_low": ..., ...},
        bars_5m=[...],   # optional, for structure bounce confirmation
    )
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.indicators.structure_levels import StructureLevelManager

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MNQ_TICK_SIZE = 0.25
EQUAL_LEVEL_THRESHOLD = 2.0  # 2 points = 8 ticks on MNQ
MAX_OBS_PER_SIDE = 5
MAX_SWEEP_LEVELS = 10
PIVOT_LOOKBACK = 20
SPEED_FAST_THRESHOLD = 1.5
SPEED_SLOW_THRESHOLD = 0.5
SPEED_FAST_BARS = 5
SPEED_SLOW_BARS = 14
DISPLACEMENT_MIN_CONSECUTIVE = 2
DISPLACEMENT_BODY_ATR_MULT = 0.75
OB_DEDUP_ATR_MULT = 0.5
OB_LOOKBACK_BARS = 10  # How far back to search for opposing candle before displacement
VOLUME_SMA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.5


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class OrderBlock:
    """Represents a detected order block zone."""

    side: str  # "bull" or "bear"
    high: float  # upper boundary of the OB zone (full candle high)
    low: float  # lower boundary of the OB zone (full candle low)
    timestamp: str = ""
    mitigated: bool = False
    created_at: float = 0.0  # epoch timestamp for time decay

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass
class SwingPivot:
    """A swing high or low detected from price action."""

    level: float
    pivot_type: str  # "high" or "low"
    bar_index: int = 0
    swept: bool = False


@dataclass
class SweepLevel:
    """A liquidity level formed by equal highs/lows or session extremes."""

    level: float
    level_type: str  # "equal_highs", "equal_lows", "session_high", "session_low"
    swept: bool = False
    swept_at: float = 0.0  # epoch timestamp of first sweep trigger


# ---------------------------------------------------------------------------
# Confluence Engine
# ---------------------------------------------------------------------------


class ConfluenceEngine:
    """6-factor confluence scoring engine for MNQ smart money setups.

    Maintains internal state for order blocks, swing pivots, and liquidity
    sweep levels.  Call ``update()`` on every new 1-min bar, then ``score()``
    when evaluating a potential entry.
    """

    # Timeframe-weighted points for structure factor
    BOUNCE_POINTS: dict[str, int] = {"1h": 1, "4h": 1, "D": 2, "W": 2}
    BOS_POINTS: dict[str, int] = {"1h": 1, "4h": 2, "D": 2, "W": 3}

    def __init__(self, structure_manager: StructureLevelManager | None = None) -> None:
        # Order blocks per side
        self._bull_obs: list[OrderBlock] = []
        self._bear_obs: list[OrderBlock] = []

        # Swing pivots (rolling window)
        self._swing_highs: list[SwingPivot] = []
        self._swing_lows: list[SwingPivot] = []

        # Liquidity sweep levels
        self._sweep_levels: list[SweepLevel] = []

        # Watermark for OB detection — avoid re-scanning entire bar history
        self._last_ob_scan_len: int = 0

        # Track used session-level sweeps (reset on session change)
        self._used_session_sweeps: set[str] = set()

        # HTF structure level manager (optional)
        self._structure_manager = structure_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, bars_1m: list[dict], atr: float) -> None:
        """Called on each new 1-min bar.  Updates OBs, pivots, sweep levels.

        Args:
            bars_1m: Recent CLOSED 1-minute bars (newest last).  Each bar has
                     keys ``open``, ``high``, ``low``, ``close``, ``volume``,
                     ``buy_volume``, ``sell_volume``, ``timestamp``.
            atr: Current 14-period ATR on 1-min bars.
        """
        if len(bars_1m) < 5:
            return

        self._detect_order_blocks(bars_1m, atr)
        self._mitigate_order_blocks(bars_1m[-1])
        self._prune_stale_order_blocks()
        self._update_pivots(bars_1m)

    def score(
        self,
        side: str,
        last_price: float,
        bars_1m: list[dict],
        atr: float,
        multi_tf_emas: dict,
        session_levels: dict,
        bars_5m: list[dict] | None = None,
    ) -> dict:
        """Score a potential entry on a 0-9 scale.

        Args:
            side: ``"long"`` or ``"short"``.
            last_price: Current price.
            bars_1m: Recent 1-min bars (newest last).
            atr: Current ATR (14-period, 1-min).
            multi_tf_emas: ``{"5m": {"ema_9": float, "ema_50": float}, ...}``
            session_levels: ``{"asian_high": float, "asian_low": float, ...}``

        Returns:
            dict with keys: ``score``, ``speed_state``, ``blocked``,
            ``block_reason``, ``factors``, ``risk_flags``.
        """
        # Fallback: if last_price is 0 (no live ticks yet after restart),
        # use the last bar's close so all factors get a real price.
        if last_price <= 0 and bars_1m:
            last_price = bars_1m[-1].get("close", 0.0)

        risk_flags: list[str] = []
        factors: dict[str, dict] = {}
        blocked = False
        block_reason = ""
        total_score = 0
        fast_market = False

        # --- 1. Market Speed (filter) ---
        speed_state = self._calc_speed_state(bars_1m)
        if speed_state == "SLOW":
            blocked = True
            block_reason = "Market speed too slow (speed_ratio < 0.5)"
        elif speed_state == "FAST":
            risk_flags.append("fast_market")
            fast_market = True

        # --- 2. Trend ---
        trend_result = self._score_trend(side, last_price, multi_tf_emas)
        factors["trend"] = trend_result
        if trend_result.get("blocked"):
            blocked = True
            block_reason = block_reason or trend_result["detail"]
        total_score += trend_result["score"]

        # --- 3. Order Block ---
        ob_result = self._score_order_block(side, last_price)
        factors["order_block"] = ob_result
        total_score += ob_result["score"]

        # --- 4. Candle Pattern ---
        candle_result = self._score_candle_pattern(side, bars_1m)
        factors["candle"] = candle_result
        total_score += candle_result["score"]

        # --- 5. Liquidity Sweep ---
        sweep_result = self._score_liquidity_sweep(side, last_price, bars_1m, atr, session_levels)
        factors["sweep"] = sweep_result
        total_score += sweep_result["score"]

        # --- 6. Volume ---
        vol_result = self._score_volume(side, bars_1m)
        factors["volume"] = vol_result
        total_score += vol_result["score"]

        # --- 7. Structure (HTF S/R) ---
        structure_result = self._score_structure(side, last_price, bars_1m, bars_5m or [])
        factors["structure"] = structure_result

        # Structure block (trading against D/W levels)
        if structure_result.get("blocked"):
            blocked = True
            block_reason = block_reason or structure_result["block_reason"]

        # Core >= 2 safeguard: structure points only count if at least 2 of 5 core factors score
        core_score = sum(factors[f]["score"] for f in ("trend", "order_block", "candle", "sweep", "volume"))
        if core_score >= 2:
            total_score += structure_result["score"]
        else:
            # Log that structure was suppressed
            if structure_result["score"] > 0:
                logger.info(
                    "confluence.structure_suppressed",
                    structure_score=structure_result["score"],
                    core_score=core_score,
                    msg=f"Structure +{structure_result['score']} suppressed (core={core_score} < 2)",
                )

        # --- FAST market: info flag only, no penalty ---
        # Strong directional moves WITH confluence = best setups. Other filters
        # (OB decay, direction-aware volume, sweep single-fire) already protect
        # against news-spike false signals. No need to penalize FAST markets.

        # --- Extract OB zone data for dynamic stop computation ---
        ob_zone = None
        if ob_result["score"] > 0:
            ob_zone = {
                "ob_zone_low": ob_result.get("ob_zone_low", 0),
                "ob_zone_high": ob_result.get("ob_zone_high", 0),
                "ob_side": ob_result.get("ob_side", ""),
            }

        # --- Extract HTF structure zone for dynamic stop (fallback when no OB) ---
        structure_zone = None
        nearest_level = structure_result.get("nearest_level")
        if nearest_level is not None and structure_result["score"] > 0:
            structure_zone = {
                "zone_low": nearest_level.zone_low,
                "zone_high": nearest_level.zone_high,
                "timeframe": nearest_level.timeframe,
                "level_type": nearest_level.level_type,
            }

        result = {
            "score": total_score,
            "speed_state": speed_state,
            "blocked": blocked,
            "block_reason": block_reason,
            "factors": factors,
            "risk_flags": risk_flags,
            "fast_market": fast_market,
            "ob_zone": ob_zone,
            "structure_zone": structure_zone,
        }

        logger.info(
            "confluence_scored",
            side=side,
            score=total_score,
            blocked=blocked,
            block_reason=block_reason or "",
            speed=speed_state,
            factors={k: v["score"] for k, v in factors.items()},
        )

        return result

    # ------------------------------------------------------------------
    # Factor 1: Market Speed
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_speed_state(bars_1m: list[dict]) -> str:
        """Compute speed ratio and return SLOW / NORMAL / FAST.

        Uses bar range (high-low) instead of body (close-open) to capture
        true volatility including wicks.  Doji bars with large ranges are
        correctly classified as FAST, not SLOW.
        """
        if len(bars_1m) < SPEED_SLOW_BARS:
            return "NORMAL"

        recent = bars_1m[-SPEED_SLOW_BARS:]

        def _avg_range(bars: list[dict]) -> float:
            ranges = [abs(b["high"] - b["low"]) for b in bars]
            return sum(ranges) / len(ranges) if ranges else 0.0

        fast_avg = _avg_range(recent[-SPEED_FAST_BARS:])
        slow_avg = _avg_range(recent)

        if slow_avg == 0:
            return "SLOW"

        ratio = fast_avg / slow_avg

        if ratio > SPEED_FAST_THRESHOLD:
            return "FAST"
        if ratio < SPEED_SLOW_THRESHOLD:
            return "SLOW"
        return "NORMAL"

    # ------------------------------------------------------------------
    # Factor 2: Trend (1 point)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_trend(side: str, last_price: float, multi_tf_emas: dict) -> dict:
        """Check multi-TF EMA alignment.

        Bullish: price > EMA9 > EMA50 on primary + 1 higher TF.
        Bearish: price < EMA9 < EMA50 on primary + 1 higher TF.
        30m disagreement is a soft penalty (score 0) not a hard block.

        When HTF EMAs haven't diverged yet (ema_9 ≈ ema_50, meaning
        insufficient bars for EMA-50 to separate), those TFs are treated
        as "insufficient data" and excluded from the alignment requirement.
        This prevents trend from being permanently 0 after restarts when
        only 120 1-min bars are persisted (= 8 fifteen-min, 4 thirty-min bars).
        """
        # Minimum EMA separation to consider a TF as "has enough data".
        # Below this, ema_9 and ema_50 are essentially identical due to
        # insufficient bars (EMA-50 needs ~50 bars to diverge meaningfully).
        MIN_EMA_SEPARATION = 1.0  # 1 point on MNQ

        tf_order = ["5m", "15m", "30m"]
        aligned_tfs: list[str] = []
        diverged_tfs: list[str] = []  # TFs with enough data to evaluate
        side_lower = side.lower()

        for tf in tf_order:
            emas = multi_tf_emas.get(tf)
            if not emas:
                continue
            ema_9 = emas.get("ema_9", 0.0)
            ema_50 = emas.get("ema_50", 0.0)
            if ema_9 == 0.0 or ema_50 == 0.0:
                continue

            # Check if this TF has enough data for EMAs to be meaningful
            if abs(ema_9 - ema_50) < MIN_EMA_SEPARATION:
                continue  # Skip — insufficient data, not a real signal

            diverged_tfs.append(tf)

            if side_lower == "long" and last_price > ema_9 > ema_50:
                aligned_tfs.append(tf)
            elif side_lower == "short" and last_price < ema_9 < ema_50:
                aligned_tfs.append(tf)

        # 30m disagreement: no longer a hard block — just prevents scoring.
        # EMAs are lagging indicators; hard-blocking kills entries during
        # fast reversals (e.g. FOMC pumps where 30m EMAs haven't crossed yet).
        # The risk manager still has a speed-conditional 30m gate.
        # Only check 30m disagreement if 30m has diverged enough to be meaningful.
        emas_30m = multi_tf_emas.get("30m")
        trend_30m_disagrees = False
        if emas_30m:
            ema_9_30 = emas_30m.get("ema_9", 0.0)
            ema_50_30 = emas_30m.get("ema_50", 0.0)
            if ema_9_30 > 0 and ema_50_30 > 0 and abs(ema_9_30 - ema_50_30) >= MIN_EMA_SEPARATION:
                if side_lower == "long" and ema_9_30 < ema_50_30:
                    trend_30m_disagrees = True
                elif side_lower == "short" and ema_9_30 > ema_50_30:
                    trend_30m_disagrees = True

        # Score trend: need alignment on all diverged TFs (min 1).
        # If only 5m has diverged (after restart), 5m alignment alone is enough.
        # If 2+ TFs diverged, need alignment on at least 2.
        # If 30m disagrees, score 0 but do NOT hard-block the signal.
        min_required = min(len(diverged_tfs), 2)  # At most require 2
        scored = (
            1
            if len(aligned_tfs) >= max(min_required, 1) and not trend_30m_disagrees
            else 0
        )

        detail_parts = []
        if aligned_tfs:
            detail_parts.append(f"aligned on {', '.join(aligned_tfs)}")
        else:
            detail_parts.append("no TF alignment")
        if len(diverged_tfs) < len(tf_order):
            insufficient = [tf for tf in tf_order if tf not in diverged_tfs]
            detail_parts.append(f"insufficient data: {', '.join(insufficient)}")
        if trend_30m_disagrees:
            detail_parts.append("30m trend disagrees (soft penalty)")

        return {
            "score": scored,
            "detail": "; ".join(detail_parts),
            "blocked": False,  # Never hard-block on trend alone
            "aligned_tfs": aligned_tfs,
        }

    # ------------------------------------------------------------------
    # Factor 3: Order Blocks (2 points)
    # ------------------------------------------------------------------

    def _detect_order_blocks(self, bars: list[dict], atr: float) -> None:
        """Scan for displacement + FVG to identify OB zones.

        Displacement: 2+ consecutive strong candles (body > 1.5x ATR).
        FVG: gap between bar[i-1].low and bar[i+1].high (bullish) or
             bar[i-1].high and bar[i+1].low (bearish).
        OB: the last opposing candle before displacement — uses FULL candle
            range [low, high] per ICT methodology.

        Uses a watermark to avoid re-scanning the entire bar history on
        every update.  Only scans new bars since the last call.
        """
        if atr <= 0 or len(bars) < 5:
            return

        body_threshold = atr * DISPLACEMENT_BODY_ATR_MULT

        # Only scan from where we left off (with small overlap for boundary)
        scan_start = max(DISPLACEMENT_MIN_CONSECUTIVE + 1, self._last_ob_scan_len - 3)
        # Stop at len(bars) - 2 because FVG check needs bars[i+1]
        scan_end = len(bars) - 2

        for i in range(scan_start, scan_end + 1):
            # Check for bullish displacement (consecutive up candles)
            if self._is_bullish_displacement(bars, i, body_threshold):
                disp_start = i - (DISPLACEMENT_MIN_CONSECUTIVE - 1)
                if disp_start - 1 >= 0 and i + 1 < len(bars):
                    gap_bottom = bars[disp_start - 1]["high"]
                    gap_top = bars[i + 1]["low"]
                    if gap_top > gap_bottom:  # FVG confirmed
                        ob_bar = self._find_opposing_candle(
                            bars, disp_start, "bearish"
                        )
                        if ob_bar is not None:
                            # Use FULL candle range [low, high] for OB zone
                            ob = OrderBlock(
                                side="bull",
                                high=ob_bar["high"],
                                low=ob_bar["low"],
                                timestamp=ob_bar.get("timestamp", ""),
                                created_at=_time.time(),
                            )
                            self._add_ob(ob, atr)

            # Check for bearish displacement (consecutive down candles)
            if self._is_bearish_displacement(bars, i, body_threshold):
                disp_start = i - (DISPLACEMENT_MIN_CONSECUTIVE - 1)
                if disp_start - 1 >= 0 and i + 1 < len(bars):
                    gap_top = bars[disp_start - 1]["low"]
                    gap_bottom = bars[i + 1]["high"]
                    if gap_top > gap_bottom:  # Bearish FVG
                        ob_bar = self._find_opposing_candle(
                            bars, disp_start, "bullish"
                        )
                        if ob_bar is not None:
                            # Use FULL candle range [low, high] for OB zone
                            ob = OrderBlock(
                                side="bear",
                                high=ob_bar["high"],
                                low=ob_bar["low"],
                                timestamp=ob_bar.get("timestamp", ""),
                                created_at=_time.time(),
                            )
                            self._add_ob(ob, atr)

        self._last_ob_scan_len = len(bars)

    @staticmethod
    def _bar_body(bar: dict) -> float:
        return abs(bar["close"] - bar["open"])

    @staticmethod
    def _is_bullish_bar(bar: dict) -> bool:
        return bar["close"] > bar["open"]

    @staticmethod
    def _is_bearish_bar(bar: dict) -> bool:
        return bar["close"] < bar["open"]

    def _is_bullish_displacement(
        self, bars: list[dict], end_idx: int, body_threshold: float
    ) -> bool:
        """Check if bars ending at end_idx form bullish displacement."""
        start = end_idx - (DISPLACEMENT_MIN_CONSECUTIVE - 1)
        if start < 0:
            return False
        for j in range(start, end_idx + 1):
            if not self._is_bullish_bar(bars[j]):
                return False
            if self._bar_body(bars[j]) < body_threshold:
                return False
        return True

    def _is_bearish_displacement(
        self, bars: list[dict], end_idx: int, body_threshold: float
    ) -> bool:
        """Check if bars ending at end_idx form bearish displacement."""
        start = end_idx - (DISPLACEMENT_MIN_CONSECUTIVE - 1)
        if start < 0:
            return False
        for j in range(start, end_idx + 1):
            if not self._is_bearish_bar(bars[j]):
                return False
            if self._bar_body(bars[j]) < body_threshold:
                return False
        return True

    def _find_opposing_candle(
        self, bars: list[dict], disp_start: int, candle_type: str
    ) -> dict | None:
        """Find the last opposing candle before displacement start.

        Args:
            bars: Price bars.
            disp_start: Index where displacement begins.
            candle_type: ``"bullish"`` or ``"bearish"`` - the type to find.
        """
        check_fn = self._is_bullish_bar if candle_type == "bullish" else self._is_bearish_bar
        for j in range(disp_start - 1, max(disp_start - OB_LOOKBACK_BARS - 1, -1), -1):
            if j < 0:
                break
            if check_fn(bars[j]):
                return bars[j]
        return None

    def _add_ob(self, ob: OrderBlock, atr: float) -> None:
        """Add an OB, dedup within 0.5x ATR, cap at MAX_OBS_PER_SIDE."""
        target = self._bull_obs if ob.side == "bull" else self._bear_obs
        dedup_threshold = atr * OB_DEDUP_ATR_MULT

        # Dedup: skip if too close to an existing OB
        for existing in target:
            if abs(existing.mid - ob.mid) < dedup_threshold:
                return

        target.append(ob)

        # Cap at max
        if len(target) > MAX_OBS_PER_SIDE:
            # Remove oldest (first added)
            target.pop(0)

    def _mitigate_order_blocks(self, latest_bar: dict) -> None:
        """Remove OBs that price has CLOSED fully through (mitigated).

        Uses bar close instead of wick to avoid premature mitigation from
        transient price spikes during the OB tap itself.
        """
        bar_close = latest_bar["close"]

        # Bull OB mitigated when price CLOSES below its low
        self._bull_obs = [
            ob for ob in self._bull_obs if not (bar_close < ob.low)
        ]
        # Bear OB mitigated when price CLOSES above its high
        self._bear_obs = [
            ob for ob in self._bear_obs if not (bar_close > ob.high)
        ]

    def _prune_stale_order_blocks(self) -> None:
        """Remove OBs older than 120 minutes.  Legacy OBs (created_at==0) are kept."""
        cutoff = _time.time() - 7200  # 120 minutes
        self._bull_obs = [ob for ob in self._bull_obs if ob.created_at == 0 or ob.created_at > cutoff]
        self._bear_obs = [ob for ob in self._bear_obs if ob.created_at == 0 or ob.created_at > cutoff]

    def _score_order_block(self, side: str, last_price: float) -> dict:
        """Check if price taps an active OB zone.  Score decays with age.

        Fresh OB (<=30 min or no timestamp): 2 points.
        Aging OB (30-60 min): 1 point.
        Stale OB (>60 min): skipped (0 points).
        """
        side_lower = side.lower()
        obs = self._bull_obs if side_lower == "long" else self._bear_obs

        for ob in obs:
            if ob.low <= last_price <= ob.high:
                # Time decay
                age_minutes = (_time.time() - ob.created_at) / 60.0 if ob.created_at > 0 else 0.0

                if age_minutes <= 30 or ob.created_at == 0:
                    score = 2  # fresh OB or legacy OB without timestamp
                elif age_minutes <= 60:
                    score = 1  # aging OB -- half score
                else:
                    continue  # stale OB -- skip, look for fresher one

                age_str = f" (age={age_minutes:.0f}min)" if ob.created_at > 0 else ""
                return {
                    "score": score,
                    "detail": (
                        f"{ob.side} OB tap at {ob.low:.2f}-{ob.high:.2f} "
                        f"(price={last_price:.2f}){age_str}"
                    ),
                    "ob_zone_low": ob.low,
                    "ob_zone_high": ob.high,
                    "ob_side": ob.side,
                }

        return {"score": 0, "detail": "no OB tap"}

    # ------------------------------------------------------------------
    # Factor 4: Candle Pattern (1 point)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_candle_pattern(side: str, bars_1m: list[dict]) -> dict:
        """Check for engulfing, rejection wick, or strong body."""
        if len(bars_1m) < 2:
            return {"score": 0, "detail": "insufficient bars"}

        current = bars_1m[-1]
        previous = bars_1m[-2]

        cur_open = current["open"]
        cur_close = current["close"]
        cur_high = current["high"]
        cur_low = current["low"]
        cur_body = abs(cur_close - cur_open)
        cur_range = cur_high - cur_low

        prev_open = previous["open"]
        prev_close = previous["close"]
        prev_body = abs(prev_close - prev_open)

        side_lower = side.lower()

        # --- Engulfing ---
        # Current bar body fully covers previous bar body
        cur_body_top = max(cur_open, cur_close)
        cur_body_bottom = min(cur_open, cur_close)
        prev_body_top = max(prev_open, prev_close)
        prev_body_bottom = min(prev_open, prev_close)

        is_engulfing = (
            cur_body_top >= prev_body_top
            and cur_body_bottom <= prev_body_bottom
            and cur_body > prev_body
        )
        if is_engulfing:
            # Directional check: bullish engulfing for longs, bearish for shorts
            if side_lower == "long" and cur_close > cur_open:
                return {"score": 1, "detail": "bullish engulfing"}
            if side_lower == "short" and cur_close < cur_open:
                return {"score": 1, "detail": "bearish engulfing"}

        # --- Rejection wick ---
        # Require wick > body AND wick > 30% of range (prevents doji false positives)
        if cur_range > 0:
            min_wick = max(cur_body, 0.3 * cur_range)
            if side_lower == "long":
                # Lower wick should be long (rejection of lower prices)
                lower_wick = min(cur_open, cur_close) - cur_low
                if lower_wick > min_wick:
                    return {"score": 1, "detail": "bullish rejection wick"}
            else:
                # Upper wick should be long (rejection of higher prices)
                upper_wick = cur_high - max(cur_open, cur_close)
                if upper_wick > min_wick:
                    return {"score": 1, "detail": "bearish rejection wick"}

        # --- Strong body ---
        if cur_range > 0 and cur_body > 0.6 * cur_range:
            if side_lower == "long" and cur_close > cur_open:
                return {"score": 1, "detail": "strong bullish body"}
            if side_lower == "short" and cur_close < cur_open:
                return {"score": 1, "detail": "strong bearish body"}

        return {"score": 0, "detail": "no confirming candle pattern"}

    # ------------------------------------------------------------------
    # Factor 5: Liquidity Sweep (1 point)
    # ------------------------------------------------------------------

    def _update_pivots(self, bars_1m: list[dict]) -> None:
        """Detect 20-bar swing highs/lows and update sweep levels."""
        if len(bars_1m) < 5:
            return

        # We need at least 5 bars to check pivot (2 before, 1 candidate, 2 after).
        # Use a sliding window on the last PIVOT_LOOKBACK bars.
        lookback = min(len(bars_1m), PIVOT_LOOKBACK + 4)
        window = bars_1m[-lookback:]

        new_swing_highs: list[SwingPivot] = []
        new_swing_lows: list[SwingPivot] = []

        # Check each bar (except first 2 and last 2) as a potential pivot
        for i in range(2, len(window) - 2):
            bar = window[i]
            # Swing high: high > 2 bars before AND 2 bars after
            if (
                bar["high"] > window[i - 1]["high"]
                and bar["high"] > window[i - 2]["high"]
                and bar["high"] > window[i + 1]["high"]
                and bar["high"] > window[i + 2]["high"]
            ):
                new_swing_highs.append(
                    SwingPivot(level=bar["high"], pivot_type="high", bar_index=i)
                )

            # Swing low: low < 2 bars before AND 2 bars after
            if (
                bar["low"] < window[i - 1]["low"]
                and bar["low"] < window[i - 2]["low"]
                and bar["low"] < window[i + 1]["low"]
                and bar["low"] < window[i + 2]["low"]
            ):
                new_swing_lows.append(
                    SwingPivot(level=bar["low"], pivot_type="low", bar_index=i)
                )

        # Merge with existing, deduplicating by level proximity to avoid
        # the same pivot being re-detected and filling all slots
        for p in new_swing_highs:
            if not any(abs(e.level - p.level) < EQUAL_LEVEL_THRESHOLD for e in self._swing_highs):
                self._swing_highs.append(p)
        self._swing_highs = self._swing_highs[-PIVOT_LOOKBACK:]

        for p in new_swing_lows:
            if not any(abs(e.level - p.level) < EQUAL_LEVEL_THRESHOLD for e in self._swing_lows):
                self._swing_lows.append(p)
        self._swing_lows = self._swing_lows[-PIVOT_LOOKBACK:]

        # Detect equal highs / equal lows and register as sweep levels
        self._detect_equal_levels()

    def _detect_equal_levels(self) -> None:
        """Find pairs of swing highs/lows within EQUAL_LEVEL_THRESHOLD."""
        new_levels: list[SweepLevel] = []

        # Equal highs (sell-side liquidity)
        for i, sh1 in enumerate(self._swing_highs):
            for sh2 in self._swing_highs[i + 1 :]:
                if abs(sh1.level - sh2.level) <= EQUAL_LEVEL_THRESHOLD:
                    avg_level = (sh1.level + sh2.level) / 2.0
                    # Dedup within EQUAL_LEVEL_THRESHOLD (not single tick)
                    if not any(
                        abs(sl.level - avg_level) < EQUAL_LEVEL_THRESHOLD
                        for sl in self._sweep_levels
                    ):
                        new_levels.append(
                            SweepLevel(level=avg_level, level_type="equal_highs")
                        )

        # Equal lows (buy-side liquidity)
        for i, sl1 in enumerate(self._swing_lows):
            for sl2 in self._swing_lows[i + 1 :]:
                if abs(sl1.level - sl2.level) <= EQUAL_LEVEL_THRESHOLD:
                    avg_level = (sl1.level + sl2.level) / 2.0
                    if not any(
                        abs(sl.level - avg_level) < EQUAL_LEVEL_THRESHOLD
                        for sl in self._sweep_levels
                    ):
                        new_levels.append(
                            SweepLevel(level=avg_level, level_type="equal_lows")
                        )

        self._sweep_levels = (self._sweep_levels + new_levels)[-MAX_SWEEP_LEVELS:]

    def _score_liquidity_sweep(
        self, side: str, last_price: float, bars_1m: list[dict],
        atr: float, session_levels: dict,
    ) -> dict:
        """Check if a liquidity sweep has occurred.

        For LONG: a recent bar's LOW dipped below a liquidity level, and
                  current price is back above it (sweep + reversal confirmed).
        For SHORT: a recent bar's HIGH spiked above a liquidity level, and
                   current price is back below it.

        Each sweep level fires only once.  It resets when price moves
        > 1.5 * ATR away from the level (becomes relevant again).
        Session-level sweeps are tracked separately via _used_session_sweeps.
        """
        if len(bars_1m) < 3:
            return {"score": 0, "detail": "insufficient bars for sweep check"}

        side_lower = side.lower()
        proximity = max(atr * 0.5, EQUAL_LEVEL_THRESHOLD) if atr > 0 else EQUAL_LEVEL_THRESHOLD

        # Reset swept flags for internal levels when price moved away
        if atr > 0:
            for sl in self._sweep_levels:
                if sl.swept and abs(last_price - sl.level) > atr * 1.5:
                    sl.swept = False

        # Check last 3 CLOSED bars for sweep evidence (exclude current/live bar
        # to prevent self-fulfilling signals where touching = sweep + reversal)
        recent_bars = bars_1m[-4:-1] if len(bars_1m) >= 4 else bars_1m[:-1]

        # --- Check internal sweep levels (equal highs/lows) ---
        if side_lower == "long":
            for sl in self._sweep_levels:
                if sl.swept:
                    continue  # already used this sweep
                if sl.level_type != "equal_lows":
                    continue
                swept = any(b["low"] < sl.level for b in recent_bars)
                reversed_above = last_price > sl.level and last_price - sl.level <= proximity
                if swept and reversed_above:
                    sl.swept = True
                    sl.swept_at = _time.time()
                    return {
                        "score": 1,
                        "detail": f"sweep of {sl.level_type} at {sl.level:.2f} (first touch)",
                    }
        else:  # short
            for sl in self._sweep_levels:
                if sl.swept:
                    continue
                if sl.level_type != "equal_highs":
                    continue
                swept = any(b["high"] > sl.level for b in recent_bars)
                reversed_below = last_price < sl.level and sl.level - last_price <= proximity
                if swept and reversed_below:
                    sl.swept = True
                    sl.swept_at = _time.time()
                    return {
                        "score": 1,
                        "detail": f"sweep of {sl.level_type} at {sl.level:.2f} (first touch)",
                    }

        # --- Check session levels (tracked via _used_session_sweeps set) ---
        session_check_keys = (
            ["asian_low", "london_low", "ny_low"]
            if side_lower == "long"
            else ["asian_high", "london_high", "ny_high"]
        )
        for key in session_check_keys:
            val = session_levels.get(key, 0.0)
            if val <= 0:
                continue
            session_key = f"session_{key}"
            if session_key in self._used_session_sweeps:
                continue  # already used this session sweep

            if side_lower == "long":
                swept = any(b["low"] < val for b in recent_bars)
                reversed_ok = last_price > val and last_price - val <= proximity
            else:
                swept = any(b["high"] > val for b in recent_bars)
                reversed_ok = last_price < val and val - last_price <= proximity

            if swept and reversed_ok:
                self._used_session_sweeps.add(session_key)
                return {
                    "score": 1,
                    "detail": f"sweep of {session_key} at {val:.2f} (first touch)",
                }

        return {"score": 0, "detail": "no liquidity sweep detected"}

    # ------------------------------------------------------------------
    # Factor 6: Volume (1 point)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_volume(side: str, bars_1m: list[dict]) -> dict:
        """Check if current bar volume is above 1.5x 20-bar SMA.

        Volume spike is required first.  Then directional confirmation
        is checked: if buy/sell volume data is available and the dominant
        side opposes the trade direction, the point is rejected.  This
        prevents high sell-volume from scoring for a long signal.  When
        buy/sell data is unavailable (both zero), direction check is
        to the entry direction as stops are hit.
        """
        if len(bars_1m) < VOLUME_SMA_PERIOD + 1:
            return {"score": 0, "detail": "insufficient bars for volume SMA"}

        current = bars_1m[-1]
        cur_volume = current.get("volume", 0)
        buy_vol = current.get("buy_volume", 0)
        sell_vol = current.get("sell_volume", 0)

        # 20-bar volume SMA (excluding current bar)
        lookback = bars_1m[-(VOLUME_SMA_PERIOD + 1) : -1]
        avg_volume = sum(b.get("volume", 0) for b in lookback) / VOLUME_SMA_PERIOD

        if avg_volume <= 0:
            return {"score": 0, "detail": "no volume data"}

        volume_ratio = cur_volume / avg_volume
        is_spike = volume_ratio >= VOLUME_SPIKE_MULT

        if is_spike:
            # Direction check: buy_volume vs sell_volume
            total_dir = buy_vol + sell_vol
            if total_dir > 0:
                if side.lower() == "long" and sell_vol > buy_vol:
                    return {
                        "score": 0,
                        "detail": f"volume spike but sell-dominated ({sell_vol}>{buy_vol})",
                    }
                if side.lower() == "short" and buy_vol > sell_vol:
                    return {
                        "score": 0,
                        "detail": f"volume spike but buy-dominated ({buy_vol}>{sell_vol})",
                    }

            dir_info = ""
            if buy_vol > 0 or sell_vol > 0:
                dir_info = f", buy={buy_vol} sell={sell_vol}"
            return {
                "score": 1,
                "detail": f"volume spike {volume_ratio:.1f}x avg{dir_info}",
            }

        return {
            "score": 0,
            "detail": f"volume {volume_ratio:.1f}x avg (need {VOLUME_SPIKE_MULT}x)",
        }

    # ------------------------------------------------------------------
    # Factor 7: HTF Structure (bounce / BOS)
    # ------------------------------------------------------------------

    def _score_structure(
        self, side: str, last_price: float, bars_1m: list[dict], bars_5m: list[dict]
    ) -> dict:
        """Score the structure factor (bounce + BOS) with timeframe weighting."""
        if self._structure_manager is None:
            return {"score": 0, "detail": "no HTF structure manager", "blocked": False, "block_reason": ""}

        # Compute volume for BOS detection from 1-min bars
        cur_volume = bars_1m[-1].get("volume", 0.0) if bars_1m else 0.0
        vol_lookback = bars_1m[-21:-1] if len(bars_1m) > 1 else []
        avg_vol = (
            sum(b.get("volume", 0) for b in vol_lookback) / len(vol_lookback)
            if vol_lookback else 0.0
        )

        result = self._structure_manager.check_proximity(
            last_price, side, bars_1m, bars_5m,
            current_volume=cur_volume,
            avg_volume=avg_vol,
        )

        score = 0
        details: list[str] = []
        level = result.get("nearest_level")
        tf = level.timeframe if level else None

        # BOS takes priority over bounce (mutually exclusive)
        # Use the actual broken level's TF for BOS points, not nearest_level's TF
        bos_tf = result.get("bos_tf")
        if result.get("bos_score") and bos_tf:
            pts = self.BOS_POINTS.get(bos_tf, 1)
            score = pts
            details.append(f"BOS through {bos_tf} level (+{pts})")
        elif result.get("bounce_score") and tf:
            pts = self.BOUNCE_POINTS.get(tf, 1)
            score = pts
            details.append(f"bounce at {tf} {level.level_type} (+{pts})")

        score = min(score, 3)  # Safety cap

        return {
            "score": score,
            "detail": ", ".join(details) if details else "no HTF structure",
            "blocked": result.get("blocked", False),
            "block_reason": result.get("block_reason", ""),
            "nearest_level": level,  # forward for dynamic stop computation
        }

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def active_bull_obs(self) -> list[OrderBlock]:
        """Return active bullish order blocks (for debugging / display)."""
        return list(self._bull_obs)

    @property
    def active_bear_obs(self) -> list[OrderBlock]:
        """Return active bearish order blocks (for debugging / display)."""
        return list(self._bear_obs)

    @property
    def sweep_levels(self) -> list[SweepLevel]:
        """Return active liquidity sweep levels."""
        return list(self._sweep_levels)

    def reset(self) -> None:
        """Clear all internal state.  Call at session boundaries."""
        self._bull_obs.clear()
        self._bear_obs.clear()
        self._swing_highs.clear()
        self._swing_lows.clear()
        self._sweep_levels.clear()
        self._used_session_sweeps.clear()
        self._last_ob_scan_len = 0
        logger.info("confluence_engine_reset")
