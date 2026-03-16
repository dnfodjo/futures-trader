"""Automatic regime classifier — heuristic market condition detection.

Classifies the current market into one of:
- TRENDING_UP:     Price above VWAP, rising highs, positive delta, directional move
- TRENDING_DOWN:   Price below VWAP, falling lows, negative delta, directional move
- BREAKOUT:        Breaking session/prior-day extremes with volume + delta confirmation
- CHOPPY:          Narrow range, oscillating around VWAP, mixed signals
- LOW_VOLUME:      RVOL < 0.7 — thin market, unreliable signals
- NEWS_DRIVEN:     During or immediately after high-impact economic events

The classifier runs every state update cycle inside StateEngine. It uses only
data already available (tick processor snapshot, recent bars, cross-market, calendar)
so it adds zero latency or API cost.

Inputs (from StateEngine context):
- Recent 1-second OHLCV bars (up to 30 bars = 30 seconds)
- Tick processor snapshot (VWAP, delta, delta_trend, session H/L, tape speed)
- Key levels (PDH, PDL, ONH, ONL, session H/L)
- RVOL
- Blackout status + upcoming events
- Cross-market (TICK index for confirmation)

Design principles:
- Avoids rapid flipping: requires sustained evidence before changing regime
- Returns confidence (0.0-1.0) reflecting signal strength
- Errors toward CHOPPY (conservative — better to miss a trend than to
  false-positive a breakout)
"""

from __future__ import annotations

from collections import deque
from typing import Any

import structlog

from src.core.types import Regime
from src.data.schemas import OHLCVBar

logger = structlog.get_logger()

# ── Thresholds ────────────────────────────────────────────────────────────────

# Trending detection
_TREND_MIN_BARS = 10  # need at least 10 bars to assess trend
_TREND_VWAP_THRESHOLD = 3.0  # must be >3 pts from VWAP for trend
_TREND_DIRECTIONAL_MOVE = 4.0  # min net move over the bar window
_TREND_HH_HL_COUNT = 3  # need 3+ higher highs/lows (or lower) in swing analysis

# Breakout detection
_BREAKOUT_PROXIMITY = 2.0  # within 2 pts of key level
_BREAKOUT_VOLUME_MULTIPLIER = 1.3  # need 30%+ above-average volume
_BREAKOUT_OVERSHOOT = 1.0  # must break through level by at least 1 pt

# Choppy detection
_CHOPPY_RANGE_MAX = 8.0  # session range < 8 pts = narrow range
_CHOPPY_VWAP_PROXIMITY = 4.0  # within 4 pts of VWAP = oscillating

# Low volume
_LOW_VOLUME_RVOL = 0.70  # RVOL below 0.70 = thin market

# Confidence
_HIGH_CONFIDENCE = 0.85
_MODERATE_CONFIDENCE = 0.65
_LOW_CONFIDENCE = 0.50
_BASELINE_CONFIDENCE = 0.40


class RegimeClassifier:
    """Heuristic regime classifier that runs on every state engine cycle.

    Usage:
        classifier = RegimeClassifier()
        regime, confidence = classifier.classify(
            tick_snap=tick_processor.snapshot(),
            recent_bars=list(recent_bars_deque),
            rvol=1.2,
            in_blackout=False,
            upcoming_high_impact=False,
            prior_day_high=19900.0,
            prior_day_low=19700.0,
            overnight_high=19850.0,
            overnight_low=19750.0,
        )
    """

    def __init__(
        self,
        stability_window: int = 3,
    ) -> None:
        """
        Args:
            stability_window: Number of consecutive classifications before
                switching regime. Prevents rapid flipping.
        """
        self._stability_window = stability_window
        self._current_regime = Regime.CHOPPY
        self._current_confidence = 0.5
        self._candidate_regime: Regime | None = None
        self._candidate_count = 0
        self._classification_count = 0

    def classify(
        self,
        tick_snap: dict[str, Any],
        recent_bars: list[OHLCVBar],
        rvol: float,
        in_blackout: bool,
        upcoming_high_impact: bool,
        prior_day_high: float = 0.0,
        prior_day_low: float = 0.0,
        overnight_high: float = 0.0,
        overnight_low: float = 0.0,
    ) -> tuple[Regime, float]:
        """Classify the current market regime.

        Returns:
            Tuple of (Regime, confidence).
        """
        self._classification_count += 1

        raw_regime, raw_confidence = self._raw_classify(
            tick_snap=tick_snap,
            recent_bars=recent_bars,
            rvol=rvol,
            in_blackout=in_blackout,
            upcoming_high_impact=upcoming_high_impact,
            prior_day_high=prior_day_high,
            prior_day_low=prior_day_low,
            overnight_high=overnight_high,
            overnight_low=overnight_low,
        )

        # ── Stability filter: prevent rapid flipping ──────────────────
        regime, confidence = self._apply_stability(raw_regime, raw_confidence)

        if regime != self._current_regime:
            logger.info(
                "regime_classifier.regime_change",
                old=self._current_regime.value,
                new=regime.value,
                confidence=round(confidence, 2),
            )

        self._current_regime = regime
        self._current_confidence = confidence
        return regime, confidence

    def _raw_classify(
        self,
        tick_snap: dict[str, Any],
        recent_bars: list[OHLCVBar],
        rvol: float,
        in_blackout: bool,
        upcoming_high_impact: bool,
        prior_day_high: float,
        prior_day_low: float,
        overnight_high: float,
        overnight_low: float,
    ) -> tuple[Regime, float]:
        """Core classification logic. Returns raw (regime, confidence) before
        the stability filter.

        Priority order (highest to lowest):
        1. NEWS_DRIVEN — overrides everything during high-impact events
        2. LOW_VOLUME — RVOL < 0.70 means signals are unreliable
        3. BREAKOUT — breaking key levels with confirmation
        4. TRENDING — sustained directional move
        5. CHOPPY — default / narrow range / mixed signals
        """
        last_price = tick_snap.get("last_price", 0.0)
        if last_price == 0.0:
            return Regime.CHOPPY, _BASELINE_CONFIDENCE

        # ── 1. NEWS_DRIVEN ────────────────────────────────────────────
        if in_blackout or upcoming_high_impact:
            return Regime.NEWS_DRIVEN, _HIGH_CONFIDENCE

        # ── 2. LOW_VOLUME ─────────────────────────────────────────────
        if rvol < _LOW_VOLUME_RVOL and rvol > 0:
            return Regime.LOW_VOLUME, min(_HIGH_CONFIDENCE, 0.5 + (0.70 - rvol))

        # ── 3. BREAKOUT ───────────────────────────────────────────────
        breakout = self._check_breakout(
            tick_snap=tick_snap,
            recent_bars=recent_bars,
            rvol=rvol,
            prior_day_high=prior_day_high,
            prior_day_low=prior_day_low,
            overnight_high=overnight_high,
            overnight_low=overnight_low,
        )
        if breakout is not None:
            return breakout

        # ── 4. TRENDING ───────────────────────────────────────────────
        trend = self._check_trend(
            tick_snap=tick_snap,
            recent_bars=recent_bars,
        )
        if trend is not None:
            return trend

        # ── 5. CHOPPY (default) ───────────────────────────────────────
        choppy_confidence = self._assess_choppiness(
            tick_snap=tick_snap,
            recent_bars=recent_bars,
        )
        return Regime.CHOPPY, choppy_confidence

    # ── Breakout Detection ────────────────────────────────────────────────────

    def _check_breakout(
        self,
        tick_snap: dict[str, Any],
        recent_bars: list[OHLCVBar],
        rvol: float,
        prior_day_high: float,
        prior_day_low: float,
        overnight_high: float,
        overnight_low: float,
    ) -> tuple[Regime, float] | None:
        """Check if price is breaking a key level with confirmation.

        Returns (BREAKOUT, confidence) or None if no breakout detected.
        """
        last_price = tick_snap.get("last_price", 0.0)
        session_high = tick_snap.get("session_high", 0.0)
        session_low = tick_snap.get("session_low", 0.0)
        delta_trend = tick_snap.get("delta_trend", "neutral")
        tape_speed = tick_snap.get("tape_speed", 0.0)

        # Key levels to check for breakout (upside and downside)
        upside_levels = [
            (prior_day_high, "PDH"),
            (overnight_high, "ONH"),
        ]
        downside_levels = [
            (prior_day_low, "PDL"),
            (overnight_low, "ONL"),
        ]

        # Check upside breakouts
        for level, name in upside_levels:
            if level <= 0:
                continue
            overshoot = last_price - level
            if overshoot >= _BREAKOUT_OVERSHOOT:
                # Price broke above level — check confirmation
                confidence = self._breakout_confidence(
                    overshoot=overshoot,
                    delta_confirms=(delta_trend in ("positive",)),
                    rvol=rvol,
                    tape_speed=tape_speed,
                    recent_bars=recent_bars,
                    direction="up",
                )
                if confidence >= _LOW_CONFIDENCE:
                    logger.debug(
                        "regime_classifier.breakout_detected",
                        level=name,
                        overshoot=round(overshoot, 2),
                        confidence=round(confidence, 2),
                    )
                    return Regime.BREAKOUT, confidence

        # Check downside breakouts
        for level, name in downside_levels:
            if level <= 0:
                continue
            overshoot = level - last_price
            if overshoot >= _BREAKOUT_OVERSHOOT:
                confidence = self._breakout_confidence(
                    overshoot=overshoot,
                    delta_confirms=(delta_trend in ("negative",)),
                    rvol=rvol,
                    tape_speed=tape_speed,
                    recent_bars=recent_bars,
                    direction="down",
                )
                if confidence >= _LOW_CONFIDENCE:
                    logger.debug(
                        "regime_classifier.breakout_detected",
                        level=name,
                        overshoot=round(overshoot, 2),
                        confidence=round(confidence, 2),
                    )
                    return Regime.BREAKOUT, confidence

        # Session high/low breakout (intraday) — lower confidence
        # Only if we have enough range to make it meaningful
        session_range = session_high - session_low if session_high > session_low else 0.0
        if session_range >= 10.0:
            # New session high with confirmation
            if (
                last_price >= session_high
                and last_price == tick_snap.get("last_price", 0.0)
                and delta_trend in ("positive",)
                and rvol >= 1.0
            ):
                return Regime.BREAKOUT, _MODERATE_CONFIDENCE * 0.85

            # New session low with confirmation
            if (
                last_price <= session_low
                and delta_trend in ("negative",)
                and rvol >= 1.0
            ):
                return Regime.BREAKOUT, _MODERATE_CONFIDENCE * 0.85

        return None

    def _breakout_confidence(
        self,
        overshoot: float,
        delta_confirms: bool,
        rvol: float,
        tape_speed: float,
        recent_bars: list[OHLCVBar],
        direction: str,
    ) -> float:
        """Score breakout confidence based on confirmation factors.

        Each factor adds to confidence. A "clean" breakout has all factors:
        - Price overshoot (already confirmed by caller)
        - Delta alignment (buyers for up, sellers for down)
        - Volume above average (RVOL > 1.3)
        - Fast tape (high activity)
        - Recent bars moving in breakout direction
        """
        score = 0.35  # base for price overshoot

        # More overshoot = more confidence
        if overshoot >= 3.0:
            score += 0.10
        elif overshoot >= 2.0:
            score += 0.05

        # Delta alignment
        if delta_confirms:
            score += 0.15

        # Volume confirmation
        if rvol >= _BREAKOUT_VOLUME_MULTIPLIER:
            score += 0.15
        elif rvol >= 1.0:
            score += 0.05

        # Fast tape = conviction
        if tape_speed >= 40:
            score += 0.10
        elif tape_speed >= 20:
            score += 0.05

        # Recent bar alignment
        if len(recent_bars) >= 5:
            last_5 = recent_bars[-5:]
            if direction == "up":
                up_count = sum(1 for b in last_5 if b.is_up)
                if up_count >= 4:
                    score += 0.10
            else:
                down_count = sum(1 for b in last_5 if not b.is_up)
                if down_count >= 4:
                    score += 0.10

        return min(score, _HIGH_CONFIDENCE)

    # ── Trend Detection ───────────────────────────────────────────────────────

    def _check_trend(
        self,
        tick_snap: dict[str, Any],
        recent_bars: list[OHLCVBar],
    ) -> tuple[Regime, float] | None:
        """Check for a sustained directional trend.

        Looks at:
        1. VWAP relationship (above = bullish, below = bearish)
        2. Net move over recent bars (directional displacement)
        3. Higher-highs/higher-lows pattern (or lower-lows/lower-highs)
        4. Delta alignment (confirms buyers/sellers in control)

        Returns (TRENDING_UP or TRENDING_DOWN, confidence) or None.
        """
        if len(recent_bars) < _TREND_MIN_BARS:
            return None

        last_price = tick_snap.get("last_price", 0.0)
        vwap = tick_snap.get("vwap", 0.0)
        delta_trend = tick_snap.get("delta_trend", "neutral")

        # Calculate net displacement
        window = recent_bars[-min(30, len(recent_bars)):]
        net_move = window[-1].close - window[0].close

        # Must have meaningful displacement
        if abs(net_move) < _TREND_DIRECTIONAL_MOVE:
            return None

        # Determine direction
        is_uptrend = net_move > 0

        # Score trend quality
        score = 0.0

        # 1. Directional displacement (base)
        if abs(net_move) >= _TREND_DIRECTIONAL_MOVE:
            score += 0.25
        if abs(net_move) >= _TREND_DIRECTIONAL_MOVE * 2:
            score += 0.10

        # 2. VWAP confirmation
        if vwap > 0:
            vwap_dist = last_price - vwap
            if is_uptrend and vwap_dist >= _TREND_VWAP_THRESHOLD:
                score += 0.20
            elif not is_uptrend and vwap_dist <= -_TREND_VWAP_THRESHOLD:
                score += 0.20
            elif is_uptrend and vwap_dist > 0:
                score += 0.10  # above VWAP but not far
            elif not is_uptrend and vwap_dist < 0:
                score += 0.10

        # 3. Swing structure (higher highs/lows or lower lows/highs)
        swing_score = self._swing_structure_score(window, is_uptrend)
        score += swing_score

        # 4. Delta alignment
        if is_uptrend and delta_trend == "positive":
            score += 0.15
        elif not is_uptrend and delta_trend == "negative":
            score += 0.15
        elif delta_trend == "flipping":
            score -= 0.10  # momentum shift = less confidence in trend

        # 5. Bar direction bias
        if len(window) >= 5:
            up_pct = sum(1 for b in window if b.is_up) / len(window)
            if is_uptrend and up_pct >= 0.65:
                score += 0.10
            elif not is_uptrend and up_pct <= 0.35:
                score += 0.10

        # Require minimum score for trend classification
        if score < _LOW_CONFIDENCE:
            return None

        regime = Regime.TRENDING_UP if is_uptrend else Regime.TRENDING_DOWN
        return regime, min(score, _HIGH_CONFIDENCE)

    def _swing_structure_score(
        self,
        bars: list[OHLCVBar],
        is_uptrend: bool,
    ) -> float:
        """Score the quality of swing structure (HH/HL or LL/LH).

        Divides bars into chunks and checks if highs/lows are monotonically
        increasing (uptrend) or decreasing (downtrend).
        """
        if len(bars) < 9:
            return 0.0

        # Divide into 3 equal segments and compare highs/lows
        chunk_size = len(bars) // 3
        segments = [
            bars[:chunk_size],
            bars[chunk_size:chunk_size * 2],
            bars[chunk_size * 2:],
        ]

        seg_highs = [max(b.high for b in seg) for seg in segments if seg]
        seg_lows = [min(b.low for b in seg) for seg in segments if seg]

        if len(seg_highs) < 3:
            return 0.0

        if is_uptrend:
            # Check for higher highs and higher lows
            hh = seg_highs[0] < seg_highs[1] < seg_highs[2]
            hl = seg_lows[0] < seg_lows[1] < seg_lows[2]
            if hh and hl:
                return 0.20  # strong swing structure
            elif hh or hl:
                return 0.10  # partial confirmation
        else:
            # Check for lower lows and lower highs
            ll = seg_lows[0] > seg_lows[1] > seg_lows[2]
            lh = seg_highs[0] > seg_highs[1] > seg_highs[2]
            if ll and lh:
                return 0.20
            elif ll or lh:
                return 0.10

        return 0.0

    # ── Choppiness Assessment ─────────────────────────────────────────────────

    def _assess_choppiness(
        self,
        tick_snap: dict[str, Any],
        recent_bars: list[OHLCVBar],
    ) -> float:
        """Score how choppy the market is (higher = more confident it's choppy).

        Strong choppy signals:
        - Narrow session range
        - Price oscillating around VWAP
        - Mixed delta (neutral or flipping)
        - No directional momentum in recent bars
        """
        score = _LOW_CONFIDENCE  # base — CHOPPY is the default

        last_price = tick_snap.get("last_price", 0.0)
        vwap = tick_snap.get("vwap", 0.0)
        session_high = tick_snap.get("session_high", 0.0)
        session_low = tick_snap.get("session_low", 0.0)
        delta_trend = tick_snap.get("delta_trend", "neutral")

        # Narrow session range
        session_range = session_high - session_low if session_high > session_low else 0.0
        if 0 < session_range < _CHOPPY_RANGE_MAX:
            score += 0.15

        # Near VWAP
        if vwap > 0 and last_price > 0:
            vwap_dist = abs(last_price - vwap)
            if vwap_dist < _CHOPPY_VWAP_PROXIMITY:
                score += 0.10

        # Mixed/neutral delta
        if delta_trend in ("neutral", "flipping"):
            score += 0.10

        # No directional momentum in recent bars
        if len(recent_bars) >= 10:
            window = recent_bars[-10:]
            net_move = abs(window[-1].close - window[0].close)
            if net_move < 2.0:
                score += 0.10

            # Mixed bar direction
            up_count = sum(1 for b in window if b.is_up)
            if 3 <= up_count <= 7:  # roughly balanced
                score += 0.05

        return min(score, _HIGH_CONFIDENCE)

    # ── Stability Filter ──────────────────────────────────────────────────────

    def _apply_stability(
        self,
        raw_regime: Regime,
        raw_confidence: float,
    ) -> tuple[Regime, float]:
        """Prevent rapid flipping by requiring N consecutive classifications.

        If the raw classifier produces the same result for `stability_window`
        consecutive cycles, switch to it. Otherwise keep the current regime.

        Exception: HIGH_CONFIDENCE classifications (≥0.80) override immediately
        for NEWS_DRIVEN and BREAKOUT which are time-sensitive.
        """
        # First classification — accept directly, no stability needed
        if self._classification_count <= 1:
            self._candidate_regime = None
            self._candidate_count = 0
            return raw_regime, raw_confidence

        # Immediate overrides for time-sensitive regimes
        if raw_regime in (Regime.NEWS_DRIVEN, Regime.BREAKOUT):
            if raw_confidence >= 0.80:
                self._candidate_regime = None
                self._candidate_count = 0
                return raw_regime, raw_confidence

        # Same as current regime — just update confidence
        if raw_regime == self._current_regime:
            self._candidate_regime = None
            self._candidate_count = 0
            return raw_regime, raw_confidence

        # Different from current — count consecutive appearances
        if raw_regime == self._candidate_regime:
            self._candidate_count += 1
        else:
            self._candidate_regime = raw_regime
            self._candidate_count = 1

        # Switch if stable for long enough
        if self._candidate_count >= self._stability_window:
            self._candidate_regime = None
            self._candidate_count = 0
            return raw_regime, raw_confidence

        # Not stable yet — keep current regime but lower confidence
        return self._current_regime, max(
            _BASELINE_CONFIDENCE, self._current_confidence * 0.9
        )

    # ── Accessors ────────────────────────────────────────────────────────────

    @property
    def current_regime(self) -> Regime:
        return self._current_regime

    @property
    def current_confidence(self) -> float:
        return self._current_confidence

    @property
    def classification_count(self) -> int:
        return self._classification_count

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "regime": self._current_regime.value,
            "confidence": round(self._current_confidence, 2),
            "classifications": self._classification_count,
            "candidate_regime": (
                self._candidate_regime.value if self._candidate_regime else None
            ),
            "candidate_count": self._candidate_count,
        }
