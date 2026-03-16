"""Setup detector — identifies high-probability trade setups from market state.

Pre-processes market data to detect actionable trading patterns BEFORE sending
to the LLM. This way the LLM receives structured setup alerts instead of having
to infer patterns from raw numbers.

Each detected setup includes:
- Type, side, and confidence score
- Trigger price and suggested stop distance
- Human-readable description for the LLM
- Confirming signals and invalidation criteria
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

from src.core.types import KeyLevels, MarketState, Regime, SessionPhase

ET = ZoneInfo("US/Eastern")


# ── Enums ────────────────────────────────────────────────────────────────────


class SetupType(str, Enum):
    VWAP_PULLBACK = "vwap_pullback"
    VWAP_REJECTION = "vwap_rejection"
    LEVEL_TEST = "level_test"
    FAILED_BREAKOUT = "failed_breakout"
    OPENING_RANGE_BREAK = "opening_range_break"
    DELTA_DIVERGENCE = "delta_divergence"
    ABSORPTION = "absorption"
    TREND_CONTINUATION = "trend_continuation"
    EXHAUSTION = "exhaustion"
    MEAN_REVERSION = "mean_reversion"


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class DetectedSetup:
    """A detected trade setup with all context needed by the LLM."""

    setup_type: SetupType
    side: str  # "long" or "short"
    confidence: float  # 0.0-1.0
    trigger_price: float
    suggested_stop_distance: float
    description: str
    confirming_signals: list[str] = field(default_factory=list)
    invalidation: str = ""


# ── Setup Detector ───────────────────────────────────────────────────────────


class SetupDetector:
    """Scans market state for actionable trade setups.

    Usage:
        detector = SetupDetector()
        setups = detector.detect(state, bars_1s=recent_bars)
        for setup in setups:
            print(f"{setup.setup_type}: {setup.description}")
    """

    # Thresholds — all in points (MNQ)
    VWAP_NEAR_THRESHOLD = 2.0       # "near VWAP" = within N pts
    VWAP_EXTENDED_THRESHOLD = 5.0   # "was extended from VWAP" = >N pts away
    LEVEL_PROXIMITY = 3.0           # "near a key level" = within N pts
    FAILED_BREAKOUT_PIERCE = 2.0    # must have pierced level by at least N pts
    MEAN_REVERSION_DISTANCE = 10.0  # extended from VWAP in choppy
    EXHAUSTION_MOVE_MIN = 15.0      # minimum move for exhaustion
    EXHAUSTION_TIME_MAX_SEC = 300   # 5 minutes
    ABSORPTION_RVOL_MIN = 2.0       # minimum RVOL for absorption
    ABSORPTION_RANGE_MAX = 1.0      # max price range for absorption

    def detect(
        self,
        state: MarketState,
        bars_1s: list[dict[str, Any]],
        bars_5min: list[dict[str, Any]] | None = None,
    ) -> list[DetectedSetup]:
        """Detect all active setups from the current market state.

        Args:
            state: Current MarketState snapshot.
            bars_1s: Recent 1-second OHLCV bars (as dicts with open/high/low/close/volume/timestamp).
            bars_5min: Optional 5-minute bars for longer-term context.

        Returns:
            List of detected setups sorted by confidence descending.
        """
        setups: list[DetectedSetup] = []

        # Run each detector
        self._detect_vwap_pullback(state, setups)
        self._detect_level_test(state, setups)
        self._detect_failed_breakout(state, setups)
        self._detect_delta_divergence(state, setups)
        self._detect_absorption(state, bars_1s, setups)
        self._detect_mean_reversion(state, setups)
        self._detect_exhaustion(state, bars_1s, setups)
        self._detect_opening_range_break(state, bars_1s, setups)
        self._detect_trend_continuation(state, bars_1s, setups)

        # Sort by confidence descending
        setups.sort(key=lambda s: s.confidence, reverse=True)
        return setups

    # ── Individual Detectors ─────────────────────────────────────────────────

    def _detect_vwap_pullback(
        self,
        state: MarketState,
        setups: list[DetectedSetup],
    ) -> None:
        """VWAP_PULLBACK: Price pulls back to VWAP in a trending market.

        Long: Price was above VWAP by >5pts (session_high implies this), pulled
              back to within 2pts of VWAP, delta still net positive.
        Short: Mirror — was below, pulled back up, delta still negative.
        """
        vwap = state.levels.vwap
        if vwap <= 0:
            return

        price = state.last_price
        dist_from_vwap = price - vwap
        session_high = state.levels.session_high
        session_low = state.levels.session_low

        # Long pullback: trending up, price near VWAP, was previously extended above
        if (
            state.regime in (Regime.TRENDING_UP, Regime.BREAKOUT)
            and abs(dist_from_vwap) <= self.VWAP_NEAR_THRESHOLD
            and session_high - vwap > self.VWAP_EXTENDED_THRESHOLD
            and state.flow.cumulative_delta > 0
        ):
            confidence = self._vwap_pullback_confidence(state, "long")
            signals = []
            if state.flow.delta_trend == "positive":
                signals.append("positive delta trend")
            if state.flow.cumulative_delta > 100:
                signals.append("strong cumulative delta")
            signals.append(f"regime: {state.regime.value}")

            setups.append(DetectedSetup(
                setup_type=SetupType.VWAP_PULLBACK,
                side="long",
                confidence=confidence,
                trigger_price=vwap,
                suggested_stop_distance=2.0,
                description=(
                    f"Price pulled back to VWAP ({vwap:.1f}) in uptrend. "
                    f"Session high {session_high:.1f} was {session_high - vwap:.1f}pts above VWAP. "
                    f"Delta remains positive ({state.flow.cumulative_delta:.0f})."
                ),
                confirming_signals=signals,
                invalidation=f"Price breaks below VWAP by 3+ pts (below {vwap - 3:.1f})",
            ))

        # Short pullback: trending down, price near VWAP, was previously extended below
        if (
            state.regime in (Regime.TRENDING_DOWN,)
            and abs(dist_from_vwap) <= self.VWAP_NEAR_THRESHOLD
            and vwap - session_low > self.VWAP_EXTENDED_THRESHOLD
            and state.flow.cumulative_delta < 0
        ):
            confidence = self._vwap_pullback_confidence(state, "short")
            signals = []
            if state.flow.delta_trend == "negative":
                signals.append("negative delta trend")
            if state.flow.cumulative_delta < -100:
                signals.append("strong negative cumulative delta")
            signals.append(f"regime: {state.regime.value}")

            setups.append(DetectedSetup(
                setup_type=SetupType.VWAP_PULLBACK,
                side="short",
                confidence=confidence,
                trigger_price=vwap,
                suggested_stop_distance=2.0,
                description=(
                    f"Price pulled back up to VWAP ({vwap:.1f}) in downtrend. "
                    f"Session low {session_low:.1f} was {vwap - session_low:.1f}pts below VWAP. "
                    f"Delta remains negative ({state.flow.cumulative_delta:.0f})."
                ),
                confirming_signals=signals,
                invalidation=f"Price breaks above VWAP by 3+ pts (above {vwap + 3:.1f})",
            ))

    def _vwap_pullback_confidence(self, state: MarketState, side: str) -> float:
        """Compute confidence for a VWAP pullback setup."""
        conf = 0.5
        # Regime confidence boost
        conf += state.regime_confidence * 0.2
        # Delta alignment boost
        if side == "long" and state.flow.delta_trend == "positive":
            conf += 0.15
        elif side == "short" and state.flow.delta_trend == "negative":
            conf += 0.15
        return min(conf, 1.0)

    def _detect_level_test(
        self,
        state: MarketState,
        setups: list[DetectedSetup],
    ) -> None:
        """LEVEL_TEST: Price approaches a key level with delta confirmation.

        Checks PDH, PDL, ONH, ONL, session H/L for proximity.
        """
        price = state.last_price
        levels = state.levels

        # Build list of (level_value, level_name, side_if_support, side_if_resistance)
        key_levels = []
        if levels.prior_day_high > 0:
            key_levels.append((levels.prior_day_high, "PDH", "resistance"))
        if levels.prior_day_low > 0:
            key_levels.append((levels.prior_day_low, "PDL", "support"))
        if levels.overnight_high > 0:
            key_levels.append((levels.overnight_high, "ONH", "resistance"))
        if levels.overnight_low > 0:
            key_levels.append((levels.overnight_low, "ONL", "support"))
        if levels.session_high > 0:
            key_levels.append((levels.session_high, "Session High", "resistance"))
        if levels.session_low > 0:
            key_levels.append((levels.session_low, "Session Low", "support"))

        for level_val, level_name, level_type in key_levels:
            dist = abs(price - level_val)
            if dist > self.LEVEL_PROXIMITY:
                continue

            # Determine side based on level type and delta
            if level_type == "support" and price >= level_val - self.LEVEL_PROXIMITY:
                # Testing support from above — look for bounce (long)
                side = "long"
                delta_confirms = state.flow.delta_1min > 0 or state.flow.delta_trend == "positive"
            elif level_type == "resistance" and price <= level_val + self.LEVEL_PROXIMITY:
                # Testing resistance from below — look for rejection (short)
                side = "short"
                delta_confirms = state.flow.delta_1min < 0 or state.flow.delta_trend == "negative"
            else:
                continue

            confidence = 0.5
            signals = [f"near {level_name} ({level_val:.1f})"]

            if delta_confirms:
                confidence += 0.15
                signals.append(f"delta confirms ({state.flow.delta_trend})")

            # Boost if rvol is elevated
            if state.flow.rvol > 1.5:
                confidence += 0.1
                signals.append(f"elevated volume (RVOL {state.flow.rvol:.1f}x)")

            confidence = min(confidence, 1.0)

            setups.append(DetectedSetup(
                setup_type=SetupType.LEVEL_TEST,
                side=side,
                confidence=confidence,
                trigger_price=level_val,
                suggested_stop_distance=3.0,
                description=(
                    f"Price testing {level_name} at {level_val:.1f} "
                    f"(currently {price:.1f}, {dist:.1f}pts away). "
                    f"Delta trend: {state.flow.delta_trend}."
                ),
                confirming_signals=signals,
                invalidation=(
                    f"Price moves through {level_name} by 3+ pts "
                    f"({'above' if side == 'short' else 'below'} "
                    f"{level_val + 3 if side == 'short' else level_val - 3:.1f})"
                ),
            ))

    def _detect_failed_breakout(
        self,
        state: MarketState,
        setups: list[DetectedSetup],
    ) -> None:
        """FAILED_BREAKOUT: Price breaks a level then reverses back through it.

        Detects by comparing session high/low to key levels — if session
        extreme exceeded a level by >2pts but current price is back on the
        original side, it's a failed breakout.
        """
        price = state.last_price
        levels = state.levels

        # (level_value, level_name, level_type)
        # "resistance" levels can have failed breakouts ABOVE them
        # "support" levels can have failed breakouts BELOW them
        key_levels: list[tuple[float, str, str]] = []
        if levels.prior_day_high > 0:
            key_levels.append((levels.prior_day_high, "PDH", "resistance"))
        if levels.prior_day_low > 0:
            key_levels.append((levels.prior_day_low, "PDL", "support"))
        if levels.overnight_high > 0:
            key_levels.append((levels.overnight_high, "ONH", "resistance"))
        if levels.overnight_low > 0:
            key_levels.append((levels.overnight_low, "ONL", "support"))

        for level_val, level_name, level_type in key_levels:
            # Failed breakout ABOVE (only for resistance levels):
            # session high exceeded level, price now below it
            if (
                level_type == "resistance"
                and levels.session_high > level_val + self.FAILED_BREAKOUT_PIERCE
                and price < level_val
            ):
                pierce_dist = levels.session_high - level_val
                fail_dist = level_val - price
                confidence = min(0.5 + (pierce_dist / 10.0) + (fail_dist / 10.0), 1.0)

                signals = [
                    f"broke above {level_name} by {pierce_dist:.1f}pts",
                    f"now {fail_dist:.1f}pts below {level_name}",
                ]
                if state.flow.delta_trend == "negative":
                    confidence = min(confidence + 0.1, 1.0)
                    signals.append("negative delta confirms reversal")

                setups.append(DetectedSetup(
                    setup_type=SetupType.FAILED_BREAKOUT,
                    side="short",
                    confidence=confidence,
                    trigger_price=level_val,
                    suggested_stop_distance=pierce_dist + 1.0,
                    description=(
                        f"Failed breakout above {level_name} ({level_val:.1f}). "
                        f"Session high reached {levels.session_high:.1f} "
                        f"but price fell back to {price:.1f}."
                    ),
                    confirming_signals=signals,
                    invalidation=f"Price reclaims {level_name} and holds above {level_val + 2:.1f}",
                ))

            # Failed breakout BELOW (only for support levels):
            # session low pierced below level, price now above it
            if (
                level_type == "support"
                and levels.session_low < level_val - self.FAILED_BREAKOUT_PIERCE
                and price > level_val
            ):
                pierce_dist = level_val - levels.session_low
                fail_dist = price - level_val
                confidence = min(0.5 + (pierce_dist / 10.0) + (fail_dist / 10.0), 1.0)

                signals = [
                    f"broke below {level_name} by {pierce_dist:.1f}pts",
                    f"now {fail_dist:.1f}pts above {level_name}",
                ]
                if state.flow.delta_trend == "positive":
                    confidence = min(confidence + 0.1, 1.0)
                    signals.append("positive delta confirms reversal")

                setups.append(DetectedSetup(
                    setup_type=SetupType.FAILED_BREAKOUT,
                    side="long",
                    confidence=confidence,
                    trigger_price=level_val,
                    suggested_stop_distance=pierce_dist + 1.0,
                    description=(
                        f"Failed breakout below {level_name} ({level_val:.1f}). "
                        f"Session low reached {levels.session_low:.1f} "
                        f"but price recovered to {price:.1f}."
                    ),
                    confirming_signals=signals,
                    invalidation=f"Price drops back below {level_name} under {level_val - 2:.1f}",
                ))

    def _detect_delta_divergence(
        self,
        state: MarketState,
        setups: list[DetectedSetup],
    ) -> None:
        """DELTA_DIVERGENCE: Price makes new extreme but delta doesn't confirm.

        Bearish divergence: price near session high but delta is negative.
        Bullish divergence: price near session low but delta is positive.
        """
        price = state.last_price
        session_high = state.levels.session_high
        session_low = state.levels.session_low

        if session_high <= 0 or session_low <= 0:
            return

        session_range = session_high - session_low
        if session_range <= 0:
            return

        range_pct = (price - session_low) / session_range

        # Bearish divergence: price near session high but delta negative
        if range_pct > 0.90 and state.flow.cumulative_delta < 0:
            div_magnitude = abs(state.flow.cumulative_delta)
            confidence = min(0.5 + (div_magnitude / 500.0), 0.9)
            if state.flow.delta_trend == "negative":
                confidence = min(confidence + 0.1, 1.0)

            setups.append(DetectedSetup(
                setup_type=SetupType.DELTA_DIVERGENCE,
                side="short",
                confidence=confidence,
                trigger_price=price,
                suggested_stop_distance=3.0,
                description=(
                    f"Bearish delta divergence: price at {price:.1f} "
                    f"(near session high {session_high:.1f}) but cumulative delta "
                    f"is negative ({state.flow.cumulative_delta:.0f})."
                ),
                confirming_signals=[
                    f"price in top 10% of session range ({range_pct:.0%})",
                    f"negative cumulative delta ({state.flow.cumulative_delta:.0f})",
                    f"delta trend: {state.flow.delta_trend}",
                ],
                invalidation=(
                    "Delta turns positive and price makes new session high "
                    f"above {session_high:.1f}"
                ),
            ))

        # Bullish divergence: price near session low but delta positive
        if range_pct < 0.10 and state.flow.cumulative_delta > 0:
            div_magnitude = state.flow.cumulative_delta
            confidence = min(0.5 + (div_magnitude / 500.0), 0.9)
            if state.flow.delta_trend == "positive":
                confidence = min(confidence + 0.1, 1.0)

            setups.append(DetectedSetup(
                setup_type=SetupType.DELTA_DIVERGENCE,
                side="long",
                confidence=confidence,
                trigger_price=price,
                suggested_stop_distance=3.0,
                description=(
                    f"Bullish delta divergence: price at {price:.1f} "
                    f"(near session low {session_low:.1f}) but cumulative delta "
                    f"is positive ({state.flow.cumulative_delta:.0f})."
                ),
                confirming_signals=[
                    f"price in bottom 10% of session range ({range_pct:.0%})",
                    f"positive cumulative delta ({state.flow.cumulative_delta:.0f})",
                    f"delta trend: {state.flow.delta_trend}",
                ],
                invalidation=(
                    "Delta turns negative and price makes new session low "
                    f"below {session_low:.1f}"
                ),
            ))

    def _detect_absorption(
        self,
        state: MarketState,
        bars_1s: list[dict[str, Any]],
        setups: list[DetectedSetup],
    ) -> None:
        """ABSORPTION: High volume at a level with minimal price movement.

        Requires RVOL > 2x AND small price range in recent bars.
        """
        if state.flow.rvol < self.ABSORPTION_RVOL_MIN:
            return

        # Check if near a key level
        price = state.last_price
        levels = state.levels
        nearby_level = self._find_nearest_key_level(price, levels)
        if nearby_level is None:
            return

        level_val, level_name, level_type = nearby_level

        # Check price range in recent bars
        if bars_1s:
            recent = bars_1s[-10:] if len(bars_1s) >= 10 else bars_1s
            highs = [b["high"] for b in recent]
            lows = [b["low"] for b in recent]
            bar_range = max(highs) - min(lows) if highs and lows else float("inf")
        else:
            # No bars — use spread as proxy for tightness
            bar_range = state.spread * 4

        # Absorption: high volume but tight range
        if bar_range <= 2.0:  # within 2pts
            side = "long" if level_type == "support" else "short"
            confidence = min(0.5 + (state.flow.rvol - 2.0) * 0.2, 0.9)

            if (side == "long" and state.flow.delta_trend == "positive") or \
               (side == "short" and state.flow.delta_trend == "negative"):
                confidence = min(confidence + 0.1, 1.0)

            total_vol = sum(b.get("volume", 0) for b in (bars_1s[-10:] if bars_1s else []))

            setups.append(DetectedSetup(
                setup_type=SetupType.ABSORPTION,
                side=side,
                confidence=confidence,
                trigger_price=level_val,
                suggested_stop_distance=2.0,
                description=(
                    f"Absorption detected at {level_name} ({level_val:.1f}). "
                    f"RVOL {state.flow.rvol:.1f}x with only {bar_range:.1f}pt range. "
                    f"Large orders appear to be absorbing {'selling' if side == 'long' else 'buying'} "
                    f"pressure."
                ),
                confirming_signals=[
                    f"RVOL {state.flow.rvol:.1f}x (>{self.ABSORPTION_RVOL_MIN}x threshold)",
                    f"tight range: {bar_range:.1f}pts",
                    f"near {level_name}",
                    f"volume: {total_vol}",
                ],
                invalidation=f"Price breaks {'below' if side == 'long' else 'above'} {level_val:.1f} with momentum",
            ))

    def _detect_mean_reversion(
        self,
        state: MarketState,
        setups: list[DetectedSetup],
    ) -> None:
        """MEAN_REVERSION: Extended from VWAP in choppy regime.

        Only fires in CHOPPY regime when price is >10pts from VWAP.
        """
        if state.regime != Regime.CHOPPY:
            return

        vwap = state.levels.vwap
        if vwap <= 0:
            return

        price = state.last_price
        dist_from_vwap = price - vwap

        if abs(dist_from_vwap) < self.MEAN_REVERSION_DISTANCE:
            return

        side = "short" if dist_from_vwap > 0 else "long"
        confidence = min(0.4 + (abs(dist_from_vwap) / 30.0), 0.85)

        # Boost if regime confidence is high (more clearly choppy)
        confidence = min(confidence + state.regime_confidence * 0.1, 0.9)

        direction = "above" if dist_from_vwap > 0 else "below"

        setups.append(DetectedSetup(
            setup_type=SetupType.MEAN_REVERSION,
            side=side,
            confidence=confidence,
            trigger_price=price,
            suggested_stop_distance=abs(dist_from_vwap) * 0.5,
            description=(
                f"Mean reversion setup: price {price:.1f} is {abs(dist_from_vwap):.1f}pts "
                f"{direction} VWAP ({vwap:.1f}) in choppy regime. "
                f"Expect reversion toward VWAP."
            ),
            confirming_signals=[
                f"choppy regime (confidence {state.regime_confidence:.0%})",
                f"{abs(dist_from_vwap):.1f}pts from VWAP",
                f"delta: {state.flow.cumulative_delta:.0f}",
            ],
            invalidation=(
                f"Regime changes to trending or price extends further "
                f"{'above' if side == 'short' else 'below'} "
                f"{price + 5 if side == 'short' else price - 5:.1f}"
            ),
        ))

    def _detect_exhaustion(
        self,
        state: MarketState,
        bars_1s: list[dict[str, Any]],
        setups: list[DetectedSetup],
    ) -> None:
        """EXHAUSTION: Rapid move with declining delta/volume.

        Requires >15pt move in <5 minutes with declining volume in the
        latter portion of the move.
        """
        if len(bars_1s) < 10:
            return

        # Look at the last 5 minutes of bars
        recent_bars = bars_1s[-300:] if len(bars_1s) > 300 else bars_1s

        if not recent_bars:
            return

        first_close = recent_bars[0].get("close", 0)
        last_close = recent_bars[-1].get("close", 0)
        move = last_close - first_close

        if abs(move) < self.EXHAUSTION_MOVE_MIN:
            return

        # Check if volume is declining in the latter half
        mid = len(recent_bars) // 2
        first_half_vol = sum(b.get("volume", 0) for b in recent_bars[:mid])
        second_half_vol = sum(b.get("volume", 0) for b in recent_bars[mid:])

        # Volume declining = second half has less volume
        volume_declining = second_half_vol < first_half_vol * 0.8

        # Delta declining relative to move direction
        delta_declining = (
            (move > 0 and state.flow.delta_1min < 0) or
            (move < 0 and state.flow.delta_1min > 0)
        )

        if not (volume_declining or delta_declining):
            return

        side = "short" if move > 0 else "long"
        confidence = 0.5
        signals = []

        if volume_declining:
            confidence += 0.15
            signals.append(
                f"volume declining: {second_half_vol} vs {first_half_vol} (first half)"
            )
        if delta_declining:
            confidence += 0.15
            signals.append(f"delta diverging from move (delta_1min: {state.flow.delta_1min:.0f})")
        if state.flow.delta_trend in ("negative" if move > 0 else "positive", "flipping"):
            confidence += 0.1
            signals.append(f"delta trend: {state.flow.delta_trend}")

        signals.append(f"move: {abs(move):.1f}pts in {len(recent_bars)}s")
        confidence = min(confidence, 0.95)

        setups.append(DetectedSetup(
            setup_type=SetupType.EXHAUSTION,
            side=side,
            confidence=confidence,
            trigger_price=state.last_price,
            suggested_stop_distance=abs(move) * 0.3,
            description=(
                f"Exhaustion signal: {abs(move):.1f}pt {'up' if move > 0 else 'down'} "
                f"move with {'declining volume' if volume_declining else 'diverging delta'}. "
                f"Potential reversal {'from highs' if move > 0 else 'from lows'}."
            ),
            confirming_signals=signals,
            invalidation=(
                f"Move resumes with fresh delta/volume "
                f"{'above' if move > 0 else 'below'} {state.last_price:.1f}"
            ),
        ))

    def _detect_opening_range_break(
        self,
        state: MarketState,
        bars_1s: list[dict[str, Any]],
        setups: list[DetectedSetup],
    ) -> None:
        """OPENING_RANGE_BREAK: Price breaks the first 15-min range after 9:45 AM ET.

        Computes the OR high/low from bars with timestamps between 9:30-9:45 ET.
        Only fires after 9:45 AM ET when price is outside the OR.
        """
        if not bars_1s:
            return

        # Determine current time in ET
        ts = state.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        now_et = ts.astimezone(ET)

        # OR is complete after 9:45
        or_end = now_et.replace(hour=9, minute=45, second=0, microsecond=0)
        if now_et < or_end:
            return

        # Find OR high/low from bars in the 9:30-9:45 window
        or_start = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        or_high = float("-inf")
        or_low = float("inf")
        or_bar_count = 0

        for bar in bars_1s:
            bar_ts = bar.get("timestamp")
            if bar_ts is None:
                continue
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.replace(tzinfo=timezone.utc)
            bar_et = bar_ts.astimezone(ET)

            if or_start <= bar_et < or_end:
                or_high = max(or_high, bar["high"])
                or_low = min(or_low, bar["low"])
                or_bar_count += 1

        if or_bar_count == 0 or or_high == float("-inf") or or_low == float("inf"):
            return

        price = state.last_price
        or_range = or_high - or_low

        # Upside break
        if price > or_high:
            break_dist = price - or_high
            confidence = min(0.5 + (break_dist / 10.0), 0.85)
            if state.flow.delta_trend == "positive":
                confidence = min(confidence + 0.1, 0.95)

            setups.append(DetectedSetup(
                setup_type=SetupType.OPENING_RANGE_BREAK,
                side="long",
                confidence=confidence,
                trigger_price=or_high,
                suggested_stop_distance=or_range,
                description=(
                    f"Opening range break to the upside. OR: {or_low:.1f}-{or_high:.1f} "
                    f"({or_range:.1f}pt range). Price {price:.1f} is {break_dist:.1f}pts "
                    f"above OR high."
                ),
                confirming_signals=[
                    f"OR range: {or_low:.1f}-{or_high:.1f}",
                    f"break distance: {break_dist:.1f}pts",
                    f"delta trend: {state.flow.delta_trend}",
                ],
                invalidation=f"Price falls back below OR high ({or_high:.1f})",
            ))

        # Downside break
        elif price < or_low:
            break_dist = or_low - price
            confidence = min(0.5 + (break_dist / 10.0), 0.85)
            if state.flow.delta_trend == "negative":
                confidence = min(confidence + 0.1, 0.95)

            setups.append(DetectedSetup(
                setup_type=SetupType.OPENING_RANGE_BREAK,
                side="short",
                confidence=confidence,
                trigger_price=or_low,
                suggested_stop_distance=or_range,
                description=(
                    f"Opening range break to the downside. OR: {or_low:.1f}-{or_high:.1f} "
                    f"({or_range:.1f}pt range). Price {price:.1f} is {break_dist:.1f}pts "
                    f"below OR low."
                ),
                confirming_signals=[
                    f"OR range: {or_low:.1f}-{or_high:.1f}",
                    f"break distance: {break_dist:.1f}pts",
                    f"delta trend: {state.flow.delta_trend}",
                ],
                invalidation=f"Price rallies back above OR low ({or_low:.1f})",
            ))

    def _detect_trend_continuation(
        self,
        state: MarketState,
        bars_1s: list[dict[str, Any]],
        setups: list[DetectedSetup],
    ) -> None:
        """TREND_CONTINUATION: Pullback in established trend (38-62% Fibonacci).

        In a trending regime, checks if price has pulled back 38-62% of the
        last swing and delta suggests resumption.
        """
        if state.regime not in (Regime.TRENDING_UP, Regime.TRENDING_DOWN):
            return

        if len(bars_1s) < 20:
            return

        # Get swing high/low from bars
        closes = [b.get("close", 0) for b in bars_1s]
        if not closes:
            return

        swing_high = max(closes)
        swing_low = min(closes)
        swing_range = swing_high - swing_low

        if swing_range < 5.0:
            return

        price = state.last_price

        if state.regime == Regime.TRENDING_UP:
            # Pullback from high: how much of the swing has retraced?
            pullback = swing_high - price
            pullback_pct = pullback / swing_range if swing_range > 0 else 0

            if 0.30 <= pullback_pct <= 0.68:
                confidence = 0.5
                signals = [f"pullback: {pullback_pct:.0%} of swing"]

                # Delta resumption (positive delta in uptrend = confirmation)
                if state.flow.delta_trend == "positive":
                    confidence += 0.15
                    signals.append("delta resuming positive")
                if state.flow.cumulative_delta > 0:
                    confidence += 0.1
                    signals.append(f"cumulative delta positive ({state.flow.cumulative_delta:.0f})")

                confidence = min(confidence + state.regime_confidence * 0.1, 0.9)
                signals.append(f"regime confidence: {state.regime_confidence:.0%}")

                setups.append(DetectedSetup(
                    setup_type=SetupType.TREND_CONTINUATION,
                    side="long",
                    confidence=confidence,
                    trigger_price=price,
                    suggested_stop_distance=pullback + 2.0,
                    description=(
                        f"Trend continuation in uptrend. Swing {swing_low:.1f}-{swing_high:.1f} "
                        f"({swing_range:.1f}pts). Price pulled back {pullback_pct:.0%} to "
                        f"{price:.1f}. Looking for resumption."
                    ),
                    confirming_signals=signals,
                    invalidation=(
                        f"Pullback exceeds 68% (below {swing_high - swing_range * 0.68:.1f})"
                    ),
                ))

        elif state.regime == Regime.TRENDING_DOWN:
            # Pullback from low (bounce): how much has retraced?
            pullback = price - swing_low
            pullback_pct = pullback / swing_range if swing_range > 0 else 0

            if 0.30 <= pullback_pct <= 0.68:
                confidence = 0.5
                signals = [f"pullback: {pullback_pct:.0%} of swing"]

                if state.flow.delta_trend == "negative":
                    confidence += 0.15
                    signals.append("delta resuming negative")
                if state.flow.cumulative_delta < 0:
                    confidence += 0.1
                    signals.append(f"cumulative delta negative ({state.flow.cumulative_delta:.0f})")

                confidence = min(confidence + state.regime_confidence * 0.1, 0.9)
                signals.append(f"regime confidence: {state.regime_confidence:.0%}")

                setups.append(DetectedSetup(
                    setup_type=SetupType.TREND_CONTINUATION,
                    side="short",
                    confidence=confidence,
                    trigger_price=price,
                    suggested_stop_distance=pullback + 2.0,
                    description=(
                        f"Trend continuation in downtrend. Swing {swing_high:.1f}-{swing_low:.1f} "
                        f"({swing_range:.1f}pts). Price bounced {pullback_pct:.0%} to "
                        f"{price:.1f}. Looking for continuation lower."
                    ),
                    confirming_signals=signals,
                    invalidation=(
                        f"Bounce exceeds 68% (above {swing_low + swing_range * 0.68:.1f})"
                    ),
                ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_nearest_key_level(
        self,
        price: float,
        levels: KeyLevels,
    ) -> tuple[float, str, str] | None:
        """Find the nearest key level within proximity threshold.

        Returns (level_value, level_name, "support"|"resistance") or None.
        """
        candidates = []
        if levels.prior_day_high > 0:
            candidates.append((levels.prior_day_high, "PDH", "resistance"))
        if levels.prior_day_low > 0:
            candidates.append((levels.prior_day_low, "PDL", "support"))
        if levels.overnight_high > 0:
            candidates.append((levels.overnight_high, "ONH", "resistance"))
        if levels.overnight_low > 0:
            candidates.append((levels.overnight_low, "ONL", "support"))
        if levels.session_high > 0:
            candidates.append((levels.session_high, "Session High", "resistance"))
        if levels.session_low > 0:
            candidates.append((levels.session_low, "Session Low", "support"))

        nearest = None
        min_dist = float("inf")
        for level_val, name, ltype in candidates:
            dist = abs(price - level_val)
            if dist <= self.LEVEL_PROXIMITY and dist < min_dist:
                min_dist = dist
                nearest = (level_val, name, ltype)

        return nearest
