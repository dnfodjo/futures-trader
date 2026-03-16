"""Price action narrative generator for LLM consumption.

Converts raw numerical MarketState data into rich narrative text that gives the LLM
contextual understanding of what price is DOING, not just what it IS.

Instead of the LLM receiving "last_price: 19850, vwap: 19840, delta: 450",
it receives: "Price is 10 points above VWAP, trending higher with strong buying pressure
(delta +450). Tape speed moderate at 8 trades/sec. Two large buy lots in the last 5 minutes.
Approaching prior day high at 19870."
"""

from __future__ import annotations

from collections import deque

from src.core.types import MarketState, Regime, SessionPhase


# ── Thresholds ────────────────────────────────────────────────────────────────

# VWAP proximity (points)
_VWAP_AT = 2.0  # within 2pts = "at VWAP"
_VWAP_NEAR = 5.0  # 2-5pts = "near VWAP"
_VWAP_MODERATE = 15.0  # 5-15pts = "moderately extended"
# >15pts = "significantly extended"

# Delta magnitude
_DELTA_STRONG = 500.0
_DELTA_MODERATE = 200.0
# <200 = "light"

# Tape speed (trades per second)
_TAPE_FAST = 10.0
_TAPE_SLOW = 5.0

# TICK thresholds
_TICK_BULLISH_EXTREME = 800
_TICK_BEARISH_EXTREME = -800
_TICK_NEUTRAL_HIGH = 400
_TICK_NEUTRAL_LOW = -400

# Key level proximity (points)
_LEVEL_PROXIMITY = 5.0


class PriceActionAnalyzer:
    """Generates narrative descriptions of price action for LLM consumption.

    Instead of the LLM receiving "last_price: 19850, vwap: 19840, delta: 450",
    it receives: "Price is 10 points above VWAP, trending higher with strong buying
    pressure (delta +450). Tape speed accelerating at 15 trades/sec. Two large buy lots
    (12 and 8 contracts) in the last minute. Approaching session high at 19855."
    """

    def __init__(self) -> None:
        self._prior_summaries: deque[str] = deque(maxlen=5)
        self._last_price: float = 0.0
        self._last_delta: float = 0.0

    def analyze(self, state: MarketState) -> str:
        """Generate a comprehensive narrative of current price action.

        Returns 3-6 sentences covering:
        1. Price location relative to key levels and VWAP
        2. Order flow interpretation (delta, tape, large lots)
        3. Cross-market context (ES, TICK, VIX)
        4. What price is doing (trending, consolidating, reversing)
        5. Any notable patterns or divergences
        """
        parts: list[str] = []

        # 1. Price location relative to VWAP and key levels
        parts.append(self._describe_price_location(state))

        # 2. Order flow reading (delta, tape, large lots)
        parts.append(self._describe_order_flow(state))

        # 3. Trend / regime context
        regime_text = self._describe_regime(state)
        if regime_text:
            parts.append(regime_text)

        # 4. Cross-market context (only if data available)
        cross_text = self._describe_cross_market(state)
        if cross_text:
            parts.append(cross_text)

        # 5. Position context (only if in a trade)
        pos_text = self._describe_position(state)
        if pos_text:
            parts.append(pos_text)

        # 6. Session phase / time context
        parts.append(self._describe_session_phase(state))

        # Update tracking state
        self._last_price = state.last_price
        self._last_delta = state.flow.cumulative_delta

        narrative = " ".join(parts)
        self._prior_summaries.append(narrative)
        return narrative

    # ── Price Location ────────────────────────────────────────────────────────

    def _describe_price_location(self, state: MarketState) -> str:
        """Describe where price is relative to VWAP, value area, and key levels."""
        price = state.last_price
        vwap = state.levels.vwap
        segments: list[str] = []

        # VWAP relationship
        if vwap > 0.0:
            vwap_dist = price - vwap
            abs_dist = abs(vwap_dist)

            if abs_dist <= _VWAP_AT:
                segments.append(f"Price at {price:.0f} is trading at VWAP ({vwap:.0f})")
            elif abs_dist <= _VWAP_NEAR:
                direction = "above" if vwap_dist > 0 else "below"
                segments.append(
                    f"Price at {price:.0f} is {abs_dist:.0f} points {direction} "
                    f"VWAP ({vwap:.0f}), near VWAP"
                )
            elif abs_dist <= _VWAP_MODERATE:
                direction = "above" if vwap_dist > 0 else "below"
                segments.append(
                    f"Price at {price:.0f} is {abs_dist:.0f} points {direction} "
                    f"VWAP ({vwap:.0f}), moderately extended"
                )
            else:
                direction = "above" if vwap_dist > 0 else "below"
                segments.append(
                    f"Price at {price:.0f} is {abs_dist:.0f} points {direction} "
                    f"VWAP ({vwap:.0f}), significantly extended"
                )
        else:
            segments.append(f"Price at {price:.0f}")

        # Value area relationship
        va_high = state.levels.value_area_high
        va_low = state.levels.value_area_low
        if va_high > 0.0 and va_low > 0.0:
            if va_low <= price <= va_high:
                segments.append("Inside value area.")
            elif price > va_high:
                segments.append(f"Above value area high ({va_high:.0f}).")
            else:
                segments.append(f"Below value area low ({va_low:.0f}).")

        # Key level proximity
        level_notes = self._find_nearby_levels(state)
        if level_notes:
            segments.append(level_notes)

        return " ".join(segments)

    def _find_nearby_levels(self, state: MarketState) -> str:
        """Identify key levels price is near."""
        price = state.last_price
        nearby: list[str] = []

        levels_map = {
            "prior day high": state.levels.prior_day_high,
            "prior day low": state.levels.prior_day_low,
            "session high": state.levels.session_high,
            "session low": state.levels.session_low,
            "overnight high": state.levels.overnight_high,
            "overnight low": state.levels.overnight_low,
            "POC": state.levels.poc,
        }

        for name, level in levels_map.items():
            if level <= 0.0:
                continue
            dist = abs(price - level)
            if dist <= _LEVEL_PROXIMITY:
                direction = "above" if price >= level else "below"
                nearby.append(f"{dist:.0f} points {direction} {name} ({level:.0f})")

        if nearby:
            return "Near: " + "; ".join(nearby) + "."
        return ""

    # ── Order Flow ────────────────────────────────────────────────────────────

    def _describe_order_flow(self, state: MarketState) -> str:
        """Describe delta, tape speed, and large lots."""
        flow = state.flow
        segments: list[str] = []

        # Delta magnitude and direction
        delta = flow.cumulative_delta
        abs_delta = abs(delta)

        if abs_delta >= _DELTA_STRONG:
            strength = "strong"
        elif abs_delta >= _DELTA_MODERATE:
            strength = "moderate"
        else:
            strength = "light"

        if delta > 0:
            segments.append(
                f"{strength.capitalize()} buying pressure (delta +{delta:.0f})."
            )
        elif delta < 0:
            segments.append(
                f"{strength.capitalize()} selling pressure (delta {delta:.0f})."
            )
        else:
            segments.append("Delta neutral, no clear order flow bias.")

        # Delta change from prior reading
        if self._last_delta != 0.0:
            delta_change = delta - self._last_delta
            if delta_change > 100:
                if self._last_delta < 0 and delta > 0:
                    segments.append("Delta flipping positive, buyers recovering control.")
                else:
                    segments.append("Delta improving, buying pressure increasing.")
            elif delta_change < -100:
                if self._last_delta > 0 and delta < 0:
                    segments.append("Delta turning negative, sellers taking over.")
                else:
                    segments.append("Delta fading, selling pressure building.")

        # Tape speed
        tape = flow.tape_speed
        if tape > 0:
            if tape >= _TAPE_FAST:
                segments.append(
                    f"Tape fast at {tape:.0f} trades/sec, high activity."
                )
            elif tape >= _TAPE_SLOW:
                segments.append(
                    f"Tape moderate at {tape:.0f} trades/sec."
                )
            else:
                segments.append(
                    f"Tape slow at {tape:.0f} trades/sec, low participation."
                )

        # Large lots
        large_lots = flow.large_lot_count_5min
        if large_lots > 0:
            lot_word = "print" if large_lots == 1 else "prints"
            direction = "buy" if delta >= 0 else "sell"
            segments.append(
                f"{large_lots} large {direction} {lot_word} in the last 5 minutes."
            )

        return " ".join(segments)

    # ── Regime / Trend ────────────────────────────────────────────────────────

    def _describe_regime(self, state: MarketState) -> str:
        """Describe the current market regime / trend character."""
        regime = state.regime
        confidence = state.regime_confidence

        conf_word = "high" if confidence >= 0.7 else "moderate" if confidence >= 0.4 else "low"

        regime_descriptions = {
            Regime.TRENDING_UP: (
                f"Market trending higher ({conf_word} confidence). "
                "Favoring continuation and pullback entries on the long side."
            ),
            Regime.TRENDING_DOWN: (
                f"Market trending lower ({conf_word} confidence). "
                "Favoring continuation and rally-sells on the short side."
            ),
            Regime.CHOPPY: (
                f"Choppy, range-bound conditions ({conf_word} confidence). "
                "Mean reversion setups preferred, avoid breakout chases."
            ),
            Regime.BREAKOUT: (
                f"Breakout in progress ({conf_word} confidence). "
                "Expansion move underway, watch for follow-through or failure."
            ),
            Regime.NEWS_DRIVEN: (
                f"News-driven volatility ({conf_word} confidence). "
                "Price action dominated by headlines, elevated risk."
            ),
            Regime.LOW_VOLUME: (
                f"Low volume environment ({conf_word} confidence). "
                "Thin tape, risk of erratic moves on small orders."
            ),
        }

        return regime_descriptions.get(regime, "")

    # ── Cross-Market ──────────────────────────────────────────────────────────

    def _describe_cross_market(self, state: MarketState) -> str:
        """Describe cross-market signals: TICK, VIX, ES."""
        cm = state.cross_market
        segments: list[str] = []

        # TICK interpretation
        tick = cm.tick_index
        if tick != 0:
            if tick > _TICK_BULLISH_EXTREME:
                segments.append(
                    f"NYSE TICK at bullish extreme ({tick:+d}), broad buying."
                )
            elif tick < _TICK_BEARISH_EXTREME:
                segments.append(
                    f"NYSE TICK at bearish extreme ({tick:+d}), broad selling."
                )
            elif _TICK_NEUTRAL_LOW <= tick <= _TICK_NEUTRAL_HIGH:
                segments.append(f"TICK neutral ({tick:+d}).")
            elif tick > _TICK_NEUTRAL_HIGH:
                segments.append(f"TICK leaning bullish ({tick:+d}).")
            else:
                segments.append(f"TICK leaning bearish ({tick:+d}).")

        # VIX interpretation
        vix = cm.vix
        vix_chg = cm.vix_change_pct
        if vix > 0:
            if vix_chg < -1.0:
                segments.append(
                    f"VIX declining ({vix:.1f}, {vix_chg:+.1f}%), risk-on environment."
                )
            elif vix_chg > 1.0:
                segments.append(
                    f"VIX rising ({vix:.1f}, {vix_chg:+.1f}%), risk-off caution."
                )
            elif vix_chg != 0.0:
                if vix_chg < 0:
                    segments.append(
                        f"VIX slightly declining ({vix:.1f}, {vix_chg:+.1f}%)."
                    )
                else:
                    segments.append(
                        f"VIX slightly rising ({vix:.1f}, {vix_chg:+.1f}%)."
                    )

        # ES context
        es_chg = cm.es_change_pct
        if cm.es_price > 0 and es_chg != 0.0:
            if es_chg > 0.1:
                segments.append(f"ES confirming strength ({es_chg:+.2f}%).")
            elif es_chg < -0.1:
                segments.append(f"ES showing weakness ({es_chg:+.2f}%).")
            else:
                segments.append(f"ES flat ({es_chg:+.2f}%).")

        if not segments:
            return ""

        return "Cross-market: " + " ".join(segments)

    # ── Position Context ──────────────────────────────────────────────────────

    def _describe_position(self, state: MarketState) -> str:
        """Describe the current position if one exists."""
        pos = state.position
        if pos is None:
            return ""

        side_str = pos.side.value  # "long" or "short"
        qty = pos.quantity
        entry = pos.avg_entry
        pnl = pos.unrealized_pnl
        hold_min = pos.hold_time_min

        pnl_word = "profit" if pnl >= 0 else "loss"
        pnl_sign = "+" if pnl >= 0 else ""

        segments = [
            f"Currently {side_str} {qty} contracts from {entry:.0f},",
            f"unrealized {pnl_sign}{pnl:.0f} ({pnl_word}),",
            f"held for {hold_min:.1f} min.",
        ]

        if pos.stop_price > 0:
            stop_dist = abs(state.last_price - pos.stop_price)
            segments.append(f"Stop at {pos.stop_price:.0f} ({stop_dist:.0f} pts away).")

        return " ".join(segments)

    # ── Session Phase ─────────────────────────────────────────────────────────

    def _describe_session_phase(self, state: MarketState) -> str:
        """Describe the current session phase and its trading implications."""
        phase = state.session_phase

        phase_descriptions = {
            SessionPhase.PRE_MARKET: (
                "Pre-market session. Monitoring overnight developments "
                "and preparing for the open."
            ),
            SessionPhase.OPEN_DRIVE: (
                "Open drive phase (first 30 min). High volatility expected, "
                "watching for initial directional commitment."
            ),
            SessionPhase.MORNING: (
                "Morning session. Continuation moves likely if the open drive "
                "established a clear direction."
            ),
            SessionPhase.MIDDAY: (
                "Midday chop zone. Typically lower volume and range-bound. "
                "Tighter stops and reduced expectations."
            ),
            SessionPhase.AFTERNOON: (
                "Afternoon session. Watching for institutional re-engagement "
                "and potential trend resumption."
            ),
            SessionPhase.CLOSE: (
                "Closing session. End of day squaring and potential "
                "last-minute directional moves."
            ),
            SessionPhase.AFTER_HOURS: (
                "After hours session. Thin liquidity, limited participation."
            ),
        }

        return phase_descriptions.get(phase, "")
