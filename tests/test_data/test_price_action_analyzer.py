"""Tests for the PriceActionAnalyzer — narrative generation for LLM consumption."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.core.types import (
    CrossMarketContext,
    KeyLevels,
    MarketState,
    OrderFlowData,
    PositionState,
    Regime,
    SessionPhase,
    Side,
)
from src.data.price_action_analyzer import PriceActionAnalyzer


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_state(**overrides: object) -> MarketState:
    """Build a default MarketState with sensible values for narrative testing."""
    defaults: dict = {
        "timestamp": datetime.now(tz=UTC),
        "symbol": "MNQM6",
        "last_price": 19850.0,
        "bid": 19849.75,
        "ask": 19850.25,
        "spread": 0.50,
        "session_phase": SessionPhase.MORNING,
        "regime": Regime.TRENDING_UP,
        "regime_confidence": 0.75,
        "levels": KeyLevels(
            prior_day_high=19870.0,
            prior_day_low=19780.0,
            prior_day_close=19845.0,
            overnight_high=19865.0,
            overnight_low=19810.0,
            session_high=19868.0,
            session_low=19825.0,
            session_open=19840.0,
            vwap=19840.0,
            poc=19845.0,
            value_area_high=19860.0,
            value_area_low=19830.0,
        ),
        "flow": OrderFlowData(
            cumulative_delta=450.0,
            delta_1min=80.0,
            delta_5min=200.0,
            delta_trend="positive",
            rvol=1.2,
            volume_1min=350,
            large_lot_count_5min=2,
            tape_speed=8.0,
        ),
        "cross_market": CrossMarketContext(
            es_price=5420.0,
            es_change_pct=0.15,
            tick_index=480,
            vix=18.2,
            vix_change_pct=-0.3,
            ten_year_yield=4.25,
            dxy=104.5,
        ),
        "position": None,
        "daily_pnl": 0.0,
        "daily_trades": 0,
    }
    defaults.update(overrides)
    return MarketState(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBasicNarrativeGeneration:
    """Core functionality: the analyzer returns a non-empty multi-sentence string."""

    def test_returns_non_empty_string(self) -> None:
        analyzer = PriceActionAnalyzer()
        state = _make_state()
        result = analyzer.analyze(state)
        assert isinstance(result, str)
        assert len(result) > 20  # meaningful content, not trivial

    def test_narrative_contains_multiple_sentences(self) -> None:
        analyzer = PriceActionAnalyzer()
        state = _make_state()
        result = analyzer.analyze(state)
        # Narratives should have at least 3 sentences (period-terminated)
        sentences = [s.strip() for s in result.split(".") if s.strip()]
        assert len(sentences) >= 3, f"Expected >= 3 sentences, got {len(sentences)}: {result}"


class TestVWAPRelationship:
    """VWAP proximity and relationship language."""

    def test_price_above_vwap(self) -> None:
        """Price 10 pts above VWAP -> 'above VWAP' in narrative."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(last_price=19850.0, levels=KeyLevels(vwap=19840.0))
        result = analyzer.analyze(state)
        assert "above VWAP" in result.lower() or "above vwap" in result.lower()

    def test_price_below_vwap(self) -> None:
        """Price 10 pts below VWAP -> 'below VWAP' in narrative."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(last_price=19830.0, levels=KeyLevels(vwap=19840.0))
        result = analyzer.analyze(state)
        assert "below VWAP" in result.lower() or "below vwap" in result.lower()

    def test_price_at_vwap(self) -> None:
        """Price within 2 pts of VWAP -> 'at VWAP' in narrative."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(last_price=19841.0, levels=KeyLevels(vwap=19840.0))
        result = analyzer.analyze(state)
        assert "at vwap" in result.lower()

    def test_significantly_extended_from_vwap(self) -> None:
        """Price >15 pts from VWAP -> 'significantly extended' in narrative."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(last_price=19860.0, levels=KeyLevels(vwap=19840.0))
        result = analyzer.analyze(state)
        assert "significantly extended" in result.lower() or "extended" in result.lower()


class TestDeltaInterpretation:
    """Order flow / delta narrative language."""

    def test_strong_buying_pressure(self) -> None:
        """Delta > 500 -> 'strong buying' or 'strong' + 'buy'."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(cumulative_delta=600.0, tape_speed=8.0)
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "strong" in lower and "buy" in lower

    def test_strong_selling_pressure(self) -> None:
        """Delta < -500 -> 'strong selling' or 'strong' + 'sell'."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(cumulative_delta=-600.0, tape_speed=8.0)
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "strong" in lower and "sell" in lower

    def test_moderate_delta(self) -> None:
        """Delta in 200-500 range -> 'moderate' in narrative."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(cumulative_delta=350.0, tape_speed=8.0)
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        assert "moderate" in result.lower()

    def test_light_delta(self) -> None:
        """Delta < 200 -> 'light' in narrative."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(cumulative_delta=100.0, tape_speed=8.0)
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        assert "light" in result.lower()

    def test_delta_change_tracked(self) -> None:
        """Sequential calls should mention delta change direction."""
        analyzer = PriceActionAnalyzer()
        # First call: negative delta
        state1 = _make_state(
            flow=OrderFlowData(cumulative_delta=-200.0, tape_speed=5.0)
        )
        analyzer.analyze(state1)

        # Second call: delta turned positive
        state2 = _make_state(
            flow=OrderFlowData(cumulative_delta=300.0, tape_speed=5.0)
        )
        result2 = analyzer.analyze(state2)
        lower = result2.lower()
        # Should reference improving/flipping/turning
        assert any(word in lower for word in ["improv", "flip", "turn", "recovering"])


class TestTapeSpeed:
    """Tape speed narrative interpretation."""

    def test_fast_tape(self) -> None:
        """Tape > 10 tps -> 'fast' or 'accelerat' in narrative."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(tape_speed=15.0, cumulative_delta=100.0)
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "fast" in lower or "accelerat" in lower

    def test_slow_tape(self) -> None:
        """Tape < 5 tps -> 'slow' in narrative."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(tape_speed=3.0, cumulative_delta=100.0)
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        assert "slow" in result.lower()


class TestCrossMarketContext:
    """Cross-market data (TICK, VIX, ES) in narrative."""

    def test_bullish_tick_extreme(self) -> None:
        """TICK > 800 -> 'bullish' in narrative."""
        analyzer = PriceActionAnalyzer()
        cm = CrossMarketContext(tick_index=900, vix=18.0)
        state = _make_state(cross_market=cm)
        result = analyzer.analyze(state)
        assert "bullish" in result.lower()

    def test_bearish_tick_extreme(self) -> None:
        """TICK < -800 -> 'bearish' in narrative."""
        analyzer = PriceActionAnalyzer()
        cm = CrossMarketContext(tick_index=-900, vix=18.0)
        state = _make_state(cross_market=cm)
        result = analyzer.analyze(state)
        assert "bearish" in result.lower()

    def test_vix_declining_risk_on(self) -> None:
        """VIX declining -> 'risk-on' or 'declining' in narrative."""
        analyzer = PriceActionAnalyzer()
        cm = CrossMarketContext(vix=18.0, vix_change_pct=-1.5)
        state = _make_state(cross_market=cm)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "risk-on" in lower or "declining" in lower or "falling" in lower

    def test_vix_spiking_risk_off(self) -> None:
        """VIX spiking -> 'risk-off' or 'spiking' in narrative."""
        analyzer = PriceActionAnalyzer()
        cm = CrossMarketContext(vix=25.0, vix_change_pct=3.0)
        state = _make_state(cross_market=cm)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "risk-off" in lower or "spik" in lower or "rising" in lower

    def test_no_cross_market_data(self) -> None:
        """Zero cross-market values -> no cross-market section in narrative."""
        analyzer = PriceActionAnalyzer()
        cm = CrossMarketContext()  # all zeros
        state = _make_state(cross_market=cm)
        result = analyzer.analyze(state)
        # Should not crash, should still produce valid narrative
        assert isinstance(result, str)
        assert len(result) > 10


class TestPositionContext:
    """When a position is open, narrative includes position-relevant info."""

    def test_long_position_context(self) -> None:
        """With a long position, narrative mentions the position."""
        analyzer = PriceActionAnalyzer()
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19840.0,
            unrealized_pnl=60.0,
            stop_price=19825.0,
            time_in_trade_sec=180,
        )
        state = _make_state(position=pos)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "long" in lower or "position" in lower

    def test_short_position_context(self) -> None:
        """With a short position, narrative mentions the position."""
        analyzer = PriceActionAnalyzer()
        pos = PositionState(
            side=Side.SHORT,
            quantity=2,
            avg_entry=19870.0,
            unrealized_pnl=40.0,
            stop_price=19885.0,
            time_in_trade_sec=120,
        )
        state = _make_state(position=pos)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "short" in lower or "position" in lower

    def test_flat_no_position_language(self) -> None:
        """Without a position, no position-specific language."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(position=None)
        result = analyzer.analyze(state)
        # Should not mention unrealized P&L or stop loss
        lower = result.lower()
        assert "unrealized" not in lower


class TestSessionPhaseContext:
    """Session phase and time context in narrative."""

    def test_open_drive_mentioned(self) -> None:
        """Open drive phase -> narrative mentions 'open' context."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(session_phase=SessionPhase.OPEN_DRIVE)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "open" in lower

    def test_midday_mentioned(self) -> None:
        """Midday phase -> narrative mentions 'midday' or 'lunch'."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(session_phase=SessionPhase.MIDDAY)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "midday" in lower or "lunch" in lower or "chop" in lower

    def test_close_mentioned(self) -> None:
        """Close phase -> narrative mentions 'close' or 'end of day'."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(session_phase=SessionPhase.CLOSE)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "close" in lower or "end of day" in lower or "closing" in lower


class TestKeyLevels:
    """Key level proximity in narrative."""

    def test_near_prior_day_high(self) -> None:
        """Price near PDH -> mentioned in narrative."""
        analyzer = PriceActionAnalyzer()
        levels = KeyLevels(
            prior_day_high=19855.0,
            prior_day_low=19780.0,
            vwap=19830.0,
            session_high=19860.0,
            session_low=19810.0,
            value_area_high=19850.0,
            value_area_low=19820.0,
        )
        state = _make_state(last_price=19853.0, levels=levels)
        result = analyzer.analyze(state)
        lower = result.lower()
        # Should reference prior day high or PDH
        assert "prior day high" in lower or "pdh" in lower or "prior high" in lower

    def test_near_session_high(self) -> None:
        """Price near session high -> mentioned in narrative."""
        analyzer = PriceActionAnalyzer()
        levels = KeyLevels(
            session_high=19852.0,
            session_low=19810.0,
            vwap=19830.0,
            value_area_high=19850.0,
            value_area_low=19820.0,
        )
        state = _make_state(last_price=19850.0, levels=levels)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "session high" in lower


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_all_zero_state(self) -> None:
        """All zeros should not crash, just produce a basic narrative."""
        analyzer = PriceActionAnalyzer()
        state = MarketState()  # all defaults / zeros
        result = analyzer.analyze(state)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_large_lot_mention(self) -> None:
        """When large lots are present, they're mentioned."""
        analyzer = PriceActionAnalyzer()
        flow = OrderFlowData(
            large_lot_count_5min=5,
            cumulative_delta=300.0,
            tape_speed=8.0,
        )
        state = _make_state(flow=flow)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "large" in lower or "lot" in lower or "print" in lower

    def test_prior_summaries_stored(self) -> None:
        """Analyzer tracks up to 5 prior summaries internally."""
        analyzer = PriceActionAnalyzer()
        for i in range(7):
            state = _make_state(
                last_price=19840.0 + i,
                flow=OrderFlowData(cumulative_delta=float(i * 100), tape_speed=5.0),
            )
            analyzer.analyze(state)
        # deque maxlen=5, so only last 5 stored
        assert len(analyzer._prior_summaries) == 5

    def test_regime_mentioned(self) -> None:
        """Market regime is reflected in the narrative."""
        analyzer = PriceActionAnalyzer()
        state = _make_state(regime=Regime.TRENDING_UP)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "trend" in lower or "uptrend" in lower or "higher" in lower

    def test_value_area_inside(self) -> None:
        """Price inside value area -> mentioned."""
        analyzer = PriceActionAnalyzer()
        levels = KeyLevels(
            value_area_high=19860.0,
            value_area_low=19830.0,
            vwap=19845.0,
            session_high=19870.0,
            session_low=19810.0,
        )
        state = _make_state(last_price=19845.0, levels=levels)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "value area" in lower or "inside" in lower

    def test_value_area_outside(self) -> None:
        """Price outside value area -> mentioned."""
        analyzer = PriceActionAnalyzer()
        levels = KeyLevels(
            value_area_high=19840.0,
            value_area_low=19820.0,
            vwap=19830.0,
            session_high=19870.0,
            session_low=19810.0,
        )
        state = _make_state(last_price=19860.0, levels=levels)
        result = analyzer.analyze(state)
        lower = result.lower()
        assert "value area" in lower or "outside" in lower or "above" in lower
