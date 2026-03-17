"""Tests for core domain types."""

from datetime import UTC, datetime

from src.core.types import (
    ActionType,
    Event,
    EventType,
    GuardrailResult,
    KeyLevels,
    LLMAction,
    MarketState,
    OrderFlowData,
    PositionState,
    Regime,
    SessionPhase,
    SessionSummary,
    Side,
    TradeRecord,
)


def test_llm_action_do_nothing():
    action = LLMAction(
        action=ActionType.DO_NOTHING,
        reasoning="No setup present",
        confidence=0.3,
    )
    assert action.action == ActionType.DO_NOTHING
    assert action.side is None
    assert action.quantity is None


def test_llm_action_enter_long():
    action = LLMAction(
        action=ActionType.ENTER,
        side=Side.LONG,
        quantity=3,
        stop_distance=15.0,
        reasoning="Strong delta confirmation at VWAP",
        confidence=0.8,
        model_used="sonnet",
    )
    assert action.side == Side.LONG
    assert action.quantity == 3
    assert action.stop_distance == 15.0
    assert action.confidence == 0.8


def test_market_state_defaults():
    state = MarketState()
    assert state.position is None
    assert state.daily_pnl == 0.0
    assert state.in_blackout is False
    assert state.regime == Regime.CHOPPY
    assert state.session_phase == SessionPhase.PRE_MARKET


def test_market_state_with_position():
    pos = PositionState(
        side=Side.LONG,
        quantity=3,
        avg_entry=19850.0,
        unrealized_pnl=60.0,
        stop_price=19835.0,
        time_in_trade_sec=180,
    )
    state = MarketState(
        last_price=19860.0,
        position=pos,
        daily_pnl=145.0,
        daily_trades=2,
        daily_winners=2,
    )
    assert state.position is not None
    assert state.position.quantity == 3
    assert state.daily_pnl == 145.0


def test_position_state():
    pos = PositionState(
        side=Side.SHORT,
        quantity=2,
        avg_entry=19900.0,
        unrealized_pnl=40.0,
        max_favorable=80.0,
        max_adverse=-20.0,
    )
    assert pos.side == Side.SHORT
    assert pos.max_favorable == 80.0


def test_trade_record_has_uuid():
    trade = TradeRecord(
        timestamp_entry=datetime.now(tz=UTC),
        side=Side.LONG,
        entry_quantity=2,
        entry_price=19850.0,
        stop_price=19835.0,
    )
    assert len(trade.id) == 36  # UUID format


def test_guardrail_result():
    allowed = GuardrailResult(allowed=True)
    assert allowed.allowed is True
    assert allowed.reason == ""

    blocked = GuardrailResult(
        allowed=False,
        reason="Max position exceeded",
        modified_quantity=3,
    )
    assert blocked.allowed is False
    assert blocked.modified_quantity == 3


def test_event():
    event = Event(
        type=EventType.ORDER_FILLED,
        data={"order_id": 123, "fill_price": 19850.0},
    )
    assert event.type == EventType.ORDER_FILLED
    assert event.data["fill_price"] == 19850.0


def test_key_levels_defaults():
    levels = KeyLevels()
    assert levels.vwap == 0.0
    assert levels.poc == 0.0


def test_order_flow_data():
    flow = OrderFlowData(
        cumulative_delta=2840.0,
        delta_5min=-320.0,
        delta_trend="weakening",
        rvol=1.3,
    )
    assert flow.delta_trend == "weakening"
    assert flow.rvol == 1.3


# ── PositionState computed properties ──


class TestPositionStateProperties:
    def test_pnl_per_contract(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19850.0,
            unrealized_pnl=90.0,
        )
        assert pos.pnl_per_contract == 30.0

    def test_pnl_per_contract_zero_qty(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=0,
            avg_entry=19850.0,
        )
        assert pos.pnl_per_contract == 0.0

    def test_risk_per_contract(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19850.0,
            stop_price=19835.0,
        )
        assert pos.risk_per_contract == 15.0

    def test_risk_per_contract_short(self):
        pos = PositionState(
            side=Side.SHORT,
            quantity=2,
            avg_entry=19900.0,
            stop_price=19920.0,
        )
        assert pos.risk_per_contract == 20.0

    def test_risk_per_contract_no_stop(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19850.0,
        )
        assert pos.risk_per_contract == 0.0

    def test_is_profitable(self):
        profitable = PositionState(
            side=Side.LONG, quantity=1, avg_entry=19850.0, unrealized_pnl=50.0
        )
        assert profitable.is_profitable is True

        losing = PositionState(
            side=Side.LONG, quantity=1, avg_entry=19850.0, unrealized_pnl=-30.0
        )
        assert losing.is_profitable is False

        breakeven = PositionState(
            side=Side.LONG, quantity=1, avg_entry=19850.0, unrealized_pnl=0.0
        )
        assert breakeven.is_profitable is False

    def test_hold_time_min(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=1,
            avg_entry=19850.0,
            time_in_trade_sec=900,
        )
        assert pos.hold_time_min == 15.0


# ── SessionSummary computed properties ──


class TestSessionSummaryProperties:
    def test_win_rate(self):
        summary = SessionSummary(date="2026-03-14", total_trades=10, winners=7)
        assert summary.win_rate == 70.0

    def test_win_rate_no_trades(self):
        summary = SessionSummary(date="2026-03-14")
        assert summary.win_rate == 0.0

    def test_profit_factor(self):
        trades = [
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19850.0,
                stop_price=19835.0,
                pnl=100.0,
            ),
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19860.0,
                stop_price=19845.0,
                pnl=50.0,
            ),
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19870.0,
                stop_price=19855.0,
                pnl=-60.0,
            ),
        ]
        summary = SessionSummary(
            date="2026-03-14", total_trades=3, winners=2, losers=1, trades=trades
        )
        # Profit factor = (100 + 50) / 60 = 2.5
        assert summary.profit_factor == 2.5

    def test_profit_factor_no_losses(self):
        trades = [
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19850.0,
                stop_price=19835.0,
                pnl=100.0,
            ),
        ]
        summary = SessionSummary(
            date="2026-03-14", total_trades=1, winners=1, trades=trades
        )
        assert summary.profit_factor == float("inf")

    def test_profit_factor_no_trades(self):
        summary = SessionSummary(date="2026-03-14")
        assert summary.profit_factor == 0.0

    def test_avg_winner_and_loser(self):
        trades = [
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19850.0,
                stop_price=19835.0,
                pnl=100.0,
            ),
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19860.0,
                stop_price=19845.0,
                pnl=200.0,
            ),
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.SHORT,
                entry_quantity=1,
                entry_price=19900.0,
                stop_price=19920.0,
                pnl=-50.0,
            ),
        ]
        summary = SessionSummary(
            date="2026-03-14", total_trades=3, winners=2, losers=1, trades=trades
        )
        assert summary.avg_winner == 150.0
        assert summary.avg_loser == -50.0

    def test_is_green_day(self):
        green = SessionSummary(date="2026-03-14", net_pnl=200.0)
        assert green.is_green_day is True

        red = SessionSummary(date="2026-03-14", net_pnl=-100.0)
        assert red.is_green_day is False

        flat = SessionSummary(date="2026-03-14", net_pnl=0.0)
        assert flat.is_green_day is False


# ── MarketState.to_llm_dict() ──


class TestMarketStateToLLMDict:
    def test_minimal_state(self):
        state = MarketState(last_price=19850.0)
        d = state.to_llm_dict()
        assert d["price"]["last"] == 19850.0
        assert d["position"] == "FLAT"
        assert "upcoming_events" not in d
        assert "in_blackout" not in d

    def test_with_position(self):
        pos = PositionState(
            side=Side.LONG,
            quantity=3,
            avg_entry=19850.0,
            unrealized_pnl=60.0,
            stop_price=19835.0,
        )
        state = MarketState(last_price=19860.0, position=pos)
        d = state.to_llm_dict()
        assert d["position"]["side"] == "long"
        assert d["position"]["qty"] == 3
        assert d["position"]["stop"] == 19835.0

    def test_strips_zero_levels(self):
        state = MarketState(
            levels=KeyLevels(vwap=19840.0, poc=19845.0)
        )
        d = state.to_llm_dict()
        # vwap and poc should be present, zeros should be stripped
        assert d["levels"]["vwap"] == 19840.0
        assert d["levels"]["poc"] == 19845.0
        assert "prior_day_high" not in d["levels"]

    def test_strips_default_flow(self):
        state = MarketState(
            flow=OrderFlowData(cumulative_delta=2840.0, delta_trend="positive")
        )
        d = state.to_llm_dict()
        assert d["flow"]["cumulative_delta"] == 2840.0
        assert d["flow"]["delta_trend"] == "positive"
        # rvol=1.0 (default) and volume_1min=0 should be stripped
        assert "rvol" not in d["flow"]
        assert "volume_1min" not in d["flow"]

    def test_includes_price_action_summary(self):
        state = MarketState(price_action_summary="MNQ rallied from 19825")
        d = state.to_llm_dict()
        assert d["price_action"] == "MNQ rallied from 19825"

    def test_excludes_empty_price_action(self):
        state = MarketState()
        d = state.to_llm_dict()
        assert "price_action" not in d

    def test_includes_blackout_when_true(self):
        state = MarketState(in_blackout=True)
        d = state.to_llm_dict()
        assert d["in_blackout"] is True

    def test_excludes_blackout_when_false(self):
        state = MarketState(in_blackout=False)
        d = state.to_llm_dict()
        assert "in_blackout" not in d

    def test_recent_trades_capped_at_five(self):
        trades = [
            TradeRecord(
                timestamp_entry=datetime.now(tz=UTC),
                side=Side.LONG,
                entry_quantity=1,
                entry_price=19850.0 + i,
                stop_price=19835.0,
                pnl=float(i * 10),
            )
            for i in range(8)
        ]
        state = MarketState(recent_trades=trades)
        d = state.to_llm_dict()
        assert len(d["recent_trades"]) == 5  # capped at 5

    def test_session_info(self):
        state = MarketState(
            session_phase=SessionPhase.MORNING,
            regime=Regime.TRENDING_UP,
            regime_confidence=0.85,
        )
        d = state.to_llm_dict()
        assert d["session"]["phase"] == "morning"
        assert d["session"]["regime"] == "trending_up"
        assert d["session"]["regime_confidence"] == 0.85
