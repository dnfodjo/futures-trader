"""Shared test fixtures for the futures-trader test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from src.core.config import (
    AnthropicConfig,
    AppConfig,
    DatabentoConfig,
    TelegramConfig,
    TradingConfig,
    TradovateConfig,
)
from src.core.events import EventBus
from src.core.types import (
    CrossMarketContext,
    EconomicEvent,
    KeyLevels,
    LLMAction,
    ActionType,
    MarketState,
    OrderFlowData,
    PositionState,
    Regime,
    SessionPhase,
    Side,
    TradeRecord,
)

ET = ZoneInfo("US/Eastern")


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def sample_config() -> AppConfig:
    """Config with test defaults — no real credentials."""
    return AppConfig(
        tradovate=TradovateConfig(
            username="test_user",
            password="test_pass",
            cid=8,
            sec="test_secret",
            use_demo=True,
        ),
        databento=DatabentoConfig(api_key="test_db_key"),
        anthropic=AnthropicConfig(api_key="test_anthropic_key"),
        telegram=TelegramConfig(bot_token="123:ABC", chat_id="456"),
        trading=TradingConfig(symbol="MNQM6"),
    )


@pytest.fixture
def sample_position_long() -> PositionState:
    return PositionState(
        side=Side.LONG,
        quantity=3,
        avg_entry=19850.0,
        unrealized_pnl=60.0,
        stop_price=19835.0,
        max_favorable=90.0,
        max_adverse=-20.0,
        time_in_trade_sec=240,
        adds_count=1,
    )


@pytest.fixture
def sample_position_short() -> PositionState:
    return PositionState(
        side=Side.SHORT,
        quantity=2,
        avg_entry=19900.0,
        unrealized_pnl=40.0,
        stop_price=19920.0,
        max_favorable=80.0,
        max_adverse=-10.0,
        time_in_trade_sec=180,
    )


@pytest.fixture
def sample_market_state(sample_position_long: PositionState) -> MarketState:
    """A realistic market state with a long position."""
    return MarketState(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        last_price=19860.0,
        bid=19859.75,
        ask=19860.25,
        spread=0.5,
        session_phase=SessionPhase.MORNING,
        regime=Regime.TRENDING_UP,
        regime_confidence=0.75,
        levels=KeyLevels(
            prior_day_high=19870.0,
            prior_day_low=19780.0,
            prior_day_close=19845.0,
            overnight_high=19865.0,
            overnight_low=19810.0,
            session_high=19868.0,
            session_low=19825.0,
            session_open=19840.0,
            vwap=19842.0,
            poc=19845.0,
            value_area_high=19860.0,
            value_area_low=19830.0,
        ),
        flow=OrderFlowData(
            cumulative_delta=2840.0,
            delta_1min=120.0,
            delta_5min=-320.0,
            delta_trend="weakening",
            rvol=1.3,
            volume_1min=450,
            large_lot_count_5min=4,
            tape_speed=12.5,
        ),
        cross_market=CrossMarketContext(
            es_price=5420.0,
            es_change_pct=0.15,
            tick_index=480,
            vix=18.2,
            vix_change_pct=-0.3,
            ten_year_yield=4.25,
            dxy=104.5,
        ),
        position=sample_position_long,
        daily_pnl=145.0,
        daily_trades=2,
        daily_winners=2,
        daily_losers=0,
        price_action_summary=(
            "MNQ rallied from 19825 to 19868 in the first hour. "
            "Currently pulling back from session high. Delta weakening."
        ),
        game_plan=(
            "Gap up 0.2% above prior close. Trend day setup if we hold VWAP. "
            "Watch for mean reversion if morning drive fails."
        ),
    )


@pytest.fixture
def sample_flat_market_state() -> MarketState:
    """Market state with no position."""
    return MarketState(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        last_price=19842.0,
        bid=19841.75,
        ask=19842.25,
        spread=0.5,
        session_phase=SessionPhase.MORNING,
        regime=Regime.CHOPPY,
        levels=KeyLevels(vwap=19840.0, session_high=19855.0, session_low=19825.0),
        flow=OrderFlowData(cumulative_delta=200.0, delta_trend="neutral", rvol=0.8),
    )


@pytest.fixture
def sample_enter_action() -> LLMAction:
    return LLMAction(
        action=ActionType.ENTER,
        side=Side.LONG,
        quantity=3,
        stop_distance=15.0,
        reasoning="VWAP holding, delta positive, ES confirming. A-quality setup.",
        confidence=0.8,
        model_used="sonnet",
    )


@pytest.fixture
def sample_do_nothing_action() -> LLMAction:
    return LLMAction(
        action=ActionType.DO_NOTHING,
        reasoning="Choppy action, no clear setup. Sitting out.",
        confidence=0.3,
        model_used="haiku",
    )
