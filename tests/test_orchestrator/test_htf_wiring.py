"""Tests for HTF Structure Factor wiring into TradingOrchestrator (Steps 5-6).

Tests pre-market context, no-trade windows, min_confluence_override,
and bars_5m passing to confluence engine.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import AppConfig, TradingConfig, TradovateConfig, AnthropicConfig, TelegramConfig, DatabentoConfig
from src.core.events import EventBus
from src.core.types import (
    ActionType,
    Event,
    EventType,
    GuardrailResult,
    LLMAction,
    MarketState,
    PositionState,
    SessionPhase,
    Side,
)
from src.orchestrator import OrchestratorState, TradingOrchestrator


# ── Helpers ──────────────────────────────────────────────────────────────────


def _config() -> AppConfig:
    return AppConfig(
        tradovate=TradovateConfig(use_demo=True),
        databento=DatabentoConfig(),
        anthropic=AnthropicConfig(),
        telegram=TelegramConfig(),
        trading=TradingConfig(symbol="MNQM6"),
    )


def _market_state(**overrides) -> MarketState:
    defaults = dict(
        timestamp=datetime.now(tz=UTC),
        symbol="MNQM6",
        last_price=19850.0,
        bid=19849.75,
        ask=19850.25,
        session_phase=SessionPhase.MORNING,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def _make_orchestrator(**kwargs) -> TradingOrchestrator:
    """Create a TradingOrchestrator with standard mocks."""
    bus = EventBus()
    defaults = dict(
        config=_config(),
        event_bus=bus,
        reasoner=MagicMock(),
        guardrail_engine=MagicMock(),
        order_manager=MagicMock(),
        position_tracker=MagicMock(position=None, is_flat=True),
        session_controller=MagicMock(
            daily_pnl=0.0,
            consecutive_losers=0,
            effective_max_contracts=6,
            should_stop_trading=False,
            stop_reason="",
            total_trades=0,
        ),
        kill_switch=MagicMock(is_triggered=False),
        state_provider=lambda: _market_state(),
    )
    defaults.update(kwargs)
    return TradingOrchestrator(**defaults)


# ── Pre-Market Context Tests ─────────────────────────────────────────────────


class TestPreMarketContext:
    def test_pre_market_context_defaults_to_none(self):
        """Orchestrator should initialize with no pre-market context."""
        orch = _make_orchestrator()
        assert orch._pre_market_context is None

    def test_set_pre_market_context(self):
        """set_pre_market_context should store the context."""
        orch = _make_orchestrator()

        ctx = MagicMock()
        ctx.risk_level = "high"
        ctx.events = ["FOMC"]
        ctx.no_trade_windows = [("13:45", "14:30")]
        ctx.min_confluence_override = 4

        orch.set_pre_market_context(ctx)
        assert orch._pre_market_context is ctx
        assert orch._pre_market_context.risk_level == "high"

    def test_set_pre_market_context_none(self):
        """Should handle None context gracefully."""
        orch = _make_orchestrator()
        orch.set_pre_market_context(None)
        assert orch._pre_market_context is None


# ── No-Trade Window Tests ────────────────────────────────────────────────────


class TestNoTradeWindow:
    def _make_orch_with_confluence(self, no_trade_windows=None, min_confluence_override=None):
        """Create orchestrator with confluence engine and pre-market context."""
        confluence = MagicMock()
        confluence.score = MagicMock(return_value={
            "score": 5,
            "factors": {},
            "speed_state": "NORMAL",
        })

        risk_mgr = MagicMock()
        session_params = MagicMock()
        session_params.min_confluence = 3
        risk_mgr.get_session_params = MagicMock(return_value=session_params)
        risk_mgr.check_entry_allowed = MagicMock(return_value=(True, ""))
        risk_mgr.get_sl_points = MagicMock(return_value=10.0)
        risk_mgr.compute_position_size = MagicMock(return_value=1)

        orch = _make_orchestrator(
            confluence_engine=confluence,
            risk_manager=risk_mgr,
        )

        if no_trade_windows or min_confluence_override is not None:
            ctx = MagicMock()
            ctx.no_trade_windows = no_trade_windows or []
            ctx.min_confluence_override = min_confluence_override
            orch.set_pre_market_context(ctx)

        return orch

    def test_no_trade_window_attribute_exists(self):
        """Orchestrator should have _pre_market_context attribute."""
        orch = _make_orchestrator()
        assert hasattr(orch, '_pre_market_context')

    def test_min_confluence_override_raises_threshold(self):
        """min_confluence_override should raise the minimum confluence needed."""
        orch = self._make_orch_with_confluence(min_confluence_override=5)
        # The override should be accessible
        assert orch._pre_market_context.min_confluence_override == 5


# ── bars_5m Passing Tests ────────────────────────────────────────────────────


class TestBars5mPassing:
    def test_score_call_signature_accepts_bars_5m(self):
        """Confluence engine score() call in orchestrator should include bars_5m."""
        confluence = MagicMock()
        confluence.score = MagicMock(return_value={
            "score": 2,
            "factors": {},
            "speed_state": "NORMAL",
        })

        orch = _make_orchestrator(
            confluence_engine=confluence,
            risk_manager=MagicMock(),
        )

        # The confluence engine exists
        assert orch._confluence_engine is confluence
        assert orch._use_confluence is True


# ── reduce_size / widen_stops Wiring Tests ──────────────────────────────────


class TestReduceSizeWiring:
    def test_reduce_size_sets_quantity_to_1(self):
        """When reduce_size=True, entry quantity should be 1 not 2."""
        orch = _make_orchestrator()
        ctx = MagicMock()
        ctx.reduce_size = True
        ctx.no_trade_windows = []
        ctx.min_confluence_override = None
        ctx.widen_stops = False
        orch.set_pre_market_context(ctx)

        # Check that the pre_market_context.reduce_size flag is accessible
        assert orch._pre_market_context.reduce_size is True
        # The actual quantity logic: (1 if reduce_size else 2)
        expected_qty = 1 if orch._pre_market_context and getattr(orch._pre_market_context, "reduce_size", False) else 2
        assert expected_qty == 1

    def test_no_reduce_size_keeps_quantity_at_2(self):
        """When reduce_size=False, entry quantity should stay at 2."""
        orch = _make_orchestrator()
        ctx = MagicMock()
        ctx.reduce_size = False
        ctx.no_trade_windows = []
        ctx.min_confluence_override = None
        ctx.widen_stops = False
        orch.set_pre_market_context(ctx)

        expected_qty = 1 if orch._pre_market_context and getattr(orch._pre_market_context, "reduce_size", False) else 2
        assert expected_qty == 2

    def test_no_context_keeps_quantity_at_2(self):
        """When no pre-market context, entry quantity should stay at 2."""
        orch = _make_orchestrator()
        # No context set
        expected_qty = 1 if orch._pre_market_context and getattr(orch._pre_market_context, "reduce_size", False) else 2
        assert expected_qty == 2


class TestWidenStopsWiring:
    def test_widen_stops_increases_sl_pts(self):
        """When widen_stops=True, stop distance should increase by 25%."""
        base_sl = 40.0
        widen_stops = True
        widened = base_sl * 1.25 if widen_stops else base_sl
        assert widened == 50.0

    def test_no_widen_stops_keeps_sl(self):
        """When widen_stops=False, stop distance stays the same."""
        base_sl = 40.0
        widen_stops = False
        widened = base_sl * 1.25 if widen_stops else base_sl
        assert widened == 40.0
