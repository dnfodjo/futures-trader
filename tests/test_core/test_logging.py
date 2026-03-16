"""Tests for structured logging configuration."""

import structlog

from src.core.logging import (
    bind_trading_context,
    clear_trading_context,
    setup_logging,
)


class TestSetupLogging:
    def test_setup_logging_returns_none(self):
        """setup_logging should configure structlog without error."""
        result = setup_logging("INFO")
        assert result is None

    def test_setup_logging_debug_level(self):
        """Should accept DEBUG level without error."""
        setup_logging("DEBUG")

    def test_setup_logging_warning_level(self):
        """Should accept WARNING level without error."""
        setup_logging("WARNING")


class TestBindTradingContext:
    def setup_method(self):
        """Clear context before each test."""
        structlog.contextvars.clear_contextvars()

    def teardown_method(self):
        """Clear context after each test."""
        structlog.contextvars.clear_contextvars()

    def test_bind_session_date(self):
        bind_trading_context(session_date="2026-03-14")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["session_date"] == "2026-03-14"

    def test_bind_daily_pnl(self):
        bind_trading_context(daily_pnl=245.50)
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["daily_pnl"] == 245.50

    def test_bind_position_long(self):
        bind_trading_context(position_side="long", position_qty=3)
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["position"] == "long 3"

    def test_bind_position_flat(self):
        bind_trading_context(position_qty=0)
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["position"] == "flat"

    def test_bind_system_state(self):
        bind_trading_context(system_state="trading")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["system_state"] == "trading"

    def test_bind_multiple_values(self):
        bind_trading_context(
            session_date="2026-03-14",
            daily_pnl=100.0,
            position_side="short",
            position_qty=2,
            system_state="trading",
        )
        ctx = structlog.contextvars.get_contextvars()
        assert ctx["session_date"] == "2026-03-14"
        assert ctx["daily_pnl"] == 100.0
        assert ctx["position"] == "short 2"
        assert ctx["system_state"] == "trading"

    def test_bind_none_values_not_added(self):
        """None values should not be bound to context."""
        bind_trading_context()
        ctx = structlog.contextvars.get_contextvars()
        assert len(ctx) == 0


class TestClearTradingContext:
    def test_clear_removes_all_context(self):
        bind_trading_context(session_date="2026-03-14", daily_pnl=100.0)
        ctx = structlog.contextvars.get_contextvars()
        assert len(ctx) > 0

        clear_trading_context()
        ctx = structlog.contextvars.get_contextvars()
        assert len(ctx) == 0

    def test_clear_when_empty_is_safe(self):
        """Clearing when nothing is bound should not raise."""
        clear_trading_context()
        ctx = structlog.contextvars.get_contextvars()
        assert len(ctx) == 0
