"""Tests for the exception hierarchy."""

import pytest

from src.core.exceptions import (
    BlackoutPeriodViolation,
    BracketLegRejectedError,
    ConnectionError,
    ConnectionTimeoutError,
    DailyLossLimitHit,
    DatabentoConnectionError,
    FlashCrashDetected,
    GuardrailViolation,
    InsufficientMarginError,
    KillSwitchTriggered,
    LLMConnectionError,
    LLMFailureThreshold,
    MaxPositionExceeded,
    OrderError,
    OrderModifyFailedError,
    OrderRejectedError,
    RateLimitExceeded,
    TradingSystemError,
    TradovateConnectionError,
)


class TestExceptionHierarchy:
    """Verify the inheritance tree is correct — matters for except clauses."""

    def test_base_exception(self):
        err = TradingSystemError("base error")
        assert isinstance(err, Exception)
        assert str(err) == "base error"

    # ── Connection errors ──

    def test_connection_error_is_trading_error(self):
        assert issubclass(ConnectionError, TradingSystemError)

    def test_tradovate_connection_is_connection(self):
        err = TradovateConnectionError("ws dropped")
        assert isinstance(err, ConnectionError)
        assert isinstance(err, TradingSystemError)

    def test_databento_connection_is_connection(self):
        err = DatabentoConnectionError("feed lost")
        assert isinstance(err, ConnectionError)

    def test_llm_connection_is_connection(self):
        err = LLMConnectionError("API timeout")
        assert isinstance(err, ConnectionError)

    # ── Order errors ──

    def test_order_error_is_trading_error(self):
        assert issubclass(OrderError, TradingSystemError)

    def test_order_rejected_is_order_error(self):
        err = OrderRejectedError("insufficient margin")
        assert isinstance(err, OrderError)

    def test_order_modify_failed_is_order_error(self):
        err = OrderModifyFailedError("silent failure")
        assert isinstance(err, OrderError)

    def test_insufficient_margin_is_order_error(self):
        assert issubclass(InsufficientMarginError, OrderError)

    def test_bracket_leg_rejected_is_order_error(self):
        err = BracketLegRejectedError("stop leg rejected")
        assert isinstance(err, OrderError)

    # ── Guardrail violations ──

    def test_guardrail_violation_is_trading_error(self):
        assert issubclass(GuardrailViolation, TradingSystemError)

    def test_max_position_exceeded(self):
        err = MaxPositionExceeded("6 contracts max")
        assert isinstance(err, GuardrailViolation)

    def test_daily_loss_limit_hit(self):
        err = DailyLossLimitHit("-$400 reached")
        assert isinstance(err, GuardrailViolation)

    def test_blackout_period_violation(self):
        err = BlackoutPeriodViolation("FOMC in 3 min")
        assert isinstance(err, GuardrailViolation)

    # ── Kill switch triggers ──

    def test_kill_switch_is_trading_error(self):
        assert issubclass(KillSwitchTriggered, TradingSystemError)

    def test_flash_crash_is_kill_switch(self):
        err = FlashCrashDetected("50pts in 45s")
        assert isinstance(err, KillSwitchTriggered)

    def test_connection_timeout_is_kill_switch(self):
        err = ConnectionTimeoutError("30s elapsed")
        assert isinstance(err, KillSwitchTriggered)

    def test_llm_failure_threshold_is_kill_switch(self):
        err = LLMFailureThreshold("3 consecutive failures")
        assert isinstance(err, KillSwitchTriggered)

    # ── Rate limiting ──

    def test_rate_limit_is_trading_error(self):
        err = RateLimitExceeded("4500/hr budget exhausted")
        assert isinstance(err, TradingSystemError)


class TestExceptionCatching:
    """Verify that broader except clauses catch specific exceptions."""

    def test_catch_all_connection_errors(self):
        """A single `except ConnectionError` should catch all three."""
        errors = [
            TradovateConnectionError("a"),
            DatabentoConnectionError("b"),
            LLMConnectionError("c"),
        ]
        for err in errors:
            with pytest.raises(ConnectionError):
                raise err

    def test_catch_all_order_errors(self):
        errors = [
            OrderRejectedError("a"),
            OrderModifyFailedError("b"),
            InsufficientMarginError("c"),
            BracketLegRejectedError("d"),
        ]
        for err in errors:
            with pytest.raises(OrderError):
                raise err

    def test_catch_all_guardrail_violations(self):
        errors = [
            MaxPositionExceeded("a"),
            DailyLossLimitHit("b"),
            BlackoutPeriodViolation("c"),
        ]
        for err in errors:
            with pytest.raises(GuardrailViolation):
                raise err

    def test_catch_all_kill_switches(self):
        errors = [
            FlashCrashDetected("a"),
            ConnectionTimeoutError("b"),
            LLMFailureThreshold("c"),
        ]
        for err in errors:
            with pytest.raises(KillSwitchTriggered):
                raise err

    def test_catch_everything_with_base(self):
        """TradingSystemError catches everything in the hierarchy."""
        all_errors = [
            TradovateConnectionError("a"),
            OrderRejectedError("b"),
            MaxPositionExceeded("c"),
            FlashCrashDetected("d"),
            RateLimitExceeded("e"),
        ]
        for err in all_errors:
            with pytest.raises(TradingSystemError):
                raise err
