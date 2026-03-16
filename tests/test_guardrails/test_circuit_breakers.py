"""Tests for circuit breaker multi-day/week/month loss protections."""

import pytest

from src.guardrails.circuit_breakers import CircuitBreakers, CircuitBreakerState


class TestCircuitBreakerEvaluation:
    """Test the evaluate() method under various conditions."""

    def test_no_history_returns_clean_state(self):
        cb = CircuitBreakers()
        state = cb.evaluate()
        assert not state.is_shutdown
        assert not state.sim_only
        assert state.max_contracts_override is None

    def test_all_green_days_no_restrictions(self):
        cb = CircuitBreakers()
        cb.load_history([
            ("2025-01-06", 150.0),
            ("2025-01-07", 200.0),
            ("2025-01-08", 100.0),
        ])
        state = cb.evaluate()
        assert not state.is_shutdown
        assert not state.sim_only
        assert state.max_contracts_override is None

    def test_one_red_day_no_restriction(self):
        cb = CircuitBreakers()
        cb.load_history([
            ("2025-01-06", 150.0),
            ("2025-01-07", -50.0),
        ])
        state = cb.evaluate()
        assert not state.is_shutdown
        assert not state.sim_only
        assert state.max_contracts_override is None

    def test_two_consecutive_red_days_reduces_size(self):
        cb = CircuitBreakers(base_max_contracts=6)
        cb.load_history([
            ("2025-01-06", 150.0),
            ("2025-01-07", -50.0),
            ("2025-01-08", -75.0),
        ])
        state = cb.evaluate()
        assert not state.is_shutdown
        assert not state.sim_only
        assert state.max_contracts_override == 3  # 50% of 6

    def test_three_consecutive_red_days_sim_only(self):
        cb = CircuitBreakers(base_max_contracts=6)
        cb.load_history([
            ("2025-01-06", -50.0),
            ("2025-01-07", -75.0),
            ("2025-01-08", -100.0),
        ])
        state = cb.evaluate()
        assert not state.is_shutdown
        assert state.sim_only

    def test_green_day_breaks_red_streak(self):
        cb = CircuitBreakers()
        cb.load_history([
            ("2025-01-06", -50.0),
            ("2025-01-07", -75.0),
            ("2025-01-08", 100.0),  # breaks streak
            ("2025-01-09", -30.0),  # only 1 red day
        ])
        state = cb.evaluate()
        assert not state.is_shutdown
        assert not state.sim_only
        assert state.max_contracts_override is None

    def test_weekly_loss_limit_shutdown(self):
        cb = CircuitBreakers(weekly_loss_limit=800.0)
        # All dates in the same week (need current week dates)
        from datetime import date, timedelta
        today = date.today()
        monday = today - timedelta(days=today.weekday())

        cb.load_history([
            (monday.isoformat(), -300.0),
            ((monday + timedelta(days=1)).isoformat(), -300.0),
            ((monday + timedelta(days=2)).isoformat(), -250.0),
        ])
        state = cb.evaluate()
        assert state.is_shutdown
        assert "Weekly" in state.shutdown_reason

    def test_monthly_loss_limit_shutdown(self):
        cb = CircuitBreakers(monthly_loss_limit=2000.0)
        from datetime import date
        today = date.today()
        month_start = today.replace(day=1)

        cb.load_history([
            (month_start.isoformat(), -500.0),
            ((month_start.replace(day=2)).isoformat(), -500.0),
            ((month_start.replace(day=3)).isoformat(), -500.0),
            ((month_start.replace(day=4)).isoformat(), -600.0),
        ])
        state = cb.evaluate()
        assert state.is_shutdown
        assert "Monthly" in state.shutdown_reason


class TestCircuitBreakerRecordDay:
    """Test incremental day recording."""

    def test_record_day_updates_state(self):
        cb = CircuitBreakers()
        cb.record_day("2025-01-06", -50.0)
        assert cb.consecutive_red_days == 1

        cb.record_day("2025-01-07", -75.0)
        assert cb.consecutive_red_days == 2

        cb.record_day("2025-01-08", 100.0)
        assert cb.consecutive_red_days == 0


class TestCircuitBreakerStats:
    """Test stats property."""

    def test_stats_returns_expected_keys(self):
        cb = CircuitBreakers()
        stats = cb.stats
        assert "consecutive_red_days" in stats
        assert "weekly_pnl" in stats
        assert "monthly_pnl" in stats
        assert "is_shutdown" in stats
        assert "sim_only" in stats
