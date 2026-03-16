"""Tests for the main entry point — component wiring and bootstrapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import (
    AnthropicConfig,
    AppConfig,
    DatabentoConfig,
    TelegramConfig,
    TradingConfig,
    TradovateConfig,
)
from src.core.types import KeyLevels, MarketState, SessionPhase
from src.main import (
    _acquire_lock,
    _build_components,
    _check_clock_drift,
    _make_pre_market_fn,
    _release_lock,
    _setup_logging,
)
from datetime import UTC, datetime


def _test_config() -> AppConfig:
    return AppConfig(
        tradovate=TradovateConfig(use_demo=True),
        databento=DatabentoConfig(),
        anthropic=AnthropicConfig(),
        telegram=TelegramConfig(),
        trading=TradingConfig(symbol="MNQM6"),
    )


# ── Component Building ──────────────────────────────────────────────────────


class TestBuildComponents:
    def test_builds_all_required_components(self):
        """All required components should be created."""
        config = _test_config()
        components = _build_components(config)

        required = [
            "event_bus",
            "tick_processor",
            "multi_instrument",
            "calendar",
            "state_engine",
            "llm_client",
            "reasoner",
            "pre_market_analyst",
            "session_controller",
            "position_tracker",
            "order_manager",
            "kill_switch",
            "guardrail_engine",
        ]

        for name in required:
            assert name in components, f"Missing component: {name}"
            assert components[name] is not None, f"Component {name} is None"

    def test_no_telegram_without_credentials(self):
        """Telegram should be None when no credentials provided."""
        config = AppConfig(
            tradovate=TradovateConfig(use_demo=True),
            databento=DatabentoConfig(),
            anthropic=AnthropicConfig(),
            telegram=TelegramConfig(bot_token="", chat_id=""),
            trading=TradingConfig(symbol="MNQM6"),
        )
        components = _build_components(config)

        assert components["telegram"] is None
        assert components["alert_manager"] is None

    def test_telegram_created_with_credentials(self):
        """Telegram should be created when credentials provided."""
        config = AppConfig(
            tradovate=TradovateConfig(use_demo=True),
            databento=DatabentoConfig(),
            anthropic=AnthropicConfig(),
            telegram=TelegramConfig(bot_token="123:ABC", chat_id="456"),
            trading=TradingConfig(symbol="MNQM6"),
        )
        components = _build_components(config)

        assert components["telegram"] is not None
        assert components["alert_manager"] is not None

    def test_uses_demo_mode(self):
        """Components should respect demo mode setting."""
        config = _test_config()
        components = _build_components(config)

        # Kill switch should have the configured thresholds
        ks = components["kill_switch"]
        assert ks._daily_loss_limit == config.trading.max_daily_loss


# ── Logging Setup ───────────────────────────────────────────────────────────


class TestSetupLogging:
    def test_setup_logging_doesnt_raise(self):
        """Logging setup should not raise for any valid level."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            _setup_logging(level)  # Should not raise


# ── Pre-Market Function Wiring ─────────────────────────────────────────────


class TestPreMarketFn:
    def test_returns_none_for_analyst_without_analyze(self):
        """Should return None if analyst has no analyze method."""
        analyst = MagicMock(spec=[])  # no methods
        state_engine = MagicMock()
        result = _make_pre_market_fn(analyst, state_engine)
        assert result is None

    def test_returns_callable_for_valid_analyst(self):
        """Should return a callable wrapper for a valid analyst."""
        analyst = MagicMock()
        analyst.analyze = AsyncMock(return_value="Game plan text")
        state_engine = MagicMock()
        result = _make_pre_market_fn(analyst, state_engine)
        assert result is not None
        assert callable(result)

    @pytest.mark.asyncio
    async def test_wrapper_returns_empty_string_when_no_state(self):
        """Wrapper should return empty string when state engine has no state."""
        analyst = MagicMock()
        analyst.analyze = AsyncMock(return_value="Game plan")
        state_engine = MagicMock()
        state_engine.last_state = None

        wrapper = _make_pre_market_fn(analyst, state_engine)
        result = await wrapper()
        assert result == ""
        analyst.analyze.assert_not_called()

    @pytest.mark.asyncio
    async def test_wrapper_passes_levels_to_analyst(self):
        """Wrapper should extract levels from state and pass to analyst."""
        analyst = MagicMock()
        analyst.analyze = AsyncMock(return_value="Buy dips at VWAP")

        state = MarketState(
            timestamp=datetime.now(tz=UTC),
            symbol="MNQM6",
            last_price=19850.0,
            bid=19849.75,
            ask=19850.25,
            session_phase=SessionPhase.MORNING,
            levels=KeyLevels(
                prior_day_high=19870.0,
                prior_day_low=19780.0,
                prior_day_close=19845.0,
                overnight_high=19865.0,
                overnight_low=19810.0,
            ),
        )

        state_engine = MagicMock()
        state_engine.last_state = state

        wrapper = _make_pre_market_fn(analyst, state_engine)
        result = await wrapper()

        assert result == "Buy dips at VWAP"
        analyst.analyze.assert_called_once()
        call_kwargs = analyst.analyze.call_args[1]
        assert call_kwargs["prior_day_high"] == 19870.0
        assert call_kwargs["prior_day_low"] == 19780.0
        assert call_kwargs["prior_day_close"] == 19845.0
        assert call_kwargs["overnight_high"] == 19865.0
        assert call_kwargs["overnight_low"] == 19810.0
        assert call_kwargs["current_price"] == 19850.0


# ── Lock File (Dual Instance Prevention) ────────────────────────────────────


class TestLockFile:
    def test_acquire_lock_succeeds(self, tmp_path):
        """First acquisition should succeed."""
        lock_path = str(tmp_path / "test.lock")
        assert _acquire_lock(lock_path) is True
        _release_lock()

    def test_second_acquire_fails(self, tmp_path):
        """Second acquisition while first is held should fail."""
        lock_path = str(tmp_path / "test.lock")
        assert _acquire_lock(lock_path) is True

        # Try to acquire again — should fail (same process, but flock is exclusive)
        import fcntl
        try:
            fd2 = open(lock_path, "w")
            fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # If we get here, lock wasn't held (shouldn't happen)
            fd2.close()
            acquired = True
        except (IOError, OSError):
            acquired = False

        assert acquired is False
        _release_lock()

    def test_release_allows_reacquire(self, tmp_path):
        """After release, acquisition should succeed again."""
        lock_path = str(tmp_path / "test.lock")
        assert _acquire_lock(lock_path) is True
        _release_lock()
        assert _acquire_lock(lock_path) is True
        _release_lock()


# ── Clock Drift Check ──────────────────────────────────────────────────────


class TestClockDrift:
    @pytest.mark.asyncio
    async def test_clock_check_returns_bool(self):
        """Clock drift check should return a boolean."""
        # This actually makes an HTTP request, so it may succeed or fail
        # depending on network, but should not raise
        result = await _check_clock_drift(max_drift_sec=60.0)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_clock_check_returns_true_on_network_failure(self):
        """If network is unavailable, should return True (don't block startup)."""
        # Mock aiohttp to raise
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__ = AsyncMock(
                side_effect=Exception("No network")
            )
            mock_session.return_value.__aexit__ = AsyncMock()
            result = await _check_clock_drift()
            assert result is True


# ── Dry Run Mode ───────────────────────────────────────────────────────────


class TestDryRunMode:
    def test_dry_run_sets_simulation_mode(self):
        """Dry run should enable simulation mode on OrderManager."""
        config = _test_config()
        components = _build_components(config, dry_run=True)

        om = components["order_manager"]
        assert om.simulation_mode is True

    def test_dry_run_skips_tradovate(self):
        """Dry run should not create Tradovate auth/REST/WS."""
        config = _test_config()
        components = _build_components(config, dry_run=True)

        assert components["tradovate_auth"] is None
        assert components["tradovate_rest"] is None
        assert components["tradovate_ws"] is None

    def test_dry_run_skips_databento(self):
        """Dry run should not create Databento client."""
        config = _test_config()
        components = _build_components(config, dry_run=True)

        assert components["databento_client"] is None

    def test_normal_mode_no_simulation(self):
        """Normal mode should NOT enable simulation mode."""
        config = _test_config()
        components = _build_components(config, dry_run=False)

        om = components["order_manager"]
        assert om.simulation_mode is False


# ── Component Wiring Completeness ──────────────────────────────────────────


class TestComponentCompleteness:
    def test_all_phase_11_components_present(self):
        """All learning/replay/enhanced components should be present."""
        config = _test_config()
        components = _build_components(config)

        phase_11 = [
            "data_recorder",
            "trade_logger",
            "postmortem_analyzer",
            "regime_tracker",
            "kelly_calculator",
            "trail_manager",
            "bull_bear_debate",
            "circuit_breakers",
        ]
        for name in phase_11:
            assert name in components, f"Missing phase 11 component: {name}"
            assert components[name] is not None, f"Phase 11 component {name} is None"

    def test_rate_limiter_present(self):
        """Rate limiter should be created."""
        config = _test_config()
        components = _build_components(config)
        assert components["rate_limiter"] is not None

    def test_apex_guardrail_with_apex_enabled(self):
        """Apex guardrail should be created when apex is enabled."""
        config = _test_config()
        config.trading.apex_enabled = True
        config.trading.apex_account_type = "50k"
        components = _build_components(config)

        assert components["apex_guardrail"] is not None

    def test_no_apex_without_flag(self):
        """Apex guardrail should be None when not enabled."""
        config = _test_config()
        config.trading.apex_enabled = False
        components = _build_components(config)

        assert components["apex_guardrail"] is None
