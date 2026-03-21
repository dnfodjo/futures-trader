"""Tests for configuration loading."""

import os
from unittest.mock import patch

import pytest

from src.core.config import (
    AppConfig,
    TradingConfig,
    TradovateConfig,
    load_config,
)


class TestTradovateConfig:
    def test_demo_urls(self):
        cfg = TradovateConfig(use_demo=True)
        assert "demo" in cfg.base_url
        assert "demo" in cfg.ws_url

    def test_live_urls(self):
        cfg = TradovateConfig(use_demo=False)
        assert "live" in cfg.base_url
        assert "live" in cfg.ws_url
        assert "demo" not in cfg.base_url

    def test_defaults(self):
        cfg = TradovateConfig()
        assert cfg.heartbeat_interval_sec == 2.5
        assert cfg.token_lifetime_min == 90
        assert cfg.use_demo is True


class TestTradingConfig:
    def test_defaults(self):
        cfg = TradingConfig()
        assert cfg.max_contracts in (6, 10)  # 6 in code, 10 if .env override
        assert cfg.max_daily_loss == 400.0
        assert cfg.max_stop_points == 25.0
        assert cfg.point_value == 2.0

    def test_profit_preservation_tiers(self):
        cfg = TradingConfig()
        # Defaults may be overridden by .env; test that tiers are properly ordered
        assert cfg.profit_preservation_tier2_pnl > cfg.profit_preservation_tier1_pnl
        assert cfg.profit_preservation_tier2_max_size < cfg.profit_preservation_tier1_max_size

    def test_circuit_breakers(self):
        cfg = TradingConfig()
        assert cfg.consecutive_red_days_half_size == 2
        assert cfg.consecutive_red_days_sim_only == 3
        assert cfg.max_weekly_loss == 800.0
        assert cfg.max_monthly_loss == 2000.0

    def test_partial_profit_defaults(self):
        cfg = TradingConfig()
        assert cfg.partial_profit_points == 15.0
        assert cfg.partial_quantity == 1
        assert cfg.partial_breakeven_offset == 1.0
        assert cfg.eth_partial_profit_points == 10.0


class TestContractRollover:
    def test_symbol_accepts_any_value(self):
        """TradingConfig no longer validates contract rollover — accepts any symbol."""
        # Old contracts, current contracts, far-future — all accepted without warning
        for sym in ("MNQH5", "MNQM6", "MNQZ9", "ES"):
            tc = TradingConfig(symbol=sym)
            assert tc.symbol == sym


class TestLoadConfig:
    def test_load_config_with_env_vars(self):
        """Verify sub-configs load from prefixed env vars."""
        env = {
            "TV_USERNAME": "env_user",
            "TV_USE_DEMO": "true",
            "DB_API_KEY": "env_db_key",
            "ANTHROPIC_API_KEY": "env_ant_key",
            "TG_BOT_TOKEN": "env_bot",
            "TG_CHAT_ID": "env_chat",
            "TRADE_MAX_CONTRACTS": "4",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config()
            assert cfg.tradovate.username == "env_user"
            assert cfg.databento.api_key == "env_db_key"
            assert cfg.anthropic.api_key == "env_ant_key"
            assert cfg.telegram.bot_token == "env_bot"
            assert cfg.trading.max_contracts == 4

    def test_load_config_defaults(self):
        """Config should load with all defaults when no env vars set."""
        cfg = load_config()
        assert cfg.tradovate.use_demo is True
        assert cfg.trading.max_daily_loss == 400.0
        assert cfg.log_level == "INFO"


class TestAppConfig:
    def test_all_sub_configs_exist(self):
        cfg = AppConfig()
        assert cfg.tradovate is not None
        assert cfg.databento is not None
        assert cfg.anthropic is not None
        assert cfg.telegram is not None
        assert cfg.trading is not None

    def test_data_dir_default(self):
        cfg = AppConfig()
        assert cfg.data_dir == "./data"
        assert cfg.journal_path == "./data/journal.db"
