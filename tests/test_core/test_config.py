"""Tests for configuration loading."""

import os
import warnings
from unittest.mock import patch

import pytest

from src.core.config import (
    AppConfig,
    TradingConfig,
    TradovateConfig,
    _get_next_contract_symbol,
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


class TestContractRollover:
    def test_expired_contract_warns(self):
        """Contract from the past should trigger a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TradingConfig(symbol="MNQH5")  # March 2025 — expired
            assert len(w) == 1
            assert "expired" in str(w[0].message).lower()

    def test_current_contract_no_warning(self):
        """A far-future contract shouldn't warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TradingConfig(symbol="MNQZ9")  # Dec 2029 — far future
            # Filter to only our warnings (not pydantic deprecation etc)
            rollover_warnings = [x for x in w if "MNQ" in str(x.message)]
            assert len(rollover_warnings) == 0

    def test_get_next_contract_symbol(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        ET = ZoneInfo("US/Eastern")

        # In January -> should get March (H)
        jan = datetime(2026, 1, 15, tzinfo=ET)
        assert _get_next_contract_symbol(jan) == "MNQH6"

        # In April -> should get June (M)
        apr = datetime(2026, 4, 15, tzinfo=ET)
        assert _get_next_contract_symbol(apr) == "MNQM6"

        # In December -> should get December (Z) same year
        dec = datetime(2026, 12, 1, tzinfo=ET)
        assert _get_next_contract_symbol(dec) == "MNQZ6"


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
