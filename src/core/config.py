"""Configuration loaded from environment variables via pydantic-settings.

Single source of truth for every tunable parameter. All values can be
overridden via .env file or environment variables.

Each sub-config reads its own prefixed env vars (TV_, DB_, ANTHROPIC_, etc.)
and is constructed independently so a flat .env file works correctly.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Month codes for MNQ quarterly contracts
_MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


class TradovateConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TV_", env_file=".env", extra="ignore")

    username: str = ""
    password: str = ""
    app_id: str = "FuturesTrader"
    app_version: str = "1.0.0"
    cid: int = 0  # client ID from API key setup
    sec: str = ""  # API secret
    device_id: str = "futures-trader-001"
    use_demo: bool = True  # Start with demo, flip for live
    account_name: str = ""  # specific account name (e.g., "APEX-12345"). Empty = use first account.

    heartbeat_interval_sec: float = 2.5
    token_refresh_buffer_min: int = 5
    token_lifetime_min: int = 90

    @property
    def base_url(self) -> str:
        if self.use_demo:
            return "https://demo.tradovateapi.com/v1"
        return "https://live.tradovateapi.com/v1"

    @property
    def ws_url(self) -> str:
        if self.use_demo:
            return "wss://demo.tradovateapi.com/v1/websocket"
        return "wss://live.tradovateapi.com/v1/websocket"


class DatabentoConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", env_file=".env", extra="ignore")

    api_key: str = ""
    dataset: str = "GLBX.MDP3"


class AnthropicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_", env_file=".env", extra="ignore")

    api_key: str = ""
    haiku_model: str = "claude-haiku-4-5-20251001"
    sonnet_model: str = "claude-sonnet-4-6-20260314"
    max_retries: int = 3
    timeout_sec: int = 30
    daily_cost_cap: float = 10.0  # hard cap on daily LLM spend
    daily_cost_alert: float = 5.0  # alert threshold


class TelegramConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TG_", env_file=".env", extra="ignore")

    bot_token: str = ""
    chat_id: str = ""
    throttle_sec: float = 10.0  # min seconds between non-critical messages
    heartbeat_interval_min: int = 30


class TradingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADE_", env_file=".env", extra="ignore")

    # Contract
    symbol: str = "MNQM6"  # update quarterly: H=Mar, M=Jun, U=Sep, Z=Dec
    point_value: float = 2.0  # $2 per point for MNQ
    tick_size: float = 0.25
    tick_value: float = 0.50
    commission_per_rt: float = 0.86  # Tradovate active plan round trip

    # Hard limits (guardrails — LLM CANNOT override)
    max_contracts: int = 6
    max_daily_loss: float = 400.0
    max_weekly_loss: float = 800.0
    max_monthly_loss: float = 2000.0
    max_stop_points: float = 25.0

    # Trading hours (all ET) — full Globex session
    # 18:05 start = 5 min after Asian session opens (skip thin open)
    # 16:50 end = 10 min before daily halt (flatten buffer)
    # Daily halt 17:00-18:00 is always blocked by clock.is_trading_hours()
    trading_start: str = "18:05"
    trading_end: str = "16:50"
    hard_flatten_time: str = "16:55"
    pre_market_analysis_time: str = "17:55"  # 5 min before Globex opens

    # Max contracts during ETH (overnight) — reduced for thin liquidity
    max_contracts_eth: int = 2

    # Max trades per 23h session (18:05→16:55)
    # With all sessions (Asian, London, Pre-RTH, RTH, Post-RTH),
    # the system can easily take 20+ trades across 23 hours.
    max_daily_trades: int = 24

    # News blackout
    news_blackout_before_min: int = 5
    news_blackout_after_min: int = 10

    # Flash crash detection
    flash_crash_threshold_points: float = 50.0
    flash_crash_window_sec: int = 60

    # Connection safety
    connection_loss_max_sec: int = 30
    llm_failure_max_consecutive: int = 3

    # Profit preservation
    profit_preservation_tier1_pnl: float = 200.0
    profit_preservation_tier1_max_size: int = 3
    profit_preservation_tier2_pnl: float = 400.0
    profit_preservation_tier2_max_size: int = 2

    # Circuit breakers
    consecutive_red_days_half_size: int = 2
    consecutive_red_days_sim_only: int = 3

    # State engine timing
    state_update_interval_no_position_sec: float = 30.0
    state_update_interval_in_position_sec: float = 10.0
    state_update_interval_critical_sec: float = 5.0
    state_update_interval_eth_no_position_sec: float = 45.0  # slower scan during ETH

    # Trail manager
    trail_min_move_points: float = 3.0  # only modify stop after 3pt move
    trail_min_interval_sec: float = 10.0

    # ETH (Extended Trading Hours) trail params — tighter for thin liquidity
    # Asian/London sessions have smaller ranges (5-15pts vs 20-40pts RTH)
    # so trails must be tighter to avoid giving back all profits
    eth_trail_distance: float = 5.0  # 5pts vs 8pts RTH
    eth_trail_activation_points: float = 2.0  # activate trail at 2pts vs 4pts RTH
    eth_mid_tighten_at_profit: float = 4.0  # mid-tier at 4pts vs 6pts
    eth_mid_tightened_distance: float = 4.0  # mid distance 4pts vs 6pts
    eth_tighten_at_profit: float = 8.0  # tight-tier at 8pts vs 12pts
    eth_tightened_distance: float = 3.0  # tight distance 3pts vs 5pts
    eth_max_stop_points: float = 12.0  # max 12pt stop during ETH (vs 25pt RTH)

    # Stop hunt avoidance
    stop_offset_from_obvious_levels: float = 2.5  # offset stops 2.5pts from round numbers

    # Time-based exit
    time_exit_reassess_min: int = 15  # reassess if trade stale after 15 min
    time_exit_force_min: int = 25  # suggest flatten after 25 min of no progress

    # Apex Trader Funding compliance
    apex_enabled: bool = False  # enable Apex rule enforcement
    apex_account_type: str = "50k"  # 25k, 50k, 100k, 150k
    apex_flatten_deadline: str = "16:54"  # ET — 5min buffer before 4:59 PM
    apex_drawdown_lockout_pct: float = 0.75  # block entries at 75% drawdown used

    @model_validator(mode="after")
    def check_contract_rollover(self) -> TradingConfig:
        """Warn if the configured contract symbol may be near expiration."""
        symbol = self.symbol.upper()
        if not symbol.startswith("MNQ") or len(symbol) < 5:
            return self

        month_code = symbol[3]
        year_digit = symbol[4]

        code_to_month = {"H": 3, "M": 6, "U": 9, "Z": 12}
        if month_code not in code_to_month:
            return self

        contract_month = code_to_month[month_code]
        contract_year = 2020 + int(year_digit)  # works through 2029

        now = datetime.now(ZoneInfo("US/Eastern"))
        # Contracts expire 3rd Friday of the expiration month.
        # Warn if within 14 days of the end of the contract month.
        expiry_approx = datetime(
            contract_year, contract_month, 20, tzinfo=ZoneInfo("US/Eastern")
        )
        days_until = (expiry_approx - now).days

        if days_until < 0:
            warnings.warn(
                f"Contract {symbol} appears to have expired. "
                f"Update TRADE_SYMBOL to the next front-month contract. "
                f"Current front months: {_get_next_contract_symbol(now)}",
                stacklevel=1,
            )
        elif days_until < 14:
            warnings.warn(
                f"Contract {symbol} expires in ~{days_until} days. "
                f"Consider rolling to {_get_next_contract_symbol(now)}",
                stacklevel=1,
            )

        return self


def _get_next_contract_symbol(now: datetime) -> str:
    """Get the next front-month MNQ contract symbol."""
    month = now.month
    year = now.year

    # Find next expiration month (Mar, Jun, Sep, Dec)
    for exp_month in [3, 6, 9, 12]:
        if exp_month >= month:
            code = _MONTH_CODES[exp_month]
            return f"MNQ{code}{year % 10}"

    # Wrap to next year
    code = _MONTH_CODES[3]  # March
    return f"MNQ{code}{(year + 1) % 10}"


class QuantLynkConfig(BaseSettings):
    """QuantLynk webhook configuration for order execution via QuantVue."""

    model_config = SettingsConfigDict(env_prefix="QL_", env_file=".env", extra="ignore")

    webhook_url: str = ""  # unique webhook URL from QuantLynk dashboard
    user_id: str = ""  # qv_user_id from QuantLynk dashboard
    alert_id: str = ""  # alert_id from QuantLynk dashboard
    timeout_sec: float = 10.0  # HTTP request timeout
    max_retries: int = 2  # retry on transient failures
    enabled: bool = True  # master switch for QuantLynk execution


class AppConfig(BaseSettings):
    """Top-level config that composes all sub-configs.

    Each sub-config loads its own env vars independently using its prefix.
    This avoids the nested model loading issue with pydantic-settings.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    log_level: str = "INFO"
    data_dir: str = "./data"
    journal_path: str = "./data/journal.db"
    lock_file: str = "/tmp/futures-trader.lock"
    timezone: str = "US/Eastern"

    # Sub-configs are NOT loaded by pydantic-settings nesting.
    # They're constructed independently in load_config().
    tradovate: TradovateConfig = Field(default_factory=TradovateConfig)
    databento: DatabentoConfig = Field(default_factory=DatabentoConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    quantlynk: QuantLynkConfig = Field(default_factory=QuantLynkConfig)


def load_config() -> AppConfig:
    """Load configuration from environment variables and .env file.

    Each sub-config is constructed independently so it reads its own
    prefixed env vars (TV_, DB_, ANTHROPIC_, TG_, TRADE_) correctly
    from the flat .env file.
    """
    return AppConfig(
        tradovate=TradovateConfig(),
        databento=DatabentoConfig(),
        anthropic=AnthropicConfig(),
        telegram=TelegramConfig(),
        trading=TradingConfig(),
        quantlynk=QuantLynkConfig(),
    )
