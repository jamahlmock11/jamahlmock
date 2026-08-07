"""Runtime configuration for the Kalshi BTC mispricing bot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SeriesConfig(BaseModel):
    ticker: str
    enabled: bool = True
    min_edge_pp: float = Field(
        8.0,
        description="Minimum edge in percentage points after fees to take a trade.",
    )
    max_contracts: int = 500
    max_notional_usd: float = 250.0


class RiskConfig(BaseModel):
    bankroll_usd: float = 1000.0
    kelly_fraction: float = 0.25
    max_open_positions: int = 8
    max_exposure_usd: float = 500.0
    max_loss_per_trade_usd: float = 75.0
    cooldown_seconds: float = 5.0
    min_seconds_to_expiry: float = 45.0
    max_seconds_to_expiry_15m: float = 14 * 60
    max_seconds_to_expiry_1h: float = 55 * 60


class SmileConfig(BaseModel):
    symbol: str = "IBIT"
    min_oi: int = 10
    min_volume: int = 0
    max_spread_pct: float = 0.25
    max_smile_age_seconds: float = 6 * 3600
    stale_edge_multiplier: float = 1.75
    cache_path: str = "data/cache/ibit_smile.json"
    prefer_expiries: int = 4
    risk_free_rate: float = 0.045
    dividend_yield: float = 0.0


class ExecutionConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    time_in_force: str = "immediate_or_cancel"
    use_taker: bool = True
    price_improve_ticks: int = 0
    dry_run: bool = True


class BotConfig(BaseModel):
    series: list[SeriesConfig] = Field(
        default_factory=lambda: [
            SeriesConfig(ticker="KXBTC15M", min_edge_pp=10.0, max_contracts=300),
            SeriesConfig(ticker="KXBTCD", min_edge_pp=8.0, max_contracts=500),
        ]
    )
    risk: RiskConfig = Field(default_factory=RiskConfig)
    smile: SmileConfig = Field(default_factory=SmileConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    scan_interval_seconds: float = 2.0
    kalshi_public_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_prod_base: str = "https://external-api.kalshi.com/trade-api/v2"
    kalshi_demo_base: str = "https://demo-api.kalshi.co/trade-api/v2"
    fee_rate: float = 0.07
    fee_multiplier: float = 1.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    kalshi_api_key_id: str | None = None
    kalshi_private_key_pem: str | None = None
    kalshi_private_key_path: str | None = None
    kalshi_env: Literal["prod", "demo", "public"] = "public"
    config_path: str = "config/default.yaml"
    log_level: str = "INFO"

    def resolve_private_key_pem(self) -> str | None:
        if self.kalshi_private_key_pem:
            return self.kalshi_private_key_pem.replace("\\n", "\n")
        if self.kalshi_private_key_path:
            return Path(self.kalshi_private_key_path).read_text()
        return None


def load_config(path: str | Path | None = None) -> BotConfig:
    cfg_path = Path(path or os.getenv("CONFIG_PATH", "config/default.yaml"))
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
        return BotConfig.model_validate(raw)
    return BotConfig()


def kalshi_base_url(settings: Settings, config: BotConfig) -> str:
    if settings.kalshi_env == "demo":
        return config.kalshi_demo_base
    if settings.kalshi_env == "prod":
        return config.kalshi_prod_base
    return config.kalshi_public_base
