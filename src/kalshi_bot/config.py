"""Runtime configuration for the Kalshi BTC 1-hour forecasting engine."""

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
        10.0,
        description="Minimum edge in percentage points after fees to take a trade.",
    )
    max_contracts: int = 500
    max_notional_usd: float = 250.0


class RiskConfig(BaseModel):
    bankroll_usd: float = 1000.0
    kelly_fraction: float = 0.15
    max_open_positions: int = 4
    max_exposure_usd: float = 300.0
    max_loss_per_trade_usd: float = 50.0
    cooldown_seconds: float = 5.0
    min_seconds_to_expiry: float = 60.0
    max_seconds_to_expiry_15m: float = 14 * 60
    # Hourly (KXBTCD): last-20-minute execution window only.
    max_seconds_to_expiry_1h: float = 20 * 60


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


class ForecastGateConfig(BaseModel):
    """Accuracy-first gates. Fail-closed → NO TRADE."""

    min_confidence: float = 0.60
    max_disagreement_pp: float = 10.0
    min_edge_pp: float = 10.0
    max_spread: float = 0.06
    min_volume: float = 50.0
    min_seconds_to_expiry: float = 60.0
    # Do not enter before the final 20 minutes of the hourly contract.
    max_seconds_to_expiry: float = 20 * 60
    hourly_only: bool = True
    # When scanning without authenticated BRTI, raise the bar.
    proxy_spot_edge_multiplier: float = 1.5
    proxy_spot_confidence_penalty: float = 0.30
    # Reject NO on deep ITM YES (spot already through strike) unless edge is huge —
    # quiet-tape vol floors can fabricate false mean-reversion edges.
    deep_itm_buffer_usd: float = 25.0
    deep_itm_extra_edge_pp: float = 8.0


class TierEdgeConfig(BaseModel):
    """Edge quality tiers — separate frequency from quality."""

    enabled_for_live: bool = True
    # Classify tiers on raw edge (before fees) — better for small bankrolls where fees eat net edge.
    use_raw_edge_for_tiers: bool = True
    # Net-edge band floors (dollars) when use_raw_edge_for_tiers=false
    edge_exceptional: float = 0.20   # ≥20¢
    edge_strong: float = 0.15        # 15–20¢
    edge_conditional: float = 0.08   # 8–15¢
    edge_experimental: float = 0.05  # 5–8¢; below = no trade
    # Minimum net EV after fees (can be slightly negative for experimental on small accounts)
    min_net_edge_strong: float = 0.0
    min_net_edge_experimental: float = -0.02
    # Confirmation for CONDITIONAL tier
    conditional_min_confidence: float = 0.50
    conditional_requires_model_agree: bool = False
    # Position size multipliers by tier (applied to Kelly)
    size_multiplier_exceptional: float = 1.0
    size_multiplier_strong: float = 1.0
    size_multiplier_conditional: float = 0.75
    size_multiplier_experimental: float = 0.50
    experimental_max_contracts: int = 1
    # Legacy aliases (diagnostics)
    min_edge_a_plus: float = 0.20
    min_edge_a: float = 0.15
    min_edge_b: float = 0.05
    min_confidence_a_plus: float = 0.65
    min_confidence_a: float = 0.55
    min_confidence_b: float = 0.45


class V6Config(BaseModel):
    """Kalshi BTC 15-Min Intelligence V6 settings."""

    enabled: bool = True
    series_ticker: str = "KXBTC15M"
    live_trading_enabled: bool = False
    # Minimum net edge floor for any consideration (10¢ experimental band).
    strict_min_gap_dollars: float = 0.05
    min_trades_per_bucket: int = 3
    monte_carlo_sims: int = 5000
    max_spread: float = 0.12
    min_liquidity_score: float = 0.10
    max_model_disagreement_pp: float = 28.0
    min_pattern_examples: int = 5
    require_pattern_evidence: bool = False
    journal_path: str = "data/v6_trade_journal.db"
    diagnostics_db_path: str = "data/diagnostics/evaluations.db"
    bankroll_usd: float = 1000.0
    kelly_fraction: float = 0.12
    max_position_usd: float = 75.0
    max_exposure_usd: float = 250.0
    max_daily_loss_usd: float = 100.0
    max_consecutive_losses: int = 4
    cooldown_after_loss_seconds: float = 120.0
    min_seconds_to_expiry: float = 60.0
    max_seconds_to_expiry: float = 840.0
    min_open_seconds: float = 30.0  # don't trade in first 30s of new market
    tiers: TierEdgeConfig = Field(default_factory=TierEdgeConfig)


class BotActionConfig(BaseModel):
    """Gap-tiered buy action policy (model prob − market price).

    Defaults match the 60% model reference matrix:
    ≥20pp Strong BUY, ≥15pp conditional, <15pp No trade.
    """

    strong_buy_min_gap_pp: float = 20.0
    conditional_min_gap_pp: float = 15.0
    # Elevated confirmation required only for the CONDITIONAL tier.
    conditional_min_confidence: float = 0.70
    conditional_max_disagreement_pp: float = 6.0


class ExecutionConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    time_in_force: str = "immediate_or_cancel"
    use_taker: bool = True
    price_improve_ticks: int = 0
    dry_run: bool = True


class BotConfig(BaseModel):
    series: list[SeriesConfig] = Field(
        default_factory=lambda: [
            SeriesConfig(ticker="KXBTCD", enabled=True, min_edge_pp=10.0, max_contracts=400),
            SeriesConfig(ticker="KXBTC15M", enabled=False, min_edge_pp=12.0, max_contracts=200),
        ]
    )
    risk: RiskConfig = Field(default_factory=RiskConfig)
    smile: SmileConfig = Field(default_factory=SmileConfig)
    forecast_gates: ForecastGateConfig = Field(default_factory=ForecastGateConfig)
    bot_action: BotActionConfig = Field(default_factory=BotActionConfig)
    v6: V6Config = Field(default_factory=V6Config)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    scan_interval_seconds: float = 5.0
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
