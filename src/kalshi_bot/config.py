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


class ArbitraryPolicyConfig(BaseModel):
    """Independent YES/NO judgment — do not blindly follow the market favorite."""

    enabled: bool = True
    favorite_threshold: float = 0.52
    underdog_threshold: float = 0.48
    uncalibrated_shrink: float = 0.85
    uncalibrated_band_pp: float = 4.0
    time_edge_bonus_max: float = 0.30
    chase_min_gap_decay_pp: float = 3.0
    chase_max_ask_rise: float = 0.02
    chase_ttl_seconds: float = 120.0
    block_favorite_without_edge: bool = True
    require_calibration_for_conditional: bool = True
    min_trades_per_bucket: int = 3


class BrtiConfig(BaseModel):
    """CF Benchmarks BRTI (Kalshi settlement index) resolution."""

    index_id: str = "BRTI"
    prefer_official: bool = True
    public_summary_enabled: bool = True
    allow_exchange_proxy: bool = True
    cf_benchmarks_username: str | None = None
    cf_benchmarks_api_key: str | None = None


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
    """Edge quality tiers — separate trade frequency from trade quality."""

    enabled_for_live: bool = True
    use_raw_edge_for_tiers: bool = True
    edge_exceptional: float | None = 0.20
    edge_strong: float | None = 0.15
    edge_conditional: float | None = 0.08
    edge_experimental: float | None = 0.05
    min_net_edge_strong: float | None = 0.0
    min_net_edge_experimental: float | None = -0.02
    conditional_min_confidence: float | None = 0.50
    conditional_requires_model_agree: bool = False
    size_multiplier_exceptional: float = 1.0
    size_multiplier_strong: float = 1.0
    size_multiplier_conditional: float = 0.75
    size_multiplier_experimental: float = 0.50
    experimental_max_contracts: int = 1
    min_edge_a_plus: float | None = 0.20
    min_edge_a: float | None = 0.15
    min_edge_b: float | None = 0.05
    min_confidence_a_plus: float | None = 0.65
    min_confidence_a: float | None = 0.55
    min_confidence_b: float | None = 0.45


def _cleared_tier_config() -> TierEdgeConfig:
    return TierEdgeConfig(
        enabled_for_live=False,
        edge_exceptional=None,
        edge_strong=None,
        edge_conditional=None,
        edge_experimental=None,
        min_net_edge_strong=None,
        min_net_edge_experimental=None,
        conditional_min_confidence=None,
        min_edge_a_plus=None,
        min_edge_a=None,
        min_edge_b=None,
        min_confidence_a_plus=None,
        min_confidence_a=None,
        min_confidence_b=None,
    )


class StrictEdgeRules(BaseModel):
    min_gap_dollars: float | None = None


class QualityRules15m(BaseModel):
    max_spread: float | None = None
    min_liquidity_score: float | None = None
    max_model_disagreement_pp: float | None = None
    require_pattern_evidence: bool = False
    min_pattern_examples: int | None = None
    stale_data_stop_trading: bool = True


class V6Config(BaseModel):
    """Kalshi BTC 15-Min Intelligence V6 settings."""

    enabled: bool = True
    series_ticker: str = "KXBTC15M"
    live_trading_enabled: bool = False
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
    min_open_seconds: float = 30.0
    tiers: TierEdgeConfig = Field(default_factory=TierEdgeConfig)


class BotActionConfig(BaseModel):
    """Gap-tiered buy action policy (model prob − market price)."""

    strong_buy_min_gap_pp: float = 20.0
    conditional_min_gap_pp: float = 15.0
    conditional_min_confidence: float = 0.70
    conditional_max_disagreement_pp: float = 6.0


class Rules15mConfig(BaseModel):
    """Trading rules for the 15-minute bot (KXBTC15M / V6 workflow)."""

    enabled: bool = False
    mode: Literal["mispricing", "legacy"] = "mispricing"
    strict_edge: StrictEdgeRules = Field(default_factory=StrictEdgeRules)
    tiers: TierEdgeConfig = Field(default_factory=_cleared_tier_config)
    arbitrary: ArbitraryPolicyConfig = Field(
        default_factory=lambda: ArbitraryPolicyConfig(enabled=False)
    )
    bot_action: BotActionConfig = Field(default_factory=BotActionConfig)
    quality: QualityRules15m = Field(default_factory=QualityRules15m)
    time_buckets: dict[str, dict] = Field(default_factory=dict)


class ExecutionConfig(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    time_in_force: str = "immediate_or_cancel"
    use_taker: bool = True
    price_improve_ticks: int = 0
    dry_run: bool = True


class ExitConfig(BaseModel):
    """Pre-expiry exit rules for open positions."""

    enabled: bool = True
    max_drawdown_pct: float = Field(
        0.45,
        description="Exit when unrealized loss exceeds this fraction of entry cost.",
    )
    exit_on_edge_flip: bool = True
    min_seconds_to_expiry: float = 30.0
    series_tickers: list[str] = Field(
        default_factory=lambda: ["KXBTC15M", "KXBTCD"],
        description="Only manage exits for these series.",
    )


class BotConfig(BaseModel):
    brti: BrtiConfig = Field(default_factory=BrtiConfig)
    arbitrary: ArbitraryPolicyConfig = Field(default_factory=ArbitraryPolicyConfig)
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
    exit: ExitConfig = Field(default_factory=ExitConfig)
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
    cf_benchmarks_api_username: str | None = None
    cf_benchmarks_api_key: str | None = None
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


def load_rules_15m(path: str | Path | None = None) -> Rules15mConfig:
    rules_path = Path(path or os.getenv("RULES_15M_PATH", "config/rules_15m.yaml"))
    if rules_path.exists():
        raw = yaml.safe_load(rules_path.read_text()) or {}
        return Rules15mConfig.model_validate(raw)
    return Rules15mConfig()


def kalshi_base_url(settings: Settings, config: BotConfig) -> str:
    if settings.kalshi_env == "demo":
        return config.kalshi_demo_base
    if settings.kalshi_env == "prod":
        return config.kalshi_prod_base
    return config.kalshi_public_base
