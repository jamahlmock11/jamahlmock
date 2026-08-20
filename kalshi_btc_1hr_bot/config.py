"""Configuration for the KXBTCD 1-hour forecasting bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Edge thresholds — same structure as 15m bot (cents)
MIN_EDGE_CENTS = 2.5
FEE_PER_CONTRACT_CENTS = 1.75

# Model / market constants
WINDOW_SECONDS = 3600
SETTLE_AVG_SECONDS = 60
ANNUALIZE_SECONDS = 365.25 * 24 * 3600
MOMENTUM_LOOKBACK_SECONDS = 90
MOMENTUM_WEIGHTS = (0.5, 0.3, 0.2)
VOL_LOOKBACK_SECONDS = 3600
VOL_REGIME_LOW = 0.35
VOL_REGIME_HIGH = 0.80
FUNDING_SIGNAL_WEIGHT = 0.08
MEAN_REVERSION_STRENGTH = 0.15
OBI_LEVELS = 5
CALIB_MODEL_PATH = str(Path(__file__).resolve().parent / "data" / "calib_model.joblib")

# CF Benchmarks BRTI — https://www.cfbenchmarks.com/data/indices/BRTI
BRTI_INDEX_ID = "BRTI"
BRTI_PREFER_OFFICIAL = True
BRTI_ALLOW_EXCHANGE_PROXY = True
PROXY_BRTI_CONFIDENCE_PENALTY = 0.75

# Forecast ensemble weights (should sum to ~1.0)
ENSEMBLE_WEIGHTS = {
    "five_layer": 0.40,
    "gbm_core": 0.20,
    "momentum": 0.15,
    "mean_reversion": 0.10,
    "funding": 0.10,
    "obi": 0.05,
}
ENSEMBLE_MIN_AGREEMENT = 0.55


@dataclass
class ModelConfig:
    """5-layer ensemble parameters (mirrors module-level constants)."""

    averaging_window_seconds: int = SETTLE_AVG_SECONDS
    min_vol: float = 0.05
    momentum_w_5m: float = MOMENTUM_WEIGHTS[0]
    momentum_w_15m: float = MOMENTUM_WEIGHTS[1]
    momentum_w_30m: float = MOMENTUM_WEIGHTS[2]
    funding_extreme_threshold: float = 0.0005
    funding_weight: float = FUNDING_SIGNAL_WEIGHT
    mr_coefficient: float = MEAN_REVERSION_STRENGTH
    vol_low_threshold: float = VOL_REGIME_LOW
    vol_high_threshold: float = VOL_REGIME_HIGH
    vol_low_weight: float = 1.0
    vol_med_weight: float = 0.85
    vol_high_weight: float = 0.65


@dataclass
class EdgeConfig:
    min_edge_cents: float = MIN_EDGE_CENTS
    fee_per_contract_cents: float = FEE_PER_CONTRACT_CENTS


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.20
    max_bankroll_pct: float = 0.04
    bankroll_usd: float = 1000.0


@dataclass
class RiskConfig:
    daily_loss_stop_pct: float = 0.06
    max_open_positions: int = 2
    min_seconds_to_expiry: float = 120.0
    max_seconds_to_expiry: float = 3600.0
    cooldown_seconds: float = 5.0


@dataclass
class BotConfig:
    series_ticker: str = "KXBTCD"
    window_seconds: int = WINDOW_SECONDS
    cycle_seconds: float = 5.0
    paper: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    kalshi_env: str = field(default_factory=lambda: os.getenv("KALSHI_ENV", "public"))
    kalshi_api_key_id: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY_ID", ""))
    kalshi_private_key_pem: str = field(default_factory=lambda: os.getenv("KALSHI_PRIVATE_KEY_PEM", ""))

    @property
    def kalshi_base_url(self) -> str:
        env = self.kalshi_env.lower()
        if env == "prod":
            return "https://api.elections.kalshi.com/trade-api/v2"
        if env == "demo":
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"


def load_config() -> BotConfig:
    bankroll = float(os.getenv("BANKROLL_USD", "1000"))
    cfg = BotConfig()
    cfg.sizing.bankroll_usd = bankroll
    cfg.paper = os.getenv("PAPER_MODE", "true").lower() != "false"
    return cfg


ROOT = Path(__file__).resolve().parent
