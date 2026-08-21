"""Configuration for the KXBTCD 1-hour forecasting bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Edge thresholds — static fallbacks; bot uses dynamic_gates.resolve_dynamic_thresholds()
MIN_EDGE_CENTS = 2.5
FEE_PER_CONTRACT_CENTS = 1.75
DYNAMIC_GATES_ENABLED = True
RISK_MIN_SECONDS = 120.0

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

# Crowd forecast system
CROWD_SYNTHESIS = "blend"  # weighted | median | trimmed | blend
CROWD_MIN_QUORUM = 5  # base quorum; dynamic gates use 4–6 by time bucket
CROWD_MIN_FAVORITE = 0.76  # legacy ceiling; dynamic floor typically 60–78% by bucket
CROWD_USE_ADAPTIVE_WEIGHTS = True
CROWD_PERFORMANCE_PATH = str(Path(__file__).resolve().parent / "data" / "crowd_performance.json")

# Top-N selection: evidence from top votes, best market from top opportunities
TOP_N_VOTES = 4
TOP_N_MARKETS = 4
MIN_EVIDENCE_MARGIN = 0.02


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
    max_bankroll_pct: float = 1.0
    max_trade_usd: float = 1.0
    bankroll_usd: float = 1.0


@dataclass
class RiskConfig:
    daily_loss_stop_pct: float = 0.06
    max_open_positions: int = 1
    min_seconds_to_expiry: float = 120.0
    max_seconds_to_expiry: float = 3600.0
    cooldown_seconds: float = 5.0


@dataclass
class NotifyConfig:
    enabled: bool = False
    phone_to: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from: str = ""
    notify_on_trade: bool = True
    notify_on_settlement: bool = True
    notify_on_order_failed: bool = True
    notify_on_paper: bool = False
    notify_on_startup: bool = False

    @property
    def configured(self) -> bool:
        return bool(
            self.enabled
            and self.phone_to
            and self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_from
        )


@dataclass
class BotConfig:
    series_ticker: str = "KXBTCD"
    window_seconds: int = WINDOW_SECONDS
    cycle_seconds: float = 2.0
    paper: bool = True

    model: ModelConfig = field(default_factory=ModelConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)

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


ROOT = Path(__file__).resolve().parent


def load_config() -> BotConfig:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT.parent / ".env")
    except ImportError:
        pass

    cfg = BotConfig()
    cfg.sizing.bankroll_usd = float(os.getenv("BANKROLL_USD", "1"))
    cfg.sizing.max_trade_usd = float(os.getenv("MAX_TRADE_USD", "1"))
    cfg.sizing.max_bankroll_pct = float(os.getenv("MAX_BANKROLL_PCT", "1.0"))
    cfg.paper = os.getenv("PAPER_MODE", "true").lower() not in ("false", "0", "no")
    cfg.kalshi_env = os.getenv("KALSHI_ENV", cfg.kalshi_env)
    cfg.kalshi_api_key_id = os.getenv("KALSHI_API_KEY_ID", cfg.kalshi_api_key_id)
    pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", cfg.kalshi_private_key_pem)
    key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
    if key_path and not pem:
        try:
            pem = Path(key_path).read_text()
        except OSError:
            pass
    cfg.kalshi_private_key_pem = pem
    if cfg.sizing.max_trade_usd <= 1.0:
        cfg.risk.max_open_positions = 1

    cfg.notify = NotifyConfig(
        enabled=os.getenv("NOTIFY_ENABLED", "false").lower() in ("true", "1", "yes"),
        phone_to=os.getenv("NOTIFY_PHONE_NUMBER", ""),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_from=os.getenv("TWILIO_FROM_NUMBER", ""),
        notify_on_trade=os.getenv("NOTIFY_ON_TRADE", "true").lower() not in ("false", "0", "no"),
        notify_on_settlement=os.getenv("NOTIFY_ON_SETTLEMENT", "true").lower() not in ("false", "0", "no"),
        notify_on_order_failed=os.getenv("NOTIFY_ON_ORDER_FAILED", "true").lower() not in ("false", "0", "no"),
        notify_on_paper=os.getenv("NOTIFY_ON_PAPER", "false").lower() in ("true", "1", "yes"),
        notify_on_startup=os.getenv("NOTIFY_ON_STARTUP", "false").lower() in ("true", "1", "yes"),
    )
    return cfg


def require_live_credentials(cfg: BotConfig) -> None:
    if cfg.paper:
        return
    if not cfg.kalshi_api_key_id or not cfg.kalshi_private_key_pem:
        raise RuntimeError(
            "Live trading requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PEM "
            "(or KALSHI_PRIVATE_KEY_PATH) in environment or .env"
        )


