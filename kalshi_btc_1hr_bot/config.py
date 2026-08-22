"""Configuration for the KXBTCD 1-hour forecasting bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Edge thresholds — static fallbacks; bot uses dynamic_gates.resolve_dynamic_thresholds()
MIN_EDGE_CENTS = 2.5
FEE_PER_CONTRACT_CENTS = 1.75  # accounting only; gates ignore fees when SUBTRACT_FEES_FROM_EDGE=false
SUBTRACT_FEES_FROM_EDGE = False
DYNAMIC_GATES_ENABLED = True
CROWD_GATES_ENABLED = False  # quorum + crowd favorite % no longer block trades
USE_ENSEMBLE_AGREEMENT = True  # agreement gate uses 5-layer ensemble, not crowd voters
RISK_MIN_SECONDS = 120.0
# Hard quality floors — gates may loosen inside buckets but never below these
GATE_ABS_MIN_CROWD = 0.60
GATE_ABS_MIN_EDGE_CENTS = 0.5  # gross edge (pre-fee) when fees excluded from gates
GATE_ABS_MIN_EVIDENCE = 0.012
GATE_ABS_MIN_AGREEMENT = 0.48

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
CROWD_MIN_FAVORITE = 0.76  # dashboard fallback only; trade gates use dynamic_gates
CROWD_USE_ADAPTIVE_WEIGHTS = True
CROWD_PERFORMANCE_PATH = str(Path(__file__).resolve().parent / "data" / "crowd_performance.json")

# Top-N selection: evidence from top votes, best market from top opportunities
TOP_N_VOTES = 4
TOP_N_MARKETS = 3
KALSHI_CARD_PICKS = 3  # only trade strikes shown on Kalshi's hourly card
KALSHI_CARD_ONLY = True
MIN_EVIDENCE_MARGIN = 0.02

# Trend-following confirmation gates (keep all existing edge/evidence/risk rules)
TREND_GATE_ENABLED = True
FLOW_CONFIRM_ENABLED = False
TREND_MIN_MOMENTUM = 0.0  # blended 5m/15m/30m drift must be positive (YES) or negative (NO)
TREND_REQUIRE_SPOT_VS_STRIKE = False  # allow BELOW above strike / ABOVE below strike (mean-reversion)
TREND_BIAS_SELECTION = True  # prefer trend+flow aligned picks among card top-N

# Daily loss stop — % of day-start bankroll (raised for small accounts)
DAILY_LOSS_STOP_PCT = 0.12
MAX_TRADES_PER_HOUR = 2  # entry attempts per hourly window (e.g. early + late, or retry after exit)

# Take-profit / stop-loss on open positions (sell at market bid before settlement)
EXIT_ENABLED = True
TAKE_PROFIT_PCT = 0.50  # exit when bid >= entry + 50% of entry cost
STOP_LOSS_PCT = 0.50  # exit when bid <= entry - 50% of entry cost (wider for hourly vol)
EXIT_MIN_HOLD_SECONDS = 0.0

# Late crowd favorite — enter with more size when hour slot unused + crowd strong near expiry
LATE_CROWD_ENABLED = True
LATE_CROWD_MIN_SECONDS = 120.0  # same as risk min — last ~2 min still allowed
LATE_CROWD_MAX_SECONDS = 600.0  # 10 min left — final window only
LATE_CROWD_MIN_FAVORITE = 0.72  # crowd must strongly favor trade side
LATE_CROWD_MIN_EDGE_CENTS = 0.3
LATE_CROWD_MIN_EVIDENCE = 0.010
LATE_CROWD_MIN_QUORUM = 4
LATE_CROWD_MIN_AGREEMENT = 0.50
LATE_CROWD_SIZE_MULTIPLIER = 1.75  # Kelly sizing boost for late conviction entries
LATE_CROWD_SKIP_FLOW = True  # crowd is the signal — don't require tape confirmation
LATE_CROWD_SKIP_TREND = True  # allow mean-reversion when crowd is decisive


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
    subtract_fees_from_edge: bool = SUBTRACT_FEES_FROM_EDGE


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.20
    max_bankroll_pct: float = 1.0
    max_trade_usd: float = 1.0
    bankroll_usd: float = 1.0
    use_live_balance: bool = True


@dataclass
class ExitConfig:
    enabled: bool = EXIT_ENABLED
    take_profit_pct: float = TAKE_PROFIT_PCT
    stop_loss_pct: float = STOP_LOSS_PCT
    min_hold_seconds: float = EXIT_MIN_HOLD_SECONDS


@dataclass
class RiskConfig:
    daily_loss_stop_pct: float = DAILY_LOSS_STOP_PCT
    max_open_positions: int = 1
    max_trades_per_hour: int = MAX_TRADES_PER_HOUR
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
    notify_on_exit: bool = True
    notify_on_order_failed: bool = True
    notify_on_paper: bool = False
    notify_on_startup: bool = False
    twilio_trial_template: str = ""  # e.g. sms_appointment_reminders on Twilio trial accounts

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
class GateConfig:
    crowd_gates_enabled: bool = CROWD_GATES_ENABLED
    use_ensemble_agreement: bool = USE_ENSEMBLE_AGREEMENT
    dynamic_gates_enabled: bool = DYNAMIC_GATES_ENABLED
    kalshi_card_only: bool = KALSHI_CARD_ONLY
    kalshi_card_picks: int = KALSHI_CARD_PICKS
    trend_gate_enabled: bool = TREND_GATE_ENABLED
    flow_confirm_enabled: bool = FLOW_CONFIRM_ENABLED
    trend_min_momentum: float = TREND_MIN_MOMENTUM
    trend_require_spot_vs_strike: bool = TREND_REQUIRE_SPOT_VS_STRIKE
    trend_bias_selection: bool = TREND_BIAS_SELECTION


@dataclass
class LateCrowdConfig:
    enabled: bool = LATE_CROWD_ENABLED
    min_seconds_to_expiry: float = LATE_CROWD_MIN_SECONDS
    max_seconds_to_expiry: float = LATE_CROWD_MAX_SECONDS
    min_crowd_favorite: float = LATE_CROWD_MIN_FAVORITE
    min_edge_cents: float = LATE_CROWD_MIN_EDGE_CENTS
    min_evidence_margin: float = LATE_CROWD_MIN_EVIDENCE
    min_quorum: int = LATE_CROWD_MIN_QUORUM
    min_agreement: float = LATE_CROWD_MIN_AGREEMENT
    size_multiplier: float = LATE_CROWD_SIZE_MULTIPLIER
    skip_flow: bool = LATE_CROWD_SKIP_FLOW
    skip_trend: bool = LATE_CROWD_SKIP_TREND


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
    gates: GateConfig = field(default_factory=GateConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    late_crowd: LateCrowdConfig = field(default_factory=LateCrowdConfig)

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


def gate_fee_cents(fee_cents: float | None = None, *, subtract: bool | None = None) -> float:
    """Fee applied to edge gates — zero when user opts out of fee-adjusted edge."""
    if subtract is None:
        subtract = SUBTRACT_FEES_FROM_EDGE
    if not subtract:
        return 0.0
    return fee_cents if fee_cents is not None else FEE_PER_CONTRACT_CENTS


def load_config() -> BotConfig:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT.parent / ".env")
    except ImportError:
        pass

    cfg = BotConfig()
    cfg.gates.crowd_gates_enabled = os.getenv(
        "CROWD_GATES_ENABLED", "false" if not CROWD_GATES_ENABLED else "true"
    ).lower() in ("true", "1", "yes")
    cfg.gates.use_ensemble_agreement = os.getenv(
        "USE_ENSEMBLE_AGREEMENT", "true" if USE_ENSEMBLE_AGREEMENT else "false"
    ).lower() in ("true", "1", "yes")
    cfg.gates.dynamic_gates_enabled = os.getenv(
        "DYNAMIC_GATES_ENABLED", "true" if DYNAMIC_GATES_ENABLED else "false"
    ).lower() in ("true", "1", "yes")
    cfg.gates.kalshi_card_only = os.getenv(
        "KALSHI_CARD_ONLY", "true" if KALSHI_CARD_ONLY else "false"
    ).lower() in ("true", "1", "yes")
    cfg.gates.kalshi_card_picks = int(os.getenv("KALSHI_CARD_PICKS", str(KALSHI_CARD_PICKS)))
    cfg.gates.trend_gate_enabled = os.getenv(
        "TREND_GATE_ENABLED", "true" if TREND_GATE_ENABLED else "false"
    ).lower() in ("true", "1", "yes")
    cfg.gates.flow_confirm_enabled = os.getenv(
        "FLOW_CONFIRM_ENABLED", "true" if FLOW_CONFIRM_ENABLED else "false"
    ).lower() in ("true", "1", "yes")
    cfg.gates.trend_min_momentum = float(os.getenv("TREND_MIN_MOMENTUM", str(TREND_MIN_MOMENTUM)))
    cfg.gates.trend_require_spot_vs_strike = os.getenv(
        "TREND_REQUIRE_SPOT_VS_STRIKE", "false" if not TREND_REQUIRE_SPOT_VS_STRIKE else "true"
    ).lower() in ("true", "1", "yes")
    cfg.gates.trend_bias_selection = os.getenv(
        "TREND_BIAS_SELECTION", "true" if TREND_BIAS_SELECTION else "false"
    ).lower() in ("true", "1", "yes")
    cfg.risk.daily_loss_stop_pct = float(os.getenv("DAILY_LOSS_STOP_PCT", str(DAILY_LOSS_STOP_PCT)))
    cfg.risk.max_trades_per_hour = int(os.getenv("MAX_TRADES_PER_HOUR", str(MAX_TRADES_PER_HOUR)))
    cfg.exit.enabled = os.getenv("EXIT_ENABLED", "true" if EXIT_ENABLED else "false").lower() in (
        "true",
        "1",
        "yes",
    )
    cfg.exit.take_profit_pct = float(os.getenv("TAKE_PROFIT_PCT", str(TAKE_PROFIT_PCT)))
    cfg.exit.stop_loss_pct = float(os.getenv("STOP_LOSS_PCT", str(STOP_LOSS_PCT)))
    cfg.exit.min_hold_seconds = float(os.getenv("EXIT_MIN_HOLD_SECONDS", str(EXIT_MIN_HOLD_SECONDS)))
    cfg.late_crowd.enabled = os.getenv(
        "LATE_CROWD_ENABLED", "true" if LATE_CROWD_ENABLED else "false"
    ).lower() in ("true", "1", "yes")
    cfg.late_crowd.min_seconds_to_expiry = float(
        os.getenv("LATE_CROWD_MIN_SECONDS", str(LATE_CROWD_MIN_SECONDS))
    )
    cfg.late_crowd.max_seconds_to_expiry = float(
        os.getenv("LATE_CROWD_MAX_SECONDS", str(LATE_CROWD_MAX_SECONDS))
    )
    cfg.late_crowd.min_crowd_favorite = float(
        os.getenv("LATE_CROWD_MIN_FAVORITE", str(LATE_CROWD_MIN_FAVORITE))
    )
    cfg.late_crowd.min_edge_cents = float(
        os.getenv("LATE_CROWD_MIN_EDGE_CENTS", str(LATE_CROWD_MIN_EDGE_CENTS))
    )
    cfg.late_crowd.min_evidence_margin = float(
        os.getenv("LATE_CROWD_MIN_EVIDENCE", str(LATE_CROWD_MIN_EVIDENCE))
    )
    cfg.late_crowd.min_quorum = int(os.getenv("LATE_CROWD_MIN_QUORUM", str(LATE_CROWD_MIN_QUORUM)))
    cfg.late_crowd.min_agreement = float(
        os.getenv("LATE_CROWD_MIN_AGREEMENT", str(LATE_CROWD_MIN_AGREEMENT))
    )
    cfg.late_crowd.size_multiplier = float(
        os.getenv("LATE_CROWD_SIZE_MULTIPLIER", str(LATE_CROWD_SIZE_MULTIPLIER))
    )
    cfg.late_crowd.skip_flow = os.getenv(
        "LATE_CROWD_SKIP_FLOW", "true" if LATE_CROWD_SKIP_FLOW else "false"
    ).lower() in ("true", "1", "yes")
    cfg.late_crowd.skip_trend = os.getenv(
        "LATE_CROWD_SKIP_TREND", "true" if LATE_CROWD_SKIP_TREND else "false"
    ).lower() in ("true", "1", "yes")
    cfg.edge.subtract_fees_from_edge = os.getenv(
        "SUBTRACT_FEES_FROM_EDGE", "false" if not SUBTRACT_FEES_FROM_EDGE else "true"
    ).lower() in ("true", "1", "yes")
    cfg.sizing.use_live_balance = os.getenv("USE_LIVE_BALANCE", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    cfg.sizing.bankroll_usd = float(os.getenv("BANKROLL_USD", "1"))
    max_trade_env = os.getenv("MAX_TRADE_USD", "0")
    cfg.sizing.max_trade_usd = float(max_trade_env) if max_trade_env else 0.0
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
    if cfg.sizing.max_trade_usd > 0 and cfg.sizing.max_trade_usd <= 1.0:
        cfg.risk.max_open_positions = 1

    cfg.notify = NotifyConfig(
        enabled=os.getenv("NOTIFY_ENABLED", "false").lower() in ("true", "1", "yes"),
        phone_to=os.getenv("NOTIFY_PHONE_NUMBER", ""),
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
        twilio_from=os.getenv("TWILIO_FROM_NUMBER", ""),
        notify_on_trade=os.getenv("NOTIFY_ON_TRADE", "true").lower() not in ("false", "0", "no"),
        notify_on_settlement=os.getenv("NOTIFY_ON_SETTLEMENT", "true").lower() not in ("false", "0", "no"),
        notify_on_exit=os.getenv("NOTIFY_ON_EXIT", "true").lower() not in ("false", "0", "no"),
        notify_on_order_failed=os.getenv("NOTIFY_ON_ORDER_FAILED", "true").lower() not in ("false", "0", "no"),
        notify_on_paper=os.getenv("NOTIFY_ON_PAPER", "false").lower() in ("true", "1", "yes"),
        notify_on_startup=os.getenv("NOTIFY_ON_STARTUP", "false").lower() in ("true", "1", "yes"),
        twilio_trial_template=os.getenv("TWILIO_TRIAL_TEMPLATE", ""),
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


