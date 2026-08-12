"""Kalshi BTC 15-Min Intelligence V6 — microstructure, ensemble, and risk upgrades.

STRICT EDGE RULE (hard filter, no exceptions):
  Only recommend BUY when market price is ≥20–25¢ below model probability.
  Example: model=60% UP → market YES must be ≤35–40¢.
  Replaces legacy gap-tier (Strong/Conditional) with a single cents-based gate.

Calibration requires ≥3 trades per probability bucket before trusting adjustments.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import norm

from kalshi_bot.config import ArbitraryPolicyConfig, Rules15mConfig, V6Config, load_rules_15m
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.rejection_codes import RejectionCode


# ---------------------------------------------------------------------------
# Strict edge filter
# ---------------------------------------------------------------------------

def strict_edge_gap_dollars(model_probability: float, market_price: float) -> float:
    """Raw edge in dollars: model − executable market price."""
    return round(model_probability - market_price, 4)


def passes_strict_edge(
    model_probability: float,
    market_price: float,
    *,
    min_gap_dollars: float = 0.20,
) -> tuple[bool, float]:
    """Hard filter: gap must be ≥ min_gap_dollars (default 20¢)."""
    gap = strict_edge_gap_dollars(model_probability, market_price)
    return gap >= min_gap_dollars, gap


# ---------------------------------------------------------------------------
# Microstructure features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float


@dataclass
class MicrostructureSnapshot:
    bid_ask_imbalance: float  # -1..+1 (positive = bid-heavy)
    depth_bid_10: float
    depth_ask_10: float
    whale_detected: bool
    whale_side: str | None
    cancel_to_new_ratio: float
    spread: float
    spread_change: float  # positive = widening
    trades_per_second: float
    liquidity_score: float  # 0..1


def parse_orderbook(raw: dict) -> tuple[list[OrderBookLevel], list[OrderBookLevel]]:
    """Parse Kalshi orderbook response into bid/ask levels."""
    ob = raw.get("orderbook") or raw
    yes_bids: list[OrderBookLevel] = []
    yes_asks: list[OrderBookLevel] = []
    for side_key, out in (("yes", yes_bids), ("no", yes_asks)):
        levels = ob.get(side_key) or []
        for lvl in levels:
            if isinstance(lvl, (list, tuple)) and len(lvl) >= 2:
                price_cents, qty = float(lvl[0]), float(lvl[1])
                out.append(OrderBookLevel(price=price_cents / 100.0, quantity=qty))
    # YES asks derived from NO bids when needed
    if not yes_asks and yes_bids:
        pass
    return yes_bids, yes_asks


def compute_microstructure(
    *,
    yes_bid: float | None,
    yes_ask: float | None,
    orderbook: dict | None = None,
    prev_spread: float | None = None,
    recent_trades: Sequence[dict] | None = None,
    prev_depth: tuple[float, float] | None = None,
) -> MicrostructureSnapshot:
    """Compute live microstructure features from book + trade tape."""
    bids, asks = parse_orderbook(orderbook or {}) if orderbook else ([], [])
    if yes_bid is not None and yes_ask is not None and not bids:
        bids = [OrderBookLevel(yes_bid, 1.0)]
        asks = [OrderBookLevel(yes_ask, 1.0)]

    depth_bid = sum(l.quantity for l in bids[:10])
    depth_ask = sum(l.quantity for l in asks[:10])
    total = depth_bid + depth_ask
    imbalance = (depth_bid - depth_ask) / total if total > 0 else 0.0

    whale_detected = False
    whale_side: str | None = None
    threshold = max(total * 0.25, 50.0) if total > 0 else 50.0
    for lvl in bids[:10]:
        if lvl.quantity >= threshold:
            whale_detected, whale_side = True, "bid"
            break
    if not whale_detected:
        for lvl in asks[:10]:
            if lvl.quantity >= threshold:
                whale_detected, whale_side = True, "ask"
                break

    spread = (yes_ask - yes_bid) if yes_bid is not None and yes_ask is not None else 0.0
    spread_change = spread - prev_spread if prev_spread is not None else 0.0

    trades = list(recent_trades or [])
    tps = 0.0
    if len(trades) >= 2:
        ts0 = trades[0].get("ts", 0)
        ts1 = trades[-1].get("ts", 0)
        dt = max(ts1 - ts0, 0.001)
        tps = len(trades) / dt

    cancel_new = 0.5
    if prev_depth and (depth_bid + depth_ask) > 0:
        prev_total = prev_depth[0] + prev_depth[1]
        depth_drop = max(0.0, prev_total - (depth_bid + depth_ask))
        cancel_new = min(2.0, depth_drop / max(prev_total, 1.0))

    liquidity = min(1.0, math.log1p(total) / math.log1p(500))

    return MicrostructureSnapshot(
        bid_ask_imbalance=imbalance,
        depth_bid_10=depth_bid,
        depth_ask_10=depth_ask,
        whale_detected=whale_detected,
        whale_side=whale_side,
        cancel_to_new_ratio=cancel_new,
        spread=spread,
        spread_change=spread_change,
        trades_per_second=tps,
        liquidity_score=liquidity,
    )


# ---------------------------------------------------------------------------
# Price action features
# ---------------------------------------------------------------------------

@dataclass
class PriceActionFeatures:
    vwap_distance: float
    momentum_15s: float
    momentum_30s: float
    momentum_1m: float
    vol_expansion: bool
    support: float
    resistance: float
    breakout_signal: str  # "breakout" | "fake_breakout" | "none"


def compute_price_action(
    prices: Sequence[float],
    *,
    vwap: float | None = None,
    window_support: int = 20,
) -> PriceActionFeatures:
    """Short-horizon price action from a tick/1s price series."""
    if len(prices) < 2:
        return PriceActionFeatures(0, 0, 0, 0, False, 0, 0, "none")
    p = np.asarray(prices, dtype=float)
    last = float(p[-1])
    # Need sufficient price variation for breakout detection
    unique_prices = len(np.unique(np.round(p, 2)))
    can_detect_breakout = unique_prices >= 5 and len(p) >= 10
    vw = vwap if vwap is not None else float(np.mean(p))
    vwap_dist = (last - vw) / vw if vw > 0 else 0.0

    def _mom(n: int) -> float:
        if len(p) <= n:
            return 0.0
        base = float(p[-n - 1])
        return (last - base) / base if base > 0 else 0.0

    m15, m30, m60 = _mom(min(15, len(p) - 1)), _mom(min(30, len(p) - 1)), _mom(min(60, len(p) - 1))
    rets = np.diff(p) / np.maximum(p[:-1], 1e-9)
    vol_recent = float(np.std(rets[-10:])) if len(rets) >= 10 else float(np.std(rets))
    vol_prior = float(np.std(rets[:-10])) if len(rets) > 20 else vol_recent
    vol_exp = vol_recent > vol_prior * 1.25

    support = float(np.min(p[-window_support:]))
    resistance = float(np.max(p[-window_support:]))
    breakout = "none"
    if can_detect_breakout:
        if last > resistance * 0.999 and m15 < 0:
            breakout = "fake_breakout"
        elif last > resistance * 0.999:
            breakout = "breakout"
        elif last < support * 1.001 and m15 > 0:
            breakout = "fake_breakout"
        elif last < support * 1.001:
            breakout = "breakout"

    return PriceActionFeatures(
        vwap_distance=vwap_dist,
        momentum_15s=m15,
        momentum_30s=m30,
        momentum_1m=m60,
        vol_expansion=vol_exp,
        support=support,
        resistance=resistance,
        breakout_signal=breakout,
    )


# ---------------------------------------------------------------------------
# Time-based features
# ---------------------------------------------------------------------------

@dataclass
class TimeFeatures:
    minutes_to_expiry: float
    day_of_week: int
    hour_utc: int
    session: str  # asia | europe | us
    historical_win_rate: float | None


def compute_time_features(
    close: datetime,
    *,
    now: datetime | None = None,
    win_rates_by_minute: dict[int, float] | None = None,
) -> TimeFeatures:
    now = now or datetime.now(timezone.utc)
    secs = max((close - now).total_seconds(), 0)
    mins = secs / 60.0
    dow = now.weekday()
    hour = now.hour
    if 0 <= hour < 8:
        session = "asia"
    elif 8 <= hour < 14:
        session = "europe"
    else:
        session = "us"
    bucket = int(mins)
    wr = (win_rates_by_minute or {}).get(bucket)
    return TimeFeatures(mins, dow, hour, session, wr)


# ---------------------------------------------------------------------------
# Multi-model ensemble agreement
# ---------------------------------------------------------------------------

@dataclass
class ModelVote:
    name: str
    probability: float
    weight: float


@dataclass
class EnsembleAgreement:
    votes: list[ModelVote]
    consensus_prob: float
    agreement_score: float  # 0..1
    models_agree: bool


def _logistic_prob(features: np.ndarray) -> float:
    w = np.array([0.35, 0.25, 0.15, 0.10, 0.15])
    z = float(np.dot(features[: len(w)], w))
    return 1.0 / (1.0 + math.exp(-z))


def _gbm_prob(spot: float, strike: float, vol: float, t_years: float) -> float:
    if spot <= 0 or strike <= 0 or vol <= 0 or t_years <= 0:
        return 0.5
    d2 = (math.log(spot / strike) - 0.5 * vol * vol * t_years) / (vol * math.sqrt(t_years))
    return float(norm.cdf(d2))


def multi_model_ensemble(
    *,
    spot: float,
    strike: float,
    vol: float,
    seconds_to_expiry: float,
    market_yes: float,
    micro: MicrostructureSnapshot,
    price_action: PriceActionFeatures,
    options_prob: float | None = None,
    max_disagreement_pp: float = 12.0,
) -> EnsembleAgreement:
    """Require agreement across gradient-boost proxy, logistic, NN proxy, and time-series."""
    t_y = max(seconds_to_expiry, 1.0) / (365.25 * 24 * 3600)
    feats = np.array([
        (spot - strike) / max(spot, 1.0),
        micro.bid_ask_imbalance,
        price_action.momentum_1m,
        micro.liquidity_score,
        market_yes - 0.5,
    ])
    votes = [
        ModelVote("gradient_boosting", _logistic_prob(feats * 2.5), 0.25),
        ModelVote("logistic_regression", _logistic_prob(feats), 0.20),
        ModelVote("neural_network", _logistic_prob(feats * 1.8 + 0.1), 0.20),
        ModelVote("time_series", _gbm_prob(spot, strike, vol, t_y), 0.20),
    ]
    if options_prob is not None:
        votes.append(ModelVote("options_implied", options_prob, 0.15))
    wsum = sum(v.weight for v in votes)
    consensus = sum(v.probability * v.weight for v in votes) / wsum
    probs = [v.probability for v in votes]
    spread_pp = (max(probs) - min(probs)) * 100 if len(probs) >= 2 else 0.0
    agreement = max(0.0, 1.0 - spread_pp / 30.0)
    agree = spread_pp <= max_disagreement_pp
    return EnsembleAgreement(votes, consensus, agreement, agree)


# ---------------------------------------------------------------------------
# Market quality / Do Not Trade score
# ---------------------------------------------------------------------------

@dataclass
class MarketQuality:
    do_not_trade_score: float  # 0=tradeable, 1=reject
    reasons: tuple[str, ...]
    tradeable: bool


def assess_market_quality(
    *,
    micro: MicrostructureSnapshot,
    price_action: PriceActionFeatures,
    ensemble: EnsembleAgreement,
    spread_limit: float = 0.08,
    min_liquidity: float = 0.15,
    manipulation_flag: bool = False,
) -> MarketQuality:
    score = 0.0
    reasons: list[str] = []
    if micro.spread > spread_limit:
        score += 0.25
        reasons.append("high_spread")
    if micro.liquidity_score < min_liquidity:
        score += 0.25
        reasons.append("low_liquidity")
    if price_action.breakout_signal == "fake_breakout":
        score += 0.20
        reasons.append("fake_breakout")
    if not ensemble.models_agree:
        score += 0.20
        reasons.append("model_disagreement")
    if manipulation_flag:
        score += 0.30
        reasons.append("manipulation_suspected")
    if micro.whale_detected and micro.whale_side == "ask":
        score += 0.10
        reasons.append("whale_ask_wall")
    score = min(1.0, score)
    return MarketQuality(score, tuple(reasons), tradeable=score < 0.55)


# ---------------------------------------------------------------------------
# Regime detection, manipulation, Monte Carlo
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    TREND = "trend"
    MEAN_REVERT = "mean_revert"
    CHOP = "chop"
    HIGH_VOL = "high_vol"


def detect_regime(price_action: PriceActionFeatures, micro: MicrostructureSnapshot) -> Regime:
    if price_action.vol_expansion and abs(price_action.momentum_1m) > 0.002:
        return Regime.HIGH_VOL
    if abs(price_action.momentum_1m) > 0.0015:
        return Regime.TREND
    if abs(price_action.momentum_15s) < 0.0003 and micro.spread_change > 0.01:
        return Regime.CHOP
    return Regime.MEAN_REVERT


def detect_manipulation(micro: MicrostructureSnapshot, price_action: PriceActionFeatures) -> bool:
    if micro.cancel_to_new_ratio > 0.8 and micro.trades_per_second < 0.1:
        return True
    if micro.whale_detected and price_action.breakout_signal == "fake_breakout":
        return True
    return False


def monte_carlo_binary(
    *,
    spot: float,
    strike: float,
    vol: float,
    seconds: float,
    n_sims: int = 5000,
    drift: float = 0.0,
) -> tuple[float, float, float]:
    """Return (mean prob YES, 5th pct, 95th pct) from GBM paths."""
    t = max(seconds, 1.0) / (365.25 * 24 * 3600)
    if vol <= 0 or spot <= 0:
        return 0.5, 0.5, 0.5
    rng = np.random.default_rng(int(time.time()) % (2**31))
    z = rng.standard_normal(n_sims)
    st = spot * np.exp((drift - 0.5 * vol * vol) * t + vol * math.sqrt(t) * z)
    above = st >= strike if strike > 0 else st > strike
    probs = above.astype(float)
    return float(np.mean(probs)), float(np.percentile(probs, 5)), float(np.percentile(probs, 95))


# ---------------------------------------------------------------------------
# Probability calibration (≥3 trades per bucket)
# ---------------------------------------------------------------------------

from kalshi_bot.strategy.probability_calibration import CalibrationBucket, ProbabilityCalibrator
from kalshi_bot.strategy.arbitrary_policy import EdgeChaseGuard


# ---------------------------------------------------------------------------
# Historical pattern matching + continuous learning
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    ticker: str
    side: str
    features: dict[str, Any]
    prediction: float
    confidence: float
    outcome: bool | None
    pnl: float
    reason: str
    ts: float = field(default_factory=time.time)


class TradeJournal:
    """SQLite-backed trade history for pattern matching and retraining."""

    def __init__(self, path: str = "data/v6_trade_journal.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT, side TEXT, features TEXT,
                    prediction REAL, confidence REAL,
                    outcome INTEGER, pnl REAL, reason TEXT, ts REAL
                )"""
            )

    def save(self, record: TradeRecord) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO trades (ticker, side, features, prediction, confidence, outcome, pnl, reason, ts) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.ticker,
                    record.side,
                    json.dumps(record.features),
                    record.prediction,
                    record.confidence,
                    int(record.outcome) if record.outcome is not None else None,
                    record.pnl,
                    record.reason,
                    record.ts,
                ),
            )

    def similar_setups(
        self,
        features: dict[str, Any],
        *,
        min_examples: int = 5,
        tolerance: float = 0.15,
    ) -> tuple[int, float | None]:
        """Count similar historical setups and their win rate."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT features, outcome FROM trades WHERE outcome IS NOT NULL"
            ).fetchall()
        if not rows:
            return 0, None
        target_imb = features.get("bid_ask_imbalance", 0)
        target_mom = features.get("momentum_1m", 0)
        matches: list[bool] = []
        for feat_json, outcome in rows:
            f = json.loads(feat_json)
            if abs(f.get("bid_ask_imbalance", 0) - target_imb) > tolerance:
                continue
            if abs(f.get("momentum_1m", 0) - target_mom) > tolerance:
                continue
            matches.append(bool(outcome))
        if len(matches) < min_examples:
            return len(matches), None
        return len(matches), sum(matches) / len(matches)


# ---------------------------------------------------------------------------
# Risk controls V6
# ---------------------------------------------------------------------------

@dataclass
class RiskStateV6:
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    exposure_usd: float = 0.0
    kill_switch: bool = False
    last_loss_ts: float = 0.0


class RiskControllerV6:
    def __init__(self, config: V6Config) -> None:
        self.config = config
        self.state = RiskStateV6()

    def record_outcome(self, pnl: float) -> None:
        self.state.daily_pnl += pnl
        if pnl < 0:
            self.state.consecutive_losses += 1
            self.state.last_loss_ts = time.time()
        else:
            self.state.consecutive_losses = 0
        if self.state.daily_pnl <= -self.config.max_daily_loss_usd:
            self.state.kill_switch = True
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self.state.kill_switch = True

    def allow_trade(self, confidence: float) -> tuple[bool, str]:
        if self.state.kill_switch:
            return False, "kill_switch_active"
        if self.state.exposure_usd >= self.config.max_exposure_usd:
            return False, "max_exposure"
        if time.time() - self.state.last_loss_ts < self.config.cooldown_after_loss_seconds:
            if self.state.consecutive_losses > 0:
                return False, "loss_cooldown"
        return True, "ok"

    def kelly_size(
        self,
        *,
        prob: float,
        price: float,
        bankroll: float,
        confidence: float,
    ) -> int:
        """Confidence-scaled fractional Kelly with 1-contract floor when affordable."""
        if price <= 0 or price >= 1 or prob <= price:
            return 0
        f_star = (prob - price) / (1.0 - price)
        frac = self.config.kelly_fraction * confidence
        dollars = bankroll * max(0.0, min(1.0, f_star * frac))
        dollars = min(dollars, self.config.max_position_usd)
        contracts = max(0, int(dollars / price))
        # With small bankrolls fractional Kelly rounds to 0 — still trade if we can afford 1.
        if contracts == 0 and bankroll >= price and dollars > 0:
            contracts = 1
        return min(contracts, int(bankroll / price))


# ---------------------------------------------------------------------------
# Signal weights + explainability
# ---------------------------------------------------------------------------

class SignalWeightLearner:
  """Online weight adjustment from trade outcomes."""

  def __init__(self) -> None:
      self.weights: dict[str, float] = {
          "microstructure": 0.20,
          "price_action": 0.20,
          "ensemble": 0.30,
          "monte_carlo": 0.15,
          "pattern_match": 0.15,
      }

  def update(self, signal_contributions: dict[str, float], won: bool) -> None:
      lr = 0.02
      direction = 1.0 if won else -1.0
      for k, v in signal_contributions.items():
          if k in self.weights:
              self.weights[k] = max(0.05, min(0.50, self.weights[k] + lr * direction * v))


def explainability_score(
    *,
    ensemble: EnsembleAgreement,
    quality: MarketQuality,
    pattern_support: int,
    strict_edge_gap: float,
) -> float:
    """0..1 how explainable / high-conviction the signal is."""
    s = ensemble.agreement_score * 0.35
    s += (1.0 - quality.do_not_trade_score) * 0.25
    s += min(1.0, pattern_support / 10.0) * 0.15
    s += min(1.0, strict_edge_gap / 0.30) * 0.25
    return round(min(1.0, s), 3)


# ---------------------------------------------------------------------------
# Strike gravity (15m open-level magnetism)
# ---------------------------------------------------------------------------

def strike_gravity_bias(
    spot: float,
    strike: float,
    seconds_to_expiry: float,
    momentum_1m: float,
) -> float:
    """Small probability adjustment toward strike as expiry nears."""
    if strike <= 0 or spot <= 0:
        return 0.0
    dist_pct = abs(spot - strike) / spot
    time_factor = max(0.0, 1.0 - seconds_to_expiry / 900.0)
    pull = -math.copysign(0.02 * time_factor * (1.0 - dist_pct * 50), spot - strike)
    pull += momentum_1m * 0.5
    return max(-0.05, min(0.05, pull))


# ---------------------------------------------------------------------------
# Institutional flow proxy
# ---------------------------------------------------------------------------

def institutional_flow_score(micro: MicrostructureSnapshot) -> float:
    """Proxy for informed flow from depth imbalance + trade velocity."""
    flow = micro.bid_ask_imbalance * 0.6
    if micro.trades_per_second > 0.5:
        flow *= 1.2
    if micro.whale_detected:
        flow += 0.15 if micro.whale_side == "bid" else -0.15
    return max(-1.0, min(1.0, flow))


# ---------------------------------------------------------------------------
# V6 decision engine
# ---------------------------------------------------------------------------

@dataclass
class V6Decision:
    verdict: str  # TRADE_YES | TRADE_NO | NO_TRADE
    model_probability: float
    market_price: float | None
    strict_gap_dollars: float
    confidence: float
    explainability: float
    regime: Regime
    monte_carlo_prob: float
    calibrated: bool
    pattern_examples: int
    pattern_win_rate: float | None
    quality: MarketQuality | None
    ensemble: EnsembleAgreement | None
    micro: MicrostructureSnapshot | None
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    contracts: int = 0
    audit_record: Any | None = None


class V6IntelligenceEngine:
    """Kalshi BTC 15-Min Intelligence V6 orchestrator."""

    def __init__(
        self,
        config: V6Config,
        client: KalshiClient | None = None,
        *,
        rules: Rules15mConfig | None = None,
    ) -> None:
        self.config = config
        self.rules = rules or load_rules_15m()
        self.client = client
        self.arbitrary_cfg = self.rules.arbitrary
        self.calibrator = ProbabilityCalibrator(config.min_trades_per_bucket)
        self.chase_guard = EdgeChaseGuard(ttl_seconds=self.arbitrary_cfg.chase_ttl_seconds)
        self.journal = TradeJournal(config.journal_path)
        self.risk = RiskControllerV6(config)
        self.weights = SignalWeightLearner()
        self._price_history: deque[float] = deque(maxlen=120)
        self._prev_spread: float | None = None
        self._prev_depth: tuple[float, float] | None = None
        self._monitor: Any | None = None  # OpportunityMonitor, lazy init

    def get_monitor(self) -> Any:
        from kalshi_bot.strategy.opportunity_monitor import OpportunityMonitor

        if self._monitor is None:
            self._monitor = OpportunityMonitor(self.config.diagnostics_db_path)
        return self._monitor

    def update_spot(self, price: float) -> None:
        self._price_history.append(price)

    def evaluate(
        self,
        market: dict,
        *,
        spot: float,
        vol: float,
        options_prob: float | None = None,
        now: datetime | None = None,
        spot_source: str = "unknown",
        spot_is_official: bool = False,
        record_diagnostics: bool = True,
    ) -> V6Decision:
        """Evaluate market; returns V6Decision backed by full audit record."""
        from kalshi_bot.strategy.v6_evaluator import evaluate_market_audited

        audit = evaluate_market_audited(
            self,
            market,
            spot=spot,
            spot_source=spot_source,
            spot_is_official=spot_is_official,
            vol=vol,
            options_prob=options_prob,
            now=now,
        )
        if record_diagnostics:
            self.get_monitor().record(audit)

        return V6Decision(
            verdict=audit.verdict,
            model_probability=audit.model_prob_up,
            market_price=(
                audit.yes_side.executable_ask
                if audit.verdict == "TRADE_YES"
                else audit.no_side.executable_ask
                if audit.verdict == "TRADE_NO"
                else None
            ),
            strict_gap_dollars=max(
                audit.yes_side.raw_edge_dollars, audit.no_side.raw_edge_dollars
            ),
            confidence=audit.model_confidence,
            explainability=audit.explainability,
            regime=Regime(audit.regime) if audit.regime in {r.value for r in Regime} else Regime.CHOP,
            monte_carlo_prob=audit.monte_carlo_prob,
            calibrated=audit.calibrated,
            pattern_examples=0,
            pattern_win_rate=None,
            quality=None,
            ensemble=None,
            micro=None,
            reasons=(
                f"{audit.edge_action}",
                f"tier={audit.edge_quality}",
                f"score={audit.opportunity_score}",
                f"best_side={audit.best_side}",
                f"best_net={audit.best_net_edge*100:.1f}¢",
            ),
            blockers=tuple(
                c.value for c in audit.all_rejection_codes if c != RejectionCode.NONE
            ),
            contracts=audit.contracts,
            audit_record=audit,
        )

    def evaluate_legacy(
        self,
        market: dict,
        *,
        spot: float,
        vol: float,
        options_prob: float | None = None,
        now: datetime | None = None,
    ) -> V6Decision:
        """Legacy evaluate path (pre-audit). Prefer evaluate()."""
        now = now or datetime.now(timezone.utc)
        self.update_spot(spot)
        ticker = str(market.get("ticker") or "")
        strike = float(market.get("strike") or spot)
        close = market.get("close_time")
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        no_ask = market.get("no_ask")
        if no_ask is None and yes_bid is not None:
            no_ask = max(0.0, 1.0 - yes_bid)

        secs = max((close - now).total_seconds(), 0) if close else 0
        orderbook = None
        if self.client and ticker:
            try:
                orderbook = self.client.get_orderbook(ticker, depth=10)
            except Exception:
                pass

        micro = compute_microstructure(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            orderbook=orderbook,
            prev_spread=self._prev_spread,
            prev_depth=self._prev_depth,
        )
        self._prev_spread = micro.spread
        self._prev_depth = (micro.depth_bid_10, micro.depth_ask_10)

        pa = compute_price_action(list(self._price_history))
        time_feat = compute_time_features(close, now=now) if close else None
        regime = detect_regime(pa, micro)
        manip = detect_manipulation(micro, pa)

        ensemble = multi_model_ensemble(
            spot=spot,
            strike=strike,
            vol=vol,
            seconds_to_expiry=secs,
            market_yes=yes_ask or 0.5,
            micro=micro,
            price_action=pa,
            options_prob=options_prob,
            max_disagreement_pp=self.config.max_model_disagreement_pp,
        )

        mc_mean, _, _ = monte_carlo_binary(
            spot=spot,
            strike=strike,
            vol=vol,
            seconds=secs,
            n_sims=self.config.monte_carlo_sims,
        )

        gravity = strike_gravity_bias(spot, strike, secs, pa.momentum_1m)
        inst_flow = institutional_flow_score(micro)
        raw_prob = (
            ensemble.consensus_prob * self.weights.weights["ensemble"]
            + mc_mean * self.weights.weights["monte_carlo"]
            + (options_prob or ensemble.consensus_prob) * 0.1
            + gravity
            + inst_flow * 0.03
        )
        raw_prob = max(0.01, min(0.99, raw_prob))
        model_prob, calibrated = self.calibrator.calibrate(raw_prob)

        quality = assess_market_quality(
            micro=micro,
            price_action=pa,
            ensemble=ensemble,
            spread_limit=self.config.max_spread,
            min_liquidity=self.config.min_liquidity_score,
            manipulation_flag=manip,
        )

        feat_dict = {
            "bid_ask_imbalance": micro.bid_ask_imbalance,
            "momentum_1m": pa.momentum_1m,
            "liquidity_score": micro.liquidity_score,
            "regime": regime.value,
        }
        pattern_n, pattern_wr = self.journal.similar_setups(
            feat_dict, min_examples=self.config.min_pattern_examples
        )

        reasons: list[str] = []
        blockers: list[str] = []
        verdict = "NO_TRADE"
        side_price: float | None = None
        strict_gap = 0.0

        allow, risk_reason = self.risk.allow_trade(ensemble.agreement_score)
        if not allow:
            blockers.append(risk_reason)
        if not quality.tradeable:
            blockers.append(f"do_not_trade_score={quality.do_not_trade_score:.2f}")
            blockers.extend(quality.reasons)
        if not ensemble.models_agree:
            blockers.append("multi_model_disagreement")
        if pattern_n < self.config.min_pattern_examples and self.config.require_pattern_evidence:
            blockers.append(f"insufficient_pattern_evidence ({pattern_n}<{self.config.min_pattern_examples})")

        min_gap = self.config.strict_min_gap_dollars

        if yes_ask is not None and 0 < yes_ask < 1:
            ok, gap = passes_strict_edge(model_prob, yes_ask, min_gap_dollars=min_gap)
            strict_gap = gap
            if ok and not blockers:
                fee = quadratic_fee_per_contract(yes_ask)
                if model_prob - yes_ask - fee > 0:
                    verdict = "TRADE_YES"
                    side_price = yes_ask
                    reasons.append(
                        f"STRICT EDGE: model={model_prob*100:.0f}% market={yes_ask*100:.0f}¢ "
                        f"gap={gap*100:.0f}¢ (min {min_gap*100:.0f}¢)"
                    )
            elif not ok:
                blockers.append(
                    f"strict_edge_fail: gap={gap*100:.0f}¢ < min {min_gap*100:.0f}¢ "
                    f"(model={model_prob*100:.0f}% vs ask={yes_ask*100:.0f}¢)"
                )

        if verdict == "NO_TRADE" and no_ask is not None and 0 < no_ask < 1:
            q = 1.0 - model_prob
            ok, gap = passes_strict_edge(q, no_ask, min_gap_dollars=min_gap)
            if ok and not blockers:
                fee = quadratic_fee_per_contract(no_ask)
                if q - no_ask - fee > 0:
                    verdict = "TRADE_NO"
                    side_price = no_ask
                    strict_gap = gap
                    reasons.append(
                        f"STRICT EDGE NO: model={q*100:.0f}% market={no_ask*100:.0f}¢ gap={gap*100:.0f}¢"
                    )
            elif not ok and "strict_edge_fail" not in " ".join(blockers):
                blockers.append(
                    f"strict_edge_fail NO: gap={gap*100:.0f}¢ < min {min_gap*100:.0f}¢"
                )

        explain = explainability_score(
            ensemble=ensemble,
            quality=quality,
            pattern_support=pattern_n,
            strict_edge_gap=strict_gap,
        )

        contracts = 0
        if verdict != "NO_TRADE" and side_price:
            contracts = self.risk.kelly_size(
                prob=model_prob if verdict == "TRADE_YES" else 1.0 - model_prob,
                price=side_price,
                bankroll=self.config.bankroll_usd,
                confidence=explain,
            )

        if time_feat and time_feat.historical_win_rate is not None:
            reasons.append(f"historical_wr@{time_feat.minutes_to_expiry:.0f}m={time_feat.historical_win_rate:.2f}")

        return V6Decision(
            verdict=verdict,
            model_probability=model_prob,
            market_price=side_price,
            strict_gap_dollars=strict_gap,
            confidence=ensemble.agreement_score,
            explainability=explain,
            regime=regime,
            monte_carlo_prob=mc_mean,
            calibrated=calibrated,
            pattern_examples=pattern_n,
            pattern_win_rate=pattern_wr,
            quality=quality,
            ensemble=ensemble,
            micro=micro,
            reasons=tuple(reasons),
            blockers=tuple(dict.fromkeys(blockers)),
            contracts=contracts,
        )

    def record_trade(
        self,
        decision: V6Decision,
        *,
        ticker: str,
        side: str,
        won: bool | None,
        pnl: float,
    ) -> None:
        self.journal.save(
            TradeRecord(
                ticker=ticker,
                side=side,
                features={
                    "bid_ask_imbalance": decision.micro.bid_ask_imbalance,
                    "momentum_1m": 0.0,
                    "liquidity_score": decision.micro.liquidity_score,
                },
                prediction=decision.model_probability,
                confidence=decision.confidence,
                outcome=won,
                pnl=pnl,
                reason="; ".join(decision.reasons),
            )
        )
        if won is not None:
            self.calibrator.record(decision.model_probability, won)
            self.risk.record_outcome(pnl)
            self.weights.update(
                {"ensemble": decision.confidence, "monte_carlo": decision.monte_carlo_prob},
                won=won,
            )
