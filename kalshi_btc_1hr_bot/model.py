"""Probability model for the 1-HOUR Kalshi KXBTCD bot.

Ensemble approach with 5 signal layers, each contributing to the final
probability forecast. The longer 1-hour window allows richer signals than
the 15-minute version:

1. GBM core: lognormal probability with averaging adjustment (same as 15m)
2. Multi-timeframe momentum: weighted blend of 5m/15m/30m drift
3. Volatility regime: classify low/med/high vol → adjust confidence
4. Funding rate signal: BTC perp funding as sentiment proxy
5. Mean reversion: BTC shows some MR over 1hr — pullback toward VWAP

The final probability is calibrated via logistic regression on backtest data.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from pathlib import Path

import numpy as np

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.data_feed import FundingRate, MarketData

log = logging.getLogger("model")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class MarketState:
    current_price: float
    strike: float
    seconds_remaining: float
    seconds_to_avg_start: float
    price_history: list  # [(timestamp, price), ...]
    ob_bids: list
    ob_asks: list
    now_ts: float = 0.0
    funding: Optional[FundingRate] = None
    vwap: float = 0.0  # volume-weighted avg price (0 = not available)


@dataclass
class ModelOutput:
    p_above: float
    p_below: float
    p_gbm: float  # layer 1: raw GBM
    p_momentum: float  # layer 2: momentum-adjusted
    p_funding: float  # layer 3: funding-adjusted
    p_mean_rev: float  # layer 4: mean-reversion adjusted
    p_calibrated: float  # layer 5: final calibrated
    sigma: float
    mu: float
    obi: float
    funding_rate: float
    vol_regime: str  # "low", "medium", "high"
    distance_bps: float
    features: np.ndarray

    @property
    def p_fair(self) -> float:
        return self.p_calibrated

    @property
    def confidence(self) -> float:
        return {"low": 1.0, "medium": 0.85, "high": 0.65}.get(self.vol_regime, 0.85)

    @property
    def layers(self) -> list[tuple[str, float]]:
        return [
            ("gbm_core", self.p_gbm),
            ("momentum", self.p_momentum),
            ("funding", self.p_funding),
            ("mean_reversion", self.p_mean_rev),
            ("calibrated", self.p_calibrated),
        ]


class ForecastModel:
    """Ensemble probability model for hourly BTC settlement."""

    def __init__(self) -> None:
        self.calibrator = None
        self._load_calibrator()

    def _load_calibrator(self) -> None:
        try:
            import joblib

            self.calibrator = joblib.load(config.CALIB_MODEL_PATH)
            log.info("Calibration model loaded")
        except Exception:
            log.info("No calibration model — using raw ensemble")

    def forecast(self, state: MarketState) -> ModelOutput:
        S = state.current_price
        K = state.strike
        T = max(state.seconds_to_avg_start, 1.0) / config.ANNUALIZE_SECONDS
        T_total = max(state.seconds_remaining, 1.0) / config.ANNUALIZE_SECONDS
        now_ts = state.now_ts if state.now_ts else time.time()

        # ── LAYER 1: GBM core ────────────────────────────────────────────────
        sigma = self._realized_vol(state.price_history, now_ts=now_ts)
        mu_gbm = self._momentum(
            state.price_history,
            lookback=config.MOMENTUM_LOOKBACK_SECONDS,
            now_ts=now_ts,
        )

        avg_factor = self._averaging_factor(
            seconds_to_avg_start=state.seconds_to_avg_start,
            avg_window=config.SETTLE_AVG_SECONDS,
            seconds_remaining=state.seconds_remaining,
        )
        sigma_eff = sigma * math.sqrt(avg_factor)

        if sigma_eff > 0 and S > 0 and K > 0:
            T_mid = T + (config.SETTLE_AVG_SECONDS / 2) / config.ANNUALIZE_SECONDS
            d2 = (math.log(S / K) + (mu_gbm - 0.5 * sigma_eff**2) * T_mid) / (
                sigma_eff * math.sqrt(T_mid)
            )
            p_gbm = norm_cdf(d2)
        else:
            p_gbm = 0.5
        p_gbm = max(0.001, min(0.999, p_gbm))

        # ── LAYER 2: Multi-timeframe momentum ───────────────────────────────
        mu_5m = self._momentum(state.price_history, lookback=300, now_ts=now_ts)
        mu_15m = self._momentum(state.price_history, lookback=900, now_ts=now_ts)
        mu_30m = self._momentum(state.price_history, lookback=1800, now_ts=now_ts)
        mu_blend = (
            mu_5m * config.MOMENTUM_WEIGHTS[0]
            + mu_15m * config.MOMENTUM_WEIGHTS[1]
            + mu_30m * config.MOMENTUM_WEIGHTS[2]
        )
        if sigma_eff > 0 and S > 0 and K > 0:
            d2_mom = (math.log(S / K) + (mu_blend - 0.5 * sigma_eff**2) * T_mid) / (
                sigma_eff * math.sqrt(T_mid)
            )
            p_momentum = norm_cdf(d2_mom)
        else:
            p_momentum = 0.5
        p_momentum = max(0.001, min(0.999, p_momentum))

        # ── LAYER 3: Funding rate signal ─────────────────────────────────────
        funding_rate = 0.0
        p_funding = p_momentum
        if state.funding:
            funding_rate = state.funding.funding_rate
            if abs(funding_rate) < 0.0005:
                funding_signal = funding_rate * 5000
            else:
                funding_signal = -math.copysign(0.2, funding_rate)
            funding_adj = config.FUNDING_SIGNAL_WEIGHT * funding_signal * min(1.0, T_total * 10)
            p_funding = max(0.001, min(0.999, p_momentum + funding_adj))

        # ── LAYER 4: Mean reversion ─────────────────────────────────────────
        p_mean_rev = p_funding
        if state.vwap > 0 and S > 0:
            vwap_dist = (S - state.vwap) / state.vwap
            mr_pull = -config.MEAN_REVERSION_STRENGTH * vwap_dist * 100 * min(1.0, T_total)
            p_mean_rev = max(0.001, min(0.999, p_funding + mr_pull))

        # ── Volatility regime classification ────────────────────────────────
        if sigma < config.VOL_REGIME_LOW:
            vol_regime = "low"
        elif sigma > config.VOL_REGIME_HIGH:
            vol_regime = "high"
        else:
            vol_regime = "medium"

        vol_confidence = {"low": 1.0, "medium": 0.85, "high": 0.65}[vol_regime]
        p_ensemble = 0.5 + (p_mean_rev - 0.5) * vol_confidence
        p_ensemble = max(0.001, min(0.999, p_ensemble))

        # ── LAYER 5: Calibration ────────────────────────────────────────────
        obi = self._order_book_imbalance(state.ob_bids, state.ob_asks)
        distance_bps = abs(S - K) / K * 10_000 if K > 0 else 0

        features = np.array([
            p_gbm,
            p_momentum,
            p_funding,
            p_mean_rev,
            p_ensemble,
            obi,
            mu_blend,
            mu_5m,
            mu_15m,
            mu_30m,
            sigma,
            funding_rate,
            float(vol_regime == "low"),
            float(vol_regime == "high"),
            distance_bps,
            state.seconds_remaining / config.WINDOW_SECONDS,
            avg_factor,
            (S - state.vwap) / state.vwap if state.vwap > 0 else 0.0,
        ])

        if self.calibrator is not None:
            try:
                p_calibrated = float(self.calibrator.predict_proba(features.reshape(1, -1))[0, 1])
            except Exception:
                p_calibrated = p_ensemble
        else:
            p_calibrated = p_ensemble

        p_calibrated = max(0.001, min(0.999, p_calibrated))

        return ModelOutput(
            p_above=p_calibrated,
            p_below=1 - p_calibrated,
            p_gbm=p_gbm,
            p_momentum=p_momentum,
            p_funding=p_funding,
            p_mean_rev=p_mean_rev,
            p_calibrated=p_calibrated,
            sigma=sigma,
            mu=mu_blend,
            obi=obi,
            funding_rate=funding_rate,
            vol_regime=vol_regime,
            distance_bps=distance_bps,
            features=features,
        )

    def _realized_vol(
        self,
        price_history: list,
        lookback: int = config.VOL_LOOKBACK_SECONDS,
        now_ts: float = 0.0,
    ) -> float:
        if now_ts == 0:
            now_ts = time.time()
        if len(price_history) < 10:
            return 0.5
        cutoff = now_ts - lookback
        prices = [(t, p) for t, p in price_history if t >= cutoff]
        if len(prices) < 10:
            return 0.5
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i][1] > 0 and prices[i - 1][1] > 0:
                log_returns.append(math.log(prices[i][1] / prices[i - 1][1]))
        if len(log_returns) < 5:
            return 0.5
        arr = np.array(log_returns)
        vol = float(np.std(arr, ddof=1)) * math.sqrt(config.ANNUALIZE_SECONDS)
        return max(0.1, min(3.0, vol))

    def _momentum(self, price_history: list, lookback: int, now_ts: float = 0.0) -> float:
        if now_ts == 0:
            now_ts = time.time()
        if len(price_history) < 5:
            return 0.0
        cutoff = now_ts - lookback
        prices = [(t, p) for t, p in price_history if t >= cutoff]
        if len(prices) < 5:
            return 0.0
        dt = prices[-1][0] - prices[0][0]
        if dt <= 0 or prices[0][1] <= 0:
            return 0.0
        log_ret = math.log(prices[-1][1] / prices[0][1])
        mu = (log_ret / dt) * config.ANNUALIZE_SECONDS
        return max(-5.0, min(5.0, mu))

    def _order_book_imbalance(self, bids: list, asks: list, levels: int = config.OBI_LEVELS) -> float:
        bid_vol = sum(b[1] for b in bids[:levels]) if bids else 0
        ask_vol = sum(a[1] for a in asks[:levels]) if asks else 0
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def _averaging_factor(
        self,
        seconds_to_avg_start: float,
        avg_window: float,
        seconds_remaining: float,
    ) -> float:
        """60-sec averaging reduces effective vol (same as 15m bot)."""
        if seconds_to_avg_start > 0:
            return min(1.0, math.sqrt(avg_window / 12) / math.sqrt(avg_window) + 0.85)
        elapsed_in_avg = avg_window - seconds_remaining
        if elapsed_in_avg <= 0:
            return 0.05
        frac_remaining = max(0.05, seconds_remaining / avg_window)
        return frac_remaining


class CalibratedPipeline:
    """Wraps StandardScaler + LogisticRegression. Must be module-level for joblib."""

    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


def train_calibrator(features: np.ndarray, labels: np.ndarray) -> object:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(X, labels)

    calibrator = CalibratedPipeline(scaler, model)
    Path(config.CALIB_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, config.CALIB_MODEL_PATH)
    log.info("Calibration model saved to %s", config.CALIB_MODEL_PATH)
    return calibrator


def seconds_to_avg_start(seconds_remaining: float) -> float:
    if seconds_remaining > config.SETTLE_AVG_SECONDS:
        return seconds_remaining - config.SETTLE_AVG_SECONDS
    return 0.0


def build_market_state(
    *,
    spot: float,
    strike: float,
    seconds_remaining: float,
    price_history: list,
    vwap: float = 0.0,
    funding: FundingRate | None = None,
    ob_bids: list | None = None,
    ob_asks: list | None = None,
    now_ts: float | None = None,
) -> MarketState:
    return MarketState(
        current_price=spot,
        strike=strike,
        seconds_remaining=seconds_remaining,
        seconds_to_avg_start=seconds_to_avg_start(seconds_remaining),
        price_history=price_history,
        ob_bids=ob_bids or [],
        ob_asks=ob_asks or [],
        now_ts=now_ts or time.time(),
        funding=funding,
        vwap=vwap,
    )


def forecast_from_market_data(
    model: ForecastModel,
    *,
    spot: float,
    strike: float,
    seconds_to_expiry: float,
    data: MarketData,
    ob_bids: list | None = None,
    ob_asks: list | None = None,
) -> ModelOutput:
    state = build_market_state(
        spot=spot,
        strike=strike,
        seconds_remaining=seconds_to_expiry,
        price_history=data.price_history,
        vwap=data.vwap,
        funding=data.funding,
        ob_bids=ob_bids,
        ob_asks=ob_asks,
        now_ts=data.timestamp,
    )
    return model.forecast(state)


# Backwards-compatible alias
HourlyForecastModel = ForecastModel
