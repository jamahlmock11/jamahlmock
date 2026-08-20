"""5-layer ensemble probability model for KXBTCD hourly contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kalshi_btc_1hr_bot.config import BotConfig, ModelConfig
from kalshi_btc_1hr_bot.data_feed import MarketData
from kalshi_btc_1hr_bot.utils import (
    clamp_prob,
    effective_vol_for_averaging,
    gbm_prob_above,
    sigmoid,
    years_to_expiry,
)


@dataclass
class LayerOutput:
    name: str
    probability: float
    weight: float
    detail: str = ""


@dataclass
class ForecastResult:
    p_fair: float
    p_raw: float
    confidence: float
    vol_regime: str
    layers: list[LayerOutput] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)


class LogisticCalibrator:
    """Simple logistic calibrator on 18 features (identity-ish weights for demo)."""

    def __init__(self) -> None:
        self.bias = 0.0
        self.weights = np.array([
            1.2, 0.8, 0.5, 0.3, 0.2,  # layer probs + drift
            0.4, 0.3, 0.2, 0.15, 0.1,  # momentum + funding
            0.25, 0.2, 0.15, 0.1, 0.05,  # MR + vol
            0.3, 0.2, 0.1,  # time + moneyness
        ])

    def calibrate(self, features: np.ndarray) -> float:
        z = self.bias + float(np.dot(self.weights[: len(features)], features))
        return clamp_prob(sigmoid(z))


class HourlyForecastModel:
    """5-layer ensemble: GBM, momentum, funding, mean reversion, vol regime."""

    def __init__(self, config: BotConfig | None = None) -> None:
        self.config = config or BotConfig()
        self.model_cfg: ModelConfig = self.config.model
        self.calibrator = LogisticCalibrator()

    def forecast(
        self,
        *,
        spot: float,
        strike: float,
        seconds_to_expiry: float,
        data: MarketData,
    ) -> ForecastResult:
        mc = self.model_cfg
        t_years = max(seconds_to_expiry, 1.0) / (365.25 * 24 * 3600)
        t_frac = min(1.0, seconds_to_expiry / self.config.window_seconds)

        # Layer 1: GBM core with averaging adjustment
        sigma_eff = effective_vol_for_averaging(
            max(data.annualized_vol, mc.min_vol),
            mc.averaging_window_seconds,
        )
        p_gbm = gbm_prob_above(spot, strike, t_years, sigma_eff, drift=0.0)
        layer1 = LayerOutput("gbm_core", p_gbm, 0.35, f"σ_eff={sigma_eff:.3f}")

        # Layer 2: Multi-timeframe momentum drift blend
        mu_blend = (
            mc.momentum_w_5m * data.mu_5m
            + mc.momentum_w_15m * data.mu_15m
            + mc.momentum_w_30m * data.mu_30m
        )
        p_mom = gbm_prob_above(spot, strike, t_years, sigma_eff, drift=mu_blend * 365.25 * 24 * 3600)
        layer2 = LayerOutput("momentum", p_mom, 0.25, f"μ_blend={mu_blend:.6f}")

        # Layer 3: Funding rate signal
        funding = data.funding_rate
        funding_adj = self._funding_adjustment(funding, t_frac)
        p_fund = clamp_prob(p_gbm + funding_adj)
        layer3 = LayerOutput("funding", p_fund, 0.15, f"rate={funding:.5f}")

        # Layer 4: Mean reversion toward VWAP
        if data.vwap > 0:
            mr_pull = -mc.mr_coefficient * (spot - data.vwap) / data.vwap * 100 * t_frac
            mr_pull = mr_pull / 100.0  # convert to probability units
        else:
            mr_pull = 0.0
        p_mr = clamp_prob(p_gbm + mr_pull)
        layer4 = LayerOutput("mean_reversion", p_mr, 0.15, f"pull={mr_pull:.4f}")

        # Layer 5: Volatility regime
        regime, regime_weight = self._vol_regime(data.annualized_vol)
        layers = [layer1, layer2, layer3, layer4]
        wsum = sum(l.weight for l in layers)
        p_blend = sum(l.probability * l.weight for l in layers) / wsum
        p_regime = clamp_prob(0.5 + regime_weight * (p_blend - 0.5))
        layer5 = LayerOutput("vol_regime", p_regime, 0.10, f"regime={regime}")
        layers.append(layer5)

        # Feature vector for logistic calibration (18 features)
        moneyness = math.log(spot / strike) if strike > 0 else 0.0
        features = np.array([
            p_gbm, p_mom, p_fund, p_mr, p_regime,
            mu_blend, data.mu_5m, data.mu_15m, data.mu_30m, funding,
            mr_pull, sigma_eff, data.annualized_vol, regime_weight, t_frac,
            moneyness, spot / strike if strike > 0 else 1.0, seconds_to_expiry / 3600.0,
        ], dtype=float)

        p_calibrated = self.calibrator.calibrate(features)
        confidence = self._confidence(layers, data.annualized_vol, regime)

        return ForecastResult(
            p_fair=p_calibrated,
            p_raw=p_blend,
            confidence=confidence,
            vol_regime=regime,
            layers=layers,
            features={f"f{i}": float(v) for i, v in enumerate(features)},
        )

    def _funding_adjustment(self, funding: float, t_frac: float) -> float:
        mc = self.model_cfg
        # Extreme funding → contrarian
        if abs(funding) > mc.funding_extreme_threshold:
            direction = -1.0 if funding > 0 else 1.0
            magnitude = min(abs(funding) / mc.funding_extreme_threshold, 2.0)
            return direction * mc.funding_weight * magnitude * (1.0 - t_frac)
        # Normal funding → sentiment
        return funding * mc.funding_weight * 100 * (1.0 - t_frac)

    def _vol_regime(self, vol: float) -> tuple[str, float]:
        mc = self.model_cfg
        if vol < mc.vol_low_threshold:
            return "low", mc.vol_low_weight
        if vol > mc.vol_high_threshold:
            return "high", mc.vol_high_weight
        return "medium", mc.vol_med_weight

    @staticmethod
    def _confidence(layers: list[LayerOutput], vol: float, regime: str) -> float:
        probs = [l.probability for l in layers]
        spread = max(probs) - min(probs)
        score = 1.0 - min(spread * 2.0, 0.5)
        if regime == "high":
            score *= 0.75
        elif regime == "medium":
            score *= 0.90
        return max(0.0, min(1.0, score))
