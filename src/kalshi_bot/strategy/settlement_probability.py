"""Settlement probability engine for KXBTC15M (finish above/below strike)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from kalshi_bot.data.btc_data_engine import BtcMarketSnapshot
from kalshi_bot.strategy.probability_calibration import ProbabilityCalibrator
from kalshi_bot.strategy.v6_upgrades import monte_carlo_binary


@dataclass(frozen=True)
class SettlementProbability:
    prob_above_strike: float
    prob_below_strike: float
    raw_prob: float
    calibrated: bool
    gbm_prob: float
    monte_carlo_prob: float
    momentum_adjustment: float
    confidence: float
    disagreement_pp: float


def _gbm_digital(spot: float, strike: float, vol: float, seconds: float) -> float:
    if spot <= 0 or strike <= 0 or vol <= 0 or seconds <= 0:
        return 0.5
    t = max(seconds, 1.0) / (365.25 * 24 * 3600)
    d2 = (math.log(spot / strike) - 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    return float(norm.cdf(d2))


def estimate_settlement_probability(
    *,
    spot: float,
    strike: float,
    seconds_to_expiry: float,
    annualized_vol: float,
    btc: BtcMarketSnapshot,
    options_prob: float | None = None,
    calibrator: ProbabilityCalibrator | None = None,
    monte_carlo_sims: int = 3000,
) -> SettlementProbability:
    """Probability BRTI settles >= strike at expiry."""
    gbm = _gbm_digital(spot, strike, annualized_vol, seconds_to_expiry)
    mc_mean, _, _ = monte_carlo_binary(
        spot=spot,
        strike=strike,
        vol=annualized_vol,
        seconds=seconds_to_expiry,
        n_sims=monte_carlo_sims,
    )

    # Short-horizon momentum tilt (mean-reverts in final minute)
    mins = seconds_to_expiry / 60.0
    mom_weight = 0.08 if mins > 3 else 0.03
    mom_adj = btc.momentum_1m * mom_weight
    if options_prob is not None:
        raw = 0.40 * gbm + 0.35 * mc_mean + 0.15 * options_prob + 0.10 * (0.5 + mom_adj)
        disagreement = abs(gbm - options_prob) * 100
    else:
        raw = 0.55 * gbm + 0.35 * mc_mean + 0.10 * (0.5 + mom_adj)
        disagreement = abs(gbm - mc_mean) * 100

    raw = max(0.01, min(0.99, raw + mom_adj))
    calibrated_flag = False
    prob = raw
    if calibrator is not None:
        prob, calibrated_flag = calibrator.calibrate(raw)

    # Confidence from feed agreement and model spread
    conf = 0.55
    conf += 0.25 * btc.cross_exchange_agreement
    conf += 0.10 * (1.0 - min(disagreement / 25.0, 1.0))
    if btc.is_official:
        conf += 0.10
    conf = max(0.0, min(1.0, conf))

    return SettlementProbability(
        prob_above_strike=prob,
        prob_below_strike=1.0 - prob,
        raw_prob=raw,
        calibrated=calibrated_flag,
        gbm_prob=gbm,
        monte_carlo_prob=mc_mean,
        momentum_adjustment=mom_adj,
        confidence=conf,
        disagreement_pp=disagreement,
    )
