"""Forecast ensemble — combines 5-layer model votes into a single probability."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.data_feed import MarketData
from kalshi_btc_1hr_bot.ensemble import EnsembleResult, ModelVote, combine_models
from kalshi_btc_1hr_bot.model import (
    ForecastModel,
    MarketState,
    ModelOutput,
    build_market_state,
)

log = logging.getLogger("forecast")


@dataclass
class ForecastEnsembleOutput:
    """Final ensemble forecast with underlying model detail."""

    p_fair: float
    confidence: float
    agreement_score: float
    uncertainty: float
    model_output: ModelOutput
    ensemble: EnsembleResult
    is_official_brti: bool

    @property
    def vol_regime(self) -> str:
        return self.model_output.vol_regime

    @property
    def layers(self) -> list[tuple[str, float]]:
        base = list(self.model_output.layers)
        base.append(("ensemble", self.p_fair))
        return base

    @property
    def votes(self) -> tuple[ModelVote, ...]:
        return self.ensemble.votes


class ForecastEnsemble:
    """Wraps the 5-layer model and combines multiple forecast votes."""

    def __init__(self) -> None:
        self.model = ForecastModel()

    def forecast(self, state: MarketState) -> ForecastEnsembleOutput:
        output = self.model.forecast(state)
        votes = self._build_votes(output, state)
        ensemble = combine_models(votes)

        confidence = ensemble.confidence
        if not state.is_official_brti:
            confidence *= config.PROXY_BRTI_CONFIDENCE_PENALTY

        # Require minimum agreement before trusting the ensemble
        if ensemble.agreement_score < config.ENSEMBLE_MIN_AGREEMENT:
            confidence *= ensemble.agreement_score

        p_fair = ensemble.prob_yes

        return ForecastEnsembleOutput(
            p_fair=p_fair,
            confidence=max(0.0, min(1.0, confidence)),
            agreement_score=ensemble.agreement_score,
            uncertainty=ensemble.uncertainty,
            model_output=output,
            ensemble=ensemble,
            is_official_brti=state.is_official_brti,
        )

    def _build_votes(self, output: ModelOutput, state: MarketState) -> list[ModelVote]:
        w = config.ENSEMBLE_WEIGHTS
        base_conf = output.confidence

        # OBI microstructure tilt: imbalance pushes prob slightly toward bid side
        obi_prob = max(0.001, min(0.999, 0.5 + output.obi * 0.12))

        votes = [
            ModelVote("five_layer", output.p_calibrated, w["five_layer"], base_conf),
            ModelVote("gbm_core", output.p_gbm, w["gbm_core"], base_conf * 0.85),
            ModelVote("momentum", output.p_momentum, w["momentum"], base_conf * 0.90),
            ModelVote("mean_reversion", output.p_mean_rev, w["mean_reversion"], base_conf * 0.85),
            ModelVote("funding", output.p_funding, w["funding"], base_conf * 0.80),
            ModelVote("obi_micro", obi_prob, w["obi"], base_conf * 0.70),
        ]

        if not state.is_official_brti:
            # Down-weight settlement-calibrated layer when spot is a proxy
            votes = [
                ModelVote(v.name, v.prob_yes, v.weight * (0.6 if v.name == "five_layer" else 1.0), v.confidence)
                for v in votes
            ]

        return votes


def forecast_ensemble_from_market_data(
    ensemble: ForecastEnsemble,
    *,
    spot: float,
    strike: float,
    seconds_to_expiry: float,
    data: MarketData,
    ob_bids: list | None = None,
    ob_asks: list | None = None,
) -> ForecastEnsembleOutput:
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
        is_official_brti=data.is_official,
    )
    return ensemble.forecast(state)
