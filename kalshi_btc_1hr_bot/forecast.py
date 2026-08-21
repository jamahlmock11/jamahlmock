"""Forecast ensemble — crowd system wraps 5-layer model + lens voters."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdForecastSystem, crowd_summary
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
    """Final crowd + ensemble forecast with underlying model detail."""

    p_fair: float
    confidence: float
    agreement_score: float
    uncertainty: float
    model_output: ModelOutput
    ensemble: EnsembleResult
    crowd: CrowdForecast
    is_official_brti: bool

    @property
    def vol_regime(self) -> str:
        return self.model_output.vol_regime

    @property
    def layers(self) -> list[tuple[str, float]]:
        base = list(self.model_output.layers)
        base.append(("crowd", self.crowd.prob_yes))
        base.append(("ensemble", self.p_fair))
        return base

    @property
    def votes(self) -> tuple[ModelVote, ...]:
        return self.crowd.top_votes  # top crowd voters drive evidence

    @property
    def quorum_met(self) -> bool:
        return self.crowd.quorum_met

    @property
    def favorite_met(self) -> bool:
        return self.crowd.favorite_met

    def favorite_met_for_side(self, side: str, *, min_favorite: float) -> bool:
        return self.crowd.side_met(side, min_favorite=min_favorite)

    @property
    def crowd_summary(self) -> dict:
        return crowd_summary(self.crowd)

    def crowd_summary_at(self, min_favorite: float | None = None) -> dict:
        return crowd_summary(self.crowd, min_favorite=min_favorite)


def agreement_score_for_gates(
    forecast: ForecastEnsembleOutput,
    *,
    use_ensemble: bool = True,
) -> float:
    """Agreement metric used for trade gates — ensemble or crowd voters."""
    if use_ensemble:
        return forecast.ensemble.agreement_score
    return forecast.agreement_score


class ForecastEnsemble:
    """5-layer model + crowd forecast system."""

    def __init__(self) -> None:
        self.model = ForecastModel()
        self.crowd_system = CrowdForecastSystem()

    def forecast(self, state: MarketState, data: MarketData | None = None) -> ForecastEnsembleOutput:
        output = self.model.forecast(state)
        crowd = self.crowd_system.forecast(output, state, data)

        # Legacy weighted ensemble for comparison / fallback
        legacy_votes = self._build_legacy_votes(output, state)
        ensemble = combine_models(legacy_votes)

        # Primary fair price from crowd synthesis (dynamic gate alignment happens in bot.py)
        p_fair = crowd.prob_yes
        confidence = crowd.confidence
        agreement_score = crowd.agreement_score
        uncertainty = crowd.uncertainty

        return ForecastEnsembleOutput(
            p_fair=p_fair,
            confidence=max(0.0, min(1.0, confidence)),
            agreement_score=agreement_score,
            uncertainty=uncertainty,
            model_output=output,
            ensemble=ensemble,
            crowd=crowd,
            is_official_brti=state.is_official_brti,
        )

    def record_settlement(self, forecast: ForecastEnsembleOutput, outcome_yes: bool) -> None:
        self.crowd_system.record_settlement(forecast.crowd, outcome_yes)

    def _build_legacy_votes(self, output: ModelOutput, state: MarketState) -> list[ModelVote]:
        w = config.ENSEMBLE_WEIGHTS
        base_conf = output.confidence
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
    return ensemble.forecast(state, data)
