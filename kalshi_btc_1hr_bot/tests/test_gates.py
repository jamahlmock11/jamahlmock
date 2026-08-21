"""Gate configuration tests — crowd off, ensemble agreement on."""

from __future__ import annotations

from kalshi_btc_1hr_bot.config import BotConfig, load_config
from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdMember
from kalshi_btc_1hr_bot.dynamic_gates import resolve_dynamic_thresholds
from kalshi_btc_1hr_bot.ensemble import EnsembleResult, ModelVote
from kalshi_btc_1hr_bot.evidence import DirectionalEvidence, evaluate_edge_with_evidence
from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput, agreement_score_for_gates
from kalshi_btc_1hr_bot.model import ModelOutput
import numpy as np


def _forecast(*, crowd_side_pct: float = 0.55, ensemble_agree: float = 0.72) -> ForecastEnsembleOutput:
    side = "yes" if crowd_side_pct >= 0.5 else "no"
    prob_yes = crowd_side_pct if side == "yes" else 1.0 - crowd_side_pct
    members = tuple(
        CrowdMember("m", prob_yes if side == "yes" else 1 - prob_yes, 1.0, 0.9, "model")
        for _ in range(3)
    )
    crowd = CrowdForecast(
        prob_yes=prob_yes,
        prob_no=1.0 - prob_yes,
        consensus_side=side,
        confidence=0.7,
        agreement_score=0.9,
        uncertainty=0.2,
        quorum_count=3,
        quorum_required=5,
        quorum_met=False,
        yes_votes=3 if side == "yes" else 0,
        no_votes=0 if side == "yes" else 3,
        synthesis="blend",
        members=members,
        top_votes=members,
        disagreeing=(),
    )
    votes = (ModelVote("five_layer", prob_yes, 0.4, 0.9),)
    mo = ModelOutput(prob_yes, 0.4, prob_yes, prob_yes, prob_yes, prob_yes, prob_yes, 0.5, 0, 0, 0, "med", 0, np.zeros(18))
    ens = EnsembleResult(prob_yes, 1 - prob_yes, 0.7, 0.2, ensemble_agree, votes)
    return ForecastEnsembleOutput(prob_yes, 0.7, 0.9, 0.2, mo, ens, crowd, True)


def test_crowd_gates_disabled_skips_quorum_and_favorite():
    forecast = _forecast(crowd_side_pct=0.55, ensemble_agree=0.72)
    th = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.72)
    direction = DirectionalEvidence("no", 0.0, 0.25, 0.25, forecast.votes)
    edge = evaluate_edge_with_evidence(
        0.45,
        0.40,
        0.62,
        0.38,
        0.60,
        direction,
        crowd_gates_enabled=False,
        use_ensemble_agreement=True,
        thresholds=th,
        forecast=forecast,
    )
    assert "Crowd quorum" not in edge.reason
    assert "Crowd BELOW" not in edge.reason
    assert "Crowd ABOVE" not in edge.reason


def test_ensemble_agreement_blocks_when_low():
    forecast = _forecast(crowd_side_pct=0.70, ensemble_agree=0.40)
    th = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.40)
    direction = DirectionalEvidence("yes", 0.25, 0.0, 0.25, forecast.votes)
    edge = evaluate_edge_with_evidence(
        0.70,
        0.40,
        0.65,
        0.38,
        0.63,
        direction,
        crowd_gates_enabled=False,
        use_ensemble_agreement=True,
        thresholds=th,
        forecast=forecast,
    )
    assert "Ensemble agreement" in edge.reason
    assert agreement_score_for_gates(forecast, use_ensemble=True) == 0.40


def test_load_config_crowd_gates_default_off():
    cfg = load_config()
    assert cfg.gates.crowd_gates_enabled is False
    assert cfg.gates.use_ensemble_agreement is True
