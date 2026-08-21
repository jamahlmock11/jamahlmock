"""Dynamic gate resolution tests."""

from __future__ import annotations

from kalshi_btc_1hr_bot.dynamic_gates import (
    HourBucket,
    apply_dynamic_thresholds,
    classify_hour_bucket,
    resolve_dynamic_thresholds,
)
from kalshi_btc_1hr_bot import config


def test_classify_hour_buckets():
    assert classify_hour_bucket(3000) == HourBucket.HOUR_EARLY
    assert classify_hour_bucket(1800) == HourBucket.HOUR_MID
    assert classify_hour_bucket(900) == HourBucket.HOUR_LATE
    assert classify_hour_bucket(300) == HourBucket.HOUR_FINAL
    assert classify_hour_bucket(60) == HourBucket.TOO_LATE
    assert classify_hour_bucket(4000) == HourBucket.TOO_EARLY


def test_early_bucket_aligned_with_mid_not_stricter():
    early = resolve_dynamic_thresholds(3000, vol_regime="med", agreement_score=0.6)
    mid = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.6)
    late = resolve_dynamic_thresholds(900, vol_regime="med", agreement_score=0.6)
    assert early.min_crowd_favorite <= mid.min_crowd_favorite + 0.03
    assert late.min_crowd_favorite <= early.min_crowd_favorite
    assert late.min_edge_cents <= early.min_edge_cents


def test_high_vol_tightens_gates():
    base = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.6)
    high = resolve_dynamic_thresholds(1800, vol_regime="high", agreement_score=0.6)
    assert high.min_crowd_favorite >= base.min_crowd_favorite
    assert high.min_edge_cents >= base.min_edge_cents


def test_strong_edge_relaxes_crowd_floor():
    weak = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.55, edge_cents=2.0)
    strong = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.55, edge_cents=5.0)
    assert strong.min_crowd_favorite <= weak.min_crowd_favorite


def test_thresholds_within_bucket_ranges():
    th = resolve_dynamic_thresholds(1200, vol_regime="low", agreement_score=0.7, edge_cents=3.0)
    cf_lo, cf_hi = th.crowd_favorite_range
    assert cf_lo <= th.min_crowd_favorite <= cf_hi
    e_lo, e_hi = th.min_edge_range
    assert e_lo <= th.min_edge_cents <= e_hi
    assert th.to_dict()["bucket_label"]


def test_apply_dynamic_thresholds_uses_trade_side_not_static_76():
    from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdMember
    from kalshi_btc_1hr_bot.ensemble import EnsembleResult
    from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput
    from kalshi_btc_1hr_bot.model import ModelOutput
    import numpy as np

    members = tuple(CrowdMember("m", 0.30, 1.0, 0.9, "model") for _ in range(6))
    crowd = CrowdForecast(
        prob_yes=0.30,
        prob_no=0.70,
        consensus_side="no",
        confidence=0.8,
        agreement_score=0.75,
        uncertainty=0.2,
        quorum_count=6,
        quorum_required=5,
        quorum_met=True,
        yes_votes=0,
        no_votes=6,
        synthesis="blend",
        members=members,
        top_votes=members[:4],
        disagreeing=(),
    )
    mo = ModelOutput(0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.0, 0.0, 0.0, "med", 0.0, np.zeros(18))
    ens = EnsembleResult(0.3, 0.7, 0.8, 0.2, 0.75, ())
    forecast = ForecastEnsembleOutput(0.3, 0.8, 0.75, 0.2, mo, ens, crowd, True)
    th = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.75, crowd_side_prob=0.70)
    aligned = apply_dynamic_thresholds(forecast, th, "no")
    assert aligned.crowd.side_met("no", min_favorite=th.min_crowd_favorite)
    assert aligned.confidence > 0.5
    assert aligned.crowd.quorum_required == th.min_quorum
    assert not any("76%" in n for n in aligned.crowd.notes)


def test_early_bucket_looser_than_before():
    early = resolve_dynamic_thresholds(3000, vol_regime="med", agreement_score=0.65)
    final = resolve_dynamic_thresholds(300, vol_regime="med", agreement_score=0.65, crowd_side_prob=0.66)
    assert early.min_crowd_favorite <= 0.72
    assert final.min_crowd_favorite <= 0.66
    assert final.min_edge_cents <= 1.6
    assert final.min_crowd_favorite >= config.GATE_ABS_MIN_CROWD
    assert final.min_edge_cents >= config.GATE_ABS_MIN_EDGE_CENTS


def test_quality_guardrails_compensate_relaxed_crowd():
    from kalshi_btc_1hr_bot.dynamic_gates import _apply_quality_guardrails

    cf_lo, cf_hi = 0.58, 0.66
    edge_lo, edge_hi = 1.25, 1.7
    base_ev = 0.012
    crowd, edge, ev, ag = _apply_quality_guardrails(
        min_crowd=0.59,
        min_edge=1.25,
        min_evidence=base_ev,
        min_agreement=0.48,
        cf_lo=cf_lo,
        cf_hi=cf_hi,
        edge_lo=edge_lo,
        edge_hi=edge_hi,
    )
    assert crowd >= config.GATE_ABS_MIN_CROWD
    assert edge >= config.GATE_ABS_MIN_EDGE_CENTS
    assert ev > base_ev


def test_dynamic_crowd_gate_allows_mid_bucket_trade():
    from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdMember
    from kalshi_btc_1hr_bot.ensemble import EnsembleResult, ModelVote
    from kalshi_btc_1hr_bot.evidence import directional_evidence, evaluate_edge_with_evidence
    from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput
    from kalshi_btc_1hr_bot.model import ModelOutput
    import numpy as np

    members = tuple(CrowdMember("m", 0.72, 1.0, 0.9, "model") for _ in range(8))
    crowd = CrowdForecast(
        prob_yes=0.72,
        prob_no=0.28,
        consensus_side="yes",
        confidence=0.7,
        agreement_score=0.8,
        uncertainty=0.2,
        quorum_count=8,
        quorum_required=5,
        quorum_met=True,
        yes_votes=8,
        no_votes=0,
        synthesis="blend",
        members=members,
        top_votes=members[:4],
        disagreeing=(),
    )
    th = resolve_dynamic_thresholds(
        1800,
        vol_regime="med",
        agreement_score=0.8,
        crowd_side_prob=0.72,
    )
    assert th.min_crowd_favorite < 0.76
    assert crowd.side_met("yes", min_favorite=th.min_crowd_favorite)

    votes = [ModelVote("m", 0.72, 0.5, 0.9)]
    direction = directional_evidence(votes, n=1)
    mo = ModelOutput(0.72, 0.4, 0.72, 0.72, 0.72, 0.72, 0.72, 0.5, 0.0, 0.0, 0.0, "med", 0.0, np.zeros(18))
    ens = EnsembleResult(0.72, 0.28, 0.7, 0.2, 0.8, tuple(votes))
    forecast = ForecastEnsembleOutput(0.72, 0.7, 0.8, 0.2, mo, ens, crowd, True)
    forecast = apply_dynamic_thresholds(forecast, th, direction.side)
    edge = evaluate_edge_with_evidence(
        0.72, 0.40, 0.65, 0.38, 0.63, direction, thresholds=th, forecast=forecast
    )
    assert edge.should_trade
