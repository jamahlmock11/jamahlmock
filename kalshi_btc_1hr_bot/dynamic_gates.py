"""Time- and regime-aware trade gates for the 1-hour KXBTCD window."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from kalshi_btc_1hr_bot import config


class HourBucket(str, Enum):
    HOUR_EARLY = "hour_early"  # 45–60 min remaining
    HOUR_MID = "hour_mid"  # 25–45 min
    HOUR_LATE = "hour_late"  # 10–25 min
    HOUR_FINAL = "hour_final"  # 2–10 min
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class BucketPolicy:
    """Base gate targets for a time slice within the hourly window."""

    crowd_favorite: tuple[float, float]  # (floor, ceiling)
    min_edge_cents: tuple[float, float]
    evidence_margin: tuple[float, float]
    min_quorum: tuple[int, int]
    min_agreement: tuple[float, float]
    label: str


# Loosen early hour to match mid/late — more quality trades across full window.
DEFAULT_BUCKET_POLICIES: dict[HourBucket, BucketPolicy] = {
    HourBucket.HOUR_EARLY: BucketPolicy(
        crowd_favorite=(0.64, 0.74),
        min_edge_cents=(0.8, 1.5),
        evidence_margin=(0.016, 0.022),
        min_quorum=(4, 5),
        min_agreement=(0.52, 0.58),
        label="45→60 min",
    ),
    HourBucket.HOUR_MID: BucketPolicy(
        crowd_favorite=(0.65, 0.76),
        min_edge_cents=(0.8, 1.5),
        evidence_margin=(0.016, 0.024),
        min_quorum=(4, 6),
        min_agreement=(0.52, 0.58),
        label="25→45 min",
    ),
    HourBucket.HOUR_LATE: BucketPolicy(
        crowd_favorite=(0.60, 0.70),
        min_edge_cents=(0.5, 1.2),
        evidence_margin=(0.013, 0.020),
        min_quorum=(4, 5),
        min_agreement=(0.50, 0.56),
        label="10→25 min",
    ),
    HourBucket.HOUR_FINAL: BucketPolicy(
        crowd_favorite=(0.58, 0.66),
        min_edge_cents=(0.5, 1.0),
        evidence_margin=(0.012, 0.017),
        min_quorum=(4, 5),
        min_agreement=(0.48, 0.54),
        label="2→10 min",
    ),
}


@dataclass(frozen=True)
class DynamicThresholds:
    """Resolved gate values for one market at a point in the hour."""

    bucket: HourBucket
    bucket_label: str
    min_crowd_favorite: float
    min_edge_cents: float
    min_evidence_margin: float
    min_quorum: int
    min_agreement: float
    crowd_favorite_range: tuple[float, float]
    min_edge_range: tuple[float, float]
    evidence_margin_range: tuple[float, float]
    quorum_range: tuple[int, int]
    agreement_range: tuple[float, float]

    @property
    def min_crowd_favorite_pct(self) -> float:
        return self.min_crowd_favorite * 100.0

    def crowd_side_met(self, side_pct: float) -> bool:
        return side_pct >= self.min_crowd_favorite_pct

    def to_dict(self) -> dict[str, float | str | list[float]]:
        return {
            "bucket": self.bucket.value,
            "bucket_label": self.bucket_label,
            "min_crowd_favorite": round(self.min_crowd_favorite, 4),
            "min_crowd_favorite_pct": round(self.min_crowd_favorite_pct, 1),
            "min_edge_cents": round(self.min_edge_cents, 2),
            "min_evidence_margin": round(self.min_evidence_margin, 4),
            "min_quorum": self.min_quorum,
            "min_agreement": round(self.min_agreement, 3),
            "crowd_favorite_range_pct": [round(r * 100, 1) for r in self.crowd_favorite_range],
            "min_edge_range": list(self.min_edge_range),
            "evidence_margin_range": list(self.evidence_margin_range),
            "quorum_range": list(self.quorum_range),
            "agreement_range_pct": [round(r * 100, 1) for r in self.agreement_range],
        }


def classify_hour_bucket(
    seconds_to_expiry: float,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> HourBucket:
    min_s = min_seconds if min_seconds is not None else config.RISK_MIN_SECONDS
    max_s = max_seconds if max_seconds is not None else config.WINDOW_SECONDS
    mins = seconds_to_expiry / 60.0
    if seconds_to_expiry < min_s:
        return HourBucket.TOO_LATE
    if seconds_to_expiry > max_s:
        return HourBucket.TOO_EARLY
    if mins > 45:
        return HourBucket.HOUR_EARLY
    if mins > 25:
        return HourBucket.HOUR_MID
    if mins > 10:
        return HourBucket.HOUR_LATE
    return HourBucket.HOUR_FINAL


def _lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * max(0.0, min(1.0, t))


def _vol_adjustment(vol_regime: str) -> dict[str, float]:
    regime = (vol_regime or "med").lower()
    if regime == "high":
        return {"crowd": 0.025, "edge": 0.35, "evidence": 0.003, "agreement": 0.03, "quorum": 1}
    if regime == "low":
        return {"crowd": -0.015, "edge": -0.20, "evidence": -0.002, "agreement": -0.02, "quorum": 0}
    return {"crowd": 0.0, "edge": 0.0, "evidence": 0.0, "agreement": 0.0, "quorum": 0}


def _apply_quality_guardrails(
    *,
    min_crowd: float,
    min_edge: float,
    min_evidence: float,
    min_agreement: float,
    cf_lo: float,
    cf_hi: float,
    edge_lo: float,
    edge_hi: float,
) -> tuple[float, float, float, float]:
    """Loosen entries but compensate so weak setups stay blocked."""
    min_crowd = max(config.GATE_ABS_MIN_CROWD, min_crowd)
    min_edge = max(config.GATE_ABS_MIN_EDGE_CENTS, min_edge)
    min_evidence = max(config.GATE_ABS_MIN_EVIDENCE, min_evidence)
    min_agreement = max(config.GATE_ABS_MIN_AGREEMENT, min_agreement)

    cf_mid = (cf_lo + cf_hi) / 2.0
    edge_mid = (edge_lo + edge_hi) / 2.0

    # Lower crowd floor → require slightly stronger evidence
    if min_crowd < cf_mid:
        relax = (cf_mid - min_crowd) / max(cf_mid - cf_lo, 0.01)
        min_evidence += 0.0025 * relax

    # Lower edge floor → require slightly stronger crowd
    if min_edge < edge_mid:
        relax = (edge_mid - min_edge) / max(edge_mid - edge_lo, 0.01)
        min_crowd += 0.008 * relax

    min_crowd = min(cf_hi, max(config.GATE_ABS_MIN_CROWD, min_crowd))
    min_edge = max(config.GATE_ABS_MIN_EDGE_CENTS, min_edge)
    min_evidence = max(config.GATE_ABS_MIN_EVIDENCE, min_evidence)
    return min_crowd, min_edge, min_evidence, min_agreement


def resolve_dynamic_thresholds(
    seconds_to_expiry: float,
    *,
    vol_regime: str = "med",
    agreement_score: float = 0.6,
    edge_cents: float | None = None,
    crowd_side_prob: float | None = None,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
) -> DynamicThresholds:
    """Pick gate values inside each bucket range using vol + signal strength."""
    bucket = classify_hour_bucket(
        seconds_to_expiry, min_seconds=min_seconds, max_seconds=max_seconds
    )
    if bucket in (HourBucket.TOO_EARLY, HourBucket.TOO_LATE):
        policy = DEFAULT_BUCKET_POLICIES[HourBucket.HOUR_MID]
    else:
        policy = DEFAULT_BUCKET_POLICIES[bucket]

    cf_lo, cf_hi = policy.crowd_favorite
    edge_lo, edge_hi = policy.min_edge_cents
    ev_lo, ev_hi = policy.evidence_margin
    q_lo, q_hi = policy.min_quorum
    ag_lo, ag_hi = policy.min_agreement

    # Midpoint bias: start from bucket center, then nudge with context.
    t_agree = max(0.0, min(1.0, (agreement_score - 0.45) / 0.35))
    t_edge = 0.0
    if edge_cents is not None:
        t_edge = max(0.0, min(1.0, (edge_cents - edge_lo) / max(edge_hi - edge_lo, 0.01)))
    t_crowd = 0.0
    if crowd_side_prob is not None:
        t_crowd = max(0.0, min(1.0, (crowd_side_prob - cf_lo) / max(cf_hi - cf_lo, 0.01)))

    vol = _vol_adjustment(vol_regime)

    # Strong edge or crowd → relax complementary gates slightly.
    min_crowd = _lerp(cf_hi, cf_lo, 0.50 + 0.30 * t_edge + 0.15 * t_agree)
    min_edge = _lerp(edge_hi, edge_lo, 0.45 + 0.35 * t_crowd + 0.20 * t_agree)
    min_evidence = _lerp(ev_hi, ev_lo, 0.45 + 0.30 * t_edge + 0.15 * t_agree)
    min_agreement = _lerp(ag_hi, ag_lo, 0.40 + 0.35 * t_crowd + 0.20 * t_edge)
    min_quorum = int(round(_lerp(float(q_hi), float(q_lo), 0.35 + 0.35 * t_agree + 0.20 * t_crowd)))

    min_crowd = max(cf_lo, min(cf_hi, min_crowd + vol["crowd"]))
    min_edge = max(edge_lo, min(edge_hi, min_edge + vol["edge"]))
    min_evidence = max(ev_lo, min(ev_hi, min_evidence + vol["evidence"]))
    min_agreement = max(ag_lo, min(ag_hi, min_agreement + vol["agreement"]))
    min_quorum = max(q_lo, min(q_hi, min_quorum + int(vol["quorum"])))

    min_crowd, min_edge, min_evidence, min_agreement = _apply_quality_guardrails(
        min_crowd=min_crowd,
        min_edge=min_edge,
        min_evidence=min_evidence,
        min_agreement=min_agreement,
        cf_lo=cf_lo,
        cf_hi=cf_hi,
        edge_lo=edge_lo,
        edge_hi=edge_hi,
    )

    return DynamicThresholds(
        bucket=bucket,
        bucket_label=policy.label,
        min_crowd_favorite=min_crowd,
        min_edge_cents=min_edge,
        min_evidence_margin=min_evidence,
        min_quorum=min_quorum,
        min_agreement=min_agreement,
        crowd_favorite_range=(cf_lo, cf_hi),
        min_edge_range=(edge_lo, edge_hi),
        evidence_margin_range=(ev_lo, ev_hi),
        quorum_range=(q_lo, q_hi),
        agreement_range=(ag_lo, ag_hi),
    )


def apply_dynamic_thresholds(
    forecast: Any,
    thresholds: DynamicThresholds,
    trade_side: str,
    *,
    crowd_gates_enabled: bool | None = None,
    use_ensemble_agreement: bool | None = None,
) -> Any:
    """Align confidence and notes with dynamic gates for the trade side."""
    from dataclasses import replace

    from kalshi_btc_1hr_bot import config
    from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast
    from kalshi_btc_1hr_bot.forecast import agreement_score_for_gates

    crowd_on = config.CROWD_GATES_ENABLED if crowd_gates_enabled is None else crowd_gates_enabled
    use_ensemble = (
        config.USE_ENSEMBLE_AGREEMENT if use_ensemble_agreement is None else use_ensemble_agreement
    )
    agree_score = agreement_score_for_gates(forecast, use_ensemble=use_ensemble)

    crowd: CrowdForecast = forecast.crowd
    side = trade_side.lower()
    side_prob = crowd.side_prob(side)
    side_quorum = sum(1 for m in crowd.members if m.side == side)
    finish = "ABOVE" if side == "yes" else "BELOW"

    total_w = sum(max(m.weight, 0.0) for m in crowd.members) or 1.0
    confidence = sum(m.confidence * m.weight for m in crowd.members) / total_w
    confidence *= agree_score

    if crowd_on and side_quorum < thresholds.min_quorum:
        confidence *= side_quorum / max(thresholds.min_quorum, 1)
    if crowd_on and side_prob < thresholds.min_crowd_favorite:
        confidence *= side_prob / max(thresholds.min_crowd_favorite, 0.01)
    if agree_score < thresholds.min_agreement:
        confidence *= agree_score / max(thresholds.min_agreement, 0.01)
    if not forecast.is_official_brti:
        confidence *= config.PROXY_BRTI_CONFIDENCE_PENALTY

    notes: list[str] = []
    if crowd_on and side_quorum < thresholds.min_quorum:
        notes.append(f"Crowd quorum {side_quorum}/{thresholds.min_quorum} on {side.upper()}")
    if crowd_on and side_prob < thresholds.min_crowd_favorite:
        notes.append(
            f"Crowd {finish} {side_prob:.0%} < {thresholds.min_crowd_favorite:.0%} "
            f"({thresholds.bucket_label})"
        )
    if agree_score < thresholds.min_agreement:
        label = "Ensemble agreement" if use_ensemble else "Agreement"
        notes.append(
            f"{label} {agree_score:.0%} < {thresholds.min_agreement:.0%} "
            f"({thresholds.bucket_label})"
        )
    if not forecast.is_official_brti:
        notes.append("Proxy BRTI — crowd confidence reduced")

    updated_crowd = replace(
        crowd,
        quorum_count=side_quorum,
        quorum_required=thresholds.min_quorum,
        quorum_met=side_quorum >= thresholds.min_quorum,
        confidence=max(0.0, min(1.0, confidence)),
        notes=tuple(notes),
    )
    return replace(
        forecast,
        crowd=updated_crowd,
        confidence=max(0.0, min(1.0, confidence)),
    )
