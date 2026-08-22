"""Crowd forecast system — wisdom-of-crowds ensemble for the 1hr bot.

Combines multiple independent forecasters (model layers + crowd lenses),
requires a quorum before trading, and optionally adapts weights from outcomes.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.data_feed import MarketData
from kalshi_btc_1hr_bot.ensemble import EnsembleResult, ModelVote, combine_models
from kalshi_btc_1hr_bot.model import MarketState, ModelOutput
from kalshi_btc_1hr_bot.config import ModelConfig
from kalshi_btc_1hr_bot.utils import gbm_prob_above

log = logging.getLogger("crowd")


@dataclass(frozen=True)
class CrowdMember:
    """One voter in the crowd."""

    name: str
    prob_yes: float
    weight: float
    confidence: float
    group: str = "model"  # model | trend | contrarian | micro | anchor

    @property
    def side(self) -> str:
        return "yes" if self.prob_yes >= 0.5 else "no"

    @property
    def strength(self) -> float:
        return self.weight * self.confidence * abs(self.prob_yes - 0.5) * 2.0


@dataclass(frozen=True)
class CrowdForecast:
    """Synthesized crowd probability with quorum metadata."""

    prob_yes: float
    prob_no: float
    consensus_side: str
    confidence: float
    agreement_score: float
    uncertainty: float
    quorum_count: int
    quorum_required: int
    quorum_met: bool
    yes_votes: int
    no_votes: int
    synthesis: str
    members: tuple[CrowdMember, ...]
    top_votes: tuple[CrowdMember, ...]
    disagreeing: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def finish_label(self) -> str:
        return "ABOVE" if self.consensus_side == "yes" else "BELOW"

    @property
    def favorite_prob(self) -> float:
        """Crowd probability on the consensus side (either YES or NO)."""
        return self.prob_yes if self.consensus_side == "yes" else self.prob_no

    @property
    def favorite_pct(self) -> float:
        return self.favorite_prob * 100.0

    @property
    def favorite_met(self) -> bool:
        return self.favorite_prob >= config.CROWD_MIN_FAVORITE

    def favorite_met_at(self, min_favorite: float) -> bool:
        return self.favorite_prob >= min_favorite

    def side_prob(self, side: str) -> float:
        return self.prob_yes if side == "yes" else self.prob_no

    def side_pct(self, side: str) -> float:
        return self.side_prob(side) * 100.0

    def side_met(self, side: str, *, min_favorite: float | None = None) -> bool:
        floor = config.CROWD_MIN_FAVORITE if min_favorite is None else min_favorite
        return self.side_prob(side) >= floor


class CrowdPerformanceTracker:
    """Track per-member Brier scores and adapt weights."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or config.CROWD_PERFORMANCE_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stats: dict[str, dict[str, float]] = self._load()

    def _load(self) -> dict[str, dict[str, float]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._stats, indent=2))

    def weight_multiplier(self, name: str) -> float:
        if not config.CROWD_USE_ADAPTIVE_WEIGHTS:
            return 1.0
        row = self._stats.get(name)
        if not row or row.get("n", 0) < 5:
            return 1.0
        # Lower Brier → higher multiplier (0.5x .. 1.5x)
        brier = row["brier"] / max(row["n"], 1)
        return max(0.5, min(1.5, 1.25 - brier * 2.0))

    def record(self, name: str, prob_yes: float, outcome_yes: bool) -> None:
        actual = 1.0 if outcome_yes else 0.0
        brier = (prob_yes - actual) ** 2
        row = self._stats.setdefault(name, {"n": 0.0, "brier": 0.0})
        row["n"] += 1
        row["brier"] += brier
        self._save()


def _clamp(p: float) -> float:
    return max(0.001, min(0.999, p))


def _member_from_vote(v: ModelVote, *, group: str = "model") -> CrowdMember:
    return CrowdMember(v.name, v.prob_yes, v.weight, v.confidence, group)


def _build_model_crowd(output: ModelOutput, state: MarketState) -> list[CrowdMember]:
    w = config.ENSEMBLE_WEIGHTS
    base_conf = output.confidence
    obi_prob = _clamp(0.5 + output.obi * 0.12)
    members = [
        CrowdMember("five_layer", output.p_calibrated, w["five_layer"], base_conf, "model"),
        CrowdMember("gbm_core", output.p_gbm, w["gbm_core"], base_conf * 0.85, "model"),
        CrowdMember("momentum", output.p_momentum, w["momentum"], base_conf * 0.90, "model"),
        CrowdMember("mean_reversion", output.p_mean_rev, w["mean_reversion"], base_conf * 0.85, "model"),
        CrowdMember("funding", output.p_funding, w["funding"], base_conf * 0.80, "model"),
        CrowdMember("obi_micro", obi_prob, w["obi"], base_conf * 0.70, "micro"),
    ]
    if not state.is_official_brti:
        members = [
            CrowdMember(m.name, m.prob_yes, m.weight * (0.6 if m.name == "five_layer" else 1.0), m.confidence, m.group)
            for m in members
        ]
    return members


def _build_lens_crowd(output: ModelOutput, state: MarketState, data: MarketData | None) -> list[CrowdMember]:
    """Independent crowd lenses — different ways to read the same market."""
    S, K = state.current_price, state.strike
    t_years = max(state.seconds_remaining, 1.0) / config.ANNUALIZE_SECONDS
    lenses: list[CrowdMember] = []

    # Strike distance / moneyness crowd
    dist_bps = (S - K) / K * 10000 if K else 0.0
    moneyness_prob = _clamp(0.5 + dist_bps / 400.0)  # ~100bps → strong tilt
    lenses.append(CrowdMember("strike_distance", moneyness_prob, 0.08, 0.75, "anchor"))

    # Trend crowd — follow blended momentum
    if data is not None:
        mu_blend = 0.5 * data.mu_5m + 0.3 * data.mu_15m + 0.2 * data.mu_30m
        trend_prob = _clamp(0.5 + mu_blend * 8000)
        lenses.append(CrowdMember("trend_crowd", trend_prob, 0.10, 0.80, "trend"))

    # VWAP anchor crowd
    vwap = state.vwap or S
    vwap_dist = (S - vwap) / vwap if vwap else 0.0
    vwap_prob = _clamp(0.5 + vwap_dist * 500 + (S - K) / max(K, 1) * 2.0)
    lenses.append(CrowdMember("vwap_anchor", vwap_prob, 0.08, 0.70, "anchor"))

    # Contrarian crowd — fade extension from VWAP
    ext = abs(S - vwap) / vwap if vwap else 0.0
    if ext > 0.002:
        fade = -math.copysign(min(0.15, ext * 20), S - vwap)
        contra_prob = _clamp(0.5 + (output.p_momentum - 0.5) * -0.5 + fade)
    else:
        contra_prob = _clamp(0.5 - (output.p_momentum - 0.5) * 0.3)
    lenses.append(CrowdMember("contrarian_crowd", contra_prob, 0.07, 0.65, "contrarian"))

    # Vol regime crowd — high vol → uncertainty, pull toward 0.5
    vol_pull = {"low": 1.0, "medium": 0.85, "high": 0.55}.get(output.vol_regime, 0.85)
    vol_prob = _clamp(0.5 + (output.p_gbm - 0.5) * vol_pull)
    lenses.append(CrowdMember("vol_crowd", vol_prob, 0.07, vol_pull, "model"))

    # Funding contrarian at extremes
    fr = output.funding_rate
    if abs(fr) > ModelConfig().funding_extreme_threshold:
        funding_contra = _clamp(0.5 - (output.p_funding - 0.5) * 0.6)
        lenses.append(CrowdMember("funding_contrarian", funding_contra, 0.06, 0.60, "contrarian"))
    else:
        lenses.append(CrowdMember("funding_crowd", output.p_funding, 0.06, 0.70, "trend"))

    # Time decay crowd — GBM with remaining time to avg window
    t_avg = max(state.seconds_to_avg_start, 1.0) / config.ANNUALIZE_SECONDS
    time_prob = gbm_prob_above(S, K, t_avg, output.sigma, output.mu)
    lenses.append(CrowdMember("time_decay", _clamp(time_prob), 0.09, 0.75, "anchor"))

    return lenses


def _weighted_prob(members: list[CrowdMember]) -> float:
    total = sum(max(m.weight, 0.0) for m in members) or 1.0
    return sum(m.prob_yes * m.weight for m in members) / total


def _median_prob(members: list[CrowdMember]) -> float:
    probs = sorted(m.prob_yes for m in members)
    n = len(probs)
    if n == 0:
        return 0.5
    mid = n // 2
    return probs[mid] if n % 2 else (probs[mid - 1] + probs[mid]) / 2.0


def _trimmed_prob(members: list[CrowdMember], trim: float = 0.15) -> float:
    if len(members) < 4:
        return _weighted_prob(members)
    ranked = sorted(members, key=lambda m: m.prob_yes)
    k = max(1, int(len(ranked) * trim))
    trimmed = ranked[k : len(ranked) - k] or ranked
    return _weighted_prob(trimmed)


def _synthesize(members: list[CrowdMember], method: str) -> tuple[float, str]:
    if method == "median":
        return _clamp(_median_prob(members)), "median"
    if method == "trimmed":
        return _clamp(_trimmed_prob(members)), "trimmed"
    if method == "blend":
        w = _weighted_prob(members)
        med = _median_prob(members)
        tri = _trimmed_prob(members)
        blend = 0.50 * w + 0.30 * med + 0.20 * tri
        return _clamp(blend), "blend"
    return _clamp(_weighted_prob(members)), "weighted"


def _top_members(members: list[CrowdMember], n: int) -> tuple[CrowdMember, ...]:
    ranked = sorted(members, key=lambda m: m.strength, reverse=True)
    return tuple(ranked[:n])


def _quorum(members: list[CrowdMember], side: str) -> int:
    return sum(1 for m in members if m.side == side)


class CrowdForecastSystem:
    """Build and synthesize the full forecast crowd."""

    def __init__(self, tracker: CrowdPerformanceTracker | None = None) -> None:
        self.tracker = tracker or CrowdPerformanceTracker()

    def forecast(
        self,
        output: ModelOutput,
        state: MarketState,
        data: MarketData | None = None,
    ) -> CrowdForecast:
        members = _build_model_crowd(output, state) + _build_lens_crowd(output, state, data)

        # Apply adaptive performance weights
        adjusted: list[CrowdMember] = []
        for m in members:
            mult = self.tracker.weight_multiplier(m.name)
            adjusted.append(
                CrowdMember(m.name, m.prob_yes, m.weight * mult, m.confidence, m.group)
            )
        members = adjusted

        prob_yes, synthesis = _synthesize(members, config.CROWD_SYNTHESIS)
        prob_no = 1.0 - prob_yes
        consensus = "yes" if prob_yes >= 0.5 else "no"

        yes_q = _quorum(members, "yes")
        no_q = _quorum(members, "no")
        quorum_count = yes_q if consensus == "yes" else no_q
        quorum_required = config.CROWD_MIN_QUORUM
        quorum_met = quorum_count >= quorum_required

        total_w = sum(max(m.weight, 0.0) for m in members) or 1.0
        yes_w = sum(m.weight for m in members if m.side == "yes")
        no_w = sum(m.weight for m in members if m.side == "no")
        agreement = max(yes_w, no_w) / total_w

        spread = max(m.prob_yes for m in members) - min(m.prob_yes for m in members)
        uncertainty = min(1.0, spread * 1.2)
        confidence = sum(m.confidence * m.weight for m in members) / total_w
        confidence *= agreement

        disagreeing = tuple(m.name for m in members if m.side != consensus)
        top = _top_members(members, config.TOP_N_VOTES)

        notes: list[str] = []
        if not state.is_official_brti:
            notes.append("Proxy BRTI — unofficial feed")

        return CrowdForecast(
            prob_yes=prob_yes,
            prob_no=prob_no,
            consensus_side=consensus,
            confidence=max(0.0, min(1.0, confidence)),
            agreement_score=agreement,
            uncertainty=uncertainty,
            quorum_count=quorum_count,
            quorum_required=quorum_required,
            quorum_met=quorum_met,
            yes_votes=yes_q,
            no_votes=no_q,
            synthesis=synthesis,
            members=tuple(members),
            top_votes=top,
            disagreeing=disagreeing,
            notes=tuple(notes),
        )

    def members_to_votes(self, crowd: CrowdForecast) -> list[ModelVote]:
        return [
            ModelVote(m.name, m.prob_yes, m.weight, m.confidence) for m in crowd.members
        ]

    def crowd_to_ensemble(self, crowd: CrowdForecast) -> EnsembleResult:
        return combine_models(self.members_to_votes(crowd))

    def record_settlement(self, crowd: CrowdForecast, outcome_yes: bool) -> None:
        for m in crowd.members:
            self.tracker.record(m.name, m.prob_yes, outcome_yes)


def crowd_summary(crowd: CrowdForecast, *, min_favorite: float | None = None) -> dict[str, Any]:
    """JSON-serializable crowd snapshot for dashboard."""
    fav_floor = config.CROWD_MIN_FAVORITE if min_favorite is None else min_favorite
    return {
        "prob_yes": round(crowd.prob_yes, 4),
        "consensus": crowd.consensus_side.upper(),
        "finish": crowd.finish_label,
        "confidence": round(crowd.confidence, 3),
        "agreement": round(crowd.agreement_score, 3),
        "quorum": f"{crowd.quorum_count}/{crowd.quorum_required}",
        "quorum_met": crowd.quorum_met,
        "favorite_pct": round(crowd.favorite_pct, 1),
        "favorite_met": crowd.favorite_met_at(fav_floor),
        "min_favorite_pct": round(fav_floor * 100, 1),
        "yes_votes": crowd.yes_votes,
        "no_votes": crowd.no_votes,
        "synthesis": crowd.synthesis,
        "notes": list(crowd.notes),
        "members": [
            {
                "name": m.name,
                "prob_yes": round(m.prob_yes, 4),
                "side": m.side.upper(),
                "weight": round(m.weight, 3),
                "confidence": round(m.confidence, 3),
                "group": m.group,
                "strength": round(m.strength, 4),
            }
            for m in crowd.members
        ],
        "top_votes": [
            {"name": m.name, "prob_yes": round(m.prob_yes, 4), "side": m.side.upper(), "weight": round(m.weight, 3)}
            for m in crowd.top_votes
        ],
    }
