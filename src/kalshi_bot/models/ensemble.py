"""Model ensemble and signal agreement scoring."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelVote:
    name: str
    prob_yes: float
    weight: float
    confidence: float


@dataclass(frozen=True)
class EnsembleResult:
    prob_yes: float
    prob_no: float
    confidence: float
    uncertainty: float
    agreement_score: float  # 0..1
    votes: tuple[ModelVote, ...] = field(default_factory=tuple)
    disagreeing_models: tuple[str, ...] = field(default_factory=tuple)

    @property
    def fair_yes(self) -> float:
        return self.prob_yes

    @property
    def fair_no(self) -> float:
        return self.prob_no


def combine_models(votes: list[ModelVote]) -> EnsembleResult:
    """Weighted ensemble with agreement score; weights should sum to ~1."""
    if not votes:
        return EnsembleResult(0.5, 0.5, 0.0, 1.0, 0.0, tuple())

    total_w = sum(max(v.weight, 0.0) for v in votes) or 1.0
    prob_yes = sum(v.prob_yes * v.weight for v in votes) / total_w
    prob_yes = min(max(prob_yes, 0.001), 0.999)
    prob_no = 1.0 - prob_yes

    # Agreement: fraction of weight on same side of 0.5
    yes_weight = sum(v.weight for v in votes if v.prob_yes >= 0.5)
    no_weight = sum(v.weight for v in votes if v.prob_yes < 0.5)
    agreement = max(yes_weight, no_weight) / total_w

    spread = max(v.prob_yes for v in votes) - min(v.prob_yes for v in votes)
    uncertainty = min(1.0, spread * 1.5)
    confidence = sum(v.confidence * v.weight for v in votes) / total_w
    confidence *= agreement

    median_side = prob_yes >= 0.5
    disagreeing = tuple(
        v.name for v in votes if (v.prob_yes >= 0.5) != median_side
    )

    return EnsembleResult(
        prob_yes=prob_yes,
        prob_no=prob_no,
        confidence=confidence,
        uncertainty=uncertainty,
        agreement_score=agreement,
        votes=tuple(votes),
        disagreeing_models=disagreeing,
    )
