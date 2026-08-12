"""Volatility regime classification for adaptive probability and sizing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VolRegime(str, Enum):
    LOW_VOL = "LOW_VOL"
    NORMAL_VOL = "NORMAL_VOL"
    HIGH_VOL = "HIGH_VOL"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"


@dataclass(frozen=True)
class RegimeAssessment:
    regime: VolRegime
    annualized_vol: float
    vol_percentile: float  # 0..1 vs recent history
    edge_multiplier: float
    size_multiplier: float
    allow_new_entries: bool
    reason: str


def classify_vol_regime(
    annualized_vol: float,
    *,
    recent_vols: list[float] | None = None,
    shock_threshold: float = 0.85,
    high_threshold: float = 0.65,
    low_threshold: float = 0.30,
) -> RegimeAssessment:
    """Classify current vol environment and return trading adjustments."""
    hist = sorted(recent_vols or [annualized_vol])
    if hist:
        rank = sum(1 for v in hist if v <= annualized_vol) / len(hist)
    else:
        rank = 0.5

    if rank >= shock_threshold or annualized_vol >= 1.0:
        return RegimeAssessment(
            VolRegime.VOLATILITY_SHOCK,
            annualized_vol,
            rank,
            edge_multiplier=1.75,
            size_multiplier=0.25,
            allow_new_entries=False,
            reason="volatility shock — new entries disabled",
        )
    if rank >= high_threshold or annualized_vol >= 0.70:
        return RegimeAssessment(
            VolRegime.HIGH_VOL,
            annualized_vol,
            rank,
            edge_multiplier=1.35,
            size_multiplier=0.55,
            allow_new_entries=True,
            reason="high volatility — wider edge required",
        )
    if rank <= low_threshold or annualized_vol <= 0.28:
        return RegimeAssessment(
            VolRegime.LOW_VOL,
            annualized_vol,
            rank,
            edge_multiplier=0.95,
            size_multiplier=0.90,
            allow_new_entries=True,
            reason="low volatility",
        )
    return RegimeAssessment(
        VolRegime.NORMAL_VOL,
        annualized_vol,
        rank,
        edge_multiplier=1.0,
        size_multiplier=1.0,
        allow_new_entries=True,
        reason="normal volatility",
    )


def neutral_regime(annualized_vol: float = 0.45) -> RegimeAssessment:
    """Vol regime disabled — no gating or size/edge adjustments."""
    return RegimeAssessment(
        VolRegime.NORMAL_VOL,
        annualized_vol,
        0.5,
        edge_multiplier=1.0,
        size_multiplier=1.0,
        allow_new_entries=True,
        reason="vol regime disabled",
    )
