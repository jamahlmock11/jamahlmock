"""Adaptive trade tiers — paper analysis only unless explicitly enabled."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_bot.config import TierEdgeConfig


class SetupTier(str, Enum):
    A_PLUS = "A_PLUS"
    A = "A"
    B = "B"
    NONE = "NONE"


@dataclass(frozen=True)
class TierAssessment:
    tier: SetupTier
    min_edge_required: float
    net_edge_dollars: float
    qualifies: bool
    detail: str


def classify_tier(
    *,
    net_edge_dollars: float,
    model_confidence: float,
    data_fresh: bool,
    liquidity_ok: bool,
    spread_ok: bool,
    model_agrees: bool,
    no_conflicts: bool,
    config: TierEdgeConfig,
) -> TierAssessment:
    """Classify setup tier based on net edge and quality signals.

    Tiers are used for RANKING and paper-trade experiments.
    Live trading uses only tiers enabled in config.
    """
    if not data_fresh or not liquidity_ok or not spread_ok:
        return TierAssessment(SetupTier.NONE, 0.0, net_edge_dollars, False, "data/execution fail")

    # A+ setup
    if (
        net_edge_dollars >= config.min_edge_a_plus
        and model_confidence >= config.min_confidence_a_plus
        and model_agrees
        and no_conflicts
    ):
        return TierAssessment(
            SetupTier.A_PLUS,
            config.min_edge_a_plus,
            net_edge_dollars,
            True,
            f"net_edge={net_edge_dollars*100:.1f}¢ ≥ A+ floor {config.min_edge_a_plus*100:.0f}¢",
        )

    # A setup
    if (
        net_edge_dollars >= config.min_edge_a
        and model_confidence >= config.min_confidence_a
        and model_agrees
    ):
        return TierAssessment(
            SetupTier.A,
            config.min_edge_a,
            net_edge_dollars,
            True,
            f"net_edge={net_edge_dollars*100:.1f}¢ ≥ A floor {config.min_edge_a*100:.0f}¢",
        )

    # B setup
    if net_edge_dollars >= config.min_edge_b and model_confidence >= config.min_confidence_b:
        return TierAssessment(
            SetupTier.B,
            config.min_edge_b,
            net_edge_dollars,
            True,
            f"net_edge={net_edge_dollars*100:.1f}¢ ≥ B floor {config.min_edge_b*100:.0f}¢",
        )

    # Below all tiers
    best_floor = config.min_edge_b
    return TierAssessment(
        SetupTier.NONE,
        best_floor,
        net_edge_dollars,
        False,
        f"net_edge={net_edge_dollars*100:.1f}¢ < B floor {config.min_edge_b*100:.0f}¢",
    )


def opportunity_score(
    *,
    net_edge_dollars: float,
    model_confidence: float,
    momentum_confirmation: float,
    order_flow_confirmation: float,
    liquidity_score: float,
    spread: float,
    data_fresh: bool,
    model_agrees: bool,
) -> float:
    """Rank opportunities. Does NOT override hard risk limits."""
    score = net_edge_dollars * 100 * 2.0  # edge weight
    score += model_confidence * 30.0
    score += abs(momentum_confirmation) * 10.0
    score += abs(order_flow_confirmation) * 15.0
    score += liquidity_score * 10.0
    if data_fresh:
        score += 5.0
    if model_agrees:
        score += 8.0
    score -= spread * 100 * 2.0  # spread penalty
    if not model_agrees:
        score -= 15.0
    return round(max(0.0, score), 2)


def estimate_slippage(
    *,
    spread: float,
    liquidity_score: float,
    contracts: int = 1,
) -> float:
    """Conservative slippage estimate in dollars per contract."""
    half_spread = spread / 2.0
    depth_penalty = max(0.0, 0.02 * (1.0 - liquidity_score))
    size_penalty = 0.005 * max(0, contracts - 1)
    return round(half_spread + depth_penalty + size_penalty, 4)
