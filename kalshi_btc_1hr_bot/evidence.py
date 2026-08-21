"""Directional evidence from top model votes and market selection."""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.dynamic_gates import DynamicThresholds
from kalshi_btc_1hr_bot.edge import TradeSignal, evaluate_edge
from kalshi_btc_1hr_bot.ensemble import ModelVote
from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput


@dataclass(frozen=True)
class DirectionalEvidence:
    """Evidence for finish above (YES) vs below (NO) from top model votes."""

    side: str  # "yes" = above strike, "no" = below strike
    above_score: float
    below_score: float
    margin: float  # winning side advantage
    top_votes: tuple[ModelVote, ...]

    @property
    def finish_label(self) -> str:
        return "ABOVE" if self.side == "yes" else "BELOW"


def top_votes(votes: tuple[ModelVote, ...] | list[ModelVote], n: int | None = None) -> list[ModelVote]:
    """Return the top-N votes ranked by weight × confidence."""
    n = n or config.TOP_N_VOTES
    ranked = sorted(votes, key=lambda v: v.weight * v.confidence, reverse=True)
    return ranked[:n]


def vote_evidence_for_side(vote: ModelVote, side: str) -> float:
    """Evidence strength one vote contributes toward finish above or below."""
    strength = vote.weight * vote.confidence
    if side == "yes":
        return strength * max(0.0, vote.prob_yes - 0.5) * 2.0
    return strength * max(0.0, 0.5 - vote.prob_yes) * 2.0


def directional_evidence(
    votes: tuple[ModelVote, ...] | list[ModelVote],
    n: int | None = None,
) -> DirectionalEvidence:
    """Sum evidence from top-N votes; pick finish above or below strike."""
    selected = top_votes(votes, n)
    above = sum(vote_evidence_for_side(v, "yes") for v in selected)
    below = sum(vote_evidence_for_side(v, "no") for v in selected)

    if above >= below:
        side = "yes"
        margin = above - below
    else:
        side = "no"
        margin = below - above

    return DirectionalEvidence(
        side=side,
        above_score=above,
        below_score=below,
        margin=margin,
        top_votes=tuple(selected),
    )


def evaluate_edge_with_evidence(
    p_fair: float,
    yes_ask: float,
    no_ask: float,
    yes_bid: float,
    no_bid: float,
    direction: DirectionalEvidence,
    *,
    fee_cents: float = config.FEE_PER_CONTRACT_CENTS,
    subtract_fees: bool | None = None,
    min_edge: float | None = None,
    min_evidence_margin: float | None = None,
    min_agreement: float | None = None,
    min_quorum: int | None = None,
    min_crowd_favorite: float | None = None,
    thresholds: DynamicThresholds | None = None,
    forecast: ForecastEnsembleOutput | None = None,
) -> TradeSignal:
    """Evaluate edge only on the evidence-backed side (above/below strike)."""
    if thresholds is not None:
        min_margin = min_evidence_margin if min_evidence_margin is not None else thresholds.min_evidence_margin
        edge_floor = min_edge if min_edge is not None else thresholds.min_edge_cents
        agree_floor = min_agreement if min_agreement is not None else thresholds.min_agreement
        quorum_floor = min_quorum if min_quorum is not None else thresholds.min_quorum
        crowd_floor = min_crowd_favorite if min_crowd_favorite is not None else thresholds.min_crowd_favorite
    else:
        min_margin = min_evidence_margin if min_evidence_margin is not None else config.MIN_EVIDENCE_MARGIN
        edge_floor = min_edge if min_edge is not None else config.MIN_EDGE_CENTS
        agree_floor = min_agreement if min_agreement is not None else config.ENSEMBLE_MIN_AGREEMENT
        quorum_floor = min_quorum if min_quorum is not None else config.CROWD_MIN_QUORUM
        crowd_floor = min_crowd_favorite if min_crowd_favorite is not None else config.CROWD_MIN_FAVORITE

    if forecast is not None and forecast.crowd.quorum_count < quorum_floor:
        crowd = forecast.crowd
        return TradeSignal(
            False,
            direction.side,
            p_fair,
            yes_ask if direction.side == "yes" else no_ask,
            0.0,
            0.0,
            f"Crowd quorum {crowd.quorum_count}/{quorum_floor} on {crowd.consensus_side.upper()}",
        )

    if forecast is not None and forecast.agreement_score < agree_floor:
        return TradeSignal(
            False,
            direction.side,
            p_fair,
            yes_ask if direction.side == "yes" else no_ask,
            0.0,
            0.0,
            f"Agreement {forecast.agreement_score:.0%} < min {agree_floor:.0%}",
        )

    if forecast is not None and not forecast.crowd.side_met(direction.side, min_favorite=crowd_floor):
        crowd = forecast.crowd
        finish = "ABOVE" if direction.side == "yes" else "BELOW"
        return TradeSignal(
            False,
            direction.side,
            p_fair,
            yes_ask if direction.side == "yes" else no_ask,
            0.0,
            0.0,
            f"Crowd {finish} {crowd.side_pct(direction.side):.1f}% < {crowd_floor * 100:.0f}%",
        )

    if direction.margin < min_margin:
        return TradeSignal(
            False,
            direction.side,
            p_fair,
            yes_ask if direction.side == "yes" else no_ask,
            0.0,
            0.0,
            f"Evidence margin {direction.margin:.3f} < min {min_margin:.3f} "
            f"(above={direction.above_score:.3f} below={direction.below_score:.3f})",
        )

    gate_fee = config.gate_fee_cents(fee_cents, subtract=subtract_fees)
    if direction.side == "yes":
        ev_dollars = p_fair - yes_ask
        edge_cents = ev_dollars * 100 - gate_fee
        price = yes_ask
    else:
        ev_dollars = (1.0 - p_fair) - no_ask
        edge_cents = ev_dollars * 100 - gate_fee
        price = no_ask

    if edge_cents < edge_floor:
        return TradeSignal(
            False,
            direction.side,
            p_fair,
            price,
            edge_cents,
            ev_dollars,
            f"Edge {edge_cents:.1f}c < min {edge_floor:.1f}c on {direction.finish_label}",
        )

    return TradeSignal(
        True,
        direction.side,
        p_fair,
        price,
        edge_cents,
        ev_dollars,
        f"Finish {direction.finish_label}: evidence={direction.margin:.3f} "
        f"edge={edge_cents:.1f}c @ {price:.2f}",
    )


@dataclass
class MarketCandidate:
    ticker: str
    strike: float
    secs_left: float
    forecast: ForecastEnsembleOutput
    direction: DirectionalEvidence
    edge: TradeSignal
    evidence_score: float
    market: dict
    thresholds: DynamicThresholds | None = None


def evidence_score(direction: DirectionalEvidence, forecast: ForecastEnsembleOutput) -> float:
    """Combined evidence strength for ranking among top markets."""
    win_score = direction.above_score if direction.side == "yes" else direction.below_score
    return win_score * forecast.confidence * forecast.agreement_score


def select_best_from_top_markets(
    candidates: list[MarketCandidate],
    n: int | None = None,
) -> MarketCandidate | None:
    """Take top-N markets by edge, return the one with strongest directional evidence."""
    n = n or config.TOP_N_MARKETS
    if not candidates:
        return None

    tradeable = [c for c in candidates if c.edge.should_trade]
    if not tradeable:
        return None

    top = sorted(tradeable, key=lambda c: c.edge.edge_cents, reverse=True)[:n]
    return max(top, key=lambda c: c.evidence_score)
