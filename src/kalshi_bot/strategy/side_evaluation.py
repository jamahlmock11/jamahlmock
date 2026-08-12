"""Executable-side edge evaluation (model vs ask, fees, slippage)."""

from __future__ import annotations

from kalshi_bot.strategy.decision_record import SideEvaluation
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.rejection_codes import RejectionCode
from kalshi_bot.strategy.tiered_edge import estimate_slippage
from kalshi_bot.strategy.v6_upgrades import strict_edge_gap_dollars


def evaluate_side(
    *,
    side: str,
    model_prob: float,
    ask: float | None,
    min_edge: float,
    fee_rate: float = 0.07,
    spread: float = 0.0,
    liquidity_score: float = 0.0,
) -> SideEvaluation:
    if ask is None or not (0 < ask < 1):
        return SideEvaluation(
            side=side,
            model_probability=model_prob,
            executable_ask=None,
            raw_edge_dollars=0.0,
            estimated_fee=0.0,
            estimated_slippage=0.0,
            net_edge_dollars=0.0,
            expected_value_per_contract=0.0,
            passes_edge_threshold=False,
            passes_net_ev=False,
            rejection_codes=[RejectionCode.MISSING_DATA],
        )

    raw = strict_edge_gap_dollars(model_prob, ask)
    fee = quadratic_fee_per_contract(ask, fee_rate=fee_rate)
    slip = estimate_slippage(spread=spread, liquidity_score=liquidity_score)
    net = raw - fee - slip
    ev = model_prob - ask - fee - slip
    passes_edge = raw >= min_edge
    passes_ev = net > 0

    rejections: list[RejectionCode] = []
    if not passes_edge:
        rejections.append(RejectionCode.EDGE_TOO_SMALL)
    if not passes_ev:
        rejections.append(RejectionCode.EXPECTED_VALUE_NEGATIVE)

    return SideEvaluation(
        side=side,
        model_probability=model_prob,
        executable_ask=ask,
        raw_edge_dollars=raw,
        estimated_fee=fee,
        estimated_slippage=slip,
        net_edge_dollars=net,
        expected_value_per_contract=ev,
        passes_edge_threshold=passes_edge,
        passes_net_ev=passes_ev,
        rejection_codes=rejections,
    )
