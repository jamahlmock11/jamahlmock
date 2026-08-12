"""Kalshi mispricing engine — fair value vs executable price after costs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.settlement_probability import SettlementProbability
from kalshi_bot.strategy.tiered_edge import estimate_slippage
from kalshi_bot.strategy.v6_upgrades import MicrostructureSnapshot


class TradeAction(str, Enum):
    BUY_YES = "BUY YES"
    BUY_NO = "BUY NO"
    WAIT = "WAIT"
    NO_TRADE = "NO TRADE"


@dataclass(frozen=True)
class SideMispricing:
    side: str  # YES | NO
    model_probability: float
    fair_value: float
    executable_ask: float | None
    spread: float
    fee: float
    slippage: float
    raw_edge_dollars: float
    net_edge_dollars: float
    expected_value: float


@dataclass(frozen=True)
class MispricingOpportunity:
    ticker: str
    strike: float
    seconds_to_expiry: float
    settlement: SettlementProbability
    yes: SideMispricing
    no: SideMispricing
    best_side: str | None
    best_net_edge: float
    kalshi_stale: bool
    liquidity_label: str
    order_flow_label: str
    volatility_label: str
    confidence_label: str

    @property
    def model_yes_pct(self) -> float:
        return self.settlement.prob_above_strike * 100

    @property
    def fair_value_yes(self) -> float:
        return self.settlement.prob_above_strike

    def best_mispricing(self) -> SideMispricing | None:
        if self.best_side == "YES":
            return self.yes
        if self.best_side == "NO":
            return self.no
        return None


def _side_mispricing(
    *,
    side: str,
    model_prob: float,
    ask: float | None,
    spread: float,
    liquidity_score: float,
    fee_rate: float = 0.07,
) -> SideMispricing:
    fair = model_prob
    if ask is None or not (0 < ask < 1):
        return SideMispricing(
            side=side,
            model_probability=model_prob,
            fair_value=fair,
            executable_ask=None,
            spread=spread,
            fee=0.0,
            slippage=0.0,
            raw_edge_dollars=0.0,
            net_edge_dollars=0.0,
            expected_value=0.0,
        )
    raw = round(model_prob - ask, 4)
    fee = quadratic_fee_per_contract(ask, fee_rate=fee_rate)
    slip = estimate_slippage(spread=spread, liquidity_score=liquidity_score)
    net = raw - fee - slip
    ev = model_prob - ask - fee - slip
    return SideMispricing(
        side=side,
        model_probability=model_prob,
        fair_value=fair,
        executable_ask=ask,
        spread=spread,
        fee=fee,
        slippage=slip,
        raw_edge_dollars=raw,
        net_edge_dollars=net,
        expected_value=ev,
    )


def _liquidity_label(score: float) -> str:
    if score >= 0.35:
        return "GOOD"
    if score >= 0.15:
        return "FAIR"
    return "THIN"


def _confidence_label(conf: float) -> str:
    if conf >= 0.70:
        return "HIGH"
    if conf >= 0.50:
        return "MEDIUM"
    return "LOW"


def evaluate_mispricing(
    *,
    ticker: str,
    strike: float,
    seconds_to_expiry: float,
    settlement: SettlementProbability,
    yes_ask: float | None,
    no_ask: float | None,
    micro: MicrostructureSnapshot,
    order_flow_label: str,
    volatility_label: str,
    kalshi_stale: bool = False,
    fee_rate: float = 0.07,
) -> MispricingOpportunity:
    spread = micro.spread
    yes = _side_mispricing(
        side="YES",
        model_prob=settlement.prob_above_strike,
        ask=yes_ask,
        spread=spread,
        liquidity_score=micro.liquidity_score,
        fee_rate=fee_rate,
    )
    no = _side_mispricing(
        side="NO",
        model_prob=settlement.prob_below_strike,
        ask=no_ask,
        spread=spread,
        liquidity_score=micro.liquidity_score,
        fee_rate=fee_rate,
    )
    best_side: str | None = None
    best_net = max(yes.net_edge_dollars, no.net_edge_dollars)
    if yes.executable_ask is not None and (
        best_side is None or yes.net_edge_dollars >= no.net_edge_dollars
    ):
        best_side = "YES" if yes.net_edge_dollars >= no.net_edge_dollars else best_side
    if no.executable_ask is not None and (
        best_side is None or no.net_edge_dollars > yes.net_edge_dollars
    ):
        best_side = "NO"
    if best_side is None:
        best_side = "YES" if yes.net_edge_dollars >= no.net_edge_dollars else "NO"

    return MispricingOpportunity(
        ticker=ticker,
        strike=strike,
        seconds_to_expiry=seconds_to_expiry,
        settlement=settlement,
        yes=yes,
        no=no,
        best_side=best_side if (yes.executable_ask or no.executable_ask) else None,
        best_net_edge=best_net,
        kalshi_stale=kalshi_stale,
        liquidity_label=_liquidity_label(micro.liquidity_score),
        order_flow_label=order_flow_label,
        volatility_label=volatility_label,
        confidence_label=_confidence_label(settlement.confidence),
    )
