"""Expected value and edge calculation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kalshi_btc_1hr_bot.config import EdgeConfig
from kalshi_btc_1hr_bot.utils import quadratic_fee


class TradeSide(str, Enum):
    YES = "yes"
    NO = "no"
    NONE = "none"


@dataclass
class EdgeResult:
    side: TradeSide
    market_price: float
    p_fair: float
    raw_edge: float
    net_edge: float
    ev_per_contract: float
    should_trade: bool
    reason: str


def evaluate_edge(
    *,
    p_fair: float,
    yes_ask: float | None,
    no_ask: float | None,
    yes_bid: float | None = None,
    no_bid: float | None = None,
    edge_cfg: EdgeConfig | None = None,
) -> EdgeResult:
    """Evaluate YES and NO sides; return best actionable edge."""
    cfg = edge_cfg or EdgeConfig()
    best = EdgeResult(TradeSide.NONE, 0.0, p_fair, 0.0, 0.0, 0.0, False, "no quotes")

    candidates: list[tuple[TradeSide, float]] = []
    if yes_ask is not None and 0 < yes_ask < 1:
        candidates.append((TradeSide.YES, yes_ask))
    if no_ask is not None and 0 < no_ask < 1:
        candidates.append((TradeSide.NO, no_ask))

    for side, price in candidates:
        if side == TradeSide.YES:
            raw = p_fair - price
            win_prob = p_fair
        else:
            raw = (1.0 - p_fair) - price
            win_prob = 1.0 - p_fair

        fee = quadratic_fee(price, cfg.fee_rate)
        net = raw - fee
        ev = win_prob * (1.0 - price) - (1.0 - win_prob) * price - fee

        if net > best.net_edge:
            should = net >= cfg.min_edge
            reason = f"net_edge={net*100:.2f}¢" + ("" if should else f" < min {cfg.min_edge*100:.1f}¢")
            best = EdgeResult(side, price, p_fair, raw, net, ev, should, reason)

    return best
