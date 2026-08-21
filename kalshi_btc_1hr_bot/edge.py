"""Edge calculation — identical structure to 15m bot.

Compares fair probability to Kalshi implied price; trades only when EV > threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kalshi_btc_1hr_bot import config

log = logging.getLogger("edge")


@dataclass
class TradeSignal:
    should_trade: bool
    side: str  # "yes" or "no"
    p_fair: float
    market_price: float  # Kalshi ask for chosen side (decimal 0-1)
    edge_cents: float
    ev_per_contract: float
    reason: str


def evaluate_edge(
    p_fair: float,
    yes_ask: float,
    no_ask: float,
    yes_bid: float,
    no_bid: float,
    fee_cents: float = config.FEE_PER_CONTRACT_CENTS,
    min_edge: float = config.MIN_EDGE_CENTS,
    *,
    subtract_fees: bool | None = None,
) -> TradeSignal:
    gate_fee = config.gate_fee_cents(fee_cents, subtract=subtract_fees)
    ev_yes_dollars = p_fair * 1.0 - yes_ask
    edge_yes_cents = ev_yes_dollars * 100 - gate_fee

    p_no_fair = 1.0 - p_fair
    ev_no_dollars = p_no_fair * 1.0 - no_ask
    edge_no_cents = ev_no_dollars * 100 - gate_fee

    if edge_yes_cents >= edge_no_cents:
        best_side, best_edge, best_price, best_ev = "yes", edge_yes_cents, yes_ask, ev_yes_dollars
    else:
        best_side, best_edge, best_price, best_ev = "no", edge_no_cents, no_ask, ev_no_dollars

    if best_edge < min_edge:
        return TradeSignal(
            False,
            best_side,
            p_fair,
            best_price,
            best_edge,
            best_ev,
            f"Edge {best_edge:.1f}c < min {min_edge:.1f}c",
        )

    return TradeSignal(
        True,
        best_side,
        p_fair,
        best_price,
        best_edge,
        best_ev,
        f"Edge {best_edge:.1f}c on {best_side.upper()} @ {best_price:.2f} fair={p_fair:.3f}",
    )


def vwap_fill_price(book_side: list[tuple[float, float]], quantity: int) -> float:
    filled, total_cost = 0, 0.0
    for price, size in book_side:
        if filled >= quantity:
            break
        take = min(size, quantity - filled)
        total_cost += price * take
        filled += take
    return total_cost / filled if filled > 0 else 0.0
