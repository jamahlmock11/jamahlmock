"""Kalshi Pro-style depth and order-flow analytics from orderbook + trade data."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from kalshi_btc_1hr_bot.edge import vwap_fill_price


@dataclass(frozen=True)
class DepthLevel:
    price: float
    size: int
    price_cents: int
    cumulative: int = 0


@dataclass
class TradePrint:
    ts: float
    side: str  # yes | no (taker)
    price: float
    count: int
    price_cents: int


@dataclass
class ProMarketAnalytics:
    ticker: str
    updated_at: float
    source: str = "rest"
    latency_ms: float | None = None
    yes_bid: float | None = None
    yes_ask: float | None = None
    no_bid: float | None = None
    no_ask: float | None = None
    yes_spread_cents: float | None = None
    no_spread_cents: float | None = None
    obi: float = 0.0
    yes_bid_depth: int = 0
    no_bid_depth: int = 0
    yes_ask_depth: int = 0  # liquidity available to buy YES (from no bids)
    no_ask_depth: int = 0
    yes_bids: list[DepthLevel] = field(default_factory=list)
    no_bids: list[DepthLevel] = field(default_factory=list)
    yes_asks: list[DepthLevel] = field(default_factory=list)  # implied from no bids
    no_asks: list[DepthLevel] = field(default_factory=list)  # implied from yes bids
    recent_trades: list[TradePrint] = field(default_factory=list)
    flow_yes_volume: int = 0
    flow_no_volume: int = 0
    flow_net_side: str = ""
    vwap_buy_yes_1: float | None = None
    vwap_buy_no_1: float | None = None
    slippage_buy_yes_1_cents: float | None = None
    execution_score: int = 0  # 0-100
    execution_note: str = ""


def parse_orderbook_levels(raw: dict[str, Any]) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Parse Kalshi orderbook_fp yes/no bid ladders."""
    book = raw.get("orderbook_fp") or raw.get("orderbook") or raw
    yes_rows = book.get("yes_dollars") or book.get("yes") or []
    no_rows = book.get("no_dollars") or book.get("no") or []

    def _rows(items: list) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        for row in items:
            if not row or len(row) < 2:
                continue
            price = float(row[0])
            qty = int(float(row[1]))
            if qty > 0 and price > 0:
                out.append((price, qty))
        out.sort(key=lambda x: -x[0])
        return out

    return _rows(yes_rows), _rows(no_rows)


def implied_ask_ladder(no_bids: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """NO bids imply YES asks at (1 - no_bid_price)."""
    asks = [(round(1.0 - price, 4), qty) for price, qty in no_bids]
    asks.sort(key=lambda x: x[0])
    return asks


def implied_no_ask_ladder(yes_bids: list[tuple[float, int]]) -> list[tuple[float, int]]:
    asks = [(round(1.0 - price, 4), qty) for price, qty in yes_bids]
    asks.sort(key=lambda x: x[0])
    return asks


def _levels_with_cumulative(rows: list[tuple[float, int]], *, limit: int = 10) -> list[DepthLevel]:
    out: list[DepthLevel] = []
    cum = 0
    for price, size in rows[:limit]:
        cum += size
        out.append(
            DepthLevel(
                price=price,
                size=size,
                price_cents=int(round(price * 100)),
                cumulative=cum,
            )
        )
    return out


def order_book_imbalance(yes_bids: list[tuple[float, int]], no_bids: list[tuple[float, int]], levels: int = 5) -> float:
    yes_vol = sum(q for _, q in yes_bids[:levels])
    no_vol = sum(q for _, q in no_bids[:levels])
    total = yes_vol + no_vol
    if total <= 0:
        return 0.0
    return (yes_vol - no_vol) / total


def execution_score(
    *,
    spread_cents: float | None,
    ask_depth: int,
    obi: float,
    side: str,
    latency_ms: float | None,
) -> tuple[int, str]:
    score = 50
    notes: list[str] = []
    if spread_cents is not None:
        if spread_cents <= 2:
            score += 25
            notes.append("tight spread")
        elif spread_cents <= 5:
            score += 12
        elif spread_cents >= 10:
            score -= 20
            notes.append("wide spread")
    if ask_depth >= 500:
        score += 20
        notes.append("deep book")
    elif ask_depth >= 100:
        score += 10
    elif ask_depth < 25:
        score -= 15
        notes.append("thin liquidity")
    if side == "yes" and obi > 0.15:
        score += 8
    elif side == "no" and obi < -0.15:
        score += 8
    if latency_ms is not None:
        if latency_ms <= 250:
            score += 5
        elif latency_ms > 2000:
            score -= 10
            notes.append("stale feed")
    score = max(0, min(100, score))
    note = ", ".join(notes) if notes else "normal conditions"
    return score, note


def build_pro_analytics(
    ticker: str,
    orderbook_raw: dict[str, Any],
    *,
    side: str = "yes",
    recent_trades: list[TradePrint] | None = None,
    source: str = "rest",
    latency_ms: float | None = None,
    depth_limit: int = 10,
    contracts: int = 1,
) -> ProMarketAnalytics:
    yes_bids_raw, no_bids_raw = parse_orderbook_levels(orderbook_raw)
    yes_asks_raw = implied_ask_ladder(no_bids_raw)
    no_asks_raw = implied_no_ask_ladder(yes_bids_raw)

    yes_bid = yes_bids_raw[0][0] if yes_bids_raw else None
    no_bid = no_bids_raw[0][0] if no_bids_raw else None
    yes_ask = yes_asks_raw[0][0] if yes_asks_raw else None
    no_ask = no_asks_raw[0][0] if no_asks_raw else None

    yes_spread = (yes_ask - yes_bid) * 100 if yes_bid is not None and yes_ask is not None else None
    no_spread = (no_ask - no_bid) * 100 if no_bid is not None and no_ask is not None else None

    obi = order_book_imbalance(yes_bids_raw, no_bids_raw)
    yes_bid_depth = sum(q for _, q in yes_bids_raw[:depth_limit])
    no_bid_depth = sum(q for _, q in no_bids_raw[:depth_limit])
    yes_ask_depth = sum(q for _, q in yes_asks_raw[:depth_limit])
    no_ask_depth = sum(q for _, q in no_asks_raw[:depth_limit])

    trades = list(recent_trades or [])
    flow_yes = sum(t.count for t in trades if t.side == "yes")
    flow_no = sum(t.count for t in trades if t.side == "no")
    flow_net = ""
    if flow_yes > flow_no:
        flow_net = "yes"
    elif flow_no > flow_yes:
        flow_net = "no"

    vwap_yes = vwap_fill_price(yes_asks_raw, contracts) if yes_asks_raw else None
    vwap_no = vwap_fill_price(no_asks_raw, contracts) if no_asks_raw else None
    slip_yes = (vwap_yes - yes_ask) * 100 if vwap_yes is not None and yes_ask is not None else None

    trade_side = side.lower()
    spread = yes_spread if trade_side == "yes" else no_spread
    ask_depth = yes_ask_depth if trade_side == "yes" else no_ask_depth
    score, note = execution_score(
        spread_cents=spread,
        ask_depth=ask_depth,
        obi=obi,
        side=trade_side,
        latency_ms=latency_ms,
    )

    return ProMarketAnalytics(
        ticker=ticker,
        updated_at=time.time(),
        source=source,
        latency_ms=latency_ms,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        yes_spread_cents=round(yes_spread, 2) if yes_spread is not None else None,
        no_spread_cents=round(no_spread, 2) if no_spread is not None else None,
        obi=round(obi, 4),
        yes_bid_depth=yes_bid_depth,
        no_bid_depth=no_bid_depth,
        yes_ask_depth=yes_ask_depth,
        no_ask_depth=no_ask_depth,
        yes_bids=_levels_with_cumulative(yes_bids_raw, limit=depth_limit),
        no_bids=_levels_with_cumulative(no_bids_raw, limit=depth_limit),
        yes_asks=_levels_with_cumulative(yes_asks_raw, limit=depth_limit),
        no_asks=_levels_with_cumulative(no_asks_raw, limit=depth_limit),
        recent_trades=trades[:20],
        flow_yes_volume=flow_yes,
        flow_no_volume=flow_no,
        flow_net_side=flow_net,
        vwap_buy_yes_1=vwap_yes,
        vwap_buy_no_1=vwap_no,
        slippage_buy_yes_1_cents=round(slip_yes, 2) if slip_yes is not None else None,
        execution_score=score,
        execution_note=note,
    )


def pro_analytics_to_dict(pro: ProMarketAnalytics) -> dict[str, Any]:
    return {
        "ticker": pro.ticker,
        "updated_at": pro.updated_at,
        "source": pro.source,
        "latency_ms": pro.latency_ms,
        "top_of_book": {
            "yes_bid_cents": int(round(pro.yes_bid * 100)) if pro.yes_bid else None,
            "yes_ask_cents": int(round(pro.yes_ask * 100)) if pro.yes_ask else None,
            "no_bid_cents": int(round(pro.no_bid * 100)) if pro.no_bid else None,
            "no_ask_cents": int(round(pro.no_ask * 100)) if pro.no_ask else None,
            "yes_spread_cents": pro.yes_spread_cents,
            "no_spread_cents": pro.no_spread_cents,
        },
        "obi": pro.obi,
        "depth": {
            "yes_bid_contracts": pro.yes_bid_depth,
            "no_bid_contracts": pro.no_bid_depth,
            "yes_ask_contracts": pro.yes_ask_depth,
            "no_ask_contracts": pro.no_ask_depth,
        },
        "yes_bids": [
            {"price_cents": lv.price_cents, "size": lv.size, "cumulative": lv.cumulative} for lv in pro.yes_bids
        ],
        "no_bids": [
            {"price_cents": lv.price_cents, "size": lv.size, "cumulative": lv.cumulative} for lv in pro.no_bids
        ],
        "yes_asks": [
            {"price_cents": lv.price_cents, "size": lv.size, "cumulative": lv.cumulative} for lv in pro.yes_asks
        ],
        "no_asks": [
            {"price_cents": lv.price_cents, "size": lv.size, "cumulative": lv.cumulative} for lv in pro.no_asks
        ],
        "order_flow": {
            "yes_volume": pro.flow_yes_volume,
            "no_volume": pro.flow_no_volume,
            "net_side": pro.flow_net_side,
            "recent": [
                {
                    "ts": t.ts,
                    "side": t.side.upper(),
                    "price_cents": t.price_cents,
                    "count": t.count,
                }
                for t in pro.recent_trades
            ],
        },
        "execution": {
            "vwap_buy_yes_cents": int(round(pro.vwap_buy_yes_1 * 100)) if pro.vwap_buy_yes_1 else None,
            "vwap_buy_no_cents": int(round(pro.vwap_buy_no_1 * 100)) if pro.vwap_buy_no_1 else None,
            "slippage_buy_yes_cents": pro.slippage_buy_yes_1_cents,
            "score": pro.execution_score,
            "note": pro.execution_note,
        },
    }
