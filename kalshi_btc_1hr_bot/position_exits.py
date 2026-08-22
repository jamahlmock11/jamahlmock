"""Take-profit and stop-loss exits for open Kalshi positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kalshi_btc_1hr_bot.config import ExitConfig
from kalshi_btc_1hr_bot.kalshi_client import normalize_market


@dataclass(frozen=True)
class ExitLevels:
    take_profit_price: float
    stop_loss_price: float
    take_profit_cents: float
    stop_loss_cents: float


@dataclass(frozen=True)
class ExitSignal:
    reason: str  # take_profit | stop_loss
    exit_price: float
    bid_price: float
    pnl_per_contract: float
    pnl_total: float
    unrealized_pct: float


def compute_exit_levels(
    entry_price: float,
    cfg: ExitConfig,
    *,
    stored_tp: float | None = None,
    stored_sl: float | None = None,
) -> ExitLevels:
    """Derive TP/SL limit prices from entry and configured percentages."""
    if stored_tp is not None and stored_sl is not None:
        entry = max(0.01, min(0.99, float(entry_price)))
        return ExitLevels(
            take_profit_price=round(float(stored_tp), 4),
            stop_loss_price=round(float(stored_sl), 4),
            take_profit_cents=round((float(stored_tp) - entry) * 100, 2),
            stop_loss_cents=round((entry - float(stored_sl)) * 100, 2),
        )
    entry = max(0.01, min(0.99, float(entry_price)))
    tp = min(0.99, entry * (1.0 + cfg.take_profit_pct))
    sl = max(0.01, entry * (1.0 - cfg.stop_loss_pct))
    return ExitLevels(
        take_profit_price=round(tp, 4),
        stop_loss_price=round(sl, 4),
        take_profit_cents=round((tp - entry) * 100, 2),
        stop_loss_cents=round((entry - sl) * 100, 2),
    )


def bid_for_side(market: dict[str, Any], side: str) -> float | None:
    """Best bid we can hit when selling our held side."""
    side = side.lower()
    if side == "yes":
        bid = market.get("yes_bid")
        ask = market.get("yes_ask")
    else:
        bid = market.get("no_bid")
        ask = market.get("no_ask")
    if bid is not None and bid > 0:
        return float(bid)
    if ask is not None and ask > 0:
        return max(0.01, float(ask) - 0.01)
    return None


def evaluate_exit(
    *,
    entry_price: float,
    bid_price: float,
    contracts: int,
    cfg: ExitConfig,
    levels: ExitLevels | None = None,
) -> ExitSignal | None:
    """Return an exit signal when bid crosses TP or SL."""
    if not cfg.enabled or contracts <= 0 or bid_price <= 0:
        return None
    lv = levels or compute_exit_levels(entry_price, cfg)
    pnl_per = bid_price - entry_price
    pnl_total = pnl_per * contracts
    unrealized_pct = pnl_per / max(entry_price, 0.01)

    if bid_price >= lv.take_profit_price:
        return ExitSignal(
            reason="take_profit",
            exit_price=bid_price,
            bid_price=bid_price,
            pnl_per_contract=pnl_per,
            pnl_total=pnl_total,
            unrealized_pct=unrealized_pct,
        )
    if bid_price <= lv.stop_loss_price:
        return ExitSignal(
            reason="stop_loss",
            exit_price=bid_price,
            bid_price=bid_price,
            pnl_per_contract=pnl_per,
            pnl_total=pnl_total,
            unrealized_pct=unrealized_pct,
        )
    return None


def market_bid_from_raw(raw: dict[str, Any], side: str) -> float | None:
    return bid_for_side(normalize_market(raw), side)
