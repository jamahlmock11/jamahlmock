"""Lightweight historical edge replay against recorded snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_bot.config import SeriesConfig, SmileConfig
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.mispricing import evaluate_market


@dataclass
class BacktestTrade:
    ticker: str
    side: str
    price: float
    true_prob: float
    edge_after_fees_pp: float
    settled_yes: bool
    pnl: float


@dataclass
class BacktestReport:
    trades: list[BacktestTrade]
    total_pnl: float
    hit_rate: float
    avg_edge_pp: float


def settle_binary(side: str, settled_yes: bool, price: float) -> float:
    """PnL per contract excluding fees (fees deducted separately if desired)."""
    if side == "yes":
        return (1.0 - price) if settled_yes else (-price)
    return (1.0 - price) if (not settled_yes) else (-price)


def run_backtest(
    snapshots: list[dict],
    *,
    smile: VolSmile,
    series_cfg: SeriesConfig,
    smile_cfg: SmileConfig,
    fee_rate: float = 0.07,
) -> BacktestReport:
    """Each snapshot: market dict + spot + settled_yes (+ optional now)."""
    trades: list[BacktestTrade] = []
    for snap in snapshots:
        market = snap["market"]
        mis = evaluate_market(
            market,
            spot=float(snap["spot"]),
            smile=smile,
            series_cfg=series_cfg,
            smile_cfg=smile_cfg,
            fee_rate=fee_rate,
            now=snap.get("now"),
        )
        if mis is None:
            continue
        settled_yes = bool(snap["settled_yes"])
        fee = quadratic_fee_per_contract(mis.kalshi_price, fee_rate=fee_rate)
        pnl = settle_binary(mis.side.value, settled_yes, mis.kalshi_price) - fee
        trades.append(
            BacktestTrade(
                ticker=mis.ticker,
                side=mis.side.value,
                price=mis.kalshi_price,
                true_prob=mis.options_prob,
                edge_after_fees_pp=mis.edge_after_fees_pp,
                settled_yes=settled_yes,
                pnl=pnl,
            )
        )
    if not trades:
        return BacktestReport([], 0.0, 0.0, 0.0)
    wins = sum(
        1
        for t in trades
        if (t.side == "yes" and t.settled_yes) or (t.side == "no" and not t.settled_yes)
    )
    return BacktestReport(
        trades=trades,
        total_pnl=sum(t.pnl for t in trades),
        hit_rate=wins / len(trades),
        avg_edge_pp=sum(t.edge_after_fees_pp for t in trades) / len(trades),
    )
