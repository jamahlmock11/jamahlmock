"""Synthetic and replay backtester for the 1-hour bot."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np

from kalshi_btc_1hr_bot.config import BotConfig, load_config
from kalshi_btc_1hr_bot.data_feed import FundingRate, MarketData, SyntheticPriceGenerator
from kalshi_btc_1hr_bot.edge import evaluate_edge
from kalshi_btc_1hr_bot.forecast import ForecastEnsemble
from kalshi_btc_1hr_bot.model import build_market_state
from kalshi_btc_1hr_bot.sizing import kelly_contracts
from kalshi_btc_1hr_bot.utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    market_idx: int
    side: str
    price: float
    p_fair: float
    net_edge: float
    contracts: int
    settled_yes: bool
    pnl: float


@dataclass
class BacktestReport:
    trades: list[BacktestTrade] = field(default_factory=list)
    markets_simulated: int = 0
    total_pnl: float = 0.0
    hit_rate: float = 0.0
    avg_edge: float = 0.0
    trade_rate: float = 0.0

    def summary(self) -> str:
        return (
            f"Markets: {self.markets_simulated} | Trades: {len(self.trades)} "
            f"({self.trade_rate:.1%}) | PnL: ${self.total_pnl:.2f} | "
            f"Hit rate: {self.hit_rate:.1%} | Avg edge: {self.avg_edge*100:.2f}¢"
        )


def _settlement_average(path: np.ndarray, window: int = 60) -> float:
    return float(np.mean(path[-window:]))


def _synthetic_market_data(
    path: np.ndarray,
    entry_idx: int,
    funding: float,
) -> MarketData:
    segment = path[max(0, entry_idx - 1800) : entry_idx + 1]
    if len(segment) < 2:
        segment = path[: entry_idx + 1]
    now = float(entry_idx)
    bar_seconds = 1.0
    history = [(now - (len(segment) - 1 - i) * bar_seconds, float(segment[i])) for i in range(len(segment))]
    spot = float(segment[-1])
    vwap = float(np.mean(segment))

    return MarketData(
        spot=spot,
        vwap=vwap,
        funding_rate=funding,
        annualized_vol=0.5,
        mu_5m=0.0,
        mu_15m=0.0,
        mu_30m=0.0,
        closes_1m=segment[-60:] if len(segment) >= 60 else segment,
        price_history=history,
        funding=FundingRate(funding_rate=funding),
        timestamp=now,
        is_official=True,
        source="synthetic_brti",
    )


def run_synthetic_backtest(
    *,
    n_markets: int = 100,
    seed: int = 42,
    config: BotConfig | None = None,
) -> BacktestReport:
    cfg = config or load_config()
    model = ForecastEnsemble()
    gen = SyntheticPriceGenerator(seed=seed)
    report = BacktestReport(markets_simulated=n_markets)

    spot0 = 65000.0
    for i in range(n_markets):
        path, funding, _ = gen.next_hour_path(spot0=spot0)
        strike = float(path[0])
        settlement = _settlement_average(path)
        settled_yes = settlement > strike

        # Enter with 20 minutes left (1200s into the hour)
        entry_idx = 2400
        secs_left = 3600 - entry_idx
        data = _synthetic_market_data(path, entry_idx, funding)

        forecast = model.forecast(
            build_market_state(
                spot=data.spot,
                strike=strike,
                seconds_remaining=secs_left,
                price_history=data.price_history,
                vwap=data.vwap,
                funding=data.funding,
                now_ts=data.timestamp,
                is_official_brti=data.is_official,
            )
        )

        # Synthetic market price: fair + noise
        noise = np.random.default_rng(seed + i).normal(0, 0.04)
        market_yes = float(np.clip(forecast.p_fair + noise, 0.05, 0.95))
        market_no = float(np.clip(1.0 - market_yes + np.random.default_rng(seed + i + 1).normal(0, 0.02), 0.05, 0.95))

        edge = evaluate_edge(
            forecast.p_fair,
            market_yes,
            market_no,
            market_yes - 0.02,
            market_no - 0.02,
            fee_cents=cfg.edge.fee_per_contract_cents,
            min_edge=cfg.edge.min_edge_cents,
        )

        if not edge.should_trade:
            spot0 = float(path[-1])
            continue

        win_prob = forecast.p_fair if edge.side == "yes" else 1.0 - forecast.p_fair
        contracts = kelly_contracts(
            win_prob=win_prob,
            price=edge.market_price,
            sizing=cfg.sizing,
            confidence=forecast.confidence,
        )
        if contracts == 0:
            spot0 = float(path[-1])
            continue

        won = (edge.side == "yes" and settled_yes) or (
            edge.side == "no" and not settled_yes
        )
        fee = cfg.edge.fee_per_contract_cents / 100.0
        if won:
            pnl = contracts * ((1.0 - edge.market_price) - fee)
        else:
            pnl = contracts * (-edge.market_price - fee)

        report.trades.append(
            BacktestTrade(
                market_idx=i,
                side=edge.side,
                price=edge.market_price,
                p_fair=forecast.p_fair,
                net_edge=edge.edge_cents / 100.0,
                contracts=contracts,
                settled_yes=settled_yes,
                pnl=pnl,
            )
        )
        spot0 = float(path[-1])

    if report.trades:
        wins = sum(
            1
            for t in report.trades
            if (t.side == "yes" and t.settled_yes) or (t.side == "no" and not t.settled_yes)
        )
        report.hit_rate = wins / len(report.trades)
        report.total_pnl = sum(t.pnl for t in report.trades)
        report.avg_edge = sum(t.net_edge for t in report.trades) / len(report.trades)
    report.trade_rate = len(report.trades) / max(n_markets, 1)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KXBTCD 1-hour bot backtester")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic backtest")
    parser.add_argument("--n-markets", type=int, default=100, help="Number of synthetic markets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    setup_logging()

    if args.synthetic:
        report = run_synthetic_backtest(n_markets=args.n_markets, seed=args.seed)
        print(report.summary())
        for t in report.trades[:10]:
            print(
                f"  #{t.market_idx} {t.side.upper()} @ {t.price*100:.0f}¢ "
                f"fair={t.p_fair:.1%} edge={t.net_edge*100:.1f}¢ "
                f"x{t.contracts} pnl=${t.pnl:.2f}"
            )
        if len(report.trades) > 10:
            print(f"  ... and {len(report.trades) - 10} more trades")
    else:
        print("Use --synthetic for demo backtest (no API keys needed)")


if __name__ == "__main__":
    main()
