"""Main paper/live trading loop for KXBTCD 1-hour bot."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from kalshi_btc_1hr_bot.config import BotConfig, load_config
from kalshi_btc_1hr_bot.data_feed import DataFeed
from kalshi_btc_1hr_bot.edge import TradeSignal, evaluate_edge
from kalshi_btc_1hr_bot.kalshi_client import KalshiClient, is_hourly_market, normalize_market
from kalshi_btc_1hr_bot.forecast import ForecastEnsemble, forecast_ensemble_from_market_data
from kalshi_btc_1hr_bot.risk import RiskManager
from kalshi_btc_1hr_bot.sizing import kelly_contracts
from kalshi_btc_1hr_bot.utils import setup_logging

logger = logging.getLogger(__name__)


class HourlyBot:
    def __init__(self, config: BotConfig | None = None) -> None:
        self.config = config or load_config()
        self.client = KalshiClient(self.config)
        self.feed = DataFeed(kalshi_client=self.client)
        self.ensemble = ForecastEnsemble()
        self.risk = RiskManager(self.config)

    def close(self) -> None:
        self.client.close()
        self.feed.close()

    def run_cycle(self) -> list[dict]:
        """Scan open KXBTCD markets and return trade decisions."""
        now = datetime.now(timezone.utc)
        data = self.feed.refresh()
        decisions: list[dict] = []
        scanned = 0

        try:
            for raw in self.client.iter_markets(self.config.series_ticker, status="open"):
                if not is_hourly_market(raw):
                    continue
                market = normalize_market(raw)
                close = market.get("close_time")
                strike = market.get("strike")
                if close is None or strike is None:
                    continue

                secs = (close - now).total_seconds()
                if secs < self.config.risk.min_seconds_to_expiry:
                    continue
                if secs > self.config.risk.max_seconds_to_expiry:
                    continue

                yes_ask = market.get("yes_ask")
                no_ask = market.get("no_ask")
                if yes_ask is None and no_ask is None:
                    continue

                scanned += 1
                ticker = str(market.get("ticker") or "")

                forecast = forecast_ensemble_from_market_data(
                    self.ensemble,
                    spot=data.spot,
                    strike=float(strike),
                    seconds_to_expiry=secs,
                    data=data,
                )

                yes_ask_f = float(yes_ask) if yes_ask is not None else 1.0
                no_ask_f = float(no_ask) if no_ask is not None else 1.0
                yes_bid_f = float(market.get("yes_bid") or yes_ask_f)
                no_bid_f = float(market.get("no_bid") or no_ask_f)

                edge = evaluate_edge(
                    forecast.p_fair,
                    yes_ask_f,
                    no_ask_f,
                    yes_bid_f,
                    no_bid_f,
                    fee_cents=self.config.edge.fee_per_contract_cents,
                    min_edge=self.config.edge.min_edge_cents,
                )

                allowed, block_reason = self.risk.allow_trade(ticker=ticker, seconds_to_expiry=secs)
                win_prob = forecast.p_fair if edge.side == "yes" else 1.0 - forecast.p_fair
                contracts = 0
                if edge.should_trade and allowed:
                    contracts = kelly_contracts(
                        win_prob=win_prob,
                        price=edge.market_price,
                        sizing=self.config.sizing,
                        confidence=forecast.confidence,
                    )

                action = "NO_TRADE"
                if edge.should_trade and allowed and contracts > 0:
                    action = f"BUY_{edge.side.upper()}"
                    self._execute(ticker, edge.side, contracts, edge.market_price)

                decision = {
                    "ticker": ticker,
                    "action": action,
                    "p_fair": forecast.p_fair,
                    "confidence": forecast.confidence,
                    "regime": forecast.vol_regime,
                    "spot": data.spot,
                    "strike": strike,
                    "secs_left": secs,
                    "edge": edge.edge_cents / 100.0,
                    "side": edge.side,
                    "price": edge.market_price,
                    "contracts": contracts,
                    "reason": edge.reason if edge.should_trade else block_reason,
                    "layers": forecast.layers,
                    "votes": [(v.name, v.prob_yes, v.weight) for v in forecast.votes],
                    "agreement": forecast.agreement_score,
                    "brti_official": forecast.is_official_brti,
                    "brti_source": data.source,
                }
                decisions.append(decision)

                logger.info(
                    "%s %s | fair=%.1f%% edge=%.1f¢ conf=%.0f%% agree=%.0f%% brti=%s t=%.0fs",
                    action,
                    ticker,
                    forecast.p_fair * 100,
                    edge.edge_cents,
                    forecast.confidence * 100,
                    forecast.agreement_score * 100,
                    data.source,
                    secs,
                )
        except Exception:
            logger.exception("market scan failed")

        if scanned == 0:
            logger.info(
                "No open hourly markets in window (brti=%.2f source=%s official=%s funding=%.5f)",
                data.spot,
                data.source,
                data.is_official,
                data.funding_rate,
            )
        return decisions

    def _execute(self, ticker: str, side: str, contracts: int, price: float) -> None:
        cost = contracts * price
        if self.config.paper:
            logger.info(
                "PAPER %s %s x%d @ %.0f¢ ($%.2f)",
                side.upper(),
                ticker,
                contracts,
                price * 100,
                cost,
            )
            self.risk.register_trade(ticker, cost)
            return

        try:
            price_cents = int(round(price * 100))
            self.client.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                price_cents=price_cents,
            )
            self.risk.register_trade(ticker, cost)
            logger.info("LIVE order placed: %s %s x%d", side.upper(), ticker, contracts)
        except Exception:
            logger.exception("order failed for %s", ticker)


def run_once() -> None:
    bot = HourlyBot()
    try:
        decisions = bot.run_cycle()
        trades = [d for d in decisions if d["action"] != "NO_TRADE"]
        if not trades:
            print("NO TRADE this cycle")
        else:
            for d in trades:
                print(
                    f"{d['action']} {d['ticker']} x{d['contracts']} "
                    f"@ {d['price']*100:.0f}¢ (fair={d['p_fair']:.1%}, edge={d['edge']*100:.1f}¢)"
                )
    finally:
        bot.close()


def run_loop(max_cycles: int | None = None) -> None:
    bot = HourlyBot()
    cycles = 0
    try:
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            bot.run_cycle()
            time.sleep(bot.config.cycle_seconds)
    except KeyboardInterrupt:
        logger.info("stopped by user")
    finally:
        bot.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KXBTCD 1-hour forecasting bot")
    parser.add_argument("--paper", action="store_true", default=True, help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading (requires API keys)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--cycles", type=int, default=None, help="Run N cycles then exit")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config()
    if args.live:
        cfg.paper = False

    if args.once:
        run_once()
    else:
        run_loop(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
