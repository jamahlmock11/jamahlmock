"""Main paper/live trading loop for KXBTCD 1-hour bot."""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone

from kalshi_btc_1hr_bot.config import BotConfig, load_config, require_live_credentials
from kalshi_btc_1hr_bot.data_feed import DataFeed
from kalshi_btc_1hr_bot.evidence import (
    MarketCandidate,
    directional_evidence,
    evaluate_edge_with_evidence,
    evidence_score,
    select_best_from_top_markets,
)
from kalshi_btc_1hr_bot.forecast import ForecastEnsemble, forecast_ensemble_from_market_data
from kalshi_btc_1hr_bot.kalshi_client import KalshiClient, is_hourly_market, normalize_market
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
        """Scan KXBTCD markets, rank top 4 by edge, trade the strongest evidence pick."""
        now = datetime.now(timezone.utc)
        data = self.feed.refresh()
        candidates: list[MarketCandidate] = []
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

                direction = directional_evidence(forecast.votes)

                yes_ask_f = float(yes_ask) if yes_ask is not None else 1.0
                no_ask_f = float(no_ask) if no_ask is not None else 1.0
                yes_bid_f = float(market.get("yes_bid") or yes_ask_f)
                no_bid_f = float(market.get("no_bid") or no_ask_f)

                edge = evaluate_edge_with_evidence(
                    forecast.p_fair,
                    yes_ask_f,
                    no_ask_f,
                    yes_bid_f,
                    no_bid_f,
                    direction,
                    fee_cents=self.config.edge.fee_per_contract_cents,
                    min_edge=self.config.edge.min_edge_cents,
                )

                candidates.append(
                    MarketCandidate(
                        ticker=ticker,
                        strike=float(strike),
                        secs_left=secs,
                        forecast=forecast,
                        direction=direction,
                        edge=edge,
                        evidence_score=evidence_score(direction, forecast),
                        market=market,
                    )
                )
        except Exception:
            logger.exception("market scan failed")

        best = select_best_from_top_markets(candidates)
        best_ticker = best.ticker if best else None

        for cand in candidates:
            allowed, block_reason = self.risk.allow_trade(
                ticker=cand.ticker, seconds_to_expiry=cand.secs_left
            )
            is_pick = cand.ticker == best_ticker
            contracts = 0
            action = "NO_TRADE"

            if is_pick and cand.edge.should_trade and allowed:
                win_prob = (
                    cand.forecast.p_fair
                    if cand.direction.side == "yes"
                    else 1.0 - cand.forecast.p_fair
                )
                contracts = kelly_contracts(
                    win_prob=win_prob,
                    price=cand.edge.market_price,
                    sizing=self.config.sizing,
                    confidence=cand.forecast.confidence,
                )
                if contracts > 0:
                    action = f"BUY_{cand.direction.side.upper()}"
                    self._execute(cand.ticker, cand.direction.side, contracts, cand.edge.market_price)

            reason = cand.edge.reason
            if not is_pick and cand.edge.should_trade:
                reason = "not top-4 evidence pick"
            elif not cand.edge.should_trade:
                reason = cand.edge.reason
            elif not allowed:
                reason = block_reason

            decision = {
                "ticker": cand.ticker,
                "action": action,
                "p_fair": cand.forecast.p_fair,
                "confidence": cand.forecast.confidence,
                "regime": cand.forecast.vol_regime,
                "spot": data.spot,
                "strike": cand.strike,
                "secs_left": cand.secs_left,
                "edge": cand.edge.edge_cents / 100.0,
                "side": cand.direction.side,
                "finish": cand.direction.finish_label,
                "evidence_above": cand.direction.above_score,
                "evidence_below": cand.direction.below_score,
                "evidence_margin": cand.direction.margin,
                "evidence_score": cand.evidence_score,
                "price": cand.edge.market_price,
                "contracts": contracts,
                "selected": is_pick and action != "NO_TRADE",
                "reason": reason,
                "layers": cand.forecast.layers,
                "votes": [(v.name, v.prob_yes, v.weight) for v in cand.direction.top_votes],
                "agreement": cand.forecast.agreement_score,
                "brti_official": cand.forecast.is_official_brti,
                "brti_source": data.source,
            }
            decisions.append(decision)

            if is_pick or cand.edge.should_trade:
                logger.info(
                    "%s %s | finish=%s ev=%.3f edge=%.1f¢ conf=%.0f%% %s",
                    action,
                    cand.ticker,
                    cand.direction.finish_label,
                    cand.evidence_score,
                    cand.edge.edge_cents,
                    cand.forecast.confidence * 100,
                    "(SELECTED)" if decision["selected"] else "",
                )

        if scanned == 0:
            logger.info(
                "No open hourly markets in window (brti=%.2f source=%s official=%s)",
                data.spot,
                data.source,
                data.is_official,
            )
        elif best is None:
            logger.info("NO TRADE — no candidate cleared edge + evidence from top %d", len(candidates))

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
            resp = self.client.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                count=contracts,
                price_cents=price_cents,
            )
            self.risk.register_trade(ticker, cost)
            logger.info(
                "LIVE order placed: %s %s x%d @ %d¢ ($%.2f) order=%s",
                side.upper(),
                ticker,
                contracts,
                price_cents,
                cost,
                resp.get("order", {}).get("order_id") or resp.get("order_id") or "ok",
            )
        except Exception:
            logger.exception("order failed for %s", ticker)


def run_once(config: BotConfig | None = None) -> None:
    cfg = config or load_config()
    require_live_credentials(cfg)
    bot = HourlyBot(cfg)
    try:
        decisions = bot.run_cycle()
        selected = [d for d in decisions if d.get("selected")]
        if not selected:
            print("NO TRADE this cycle")
        else:
            d = selected[0]
            print(
                f"{d['action']} {d['ticker']} x{d['contracts']} "
                f"finish={d['finish']} @ {d['price']*100:.0f}¢ "
                f"(evidence={d['evidence_score']:.3f}, edge={d['edge']*100:.1f}¢)"
            )
    finally:
        bot.close()


def run_loop(max_cycles: int | None = None, config: BotConfig | None = None) -> None:
    cfg = config or load_config()
    require_live_credentials(cfg)
    bot = HourlyBot(cfg)
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
    parser.add_argument("--max-trade-usd", type=float, default=None, help="Hard cap per trade (default $1)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--cycles", type=int, default=None, help="Run N cycles then exit")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config()
    if args.live:
        cfg.paper = False
        cfg.kalshi_env = os.getenv("KALSHI_ENV", "prod")
    if args.max_trade_usd is not None:
        cfg.sizing.max_trade_usd = args.max_trade_usd
        cfg.sizing.bankroll_usd = args.max_trade_usd

    logger.info(
        "mode=%s env=%s max_trade=$%.2f bankroll=$%.2f",
        "PAPER" if cfg.paper else "LIVE",
        cfg.kalshi_env,
        cfg.sizing.max_trade_usd,
        cfg.sizing.bankroll_usd,
    )

    if args.once:
        run_once(cfg)
    else:
        run_loop(max_cycles=args.cycles, config=cfg)


if __name__ == "__main__":
    main()
