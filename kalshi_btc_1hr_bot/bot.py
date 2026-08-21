"""Main paper/live trading loop for KXBTCD 1-hour bot."""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.config import BotConfig, gate_fee_cents, load_config, require_live_credentials
from kalshi_btc_1hr_bot.dashboard_state import build_snapshot, save_snapshot
from kalshi_btc_1hr_bot.data_feed import DataFeed
from kalshi_btc_1hr_bot.dynamic_gates import apply_dynamic_thresholds, resolve_dynamic_thresholds
from kalshi_btc_1hr_bot.evidence import (
    MarketCandidate,
    directional_evidence,
    evaluate_edge_with_evidence,
    evidence_score,
    select_best_from_top_markets,
)
from kalshi_btc_1hr_bot.forecast import ForecastEnsemble, forecast_ensemble_from_market_data, agreement_score_for_gates
from kalshi_btc_1hr_bot.kalshi_card import select_kalshi_card_markets
from kalshi_btc_1hr_bot.kalshi_client import KalshiClient, is_hourly_market, normalize_market
from kalshi_btc_1hr_bot.notifications import NotifyConfig, PhoneNotifier
from kalshi_btc_1hr_bot.risk import RiskManager
from kalshi_btc_1hr_bot.sizing import kelly_contracts
from kalshi_btc_1hr_bot.trade_journal import TradeJournal
from kalshi_btc_1hr_bot.utils import setup_logging

logger = logging.getLogger(__name__)


class HourlyBot:
    def __init__(self, config: BotConfig | None = None) -> None:
        self.config = config or load_config()
        self.client = KalshiClient(self.config)
        self.feed = DataFeed(kalshi_client=self.client)
        self.ensemble = ForecastEnsemble()
        self.risk = RiskManager(self.config)
        self.journal = TradeJournal()
        self.notifier = PhoneNotifier(self.config.notify)
        if not self.config.paper and self.config.sizing.use_live_balance:
            self._sync_live_sizing()

    def _balance_usd(self) -> float | None:
        if not self.client.authenticated:
            return None
        try:
            data = self.client.get_balance()
            if data.get("balance_dollars") is not None:
                return float(data["balance_dollars"])
            if data.get("balance") is not None:
                return float(data["balance"]) / 100.0
        except Exception:
            logger.debug("balance fetch failed", exc_info=True)
        return None

    def _sync_live_sizing(self) -> None:
        """Pull bankroll and trade cap from live Kalshi balance when configured."""
        if not self.config.sizing.use_live_balance or self.config.paper:
            return
        balance = self._balance_usd()
        if balance is None or balance <= 0:
            return
        self.config.sizing.bankroll_usd = balance
        if self.config.sizing.max_trade_usd <= 0:
            self.config.sizing.max_trade_usd = balance * self.config.sizing.max_bankroll_pct

    def close(self) -> None:
        self.client.close()
        self.feed.close()
        self.notifier.close()

    def _evaluate_market(
        self,
        *,
        market: dict,
        data: Any,
        secs: float,
        strike: float,
        ticker: str,
    ) -> MarketCandidate:
        forecast = forecast_ensemble_from_market_data(
            self.ensemble,
            spot=data.spot,
            strike=float(strike),
            seconds_to_expiry=secs,
            data=data,
        )
        direction = directional_evidence(forecast.votes)

        yes_ask = market.get("yes_ask")
        no_ask = market.get("no_ask")
        yes_ask_f = float(yes_ask) if yes_ask is not None else 1.0
        no_ask_f = float(no_ask) if no_ask is not None else 1.0
        yes_bid_f = float(market.get("yes_bid") or yes_ask_f)
        no_bid_f = float(market.get("no_bid") or no_ask_f)

        side_prob = forecast.crowd.side_prob(direction.side)
        agree_score = agreement_score_for_gates(
            forecast, use_ensemble=self.config.gates.use_ensemble_agreement
        )
        thresholds = resolve_dynamic_thresholds(
            secs,
            vol_regime=forecast.vol_regime,
            agreement_score=agree_score,
            crowd_side_prob=side_prob if self.config.gates.crowd_gates_enabled else None,
        )
        aligned = apply_dynamic_thresholds(
            forecast,
            thresholds,
            direction.side,
            crowd_gates_enabled=self.config.gates.crowd_gates_enabled,
            use_ensemble_agreement=self.config.gates.use_ensemble_agreement,
        )
        edge_fee = gate_fee_cents(
            self.config.edge.fee_per_contract_cents,
            subtract=self.config.edge.subtract_fees_from_edge,
        )
        edge = evaluate_edge_with_evidence(
            aligned.p_fair,
            yes_ask_f,
            no_ask_f,
            yes_bid_f,
            no_bid_f,
            direction,
            fee_cents=edge_fee,
            subtract_fees=self.config.edge.subtract_fees_from_edge,
            crowd_gates_enabled=self.config.gates.crowd_gates_enabled,
            use_ensemble_agreement=self.config.gates.use_ensemble_agreement,
            thresholds=thresholds,
            forecast=aligned,
        )
        edge_pass_threshold = config.GATE_ABS_MIN_EDGE_CENTS
        if edge.edge_cents >= edge_pass_threshold and not edge.should_trade:
            thresholds = resolve_dynamic_thresholds(
                secs,
                vol_regime=forecast.vol_regime,
                agreement_score=agree_score,
                edge_cents=edge.edge_cents,
                crowd_side_prob=side_prob if self.config.gates.crowd_gates_enabled else None,
            )
            aligned = apply_dynamic_thresholds(
                forecast,
                thresholds,
                direction.side,
                crowd_gates_enabled=self.config.gates.crowd_gates_enabled,
                use_ensemble_agreement=self.config.gates.use_ensemble_agreement,
            )
            edge = evaluate_edge_with_evidence(
                aligned.p_fair,
                yes_ask_f,
                no_ask_f,
                yes_bid_f,
                no_bid_f,
                direction,
                fee_cents=edge_fee,
                subtract_fees=self.config.edge.subtract_fees_from_edge,
                crowd_gates_enabled=self.config.gates.crowd_gates_enabled,
                use_ensemble_agreement=self.config.gates.use_ensemble_agreement,
                thresholds=thresholds,
                forecast=aligned,
            )

        return MarketCandidate(
            ticker=ticker,
            strike=float(strike),
            secs_left=secs,
            forecast=aligned,
            direction=direction,
            edge=edge,
            evidence_score=evidence_score(direction, aligned),
            market=market,
            thresholds=thresholds,
        )

    def run_cycle(self) -> list[dict]:
        """Scan KXBTCD, evaluate Kalshi card strikes only, trade best of top 3."""
        self._sync_live_sizing()
        now = datetime.now(timezone.utc)
        data = self.feed.refresh()
        candidates: list[MarketCandidate] = []
        decisions: list[dict] = []
        window_markets: list[dict[str, Any]] = []
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

                if market.get("yes_ask") is None and market.get("no_ask") is None:
                    continue

                window_markets.append(
                    {
                        "ticker": str(market.get("ticker") or ""),
                        "strike": float(strike),
                        "secs_left": secs,
                        "close_time": close,
                        "market": market,
                    }
                )

            scanned = len(window_markets)
            if self.config.gates.kalshi_card_only:
                card_rows = select_kalshi_card_markets(
                    window_markets,
                    data.spot,
                    n=self.config.gates.kalshi_card_picks,
                )
                logger.info(
                    "Kalshi card top %d @ spot $%.0f: %s",
                    len(card_rows),
                    data.spot,
                    ", ".join(f"${r['strike']:,.0f}" for r in card_rows),
                )
            else:
                card_rows = window_markets

            for row in card_rows:
                cand = self._evaluate_market(
                    market=row["market"],
                    data=data,
                    secs=row["secs_left"],
                    strike=row["strike"],
                    ticker=row["ticker"],
                )
                candidates.append(cand)
        except Exception:
            logger.exception("market scan failed")
            scanned = len(window_markets)

        pick_n = min(self.config.gates.kalshi_card_picks, config.TOP_N_MARKETS)
        best = select_best_from_top_markets(candidates, n=pick_n)
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
                    self._execute(
                        cand.ticker,
                        cand.direction.side,
                        contracts,
                        cand.edge.market_price,
                        meta={
                            "edge_cents": cand.edge.edge_cents,
                            "evidence_score": cand.evidence_score,
                            "p_fair": cand.forecast.p_fair,
                            "confidence": cand.forecast.confidence,
                            "strike": cand.strike,
                            "spot": data.spot,
                            "finish": cand.direction.finish_label,
                        },
                    )

            reason = cand.edge.reason
            if not is_pick and cand.edge.should_trade:
                reason = f"not top-{pick_n} Kalshi card pick"
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
                "crowd": cand.forecast.crowd_summary_at(
                    min_favorite=cand.thresholds.min_crowd_favorite if cand.thresholds else None
                ),
                "quorum_met": cand.forecast.quorum_met,
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
            logger.info(
                "NO TRADE — no Kalshi card pick cleared edge + evidence from %d strikes",
                len(candidates),
            )

        self._publish_dashboard(
            data=data,
            candidates=candidates,
            decisions=decisions,
            best=best,
            best_ticker=best_ticker,
            markets_scanned=scanned,
        )

        return decisions

    def _publish_dashboard(
        self,
        *,
        data: Any,
        candidates: list[MarketCandidate],
        decisions: list[dict],
        best: MarketCandidate | None,
        best_ticker: str | None,
        markets_scanned: int,
    ) -> None:
        mode = "PAPER" if self.config.paper else "LIVE"
        balance = self._balance_usd()
        try:
            settlements = self.journal.poll_settlements(self.client)
            for item in settlements:
                outcome_yes = str(item.get("result", "")).lower() == "yes"
                self.notifier.notify_settlement(
                    ticker=str(item.get("ticker") or ""),
                    side=str(item.get("side") or ""),
                    won=bool(item.get("won")),
                    pnl=float(item.get("pnl") or 0.0),
                    result=str(item.get("result") or ""),
                )
                # Update crowd adaptive weights when we know outcome
                for cand in candidates:
                    if cand.ticker == item.get("ticker"):
                        self.ensemble.record_settlement(cand.forecast, outcome_yes)
                        break
        except Exception:
            settlements = []

        snapshot = build_snapshot(
            cfg=self.config,
            data=data,
            candidates=candidates,
            decisions=decisions,
            best=best,
            best_ticker=best_ticker,
            risk=self.risk,
            markets_scanned=markets_scanned,
            balance_usd=balance,
            mode=mode,
            recent_settlements=settlements,
        )
        save_snapshot(snapshot)

        selected = next((d for d in decisions if d.get("selected")), None)
        best_dec = next((d for d in decisions if d.get("ticker") == best_ticker), None)
        reason = ""
        if selected:
            reason = "Trade executed"
        elif best_dec:
            reason = str(best_dec.get("reason") or "")
        elif best is None:
            reason = "No market cleared edge + evidence"

        self.journal.record_cycle(
            mode=mode,
            status=snapshot.cycle_status,
            markets_scanned=markets_scanned,
            candidates=len(candidates),
            best_ticker=best_ticker or "",
            best_action=str(best_dec.get("action") if best_dec else "NO_TRADE"),
            reason=reason,
            selected=bool(selected),
            spot=float(data.spot),
            readiness_pct=snapshot.readiness_pct,
            payload={"blockers": snapshot.blockers, "best_pick": snapshot.best_pick},
        )

    def _execute(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
        *,
        meta: dict | None = None,
    ) -> None:
        cost = contracts * price
        mode = "PAPER" if self.config.paper else "LIVE"
        meta = meta or {}
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
            self.journal.record_trade(
                ticker=ticker,
                side=side,
                contracts=contracts,
                entry_price=price,
                mode=mode,
                passed=True,
                edge_cents=float(meta.get("edge_cents", 0)),
                evidence_score=float(meta.get("evidence_score", 0)),
                p_fair=float(meta.get("p_fair", 0)),
                confidence=float(meta.get("confidence", 0)),
                strike=float(meta.get("strike", 0)),
                spot=float(meta.get("spot", 0)),
                finish=str(meta.get("finish", "")),
            )
            self.notifier.notify_trade(
                mode=mode,
                ticker=ticker,
                side=side,
                contracts=contracts,
                price=price,
                finish=str(meta.get("finish", "")),
                edge_cents=float(meta.get("edge_cents", 0)),
            )
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
            order_id = str(
                resp.get("order", {}).get("order_id")
                or resp.get("order_id")
                or resp.get("order", {}).get("id")
                or resp.get("id")
                or "ok"
            )
            self.risk.register_trade(ticker, cost)
            self.journal.record_trade(
                ticker=ticker,
                side=side,
                contracts=contracts,
                entry_price=price,
                mode=mode,
                order_id=order_id,
                passed=True,
                edge_cents=float(meta.get("edge_cents", 0)),
                evidence_score=float(meta.get("evidence_score", 0)),
                p_fair=float(meta.get("p_fair", 0)),
                confidence=float(meta.get("confidence", 0)),
                strike=float(meta.get("strike", 0)),
                spot=float(meta.get("spot", 0)),
                finish=str(meta.get("finish", "")),
            )
            logger.info(
                "LIVE order placed: %s %s x%d @ %d¢ ($%.2f) order=%s",
                side.upper(),
                ticker,
                contracts,
                price_cents,
                cost,
                order_id,
            )
            self.notifier.notify_trade(
                mode=mode,
                ticker=ticker,
                side=side,
                contracts=contracts,
                price=price,
                finish=str(meta.get("finish", "")),
                edge_cents=float(meta.get("edge_cents", 0)),
                order_id=order_id,
            )
        except Exception:
            logger.exception("order failed for %s", ticker)
            self.notifier.notify_order_failed(
                ticker=ticker,
                side=side,
                contracts=contracts,
                price=price,
            )
            self.journal.record_trade(
                ticker=ticker,
                side=side,
                contracts=contracts,
                entry_price=price,
                mode=mode,
                passed=False,
                block_reason="order_failed",
                edge_cents=float(meta.get("edge_cents", 0)),
                evidence_score=float(meta.get("evidence_score", 0)),
                p_fair=float(meta.get("p_fair", 0)),
                confidence=float(meta.get("confidence", 0)),
                strike=float(meta.get("strike", 0)),
                spot=float(meta.get("spot", 0)),
                finish=str(meta.get("finish", "")),
            )


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
    if cfg.notify.notify_on_startup and bot.notifier.config.configured:
        bot.notifier.notify_startup(
            mode="PAPER" if cfg.paper else "LIVE",
            max_trade_usd=cfg.sizing.max_trade_usd,
        )
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
    parser.add_argument("--max-trade-usd", type=float, default=None, help="Hard cap per trade (0 = use live balance)")
    parser.add_argument("--once", action="store_true", help="Run a single cycle")
    parser.add_argument("--cycles", type=int, default=None, help="Run N cycles then exit")
    parser.add_argument("--test-notify", action="store_true", help="Send a test SMS and exit")
    args = parser.parse_args(argv)
    setup_logging()

    cfg = load_config()
    if args.test_notify:
        notifier = PhoneNotifier(cfg.notify)
        if not cfg.notify.configured:
            raise SystemExit(
                "SMS not configured. Set NOTIFY_ENABLED=true, NOTIFY_PHONE_NUMBER, "
                "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER in .env"
            )
        ok = notifier.notify_startup(mode="TEST", max_trade_usd=cfg.sizing.max_trade_usd)
        notifier.close()
        raise SystemExit(0 if ok else 1)
    if args.live:
        cfg.paper = False
        cfg.kalshi_env = os.getenv("KALSHI_ENV", "prod")
    if args.max_trade_usd is not None:
        cfg.sizing.max_trade_usd = args.max_trade_usd
        if not cfg.sizing.use_live_balance:
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
