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
from kalshi_btc_1hr_bot.late_crowd import (
    LateCrowdQualification,
    resolve_late_crowd_context,
    select_late_crowd_pick,
)
from kalshi_btc_1hr_bot.notifications import NotifyConfig, PhoneNotifier
from kalshi_btc_1hr_bot.position_exits import (
    bid_for_side,
    compute_exit_levels,
    evaluate_exit,
)
from kalshi_btc_1hr_bot.risk import RiskManager
from kalshi_btc_1hr_bot.sizing import kelly_contracts
from kalshi_btc_1hr_bot.trade_journal import TradeJournal, TradeRecord
from kalshi_btc_1hr_bot.trend_gates import FlowSnapshot, apply_confirmation_gates, fetch_flow_snapshot
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
        self._sync_risk_from_journal()
        if not self.config.paper and self.config.sizing.use_live_balance:
            self._sync_live_sizing()

    def _sync_risk_from_journal(self) -> None:
        """Restore in-memory risk state from open journal trades after restart."""
        balance = self._balance_usd()
        self.risk.sync_from_journal(self.journal, balance_usd=balance)
        pending = [t for t in self.journal.pending_trades() if t.passed]
        self.risk.state.open_positions = len(pending)
        for trade in pending:
            self.risk.state.traded_tickers.add(trade.ticker)
            if trade.opened_ts > self.risk.state.last_trade_ts:
                self.risk.state.last_trade_ts = trade.opened_ts

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
        flow: FlowSnapshot | None = None,
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

        edge, trend_ok, flow_ok, trend_detail, flow_detail = apply_confirmation_gates(
            edge,
            direction,
            data=data,
            strike=float(strike),
            flow=flow,
            cfg=self.config,
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
            trend_aligned=trend_ok,
            flow_aligned=flow_ok,
            trend_detail=trend_detail,
            flow_detail=flow_detail,
        )

    def run_cycle(self) -> list[dict]:
        """Scan KXBTCD, evaluate Kalshi card strikes only, trade best of top 3."""
        self._sync_live_sizing()
        balance = self._balance_usd()
        self.risk.sync_from_journal(self.journal, balance_usd=balance)
        early_exits = self._manage_open_positions()
        now = datetime.now(timezone.utc)
        data = self.feed.refresh()
        candidates: list[MarketCandidate] = []
        decisions: list[dict] = []
        window_markets: list[dict[str, Any]] = []
        card_rows: list[dict[str, Any]] = []
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
                flow = None
                if self.config.gates.flow_confirm_enabled:
                    flow = fetch_flow_snapshot(self.client, row["ticker"])
                cand = self._evaluate_market(
                    market=row["market"],
                    data=data,
                    secs=row["secs_left"],
                    strike=row["strike"],
                    ticker=row["ticker"],
                    flow=flow,
                )
                candidates.append(cand)
        except Exception:
            logger.exception("market scan failed")
            scanned = len(window_markets)

        pick_n = min(self.config.gates.kalshi_card_picks, config.TOP_N_MARKETS)
        hour_close = card_rows[0]["close_time"] if card_rows else None
        ref_secs = card_rows[0]["secs_left"] if card_rows else 0.0
        late_context = resolve_late_crowd_context(
            cfg=self.config,
            secs_left=ref_secs,
            open_positions=self.risk.state.open_positions,
            journal=self.journal,
            window_markets=window_markets,
            hour_close=hour_close,
        )

        normal_best = select_best_from_top_markets(
            candidates, n=pick_n, trend_bias=self.config.gates.trend_bias_selection
        )
        late_pick: MarketCandidate | None = None
        late_qual: LateCrowdQualification | None = None
        edge_fee = gate_fee_cents(
            self.config.edge.fee_per_contract_cents,
            subtract=self.config.edge.subtract_fees_from_edge,
        )

        if late_context.active:
            late_pick, late_qual = select_late_crowd_pick(
                candidates,
                cfg=self.config,
                fee_cents=edge_fee,
                subtract_fees=self.config.edge.subtract_fees_from_edge,
            )
            if late_pick and late_qual and late_qual.qualified:
                logger.info(
                    "Late crowd armed (%s) — pick %s: %s",
                    late_context.reason,
                    late_pick.ticker,
                    late_qual.reason,
                )

        best = normal_best
        if (
            late_context.active
            and late_pick
            and late_qual
            and late_qual.qualified
            and (normal_best is None or not normal_best.edge.should_trade)
        ):
            best = late_pick
        best_ticker = best.ticker if best else None

        for cand in candidates:
            allowed, block_reason = self.risk.allow_trade(
                ticker=cand.ticker, seconds_to_expiry=cand.secs_left
            )
            is_pick = cand.ticker == best_ticker
            is_late_entry = (
                is_pick
                and late_pick is not None
                and late_qual is not None
                and late_qual.qualified
                and cand.ticker == late_pick.ticker
            )
            trade_edge = (
                late_qual.edge
                if is_late_entry and late_qual and late_qual.edge is not None
                else cand.edge
            )
            contracts = 0
            action = "NO_TRADE"

            if is_pick and trade_edge.should_trade and allowed:
                win_prob = (
                    cand.forecast.p_fair
                    if cand.direction.side == "yes"
                    else 1.0 - cand.forecast.p_fair
                )
                size_mult = (
                    self.config.late_crowd.size_multiplier if is_late_entry else 1.0
                )
                contracts = kelly_contracts(
                    win_prob=win_prob,
                    price=trade_edge.market_price,
                    sizing=self.config.sizing,
                    confidence=cand.forecast.confidence,
                    size_multiplier=size_mult,
                )
                if contracts > 0:
                    action = f"BUY_{cand.direction.side.upper()}"
                    self._execute(
                        cand.ticker,
                        cand.direction.side,
                        contracts,
                        trade_edge.market_price,
                        meta={
                            "edge_cents": trade_edge.edge_cents,
                            "evidence_score": cand.evidence_score,
                            "p_fair": cand.forecast.p_fair,
                            "confidence": cand.forecast.confidence,
                            "strike": cand.strike,
                            "spot": data.spot,
                            "finish": cand.direction.finish_label,
                            "late_crowd": is_late_entry,
                            "late_crowd_reason": late_qual.reason if is_late_entry and late_qual else "",
                            "size_multiplier": size_mult if is_late_entry else 1.0,
                        },
                    )

            reason = trade_edge.reason
            if not is_pick and trade_edge.should_trade:
                reason = f"not top-{pick_n} Kalshi card pick"
            elif not trade_edge.should_trade:
                reason = trade_edge.reason
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
                "edge": trade_edge.edge_cents / 100.0,
                "side": cand.direction.side,
                "finish": cand.direction.finish_label,
                "evidence_above": cand.direction.above_score,
                "evidence_below": cand.direction.below_score,
                "evidence_margin": cand.direction.margin,
                "evidence_score": cand.evidence_score,
                "price": trade_edge.market_price,
                "contracts": contracts,
                "selected": is_pick and action != "NO_TRADE",
                "late_crowd": is_late_entry,
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

            if is_pick or trade_edge.should_trade:
                logger.info(
                    "%s %s | finish=%s ev=%.3f edge=%.1f¢ conf=%.0f%% %s%s",
                    action,
                    cand.ticker,
                    cand.direction.finish_label,
                    cand.evidence_score,
                    trade_edge.edge_cents,
                    cand.forecast.confidence * 100,
                    "(SELECTED)" if decision["selected"] else "",
                    " [LATE CROWD]" if is_late_entry else "",
                )

        if scanned == 0:
            logger.info(
                "No open hourly markets in window (brti=%.2f source=%s official=%s)",
                data.spot,
                data.source,
                data.is_official,
            )
        elif best is None:
            if late_context.active and late_qual is None:
                logger.info(
                    "NO TRADE — late crowd window but no strike cleared crowd favorite from %d card picks",
                    len(candidates),
                )
            else:
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
            early_exits=early_exits,
            late_context=late_context,
            late_qual=late_qual,
        )

        return decisions

    def _build_open_positions_view(self) -> list[dict[str, Any]]:
        views: list[dict[str, Any]] = []
        for trade in self.journal.pending_trades():
            if not trade.passed:
                continue
            levels = compute_exit_levels(
                trade.entry_price,
                self.config.exit,
                stored_tp=trade.tp_price,
                stored_sl=trade.sl_price,
            )
            bid = None
            try:
                raw = self.client._request("GET", f"/markets/{trade.ticker}")
                market = normalize_market(raw.get("market", raw))
                bid = bid_for_side(market, trade.side)
            except Exception:
                logger.debug("open position quote failed for %s", trade.ticker, exc_info=True)
            unrealized = None
            if bid is not None:
                unrealized = (bid - trade.entry_price) * trade.contracts
            views.append(
                {
                    "trade_id": trade.id,
                    "ticker": trade.ticker,
                    "side": trade.side,
                    "finish": trade.finish,
                    "contracts": trade.contracts,
                    "entry_price": trade.entry_price,
                    "entry_cents": round(trade.entry_price * 100),
                    "tp_price": levels.take_profit_price,
                    "tp_cents": round(levels.take_profit_price * 100),
                    "sl_price": levels.stop_loss_price,
                    "sl_cents": round(levels.stop_loss_price * 100),
                    "bid_price": bid,
                    "bid_cents": round(bid * 100) if bid is not None else None,
                    "unrealized_pnl_usd": round(unrealized, 4) if unrealized is not None else None,
                    "exit_enabled": self.config.exit.enabled,
                    "take_profit_pct": self.config.exit.take_profit_pct,
                    "stop_loss_pct": self.config.exit.stop_loss_pct,
                }
            )
        return views

    def _manage_open_positions(self) -> list[dict[str, Any]]:
        """Check open positions each cycle and sell when TP/SL triggers."""
        if not self.config.exit.enabled:
            return []
        exits: list[dict[str, Any]] = []
        now = time.time()
        for trade in self.journal.pending_trades():
            if not trade.passed:
                continue
            if now - trade.opened_ts < self.config.exit.min_hold_seconds:
                continue
            try:
                raw = self.client._request("GET", f"/markets/{trade.ticker}")
            except Exception:
                logger.debug("exit check failed for %s", trade.ticker, exc_info=True)
                continue
            market = normalize_market(raw.get("market", raw))
            bid = bid_for_side(market, trade.side)
            if bid is None:
                continue
            levels = compute_exit_levels(
                trade.entry_price,
                self.config.exit,
                stored_tp=trade.tp_price,
                stored_sl=trade.sl_price,
            )
            signal = evaluate_exit(
                entry_price=trade.entry_price,
                bid_price=bid,
                contracts=trade.contracts,
                cfg=self.config.exit,
                levels=levels,
            )
            if signal is None:
                continue
            result = self._execute_exit(trade, signal)
            if result:
                exits.append(result)
        return exits

    def _execute_exit(self, trade: TradeRecord, signal: Any) -> dict[str, Any] | None:
        mode = "PAPER" if self.config.paper else "LIVE"
        price_cents = max(1, int(round(signal.exit_price * 100)))
        proceeds = signal.exit_price * trade.contracts

        if self.config.paper:
            self.journal.close_trade_early(
                trade.id,
                exit_price=signal.exit_price,
                exit_reason=signal.reason,
                pnl=signal.pnl_total,
            )
            self.risk.close_position(trade.ticker, proceeds)
            logger.info(
                "PAPER exit %s %s x%d @ %.0f¢ (%s) pnl=$%.2f",
                trade.side.upper(),
                trade.ticker,
                trade.contracts,
                signal.exit_price * 100,
                signal.reason,
                signal.pnl_total,
            )
            self.notifier.notify_exit(
                mode=mode,
                ticker=trade.ticker,
                side=trade.side,
                contracts=trade.contracts,
                entry_price=trade.entry_price,
                exit_price=signal.exit_price,
                reason=signal.reason,
                pnl=signal.pnl_total,
            )
            return {
                "ticker": trade.ticker,
                "side": trade.side,
                "reason": signal.reason,
                "exit_price": signal.exit_price,
                "pnl": signal.pnl_total,
                "mode": mode,
            }

        try:
            resp = self.client.place_order(
                ticker=trade.ticker,
                side=trade.side,
                action="sell",
                count=trade.contracts,
                price_cents=price_cents,
            )
            order_id = str(
                resp.get("order", {}).get("order_id")
                or resp.get("order_id")
                or resp.get("order", {}).get("id")
                or resp.get("id")
                or "ok"
            )
            self.journal.close_trade_early(
                trade.id,
                exit_price=signal.exit_price,
                exit_reason=signal.reason,
                pnl=signal.pnl_total,
                exit_order_id=order_id,
            )
            self.risk.close_position(trade.ticker, proceeds)
            logger.info(
                "LIVE exit %s %s x%d @ %d¢ (%s) pnl=$%.2f order=%s",
                trade.side.upper(),
                trade.ticker,
                trade.contracts,
                price_cents,
                signal.reason,
                signal.pnl_total,
                order_id,
            )
            self.notifier.notify_exit(
                mode=mode,
                ticker=trade.ticker,
                side=trade.side,
                contracts=trade.contracts,
                entry_price=trade.entry_price,
                exit_price=signal.exit_price,
                reason=signal.reason,
                pnl=signal.pnl_total,
                order_id=order_id,
            )
            return {
                "ticker": trade.ticker,
                "side": trade.side,
                "reason": signal.reason,
                "exit_price": signal.exit_price,
                "pnl": signal.pnl_total,
                "order_id": order_id,
                "mode": mode,
            }
        except Exception:
            logger.exception("exit order failed for %s", trade.ticker)
            self.notifier.notify_order_failed(
                ticker=trade.ticker,
                side=trade.side,
                contracts=trade.contracts,
                price=signal.exit_price,
            )
            return None

    def _publish_dashboard(
        self,
        *,
        data: Any,
        candidates: list[MarketCandidate],
        decisions: list[dict],
        best: MarketCandidate | None,
        best_ticker: str | None,
        markets_scanned: int,
        early_exits: list[dict[str, Any]] | None = None,
        late_context: Any = None,
        late_qual: LateCrowdQualification | None = None,
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
                self.risk.close_position(
                    str(item.get("ticker") or ""),
                    float(item.get("pnl") or 0.0) + float(item.get("cost_usd") or 0.0),
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
            open_positions=self._build_open_positions_view(),
            early_exits=early_exits or [],
            late_context=late_context,
            late_qual=late_qual,
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
        levels = compute_exit_levels(price, self.config.exit) if self.config.exit.enabled else None
        tp_price = levels.take_profit_price if levels else None
        sl_price = levels.stop_loss_price if levels else None
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
                tp_price=tp_price,
                sl_price=sl_price,
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
                tp_price=tp_price,
                sl_price=sl_price,
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
