"""KXBTC15M settlement-mispricing strategy engine."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from kalshi_bot.config import BotConfig, Rules15mConfig, V6Config
from kalshi_bot.data.brti import resolve_spot
from kalshi_bot.data.btc_data_engine import BtcDataEngine
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.data.kalshi_trade_tape import KalshiTradeTapeService
from kalshi_bot.data.realized_vol import estimate_realized_vol
from kalshi_bot.models.ensemble import ModelVote, combine_models
from kalshi_bot.models.probability import options_implied_prob_up
from kalshi_bot.models.volatility_regime import classify_vol_regime, neutral_regime
from kalshi_bot.platform.decision_states import TradeSignal, signal_from_action
from kalshi_bot.platform.observation import DataQuality, ObservationBundle
from kalshi_bot.strategies.base import MarketDecision, StrategyEngine
from kalshi_bot.strategy.mispricing_evaluator import evaluate_market_mispricing
from kalshi_bot.strategy.mispricing_engine import TradeAction
from kalshi_bot.strategy.price_patterns import PatternAssessment, detect_price_pattern
from kalshi_bot.strategy.stale_price_detector import assess_stale_kalshi_price
from kalshi_bot.strategy.trade_gates import evaluate_trade_gates
from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine


class Kxbtc15mStrategy(StrategyEngine):
    name = "KXBTC15M"
    series = "KXBTC15M"

    def __init__(
        self,
        client: KalshiClient,
        config: BotConfig,
        rules: Rules15mConfig,
        engine: V6IntelligenceEngine,
        btc_engine: BtcDataEngine,
        trade_tape: KalshiTradeTapeService | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.rules = rules
        self.engine = engine
        self.v6 = config.v6
        self.btc_engine = btc_engine
        self.trade_tape = trade_tape
        self._smile = None
        self._prev_btc: float | None = None
        self._prev_yes_mid: dict[str, float] = {}

    def evaluate_all(self) -> list[MarketDecision]:
        t0 = time.time()
        now = datetime.now(timezone.utc)
        obs = ObservationBundle()
        fallback = self._smile.spot_btc if self._smile else None
        t_spot = time.time()
        spot_snap = resolve_spot(self.client, fallback_btc=fallback, brti_cfg=self.config.brti)
        obs.add(
            "brti",
            spot_snap.brti,
            source=spot_snap.source,
            quality=DataQuality.FRESH if spot_snap.is_official else DataQuality.DEGRADED,
            latency_ms=(time.time() - t_spot) * 1000,
        )
        spot = spot_snap.brti
        self.engine.update_spot(spot)

        rv = estimate_realized_vol(horizon_seconds=self.v6.max_seconds_to_expiry)
        vol = rv.annualized_vol
        if self._smile is None:
            try:
                self._smile = load_ibit_smile(self.config.smile, allow_synthetic=False)
            except Exception:
                self._smile = None
        if self._smile is not None:
            vol = 0.6 * vol + 0.4 * self._smile.atm_iv
            obs.add("ibit_atm_iv", self._smile.atm_iv, source="ibit_smile", quality=DataQuality.FRESH)

        if self.config.platform.enable_vol_regime:
            regime = classify_vol_regime(vol, recent_vols=[rv.annualized_vol])
        else:
            regime = neutral_regime(vol)
        obs.add("vol_regime", regime.regime.value, source="volatility_regime", detail=regime.reason)

        btc = self.btc_engine.refresh(
            reference_price=spot,
            reference_source=spot_snap.source,
            is_official=spot_snap.is_official,
            annualized_vol=vol,
        )
        obs.add(
            "btc_cross_agreement",
            btc.cross_exchange_agreement,
            source="btc_data_engine",
            quality=DataQuality.STALE if btc.stale else DataQuality.FRESH,
        )

        if self.trade_tape is not None:
            ws = self.trade_tape.feed_status()
            obs.add(
                "kalshi_ws",
                ws.get("connected", False),
                source="kalshi_websocket",
                quality=DataQuality.FRESH if ws.get("connected") else DataQuality.STALE,
            )

        decisions: list[MarketDecision] = []
        for raw in self.client.iter_markets(self.v6.series_ticker, status="open"):
            market = normalize_market(raw)
            close = market.get("close_time")
            if close is None:
                continue
            secs = (close - now).total_seconds()
            if secs < self.v6.min_seconds_to_expiry or secs > self.v6.max_seconds_to_expiry:
                continue
            if not market.get("yes_ask") and not market.get("yes_bid"):
                continue

            ticker = str(market.get("ticker") or "")
            options_prob = None
            strike = market.get("strike")
            if self._smile is not None and strike is not None:
                try:
                    options_prob = options_implied_prob_up(
                        spot_btc=spot,
                        open_level=float(strike),
                        close_time=close,
                        smile=self._smile,
                        rate=self.config.smile.risk_free_rate,
                        dividend=self.config.smile.dividend_yield,
                        now=now,
                    ).probability
                except Exception:
                    pass

            recent_trades = None
            orderbook = None
            orderbook_source = "rest"
            kalshi_stale = False
            if self.trade_tape is not None:
                self.trade_tape.ensure_subscription([ticker])
                recent_trades = self.trade_tape.recent_trades(ticker)
                ob_quote = self.trade_tape.orderbook_quote(ticker)
                if ob_quote is not None:
                    market = dict(market)
                    if ob_quote.yes_bid is not None:
                        market["yes_bid"] = ob_quote.yes_bid
                    if ob_quote.yes_ask is not None:
                        market["yes_ask"] = ob_quote.yes_ask
                    if ob_quote.no_ask is not None:
                        market["no_ask"] = ob_quote.no_ask
                    orderbook = self.trade_tape.orderbook_dict(ticker)
                    orderbook_source = "ws"
                    kalshi_stale = ob_quote.stale

            if not regime.allow_new_entries:
                decisions.append(self._blocked_decision(market, spot, secs, regime.reason, obs))
                continue

            yes_mid = None
            if market.get("yes_bid") is not None and market.get("yes_ask") is not None:
                yes_mid = (float(market["yes_bid"]) + float(market["yes_ask"])) / 2.0
            elif market.get("yes_ask") is not None:
                yes_mid = float(market["yes_ask"])
            stale_assess = assess_stale_kalshi_price(
                prev_btc=self._prev_btc,
                curr_btc=spot,
                prev_yes_mid=self._prev_yes_mid.get(ticker),
                curr_yes_mid=yes_mid,
                model_prob_yes=0.5,
                strike=float(strike or spot),
                seconds_to_expiry=secs,
            )
            if yes_mid is not None:
                self._prev_yes_mid[ticker] = yes_mid

            pattern = detect_price_pattern(
                btc,
                spot=spot,
                strike=float(strike or spot),
                seconds_to_expiry=secs,
            )

            audit, opp, trade_dec = evaluate_market_mispricing(
                self.engine,
                market,
                spot=spot,
                spot_source=spot_snap.source,
                spot_is_official=spot_snap.is_official,
                vol=vol,
                btc=btc,
                options_prob=options_prob,
                now=now,
                fee_rate=self.config.fee_rate,
                recent_trades=recent_trades,
                kalshi_stale=kalshi_stale,
                orderbook=orderbook,
                orderbook_source=orderbook_source,
                use_settlement_model=self.config.platform.enable_settlement_model,
            )
            self.engine.get_monitor().record(audit)

            gates = evaluate_trade_gates(
                model_prob_yes=audit.model_prob_up,
                yes_net_ev=audit.yes_side.net_edge_dollars,
                no_net_ev=audit.no_side.net_edge_dollars,
                yes_ask=market.get("yes_ask"),
                yes_bid=market.get("yes_bid"),
                no_ask=market.get("no_ask"),
                seconds_to_expiry=secs,
                uncertainty_pct=audit.model_disagreement_pp,
                contracts=audit.contracts,
                min_seconds=self.v6.min_seconds_to_expiry,
                max_seconds=self.v6.max_seconds_to_expiry,
                bucket_overrides=self.rules.time_buckets or None,
                gates_cfg=self.rules.gates,
                arbitrary_cfg=self.rules.arbitrary,
            )

            votes = []
            if self.config.platform.enable_settlement_model:
                votes.append(ModelVote("settlement", audit.model_prob_up, 0.45, audit.model_confidence))
            votes.append(ModelVote("monte_carlo", audit.monte_carlo_prob, 0.50 if not self.config.platform.enable_settlement_model else 0.25, audit.model_confidence))
            if options_prob is not None:
                votes.append(ModelVote("options_implied", options_prob, 0.20, 0.6))
            votes.append(
                ModelVote(
                    "microstructure",
                    0.5 + 0.1 * (1 if audit.yes_side.net_edge_dollars > audit.no_side.net_edge_dollars else -1),
                    0.10,
                    audit.model_confidence * 0.8,
                )
            )
            ensemble = combine_models(votes)

            action = trade_dec.action.value
            net_edge = opp.best_net_edge
            signal = signal_from_action(action, net_edge=net_edge)
            if ensemble.agreement_score < 0.55 and signal in (TradeSignal.BUY_YES, TradeSignal.BUY_NO):
                signal = TradeSignal.WAIT
                trade_dec_reason = f"model disagreement (agreement={ensemble.agreement_score:.0%})"
            else:
                trade_dec_reason = trade_dec.reason

            why_not = ""
            why_trade = ""
            if signal in (TradeSignal.BUY_YES, TradeSignal.BUY_NO, TradeSignal.STRONG_BUY_YES, TradeSignal.STRONG_BUY_NO):
                regime_label = regime.regime.value if self.config.platform.enable_vol_regime else "off"
                why_trade = (
                    f"P(YES)={audit.model_prob_up:.1%} vs executable; "
                    f"net edge={net_edge*100:.1f}¢; conf={audit.model_confidence:.0%}; "
                    f"regime={regime_label}"
                )
            else:
                why_not = self._explain_no_trade(audit, trade_dec_reason, net_edge, regime.edge_multiplier, pattern, opp)

            decisions.append(
                MarketDecision(
                    ticker=ticker,
                    series=self.series,
                    signal=signal.value,
                    reason=trade_dec_reason if signal != TradeSignal.NO_TRADE else why_not.split("\n")[0],
                    model_prob_yes=audit.model_prob_up,
                    model_prob_no=audit.model_prob_down,
                    fair_yes=ensemble.fair_yes,
                    fair_no=ensemble.fair_no,
                    executable_yes=audit.yes_side.executable_ask,
                    executable_no=audit.no_side.executable_ask,
                    raw_edge=max(audit.yes_side.raw_edge_dollars, audit.no_side.raw_edge_dollars),
                    net_edge=net_edge,
                    confidence=ensemble.confidence,
                    regime="off" if not self.config.platform.enable_vol_regime else regime.regime.value,
                    strike=float(strike or spot),
                    seconds_to_expiry=secs,
                    btc_spot=spot,
                    yes_price=market.get("yes_ask"),
                    no_price=market.get("no_ask"),
                    why_trade=why_trade,
                    why_not_trade=why_not,
                    observations=obs,
                    features={
                        "agreement_score": ensemble.agreement_score,
                        "orderbook_source": orderbook_source,
                        "micro_liquidity": audit.liquidity_score,
                        "liquidity_score": audit.liquidity_score,
                        "bid_ask_imbalance": audit.bid_ask_imbalance,
                        "spread": audit.spread,
                        "scan_latency_ms": (time.time() - t0) * 1000,
                        "stale_lag_pp": stale_assess.lag_pp,
                        "price_pattern": pattern.pattern.value,
                        "pattern_finish_side": pattern.finish_side,
                        "pattern_detail": pattern.detail,
                        "yes_net_edge": audit.yes_side.net_edge_dollars,
                        "no_net_edge": audit.no_side.net_edge_dollars,
                        "executable_no": audit.no_side.executable_ask,
                        "gates": gates.to_dict(),
                        "gates_ready_side": gates.ready_side,
                    },
                    execute_verdict=audit.verdict if audit.verdict in ("TRADE_YES", "TRADE_NO") else None,
                    contracts=audit.contracts,
                )
            )

        decisions.sort(key=lambda d: d.net_edge, reverse=True)
        self._prev_btc = spot
        return decisions

    def _blocked_decision(self, market: dict, spot: float, secs: float, reason: str, obs: ObservationBundle) -> MarketDecision:
        ticker = str(market.get("ticker") or "")
        return MarketDecision(
            ticker=ticker,
            series=self.series,
            signal=TradeSignal.NO_TRADE.value,
            reason=reason,
            model_prob_yes=0.5,
            model_prob_no=0.5,
            fair_yes=0.5,
            fair_no=0.5,
            executable_yes=market.get("yes_ask"),
            executable_no=market.get("no_ask"),
            raw_edge=0.0,
            net_edge=0.0,
            confidence=0.0,
            regime="VOLATILITY_SHOCK",
            strike=float(market.get("strike") or spot),
            seconds_to_expiry=secs,
            btc_spot=spot,
            yes_price=market.get("yes_ask"),
            no_price=market.get("no_ask"),
            why_trade="",
            why_not_trade=reason,
            observations=obs,
        )

    @staticmethod
    def _explain_no_trade(audit, reason: str, net_edge: float, edge_mult: float, pattern: PatternAssessment, opp) -> str:
        yes_ask = audit.yes_side.executable_ask
        no_ask = audit.no_side.executable_ask
        lines = [
            f"Pattern: {pattern.pattern.value} ({pattern.detail})",
            f"Model probability: {audit.model_prob_up*100:.1f}%",
            f"Executable YES: {(yes_ask or 0)*100:.0f}¢ (net {audit.yes_side.net_edge_dollars*100:.1f}¢)",
            f"Executable NO: {(no_ask or 0)*100:.0f}¢ (net {audit.no_side.net_edge_dollars*100:.1f}¢)",
            f"Best net edge: {net_edge*100:.1f}¢",
            f"Confidence: {audit.model_confidence:.0%}",
            f"Reason: {reason}",
        ]
        return "\n".join(lines)
