"""Production platform runner — dual strategy orchestration with live safety."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from kalshi_bot.config import BotConfig, Rules15mConfig, Settings, kalshi_base_url, load_config, load_rules_15m
from kalshi_bot.data.btc_data_engine import BtcDataEngine
from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.data.kalshi_trade_tape import KalshiTradeTapeService
from kalshi_bot.calibration.microstructure import MicrostructureCalibrator
from kalshi_bot.calibration.time_bucket_analytics import SettledTrade, summarize_time_buckets, time_bucket_performance
from kalshi_bot.database.decision_store import DecisionSnapshot, DecisionStore
from kalshi_bot.execution.executor import Executor
from kalshi_bot.execution.position_monitor import PositionMonitor
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.learning.settlement_ingestion import SettlementIngestor
from kalshi_bot.platform.observation import ObservationBundle
from kalshi_bot.platform.safety import LiveSafetyGate
from kalshi_bot.strategies.base import MarketDecision, StrategyEngine
from kalshi_bot.strategies.kxbtc15m.engine import Kxbtc15mStrategy
from kalshi_bot.strategies.kxbtcd.engine import KxbtcdStrategy
from kalshi_bot.strategy.mispricing import Mispricing, Side
from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine
from kalshi_bot.utils.logging import setup_logging
from kalshi_bot.web.scan_state import GLOBAL_SCAN_STATE, ScanSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PlatformCycleResult:
    asof: float
    decisions_15m: list[MarketDecision] = field(default_factory=list)
    decisions_1h: list[MarketDecision] = field(default_factory=list)
    settlements: list[dict] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)


def _decision_to_mispricing(d: MarketDecision) -> Mispricing | None:
    if d.execute_verdict not in ("TRADE_YES", "TRADE_NO"):
        return None
    side = Side.YES if d.execute_verdict == "TRADE_YES" else Side.NO
    price = d.executable_yes if side == Side.YES else d.executable_no
    if price is None:
        return None
    prob = d.model_prob_yes if side == Side.YES else d.model_prob_no
    return Mispricing(
        ticker=d.ticker,
        series=d.series,
        side=side,
        kalshi_price=price,
        options_prob=prob,
        edge_pp=d.raw_edge * 100,
        edge_after_fees_pp=d.net_edge * 100,
        strike=d.strike,
        spot=d.btc_spot,
        vol=0.0,
        seconds_to_expiry=d.seconds_to_expiry,
        yes_bid=None,
        yes_ask=d.yes_price,
        reason=d.why_trade or d.reason,
    )


class ProductionPlatform:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        enable_15m: bool = True,
        enable_1h: bool = True,
    ) -> None:
        settings = settings or Settings()
        self.settings = settings
        self.config = load_config(settings.config_path)
        self.rules = load_rules_15m()
        setup_logging(settings.log_level)

        self.client = KalshiClient(
            base_url=kalshi_base_url(settings, self.config),
            api_key_id=settings.kalshi_api_key_id,
            private_key_pem=settings.resolve_private_key_pem(),
        )
        self.btc_engine = BtcDataEngine()
        self.engine = V6IntelligenceEngine(
            self.config.v6, client=self.client, rules=self.rules, btc_engine=self.btc_engine
        )
        self.trade_tape = KalshiTradeTapeService(self.client)
        self.micro_calibrator = MicrostructureCalibrator()
        self.settlement = SettlementIngestor("data/settlement_pending.db", self.micro_calibrator)
        self.decisions = DecisionStore("data/decisions.db")
        self.risk = RiskManager(self.config)
        self.executor = Executor(self.client, self.config, self.risk)
        self.position_monitor = PositionMonitor(self.client, self.config, self.executor)

        platform_cfg = self.config.platform
        live_mode = (
            self.config.execution.mode == "live"
            and not self.config.execution.dry_run
            and self.client.authenticated
        )
        self.safety = LiveSafetyGate(
            trading_enabled=platform_cfg.trading_enabled and self.config.v6.live_trading_enabled,
            live_mode=live_mode,
            max_data_age_seconds=platform_cfg.max_data_age_seconds,
            daily_loss_limit_usd=platform_cfg.daily_loss_limit_usd,
            model_version=platform_cfg.model_version,
        )
        self.safety.candidate_model_version = platform_cfg.candidate_model_version

        self.strategies: list[StrategyEngine] = []
        if enable_15m and platform_cfg.enable_kxbtc15m and self.config.v6.enabled:
            self.strategies.append(
                Kxbtc15mStrategy(
                    self.client,
                    self.config,
                    self.rules,
                    self.engine,
                    self.btc_engine,
                    self.trade_tape,
                )
            )
        if enable_1h and platform_cfg.enable_kxbtcd:
            self.strategies.append(KxbtcdStrategy(self.client, self.config))

    def close(self) -> None:
        self.trade_tape.close()
        self.client.close()

    def run_cycle(self, *, execute: bool = False) -> PlatformCycleResult:
        now = time.time()
        api_ok = self.client.authenticated
        ws = self.trade_tape.feed_status()
        self.safety.update_connectivity(api_ok=api_ok, market_data_ok=ws.get("connected", False))

        balance_usd = None
        if api_ok:
            try:
                bal = self.client.get_balance()
                balance_usd = float(bal.get("balance", 0)) / 100.0
                self.safety.update_balance(balance_usd)
            except Exception as exc:
                logger.warning("balance fetch failed: %s", exc)
                self.safety.update_connectivity(api_ok=False, market_data_ok=ws.get("connected", False))

        try:
            self.position_monitor.manage_open_positions(smile=None)
        except Exception:
            logger.exception("position monitor failed")

        decisions_15m: list[MarketDecision] = []
        decisions_1h: list[MarketDecision] = []
        obs = ObservationBundle()

        for strat in self.strategies:
            try:
                rows = strat.evaluate_all()
            except Exception:
                logger.exception("strategy %s failed", strat.name)
                continue
            for d in rows:
                self.decisions.save(
                    DecisionSnapshot(
                        ts=now,
                        strategy=strat.name,
                        series=d.series,
                        ticker=d.ticker,
                        signal=d.signal,
                        reason=d.reason,
                        model_version=self.safety.model_version,
                        time_remaining_s=d.seconds_to_expiry,
                        model_prob_yes=d.model_prob_yes,
                        model_prob_no=d.model_prob_no,
                        market_yes=d.yes_price,
                        market_no=d.no_price,
                        executable_yes=d.executable_yes,
                        executable_no=d.executable_no,
                        raw_edge=d.raw_edge,
                        net_edge=d.net_edge,
                        confidence=d.confidence,
                        regime=d.regime,
                        features=d.features,
                        observations=d.observations.to_dict(),
                        why_trade=d.why_trade,
                        why_not_trade=d.why_not_trade,
                    )
                )
            if strat.series == "KXBTC15M":
                decisions_15m = rows
            else:
                decisions_1h = rows

        settlements: list[dict] = []
        if self.config.platform.enable_settlement_ingestion:
            settlements = self.settlement.ingest(self.client, self.engine)

        if execute:
            self._maybe_execute(decisions_15m + decisions_1h, obs)

        result = PlatformCycleResult(
            asof=now,
            decisions_15m=decisions_15m,
            decisions_1h=decisions_1h,
            settlements=settlements,
            status=self.safety.status().to_dict(),
        )
        self._publish_dashboard(result, balance_usd)
        return result

    def _maybe_execute(self, decisions: list[MarketDecision], obs: ObservationBundle) -> None:
        allow, reason = self.safety.allow_new_orders(obs)
        if not allow:
            logger.info("execution blocked: %s", reason)
            return
        for d in decisions:
            if d.signal not in ("BUY YES", "STRONG BUY YES", "BUY NO", "STRONG BUY NO"):
                continue
            mis = _decision_to_mispricing(d)
            if mis is None:
                continue
            size = d.contracts or self.risk.size(mis)
            if size <= 0:
                continue
            fill = self.executor.execute(mis, size, ignore_cooldown=True)
            if fill and fill.mode == "live":
                self.safety.record_order_confirmed()
                if self.config.platform.enable_settlement_ingestion:
                    self.settlement.record_entry(
                        ticker=d.ticker,
                        side="yes" if mis.side == Side.YES else "no",
                        entry_price=fill.price,
                        contracts=fill.contracts,
                        prediction=d.model_prob_yes if mis.side == Side.YES else d.model_prob_no,
                        confidence=d.confidence,
                        features=d.features,
                        seconds_to_expiry=d.seconds_to_expiry,
                        volatility=d.features.get("realized_vol", 0.45),
                        net_edge=d.net_edge,
                        reason=d.why_trade,
                    )

    def _publish_dashboard(self, result: PlatformCycleResult, balance_usd: float | None) -> None:
        from datetime import datetime, timezone

        from kalshi_bot.calibration.metrics import calibration_table

        def _row(d: MarketDecision) -> dict:
            return {
                "strategy": d.series,
                "ticker": d.ticker,
                "btc": d.btc_spot,
                "strike": d.strike,
                "seconds_to_expiry": d.seconds_to_expiry,
                "model_yes": d.model_prob_yes * 100,
                "model_no": d.model_prob_no * 100,
                "yes_ask": d.executable_yes or d.yes_price,
                "no_ask": d.executable_no or d.no_price,
                "fair_value": d.fair_yes,
                "fair_no": d.fair_no,
                "net_edge": d.net_edge,
                "raw_edge": d.raw_edge,
                "confidence": f"{d.confidence:.0%}",
                "regime": d.regime,
                "decision": d.signal,
                "reason": d.reason,
                "why_trade": d.why_trade,
                "why_not_trade": d.why_not_trade,
                "agreement_score": d.features.get("agreement_score"),
                "orderbook_source": d.features.get("orderbook_source", "rest"),
                "price_pattern": d.features.get("price_pattern"),
                "yes_net_edge": d.features.get("yes_net_edge"),
                "no_net_edge": d.features.get("no_net_edge"),
                "gates": d.features.get("gates"),
                "gates_ready_side": d.features.get("gates_ready_side"),
            }

        opps_15m = [_row(d) for d in result.decisions_15m]
        opps_1h = [_row(d) for d in result.decisions_1h]
        opps = opps_15m + opps_1h
        opps.sort(key=lambda x: x["net_edge"], reverse=True)

        spot = opps[0]["btc"] if opps else 0.0
        cal_records: list[tuple[float, bool]] = []
        settled: list[SettledTrade] = []
        if self.config.platform.enable_settlement_ingestion:
            for row in self.settlement.settled_trades(limit=500):
                cal_records.append((float(row["prediction"]), bool(row["won"])))
            for s in result.settlements:
                pred = s.get("prediction")
                won = s.get("won")
                if pred is not None and won is not None:
                    cal_records.append((float(pred), bool(won)))
            settled = [
                SettledTrade(
                    ticker=r["ticker"],
                    side=r["side"],
                    prediction=float(r["prediction"]),
                    won=bool(r["won"]),
                    pnl=float(r["pnl"] or 0),
                    net_edge=float(r["net_edge"] or 0),
                    seconds_to_expiry=float(r["seconds_to_expiry"] or 0),
                    confidence=float(r.get("confidence") or 0),
                )
                for r in self.settlement.settled_trades(limit=500)
            ]
        calibration = calibration_table(cal_records) if cal_records else []
        if not calibration and self.config.platform.enable_settlement_ingestion:
            calibration = self.settlement.calibration_summary(self.engine.calibrator)
        time_buckets = time_bucket_performance(
            settled,
            min_seconds=self.config.v6.min_seconds_to_expiry,
            max_seconds=self.config.v6.max_seconds_to_expiry,
        )
        bucket_summary = summarize_time_buckets(time_buckets)
        micro_report = (
            self.micro_calibrator.report()
            if self.config.platform.enable_settlement_ingestion
            else {"status": "disabled", "n_total": 0}
        )

        total_pnl = sum(t.pnl for t in settled)
        wins = sum(1 for t in settled if t.won)
        perf = {
            "KXBTC15M": {
                "trade_count": len(result.decisions_15m),
                "signals": {},
                "settled_trades": len(settled),
                "win_rate": wins / len(settled) if settled else None,
                "total_pnl": total_pnl,
            },
            "KXBTCD": {"trade_count": len(result.decisions_1h), "signals": {}},
            "time_buckets": time_buckets,
            "time_bucket_summary": bucket_summary,
            "microstructure": micro_report,
        }
        for label, rows in (("KXBTC15M", result.decisions_15m), ("KXBTCD", result.decisions_1h)):
            for d in rows:
                perf[label]["signals"][d.signal] = perf[label]["signals"].get(d.signal, 0) + 1

        ws = self.trade_tape.feed_status()
        freshness = {
            **result.status,
            "scan_duration_ms": 0,
            "kalshi_ws_connected": ws.get("connected", False),
            "kalshi_ws_last_message_age": ws.get("last_message_age_s"),
            "brti_official": True,
            "brti_source": "platform",
            "live_trading_enabled": self.safety.trading_enabled and self.safety.live_mode,
            "platform_trading_enabled": self.config.platform.trading_enabled,
        }

        GLOBAL_SCAN_STATE.update(
            ScanSnapshot(
                asof=datetime.fromtimestamp(result.asof, tz=timezone.utc),
                spot=spot,
                spot_source="platform",
                balance_usd=balance_usd,
                markets_scanned=len(opps),
                opportunities=opps,
                opportunities_15m=opps_15m,
                opportunities_1h=opps_1h,
                tape=ws,
                settlements=result.settlements,
                calibration=calibration,
                performance=perf,
                microstructure_calibration=micro_report,
                time_bucket_performance=time_buckets,
                safety=result.status,
                freshness=freshness,
            )
        )
