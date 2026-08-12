#!/usr/bin/env python3
"""Kalshi BTC 15-Min Intelligence V6 — mispricing workflow runner.

Finds KXBTC15M contracts where calibrated settlement probability differs
meaningfully from executable Kalshi prices after fees, spread, slippage, and risk.

Usage:
  python run.py                  # single scan + dashboard
  python run.py --loop           # continuous loop
  python run.py --loop -n 30     # 30 iterations
  python run.py --execute        # live execution (requires prod keys + execution.mode: live)
  python run.py --web            # web dashboard + background scan loop
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kalshi_bot.config import BotConfig, Settings, V6Config, kalshi_base_url, load_config, load_rules_15m
from kalshi_bot.data.btc_data_engine import BtcDataEngine, BtcMarketSnapshot
from kalshi_bot.data.brti import resolve_spot
from kalshi_bot.data.kalshi_trade_tape import KalshiTradeTapeService
from kalshi_bot.learning.settlement_ingestion import SettlementIngestor
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.data.realized_vol import estimate_realized_vol
from kalshi_bot.execution.executor import Executor
from kalshi_bot.execution.position_monitor import PositionMonitor
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.models.probability import options_implied_prob_up
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.dashboard import (
    print_mispricing_dashboard,
    print_opportunities_table,
    print_scan_summary,
)
from kalshi_bot.strategy.mispricing_engine import MispricingOpportunity, TradeAction
from kalshi_bot.strategy.mispricing_evaluator import evaluate_market_mispricing
from kalshi_bot.strategy.trade_filter import TradeDecision
from kalshi_bot.strategy.mispricing import Mispricing, Side
from kalshi_bot.strategy.v6_upgrades import Regime, V6Decision, V6IntelligenceEngine
from kalshi_bot.utils.logging import setup_logging
from kalshi_bot.web.scan_state import GLOBAL_SCAN_STATE, ScanSnapshot

logger = logging.getLogger(__name__)
console = Console()

WORKFLOW_NAME = "KXBTC15M Mispricing"
WORKFLOW_VERSION = "V7"


# ---------------------------------------------------------------------------
# Scan result
# ---------------------------------------------------------------------------

@dataclass
class V6TradeCandidate:
    ticker: str
    decision: V6Decision


@dataclass
class MispricingCandidate:
    ticker: str
    decision: V6Decision
    opportunity: MispricingOpportunity
    trade: TradeDecision


@dataclass
class V6ScanResult:
    spot: float
    spot_source: str
    spot_official: bool
    markets_scanned: int
    trades: list[V6TradeCandidate]
    no_trades: list[V6TradeCandidate]
    asof: datetime
    vol_ann: float
    top_blockers: list[tuple[str, int]] = field(default_factory=list)
    btc: BtcMarketSnapshot | None = None
    balance_usd: float | None = None
    mispricing_rows: list[MispricingCandidate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# V6 scanner
# ---------------------------------------------------------------------------

class V6Scanner:
    """Scan KXBTC15M markets for mispricing vs calibrated settlement probability."""

    def __init__(
        self,
        client: KalshiClient,
        config: BotConfig,
        engine: V6IntelligenceEngine,
        btc_engine: BtcDataEngine,
        *,
        trade_tape: KalshiTradeTapeService | None = None,
        settlement: SettlementIngestor | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.engine = engine
        self.v6 = config.v6
        self.btc_engine = btc_engine
        self.trade_tape = trade_tape
        self.settlement = settlement
        self.rules = engine.rules

    def scan(self, smile: VolSmile | None = None) -> V6ScanResult:
        now = datetime.now(timezone.utc)
        fallback = smile.spot_btc if smile else None
        spot_snap = resolve_spot(self.client, fallback_btc=fallback, brti_cfg=self.config.brti)
        spot = spot_snap.brti
        self.engine.update_spot(spot)

        rv = estimate_realized_vol(horizon_seconds=self.v6.max_seconds_to_expiry)
        vol = rv.annualized_vol
        if smile is not None:
            vol = 0.6 * vol + 0.4 * smile.atm_iv

        btc = self.btc_engine.refresh(
            reference_price=spot,
            reference_source=spot_snap.source,
            is_official=spot_snap.is_official,
            annualized_vol=vol,
        )

        balance_usd: float | None = None
        if self.client.authenticated:
            try:
                bal = self.client.get_balance()
                balance_usd = float(bal.get("balance", 0)) / 100.0
            except Exception as exc:
                logger.warning("balance fetch failed: %s", exc)

        trades: list[V6TradeCandidate] = []
        no_trades: list[V6TradeCandidate] = []
        mispricing_rows: list[MispricingCandidate] = []
        scanned = 0
        blocker_counts: Counter[str] = Counter()
        use_mispricing = self.rules.enabled and self.rules.mode == "mispricing"
        open_tickers: list[str] = []

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
            scanned += 1
            ticker = str(market.get("ticker") or "")
            open_tickers.append(ticker)

            options_prob = None
            strike = market.get("strike")
            if smile is not None and strike is not None:
                try:
                    options_prob = options_implied_prob_up(
                        spot_btc=spot,
                        open_level=float(strike),
                        close_time=close,
                        smile=smile,
                        rate=self.config.smile.risk_free_rate,
                        dividend=self.config.smile.dividend_yield,
                        now=now,
                    ).probability
                except Exception:
                    pass

            if use_mispricing:
                recent_trades = None
                tape_stats = None
                if self.trade_tape is not None:
                    self.trade_tape.ensure_subscription([ticker])
                    recent_trades = self.trade_tape.recent_trades(ticker)
                    tape_stats = self.trade_tape.tape_stats(ticker)
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
                    kalshi_stale=tape_stats.stale if tape_stats else False,
                )
                self.engine.get_monitor().record(audit)
                decision = V6Decision(
                    verdict=audit.verdict,
                    model_probability=audit.model_prob_up,
                    market_price=(
                        audit.yes_side.executable_ask
                        if audit.verdict == "TRADE_YES"
                        else audit.no_side.executable_ask
                        if audit.verdict == "TRADE_NO"
                        else None
                    ),
                    strict_gap_dollars=max(
                        audit.yes_side.raw_edge_dollars, audit.no_side.raw_edge_dollars
                    ),
                    confidence=audit.model_confidence,
                    explainability=audit.explainability,
                    regime=Regime.CHOP,
                    monte_carlo_prob=audit.monte_carlo_prob,
                    calibrated=audit.calibrated,
                    pattern_examples=0,
                    pattern_win_rate=None,
                    quality=None,
                    ensemble=None,
                    micro=None,
                    reasons=(audit.trade_reason,),
                    blockers=tuple(
                        c.value for c in audit.all_rejection_codes if c.value != "NONE"
                    ),
                    contracts=audit.contracts,
                    audit_record=audit,
                )
                mispricing_rows.append(
                    MispricingCandidate(ticker=ticker, decision=decision, opportunity=opp, trade=trade_dec)
                )
            else:
                decision = self.engine.evaluate(
                    market,
                    spot=spot,
                    vol=vol,
                    options_prob=options_prob,
                    now=now,
                    spot_source=spot_snap.source,
                    spot_is_official=spot_snap.is_official,
                    btc_snapshot=btc,
                )

            candidate = V6TradeCandidate(ticker=ticker, decision=decision)
            if decision.verdict != "NO_TRADE":
                trades.append(candidate)
            else:
                no_trades.append(candidate)
                primary = (
                    decision.audit_record.primary_rejection.value
                    if decision.audit_record
                    else (decision.blockers[0] if decision.blockers else "UNKNOWN")
                )
                blocker_counts[primary] += 1

        trades.sort(key=lambda c: c.decision.strict_gap_dollars, reverse=True)
        mispricing_rows.sort(key=lambda r: r.opportunity.best_net_edge, reverse=True)
        top_blockers = blocker_counts.most_common(8)

        settlements: list[dict] = []
        if self.settlement is not None:
            settlements = self.settlement.ingest(self.client, self.engine)

        result = V6ScanResult(
            spot=spot,
            spot_source=spot_snap.source,
            spot_official=spot_snap.is_official,
            markets_scanned=scanned,
            trades=trades,
            no_trades=no_trades,
            asof=now,
            vol_ann=vol,
            top_blockers=top_blockers,
            btc=btc,
            balance_usd=balance_usd,
            mispricing_rows=mispricing_rows,
        )
        _publish_scan_state(
            result,
            trade_tape=self.trade_tape,
            engine=self.engine,
            settlement=self.settlement,
            settlements=settlements,
        )
        logger.info(
            "%s %s scan: markets=%d trades=%d no_trade=%d spot=%.2f vol=%.1f%%",
            WORKFLOW_NAME,
            WORKFLOW_VERSION,
            scanned,
            len(trades),
            len(no_trades),
            spot,
            vol * 100,
        )
        return result


# ---------------------------------------------------------------------------
# Scan state (web dashboard)
# ---------------------------------------------------------------------------

def _publish_scan_state(
    result: V6ScanResult,
    *,
    trade_tape: KalshiTradeTapeService | None = None,
    engine: V6IntelligenceEngine | None = None,
    settlement: SettlementIngestor | None = None,
    settlements: list[dict] | None = None,
) -> None:
    tape: dict[str, dict] = {}
    opps: list[dict] = []
    for row in result.mispricing_rows:
        ticker = row.ticker
        tape_tps = 0.0
        if trade_tape is not None:
            tstats = trade_tape.tape_stats(ticker)
            tape[ticker] = {
                "tps": tstats.trades_per_second,
                "buy_pressure": tstats.buy_pressure,
                "volume_1m": tstats.volume_1m,
                "last_price": tstats.last_price,
                "stale": tstats.stale,
                "source": tstats.source,
            }
            tape_tps = tstats.trades_per_second
        opps.append(
            {
                "ticker": ticker,
                "btc": result.spot,
                "strike": row.opportunity.strike,
                "seconds_to_expiry": row.opportunity.seconds_to_expiry,
                "model_yes": row.opportunity.model_yes_pct,
                "yes_ask": row.opportunity.yes.executable_ask,
                "fair_value": row.opportunity.fair_value_yes,
                "net_edge": row.opportunity.best_net_edge,
                "confidence": row.opportunity.confidence_label,
                "volatility": row.opportunity.volatility_label,
                "order_flow": row.opportunity.order_flow_label,
                "liquidity": row.opportunity.liquidity_label,
                "tape_tps": tape_tps,
                "decision": row.trade.action.value,
                "reason": row.trade.reason,
            }
        )
    calibration: list[dict] = []
    if engine is not None and settlement is not None:
        calibration = settlement.calibration_summary(engine.calibrator)
    GLOBAL_SCAN_STATE.update(
        ScanSnapshot(
            asof=result.asof,
            spot=result.spot,
            spot_source=result.spot_source,
            balance_usd=result.balance_usd,
            markets_scanned=result.markets_scanned,
            opportunities=opps,
            tape=tape,
            settlements=settlements or [],
            calibration=calibration,
        )
    )


def _record_fill_entry(
    settlement: SettlementIngestor | None,
    *,
    ticker: str,
    side: str,
    price: float,
    contracts: int,
    prediction: float,
    confidence: float,
    seconds_to_expiry: float,
    volatility: float,
    net_edge: float,
    reason: str,
    micro_features: dict | None = None,
) -> None:
    if settlement is None:
        return
    settlement.record_entry(
        ticker=ticker,
        side=side,
        entry_price=price,
        contracts=contracts,
        prediction=prediction,
        confidence=confidence,
        features=micro_features or {},
        seconds_to_expiry=seconds_to_expiry,
        volatility=volatility,
        net_edge=net_edge,
        reason=reason,
    )


def _record_fill_from_candidate(
    settlement: SettlementIngestor | None,
    candidate: V6TradeCandidate,
    *,
    price: float,
    contracts: int,
    vol_ann: float,
) -> None:
    if settlement is None:
        return
    d = candidate.decision
    audit = d.audit_record
    side = "yes" if d.verdict == "TRADE_YES" else "no"
    features: dict = {}
    if audit is not None:
        features = {
            "liquidity_score": audit.liquidity_score,
            "bid_ask_imbalance": audit.bid_ask_imbalance,
            "order_book_depth_bid": audit.order_book_depth_bid,
            "order_book_depth_ask": audit.order_book_depth_ask,
            "spread": audit.spread,
        }
    _record_fill_entry(
        settlement,
        ticker=candidate.ticker,
        side=side,
        price=price,
        contracts=contracts,
        prediction=d.model_probability if side == "yes" else 1.0 - d.model_probability,
        confidence=d.confidence,
        seconds_to_expiry=audit.seconds_to_expiry if audit else 0.0,
        volatility=vol_ann,
        net_edge=audit.best_net_edge if audit else d.strict_gap_dollars,
        reason="; ".join(d.reasons) if d.reasons else d.verdict,
        micro_features=features,
    )


# ---------------------------------------------------------------------------
# Decision → executor adapter
# ---------------------------------------------------------------------------

def v6_to_mispricing(ticker: str, decision: V6Decision) -> Mispricing | None:
    if decision.verdict == "NO_TRADE" or decision.market_price is None:
        return None
    side = Side.YES if decision.verdict == "TRADE_YES" else Side.NO
    prob = decision.model_probability if side == Side.YES else 1.0 - decision.model_probability
    gap_pp = decision.strict_gap_dollars * 100
    return Mispricing(
        ticker=ticker,
        series="KXBTC15M",
        side=side,
        kalshi_price=decision.market_price,
        options_prob=prob,
        edge_pp=gap_pp,
        edge_after_fees_pp=gap_pp,
        strike=0.0,
        spot=0.0,
        vol=0.0,
        seconds_to_expiry=0.0,
        yes_bid=None,
        yes_ask=decision.market_price if side == Side.YES else None,
        implied=None,  # type: ignore[arg-type]
        reason="; ".join(decision.reasons),
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_mispricing_results(result: V6ScanResult, top: int) -> None:
    if result.btc is None:
        _print_header(result)
        return
    print_scan_summary(
        asof=result.asof,
        markets_scanned=result.markets_scanned,
        trades=len(result.trades),
        btc=result.btc,
        balance_usd=result.balance_usd,
    )
    if result.mispricing_rows:
        print_opportunities_table(
            [(r.ticker, r.opportunity, r.trade) for r in result.mispricing_rows[:top]]
        )
        best = result.mispricing_rows[0]
        print_mispricing_dashboard(
            series="KXBTC15M",
            btc=result.btc,
            opp=best.opportunity,
            decision=best.trade,
            balance_usd=result.balance_usd,
        )
    elif result.trades:
        row = result.trades[0]
        console.print(f"[green]Best trade:[/green] {row.decision.verdict} {row.ticker}")
    else:
        console.print("[yellow]No qualifying mispricing opportunities this cycle.[/yellow]")


def _print_header(result: V6ScanResult) -> None:
    official = "official BRTI" if result.spot_official else "PROXY"
    console.print(
        Panel(
            f"[bold]{WORKFLOW_NAME}[/bold]  [cyan]{WORKFLOW_VERSION}[/cyan]\n"
            f"spot={result.spot:,.2f} ({result.spot_source}; {official})  "
            f"scanned={result.markets_scanned}  σ={result.vol_ann*100:.1f}%",
            border_style="blue",
        )
    )


def _print_trades(candidates: list[V6TradeCandidate], top: int) -> None:
    if not candidates:
        return
    table = Table(title="V6 TRADE candidates (strict edge + quality gates)")
    table.add_column("Verdict")
    table.add_column("Ticker")
    table.add_column("Model")
    table.add_column("Market")
    table.add_column("Gap ¢")
    table.add_column("Explain")
    table.add_column("Regime")
    table.add_column("Contracts")
    for c in candidates[:top]:
        d = c.decision
        table.add_row(
            d.verdict,
            c.ticker,
            f"{d.model_probability*100:.1f}%",
            f"{(d.market_price or 0)*100:.0f}¢",
            f"{d.strict_gap_dollars*100:.0f}",
            f"{d.explainability:.2f}",
            d.regime.value,
            str(d.contracts),
        )
    console.print(table)


def _print_opportunity_monitor(candidates: list[V6TradeCandidate]) -> None:
    """Dashboard row per evaluated market."""
    if not candidates:
        return
    table = Table(title="Opportunity Monitor (both sides evaluated)")
    table.add_column("Ticker")
    table.add_column("t_rem")
    table.add_column("Model↑")
    table.add_column("YES¢")
    table.add_column("YES net")
    table.add_column("NO¢")
    table.add_column("NO net")
    table.add_column("Conf")
    table.add_column("Edge tier")
    table.add_column("Action")
    table.add_column("Decision")
    table.add_column("Reason")
    for c in candidates:
        d = c.decision
        audit = d.audit_record
        if audit:
            table.add_row(
                c.ticker[-20:],
                f"{audit.minutes_to_expiry:.1f}m",
                f"{audit.model_prob_up*100:.0f}%",
                f"{(audit.yes_ask or 0)*100:.0f}",
                f"{audit.yes_side.net_edge_dollars*100:+.0f}",
                f"{(audit.no_ask or 0)*100:.0f}",
                f"{audit.no_side.net_edge_dollars*100:+.0f}",
                f"{audit.model_confidence*100:.0f}%",
                audit.edge_quality,
                audit.edge_action[:30],
                audit.verdict,
                audit.primary_rejection.value,
            )
    console.print(table)


def _print_near_misses(candidates: list[V6TradeCandidate], top: int) -> None:
    near = [
        c for c in candidates
        if c.decision.audit_record
        and (
            c.decision.audit_record.best_net_edge > 0.03
            or c.decision.model_probability > 0.45
        )
    ][:top]
    if not near:
        return
    table = Table(title="Near misses (NO TRADE)")
    table.add_column("Ticker")
    table.add_column("Model")
    table.add_column("Best net¢")
    table.add_column("Tier")
    table.add_column("Primary rejection")
    for c in near:
        audit = c.decision.audit_record
        if not audit:
            continue
        table.add_row(
            c.ticker,
            f"{audit.model_prob_up*100:.1f}%",
            f"{audit.best_net_edge*100:.0f}",
            audit.setup_tier,
            audit.primary_rejection.value,
        )
    console.print(table)


def _print_verdict(result: V6ScanResult) -> None:
    if result.mispricing_rows:
        best_row = result.mispricing_rows[0]
        if best_row.trade.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
            console.print(f"\n[green bold]VERDICT: {best_row.trade.action.value}[/green bold]")
        elif best_row.trade.action == TradeAction.WAIT:
            console.print(f"\n[yellow bold]VERDICT: WAIT[/yellow bold] — {best_row.trade.reason}")
        else:
            console.print(f"\n[red bold]VERDICT: NO TRADE[/red bold] — {best_row.trade.reason}")
        return
    if result.trades:
        best = result.trades[0]
        d = best.decision
        console.print(
            f"\n[green bold]ENGINE VERDICT: {d.verdict}[/green bold] {best.ticker} "
            f"@ {(d.market_price or 0)*100:.0f}¢ | gap={d.strict_gap_dollars*100:.0f}¢ "
            f"| explain={d.explainability:.2f}"
        )
        for r in d.reasons:
            console.print(f"  · {r}")
    else:
        console.print("\n[yellow bold]ENGINE VERDICT: NO TRADE[/yellow bold]")
        if result.top_blockers:
            console.print(f"Top blocker: {result.top_blockers[0][0]}")


# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------

def _assert_live_only(config: BotConfig) -> None:
    if config.execution.mode != "live" or config.execution.dry_run:
        raise SystemExit(
            "15-minute bot requires live execution: set execution.mode=live and dry_run=false"
        )


def build_v6_runtime(
    settings: Settings | None = None,
    *,
    strict_gap: float | None = None,
) -> tuple[
    Settings,
    BotConfig,
    KalshiClient,
    V6IntelligenceEngine,
    V6Scanner,
    Executor,
    BtcDataEngine,
    KalshiTradeTapeService,
    SettlementIngestor,
]:
    settings = settings or Settings()
    config = load_config(settings.config_path)
    rules = load_rules_15m()
    if strict_gap is not None:
        rules.enabled = True
        rules.strict_edge.min_gap_dollars = strict_gap
    setup_logging(settings.log_level)
    client = KalshiClient(
        base_url=kalshi_base_url(settings, config),
        api_key_id=settings.kalshi_api_key_id,
        private_key_pem=settings.resolve_private_key_pem(),
    )
    if not client.authenticated:
        logger.warning("Kalshi client not authenticated — balance/execution unavailable")
    btc_engine = BtcDataEngine()
    engine = V6IntelligenceEngine(config.v6, client=client, rules=rules, btc_engine=btc_engine)
    trade_tape = KalshiTradeTapeService(client)
    settlement = SettlementIngestor("data/settlement_pending.db")
    scanner = V6Scanner(
        client,
        config,
        engine,
        btc_engine,
        trade_tape=trade_tape,
        settlement=settlement,
    )
    risk = RiskManager(config)
    executor = Executor(client, config, risk)
    return settings, config, client, engine, scanner, executor, btc_engine, trade_tape, settlement


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_v6_once(
    *,
    allow_synthetic_smile: bool = False,
    execute: bool = False,
    strict_gap: float | None = None,
    top: int = 10,
) -> V6ScanResult:
    """Single V6 intelligence scan cycle."""
    settings, config, client, engine, scanner, executor, _btc, _tape, settlement = build_v6_runtime(
        strict_gap=strict_gap,
    )
    try:
        smile = None
        try:
            smile = load_ibit_smile(config.smile, allow_synthetic=allow_synthetic_smile)
        except Exception as exc:
            logger.warning("smile unavailable (%s); realized-vol only", exc)

        result = scanner.scan(smile)

        _print_mispricing_results(result, top)
        _print_trades(result.trades, top)
        _print_near_misses(result.no_trades, top)

        if result.top_blockers:
            console.print("Primary rejection frequency:")
            for name, count in result.top_blockers:
                console.print(f"  {count:4d}  {name}")

        # Show detailed audit for best near-miss
        if result.no_trades and result.no_trades[0].decision.audit_record:
            console.print("\n[dim]Sample audit record:[/dim]")
            console.print(result.no_trades[0].decision.audit_record.summary_text())

        _print_verdict(result)

        if execute and result.trades and config.v6.live_trading_enabled:
            for c in result.trades:
                mis = v6_to_mispricing(c.ticker, c.decision)
                if mis is None:
                    continue
                size = c.decision.contracts or executor.risk.size(mis)
                if size > 0:
                    fill = executor.execute(mis, size, ignore_cooldown=True)
                    if fill:
                        _record_fill_from_candidate(
                            settlement,
                            c,
                            price=fill.price,
                            contracts=fill.contracts,
                            vol_ann=result.vol_ann,
                        )
                    logger.info("executed %s %s x%d", c.decision.verdict, c.ticker, size)
        elif execute and result.trades and not config.v6.live_trading_enabled:
            console.print("[yellow]Trades found but live_trading_enabled=false — paper only[/yellow]")

        return result
    finally:
        _tape.close()
        client.close()


def run_v6_loop(
    *,
    max_iterations: int | None = None,
    interval: float | None = None,
    execute: bool = False,
    strict_gap: float | None = None,
) -> None:
    """Continuous V6 intelligence loop."""
    settings, config, client, engine, scanner, executor, _btc, _tape, settlement = build_v6_runtime(
        strict_gap=strict_gap,
    )
    sleep_s = interval or config.scan_interval_seconds
    position_monitor = PositionMonitor(client, config, executor)
    iterations = 0
    try:
        smile = None
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                if smile is None:
                    try:
                        smile = load_ibit_smile(config.smile, allow_synthetic=False)
                    except Exception:
                        smile = None
                position_monitor.manage_open_positions(smile=smile)
                result = scanner.scan(smile)
                if result.trades and config.v6.live_trading_enabled:
                    best = result.trades[0]
                    if execute:
                        mis = v6_to_mispricing(best.ticker, best.decision)
                        if mis is None:
                            logger.warning("skip %s: could not build order", best.ticker)
                        else:
                            size = best.decision.contracts or executor.risk.size(mis)
                            if size <= 0:
                                logger.info(
                                    "skip %s %s: size=0 (bankroll=$%.2f price=%.0f¢)",
                                    best.decision.verdict,
                                    best.ticker,
                                    config.risk.bankroll_usd,
                                    (mis.kalshi_price or 0) * 100,
                                )
                            else:
                                fill = executor.execute(mis, size, ignore_cooldown=True)
                                if fill:
                                    _record_fill_from_candidate(
                                        settlement,
                                        best,
                                        price=fill.price,
                                        contracts=fill.contracts,
                                        vol_ann=result.vol_ann,
                                    )
                                    logger.info(
                                        "executed %s %s x%d mode=%s",
                                        best.decision.verdict,
                                        best.ticker,
                                        size,
                                        fill.mode,
                                    )
                    console.print(
                        f"[green]{best.decision.verdict}[/green] {best.ticker} "
                        f"gap={best.decision.strict_gap_dollars*100:.0f}¢"
                    )
                else:
                    logger.info("NO TRADE cycle %d", iterations)
            except Exception:
                logger.exception("V6 scan iteration %d failed", iterations)
            time.sleep(sleep_s)
    finally:
        _tape.close()
        client.close()


def run_web(
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    interval: float | None = None,
    execute: bool = False,
    strict_gap: float | None = None,
) -> None:
    """Start web dashboard with background mispricing scan loop."""
    import threading

    from kalshi_bot.web.web_server import run_server

    settings, config, client, _engine, scanner, executor, _btc, trade_tape, settlement = build_v6_runtime(
        strict_gap=strict_gap,
    )
    sleep_s = interval or config.scan_interval_seconds
    stop = threading.Event()

    def scan_loop() -> None:
        smile = None
        while not stop.is_set():
            try:
                if smile is None:
                    try:
                        smile = load_ibit_smile(config.smile, allow_synthetic=False)
                    except Exception:
                        smile = None
                result = scanner.scan(smile)
                if execute and result.trades and config.v6.live_trading_enabled:
                    best = result.trades[0]
                    mis = v6_to_mispricing(best.ticker, best.decision)
                    if mis is not None:
                        size = best.decision.contracts or executor.risk.size(mis)
                        if size > 0:
                            fill = executor.execute(mis, size, ignore_cooldown=True)
                            if fill:
                                _record_fill_from_candidate(
                                    settlement,
                                    best,
                                    price=fill.price,
                                    contracts=fill.contracts,
                                    vol_ann=result.vol_ann,
                                )
            except Exception:
                logger.exception("web scan loop failed")
            stop.wait(sleep_s)

    worker = threading.Thread(target=scan_loop, name="web-scan", daemon=True)
    worker.start()
    console.print(
        f"[green]Web dashboard at http://localhost:{port}[/green] "
        f"(scan every {sleep_s:.0f}s)\n"
        "[dim]Cloud Agent: open the forwarded port URL from the Ports panel, not your local localhost[/dim]"
    )
    try:
        run_server(host=host, port=port, with_scan=False)
    finally:
        stop.set()
        trade_tape.close()
        client.close()


def collect_diagnostics(
    *,
    target: int = 100,
    interval: float = 3.0,
    max_iterations: int = 500,
    allow_synthetic_smile: bool = False,
) -> None:
    """Paper-mode diagnostic collection until target evaluations reached."""
    settings, config, client, engine, scanner, _executor, _btc, _tape, _settlement = build_v6_runtime()
    monitor = engine.get_monitor()
    collected = 0
    iterations = 0
    smile = None
    try:
        while collected < target and iterations < max_iterations:
            iterations += 1
            try:
                if smile is None:
                    try:
                        smile = load_ibit_smile(config.smile, allow_synthetic=allow_synthetic_smile)
                    except Exception:
                        smile = None
                result = scanner.scan(smile)
                n = len(result.trades) + len(result.no_trades)
                collected = monitor.rejection_breakdown().total_evaluations
                logger.info(
                    "collect iter=%d batch=%d total=%d/%d",
                    iterations,
                    n,
                    collected,
                    target,
                )
            except Exception:
                logger.exception("collect iteration %d failed", iterations)
            time.sleep(interval)
    finally:
        _tape.close()
        client.close()

    breakdown = monitor.rejection_breakdown()
    console.print(Panel(breakdown.summary_text(), title="Diagnostic Collection Complete"))
    edge_dist = monitor.edge_distribution()
    console.print("Edge distribution (best net edge per eval):")
    for bucket, count in edge_dist.items():
        console.print(f"  {bucket}: {count}")
    tier_hyp = monitor.tier_hypothetical_breakdown()
    tier_edge_only = monitor.hypothetical_trades_by_tier()
    filter_attr = monitor.filter_attribution()
    if tier_hyp or tier_edge_only:
        console.print("Hypothetical tier qualifications:")
        for tier, count in sorted(tier_hyp.items()):
            console.print(f"  {tier}: {count}")
    if tier_edge_only:
        console.print("If model block removed (edge-only tiers):")
        for tier, count in sorted(tier_edge_only.items()):
            console.print(f"  {tier}: {count}")
    if filter_attr:
        console.print("Filter failure attribution:")
        for name, count in sorted(filter_attr.items(), key=lambda x: -x[1]):
            console.print(f"  {name}: {count}")
    report_path = "data/diagnostics/rejection_report.json"
    monitor.export_report(report_path)
    console.print(f"\nFull report saved to {report_path}")


def print_diagnostic_report() -> None:
    """Print rejection breakdown from stored evaluations."""
    from kalshi_bot.strategy.opportunity_monitor import OpportunityMonitor

    config = load_config()
    monitor = OpportunityMonitor(config.v6.diagnostics_db_path)
    breakdown = monitor.rejection_breakdown()
    console.print(Panel(breakdown.summary_text(), title="Rejection Breakdown"))
    console.print("Edge distribution:", monitor.edge_distribution())
    console.print("Tier hypothetical:", monitor.tier_hypothetical_breakdown())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"{WORKFLOW_NAME} {WORKFLOW_VERSION}",
    )
    parser.add_argument("--loop", action="store_true", help="Run continuous scan loop")
    parser.add_argument("-n", "--iterations", type=int, default=None, help="Max loop iterations")
    parser.add_argument("--interval", type=float, default=None, help="Scan interval seconds")
    parser.add_argument(
        "--strict",
        type=float,
        default=None,
        help="Strict min gap in dollars (0.20=20¢, 0.25=25¢)",
    )
    parser.add_argument("--execute", action="store_true", help="Execute trades (paper/live per config)")
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic smile fallback")
    parser.add_argument("--top", type=int, default=10, help="Rows to display")
    parser.add_argument(
        "--collect",
        type=int,
        default=None,
        metavar="N",
        help="Paper diagnostic collection: evaluate N markets",
    )
    parser.add_argument("--report", action="store_true", help="Print rejection breakdown report")
    parser.add_argument("--web", action="store_true", help="Start web dashboard with background scan loop")
    parser.add_argument("--host", default="0.0.0.0", help="Web server bind host (with --web)")
    parser.add_argument("--port", type=int, default=8080, help="Web server port (with --web)")
    args = parser.parse_args(argv)

    if args.report:
        print_diagnostic_report()
        return

    if args.collect:
        collect_diagnostics(target=args.collect, interval=args.interval or 3.0)
        return

    if not load_config().v6.enabled:
        console.print("[red]V6 workflow disabled in config (v6.enabled=false)[/red]")
        sys.exit(1)

    config = load_config()
    if args.execute or args.loop:
        _assert_live_only(config)

    if args.web:
        run_web(
            host=args.host,
            port=args.port,
            interval=args.interval,
            execute=args.execute,
            strict_gap=args.strict,
        )
    elif args.loop:
        run_v6_loop(
            max_iterations=args.iterations,
            interval=args.interval,
            execute=args.execute,
            strict_gap=args.strict,
        )
    else:
        run_v6_once(
            allow_synthetic_smile=args.allow_synthetic,
            execute=args.execute,
            strict_gap=args.strict,
            top=args.top,
        )


if __name__ == "__main__":
    main()
