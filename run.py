#!/usr/bin/env python3
"""Kalshi BTC 15-Min Intelligence V6 — integrated workflow runner.

Workflow: "Kalshi BTC 15-Min Intelligence" (V6, activated)

STRICT EDGE RULE (hard filter):
  Only recommend BUY when market price is ≥20–25¢ below model probability.
  Example: agent thinks 60% UP → market YES must be ≤35–40¢.
  No exceptions for A/B setups; replaces legacy 6pp/3pp gap tiers.

Usage:
  python run.py                  # single V6 scan cycle
  python run.py --loop           # continuous 15m intelligence loop
  python run.py --loop -n 30     # 30 iterations then exit
  python run.py --strict 0.25     # 25¢ minimum gap (extra strict)
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

from kalshi_bot.config import BotConfig, Settings, V6Config, kalshi_base_url, load_config
from kalshi_bot.data.brti import resolve_spot
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.data.realized_vol import estimate_realized_vol
from kalshi_bot.execution.executor import Executor
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.models.probability import options_implied_prob_up
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.mispricing import Mispricing, Side
from kalshi_bot.strategy.v6_upgrades import V6Decision, V6IntelligenceEngine
from kalshi_bot.utils.logging import setup_logging

logger = logging.getLogger(__name__)
console = Console()

WORKFLOW_NAME = "Kalshi BTC 15-Min Intelligence"
WORKFLOW_VERSION = "V6"


# ---------------------------------------------------------------------------
# Scan result
# ---------------------------------------------------------------------------

@dataclass
class V6TradeCandidate:
    ticker: str
    decision: V6Decision


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


# ---------------------------------------------------------------------------
# V6 scanner
# ---------------------------------------------------------------------------

class V6Scanner:
    """Scan KXBTC15M markets through the V6 intelligence engine."""

    def __init__(
        self,
        client: KalshiClient,
        config: BotConfig,
        engine: V6IntelligenceEngine,
    ) -> None:
        self.client = client
        self.config = config
        self.engine = engine
        self.v6 = config.v6

    def scan(self, smile: VolSmile | None = None) -> V6ScanResult:
        now = datetime.now(timezone.utc)
        fallback = smile.spot_btc if smile else None
        spot_snap = resolve_spot(self.client, fallback_btc=fallback)
        spot = spot_snap.brti
        self.engine.update_spot(spot)

        rv = estimate_realized_vol(horizon_seconds=self.v6.max_seconds_to_expiry)
        vol = rv.annualized_vol
        if smile is not None:
            vol = 0.6 * vol + 0.4 * smile.atm_iv

        trades: list[V6TradeCandidate] = []
        no_trades: list[V6TradeCandidate] = []
        scanned = 0
        blocker_counts: Counter[str] = Counter()

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

            decision = self.engine.evaluate(
                market,
                spot=spot,
                vol=vol,
                options_prob=options_prob,
                now=now,
                spot_source=spot_snap.source,
                spot_is_official=spot_snap.is_official,
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
        top_blockers = blocker_counts.most_common(8)

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
        return V6ScanResult(
            spot=spot,
            spot_source=spot_snap.source,
            spot_official=spot_snap.is_official,
            markets_scanned=scanned,
            trades=trades,
            no_trades=no_trades,
            asof=now,
            vol_ann=vol,
            top_blockers=top_blockers,
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
        console.print("Strict 20–25¢ edge rule + quality gates not satisfied.")


# ---------------------------------------------------------------------------
# Runtime builder
# ---------------------------------------------------------------------------

def build_v6_runtime(
    settings: Settings | None = None,
    *,
    strict_gap: float | None = None,
) -> tuple[Settings, BotConfig, KalshiClient, V6IntelligenceEngine, V6Scanner, Executor]:
    settings = settings or Settings()
    config = load_config(settings.config_path)
    if strict_gap is not None:
        config.v6.strict_min_gap_dollars = strict_gap
    setup_logging(settings.log_level)
    client = KalshiClient(
        base_url=kalshi_base_url(settings, config),
        api_key_id=settings.kalshi_api_key_id,
        private_key_pem=settings.resolve_private_key_pem(),
    )
    engine = V6IntelligenceEngine(config.v6, client=client)
    scanner = V6Scanner(client, config, engine)
    risk = RiskManager(config)
    executor = Executor(client, config, risk)
    return settings, config, client, engine, scanner, executor


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
    settings, config, client, engine, scanner, executor = build_v6_runtime(
        strict_gap=strict_gap,
    )
    try:
        smile = None
        try:
            smile = load_ibit_smile(config.smile, allow_synthetic=allow_synthetic_smile)
        except Exception as exc:
            logger.warning("smile unavailable (%s); realized-vol only", exc)

        result = scanner.scan(smile)

        _print_header(result)
        _print_opportunity_monitor(result.trades + result.no_trades)
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
                    executor.execute(mis, size, ignore_cooldown=True)
                    logger.info("executed %s %s x%d", c.decision.verdict, c.ticker, size)
        elif execute and result.trades and not config.v6.live_trading_enabled:
            console.print("[yellow]Trades found but live_trading_enabled=false — paper only[/yellow]")

        return result
    finally:
        client.close()


def run_v6_loop(
    *,
    max_iterations: int | None = None,
    interval: float | None = None,
    execute: bool = False,
    strict_gap: float | None = None,
) -> None:
    """Continuous V6 intelligence loop."""
    settings, config, client, engine, scanner, executor = build_v6_runtime(
        strict_gap=strict_gap,
    )
    sleep_s = interval or config.scan_interval_seconds
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
                result = scanner.scan(smile)
                if result.trades and config.v6.live_trading_enabled:
                    best = result.trades[0]
                    if execute:
                        mis = v6_to_mispricing(best.ticker, best.decision)
                        if mis:
                            size = best.decision.contracts or executor.risk.size(mis)
                            executor.execute(mis, size, ignore_cooldown=True)
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
        client.close()


def collect_diagnostics(
    *,
    target: int = 100,
    interval: float = 3.0,
    max_iterations: int = 500,
    allow_synthetic_smile: bool = False,
) -> None:
    """Paper-mode diagnostic collection until target evaluations reached."""
    settings, config, client, engine, scanner, _executor = build_v6_runtime()
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

    if args.loop:
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
