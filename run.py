#!/usr/bin/env python3
"""Kalshi 15-Min Intelligence V6 — multi-asset workflow runner.

Scans all configured 15-minute markets (BTC, ETH, SOL, XRP, DOGE, BNB, HYPE,
plus optional commodities/indices). Each market is evaluated independently on
both YES (up) and NO (down) sides — the agent does not blindly follow the
market favorite or only trade the single best global opportunity.

Usage:
  python run.py                  # single V6 scan cycle (all 15M series)
  python run.py --loop           # continuous multi-asset intelligence loop
  python run.py --loop -n 30     # 30 iterations then exit
  python run.py --series KXETH15M,KXSOL15M   # limit to specific series
  python run.py --discover       # auto-discover all Kalshi 15M series
  python run.py --execute        # execute every qualifying trade per cycle
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
from kalshi_bot.data.brti import resolve_series_spot, resolve_spot
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.data.markets_15m import (
    Series15mSpec,
    is_up_down_15m,
    resolve_enabled_series,
)
from kalshi_bot.data.realized_vol import estimate_realized_vol
from kalshi_bot.execution.executor import Executor
from kalshi_bot.execution.position_monitor import PositionMonitor
from kalshi_bot.execution.risk import RiskManager
from kalshi_bot.models.probability import options_implied_prob_up
from kalshi_bot.models.smile import VolSmile
from kalshi_bot.strategy.mispricing import Mispricing, Side
from kalshi_bot.strategy.v6_upgrades import V6Decision, V6IntelligenceEngine
from kalshi_bot.utils.logging import setup_logging

logger = logging.getLogger(__name__)
console = Console()

WORKFLOW_NAME = "Kalshi 15-Min Intelligence"
WORKFLOW_VERSION = "V6-multi"


@dataclass
class AssetContext:
    spec: Series15mSpec
    spot: float
    spot_source: str
    spot_official: bool
    vol_ann: float
    vol_source: str


@dataclass
class V6TradeCandidate:
    ticker: str
    series: str
    asset: str
    decision: V6Decision
    spot: float
    open_level: float | None
    movement_pct: float | None


@dataclass
class V6ScanResult:
    assets: list[AssetContext]
    markets_scanned: int
    trades: list[V6TradeCandidate]
    no_trades: list[V6TradeCandidate]
    asof: datetime
    top_blockers: list[tuple[str, int]] = field(default_factory=list)


def movement_vs_open(spot: float, open_level: float | None) -> float | None:
    if open_level is None or open_level <= 0:
        return None
    return (spot - open_level) / open_level * 100.0


class V6Scanner:
    """Scan all enabled 15M series through the V6 intelligence engine."""

    def __init__(
        self,
        client: KalshiClient,
        config: BotConfig,
        engine: V6IntelligenceEngine,
        series_specs: list[Series15mSpec],
    ) -> None:
        self.client = client
        self.config = config
        self.engine = engine
        self.v6 = config.v6
        self.series_specs = series_specs

    def _asset_context(self, spec: Series15mSpec, smile: VolSmile | None) -> AssetContext:
        fallback = smile.spot_btc if smile and spec.asset == "BTC" else None
        snap = resolve_series_spot(
            self.client,
            spec,
            fallback=fallback,
            brti_cfg=self.config.brti,
        )
        self.engine.update_spot(snap.brti, series_ticker=spec.ticker)
        kraken_pair = spec.kraken_pair or "XBTUSD"
        rv = estimate_realized_vol(
            horizon_seconds=self.v6.max_seconds_to_expiry,
            kraken_pair=kraken_pair,
        )
        vol = rv.annualized_vol
        if smile is not None and spec.asset == "BTC":
            vol = 0.6 * vol + 0.4 * smile.atm_iv
        return AssetContext(
            spec=spec,
            spot=snap.brti,
            spot_source=snap.source,
            spot_official=snap.is_official,
            vol_ann=vol,
            vol_source=rv.source,
        )

    def scan(self, smile: VolSmile | None = None) -> V6ScanResult:
        now = datetime.now(timezone.utc)
        trades: list[V6TradeCandidate] = []
        no_trades: list[V6TradeCandidate] = []
        scanned = 0
        blocker_counts: Counter[str] = Counter()
        asset_contexts: list[AssetContext] = []

        for spec in self.series_specs:
            if not is_up_down_15m(spec.ticker):
                logger.debug("skip non up/down series %s", spec.ticker)
                continue
            try:
                ctx = self._asset_context(spec, smile)
            except Exception as exc:
                logger.warning("spot/vol unavailable for %s: %s", spec.ticker, exc)
                continue
            asset_contexts.append(ctx)

            for raw in self.client.iter_markets(spec.ticker, status="open"):
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
                strike = market.get("strike")

                options_prob = None
                if smile is not None and spec.asset == "BTC" and strike is not None:
                    try:
                        options_prob = options_implied_prob_up(
                            spot_btc=ctx.spot,
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
                    spot=ctx.spot,
                    vol=ctx.vol_ann,
                    options_prob=options_prob,
                    now=now,
                    spot_source=ctx.spot_source,
                    spot_is_official=ctx.spot_official,
                )
                candidate = V6TradeCandidate(
                    ticker=ticker,
                    series=spec.ticker,
                    asset=spec.asset,
                    decision=decision,
                    spot=ctx.spot,
                    open_level=float(strike) if strike is not None else None,
                    movement_pct=movement_vs_open(ctx.spot, float(strike) if strike else None),
                )
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
        logger.info(
            "%s %s scan: series=%d markets=%d trades=%d no_trade=%d",
            WORKFLOW_NAME,
            WORKFLOW_VERSION,
            len(asset_contexts),
            scanned,
            len(trades),
            len(no_trades),
        )
        return V6ScanResult(
            assets=asset_contexts,
            markets_scanned=scanned,
            trades=trades,
            no_trades=no_trades,
            asof=now,
            top_blockers=blocker_counts.most_common(8),
        )


def v6_to_mispricing(candidate: V6TradeCandidate) -> Mispricing | None:
    decision = candidate.decision
    if decision.verdict == "NO_TRADE" or decision.market_price is None:
        return None
    side = Side.YES if decision.verdict == "TRADE_YES" else Side.NO
    prob = decision.model_probability if side == Side.YES else 1.0 - decision.model_probability
    gap_pp = decision.strict_gap_dollars * 100
    return Mispricing(
        ticker=candidate.ticker,
        series=candidate.series,
        side=side,
        kalshi_price=decision.market_price,
        options_prob=prob,
        edge_pp=gap_pp,
        edge_after_fees_pp=gap_pp,
        strike=candidate.open_level or 0.0,
        spot=candidate.spot,
        vol=0.0,
        seconds_to_expiry=0.0,
        yes_bid=None,
        yes_ask=decision.market_price if side == Side.YES else None,
        implied=None,  # type: ignore[arg-type]
        reason="; ".join(decision.reasons),
    )


def _print_header(result: V6ScanResult) -> None:
    lines = [f"[bold]{WORKFLOW_NAME}[/bold]  [cyan]{WORKFLOW_VERSION}[/cyan]"]
    for ctx in result.assets:
        official = "official RTI" if ctx.spot_official else "PROXY"
        lines.append(
            f"{ctx.spec.asset}: {ctx.spot:,.4f} ({ctx.spot_source}; {official}) σ={ctx.vol_ann*100:.1f}%"
        )
    lines.append(f"scanned={result.markets_scanned}  trades={len(result.trades)}")
    console.print(Panel("\n".join(lines), border_style="blue"))


def _print_opportunity_monitor(candidates: list[V6TradeCandidate]) -> None:
    if not candidates:
        return
    table = Table(title="Opportunity Monitor (each market — YES & NO evaluated independently)")
    table.add_column("Asset")
    table.add_column("Ticker")
    table.add_column("t_rem")
    table.add_column("Move%")
    table.add_column("Spot")
    table.add_column("Open")
    table.add_column("Model↑")
    table.add_column("YES¢")
    table.add_column("YES net")
    table.add_column("NO¢")
    table.add_column("NO net")
    table.add_column("Action")
    table.add_column("Decision")
    for c in candidates:
        d = c.decision
        audit = d.audit_record
        if not audit:
            continue
        move = f"{c.movement_pct:+.3f}%" if c.movement_pct is not None else "—"
        open_s = f"{c.open_level:,.2f}" if c.open_level is not None else "—"
        table.add_row(
            c.asset,
            c.ticker[-22:],
            f"{audit.minutes_to_expiry:.1f}m",
            move,
            f"{c.spot:,.2f}",
            open_s,
            f"{audit.model_prob_up*100:.0f}%",
            f"{(audit.yes_ask or 0)*100:.0f}",
            f"{audit.yes_side.net_edge_dollars*100:+.0f}",
            f"{(audit.no_ask or 0)*100:.0f}",
            f"{audit.no_side.net_edge_dollars*100:+.0f}",
            audit.edge_action[:28],
            audit.verdict,
        )
    console.print(table)


def _print_trades(candidates: list[V6TradeCandidate], top: int) -> None:
    if not candidates:
        return
    table = Table(title="V6 TRADE candidates (independent per-market decisions)")
    table.add_column("Asset")
    table.add_column("Verdict")
    table.add_column("Ticker")
    table.add_column("Move%")
    table.add_column("Model")
    table.add_column("Market")
    table.add_column("Gap ¢")
    table.add_column("Contracts")
    for c in candidates[:top]:
        d = c.decision
        move = f"{c.movement_pct:+.2f}%" if c.movement_pct is not None else "—"
        table.add_row(
            c.asset,
            d.verdict,
            c.ticker[-22:],
            move,
            f"{d.model_probability*100:.1f}%",
            f"{(d.market_price or 0)*100:.0f}¢",
            f"{d.strict_gap_dollars*100:.0f}",
            str(d.contracts),
        )
    console.print(table)


def _print_verdict(result: V6ScanResult) -> None:
    if result.trades:
        console.print(f"\n[green bold]ENGINE VERDICT: {len(result.trades)} trade(s)[/green bold]")
        for c in result.trades[:5]:
            d = c.decision
            console.print(
                f"  {c.asset} {d.verdict} {c.ticker} @ {(d.market_price or 0)*100:.0f}¢ "
                f"gap={d.strict_gap_dollars*100:.0f}¢ move={c.movement_pct:+.3f}%"
                if c.movement_pct is not None
                else f"  {c.asset} {d.verdict} {c.ticker}"
            )
    else:
        console.print("\n[yellow bold]ENGINE VERDICT: NO TRADE[/yellow bold]")
        console.print("No market cleared strict edge + quality gates this cycle.")


def resolve_scan_series(
    v6: V6Config,
    *,
    cli_series: list[str] | None = None,
    discover: bool = False,
) -> list[Series15mSpec]:
    configured = cli_series or (v6.series_tickers if not discover else None)
    return resolve_enabled_series(
        configured=configured,
        auto_discover=discover or v6.auto_discover_15m,
        include_commodities=v6.include_commodity_15m,
        include_head_to_head=v6.include_head_to_head_15m,
    )


def build_v6_runtime(
    settings: Settings | None = None,
    *,
    strict_gap: float | None = None,
    series_specs: list[Series15mSpec] | None = None,
) -> tuple[
    Settings,
    BotConfig,
    KalshiClient,
    V6IntelligenceEngine,
    V6Scanner,
    Executor,
    list[Series15mSpec],
]:
    settings = settings or Settings()
    config = load_config(settings.config_path)
    if strict_gap is not None:
        config.v6.strict_min_gap_dollars = strict_gap
    specs = series_specs or resolve_scan_series(config.v6)
    setup_logging(settings.log_level)
    client = KalshiClient(
        base_url=kalshi_base_url(settings, config),
        api_key_id=settings.kalshi_api_key_id,
        private_key_pem=settings.resolve_private_key_pem(),
    )
    engine = V6IntelligenceEngine(config.v6, client=client, arbitrary_cfg=config.arbitrary)
    scanner = V6Scanner(client, config, engine, specs)
    risk = RiskManager(config)
    executor = Executor(client, config, risk)
    return settings, config, client, engine, scanner, executor, specs


def execute_qualifying_trades(
    executor: Executor,
    config: BotConfig,
    trades: list[V6TradeCandidate],
    *,
    execute_all: bool,
) -> int:
    """Execute qualifying trades — all or best-only per config."""
    if not trades or not config.v6.live_trading_enabled:
        return 0
    to_run = trades if execute_all else trades[:1]
    executed = 0
    for c in to_run:
        mis = v6_to_mispricing(c)
        if mis is None:
            continue
        size = c.decision.contracts or executor.risk.size(mis)
        if size <= 0:
            logger.info("skip %s %s: size=0", c.decision.verdict, c.ticker)
            continue
        fill = executor.execute(mis, size, ignore_cooldown=True)
        if fill:
            executed += 1
            logger.info(
                "executed %s %s %s x%d mode=%s",
                c.asset,
                c.decision.verdict,
                c.ticker,
                size,
                fill.mode,
            )
    return executed


def run_v6_once(
    *,
    allow_synthetic_smile: bool = False,
    execute: bool = False,
    strict_gap: float | None = None,
    top: int = 20,
    series_specs: list[Series15mSpec] | None = None,
) -> V6ScanResult:
    settings, config, client, engine, scanner, executor, _specs = build_v6_runtime(
        strict_gap=strict_gap,
        series_specs=series_specs,
    )
    try:
        smile = None
        try:
            smile = load_ibit_smile(config.smile, allow_synthetic=allow_synthetic_smile)
        except Exception as exc:
            logger.warning("IBIT smile unavailable (%s); realized-vol only", exc)

        result = scanner.scan(smile)
        _print_header(result)
        _print_opportunity_monitor(result.trades + result.no_trades)
        _print_trades(result.trades, top)

        if result.top_blockers:
            console.print("Primary rejection frequency:")
            for name, count in result.top_blockers:
                console.print(f"  {count:4d}  {name}")

        _print_verdict(result)

        if execute:
            n = execute_qualifying_trades(
                executor,
                config,
                result.trades,
                execute_all=config.v6.execute_all_qualifying,
            )
            if n == 0 and result.trades:
                console.print("[yellow]Trades found but none executed (size/risk gates)[/yellow]")
        elif result.trades and not config.v6.live_trading_enabled:
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
    series_specs: list[Series15mSpec] | None = None,
) -> None:
    settings, config, client, engine, scanner, executor, _specs = build_v6_runtime(
        strict_gap=strict_gap,
        series_specs=series_specs,
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
                if result.trades and config.v6.live_trading_enabled and execute:
                    n = execute_qualifying_trades(
                        executor,
                        config,
                        result.trades,
                        execute_all=config.v6.execute_all_qualifying,
                    )
                    console.print(f"[green]Executed {n}/{len(result.trades)} qualifying trade(s)[/green]")
                elif result.trades:
                    best = result.trades[0]
                    console.print(
                        f"[green]{best.asset} {best.decision.verdict}[/green] "
                        f"{best.ticker} gap={best.decision.strict_gap_dollars*100:.0f}¢"
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
    series_specs: list[Series15mSpec] | None = None,
) -> None:
    settings, config, client, engine, scanner, _executor, _specs = build_v6_runtime(
        series_specs=series_specs,
    )
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
                collected = monitor.rejection_breakdown().total_evaluations
                logger.info(
                    "collect iter=%d markets=%d total=%d/%d",
                    iterations,
                    result.markets_scanned,
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


def print_diagnostic_report() -> None:
    from kalshi_bot.strategy.opportunity_monitor import OpportunityMonitor

    config = load_config()
    monitor = OpportunityMonitor(config.v6.diagnostics_db_path)
    breakdown = monitor.rejection_breakdown()
    console.print(Panel(breakdown.summary_text(), title="Rejection Breakdown"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"{WORKFLOW_NAME} {WORKFLOW_VERSION}")
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
    parser.add_argument("--top", type=int, default=20, help="Rows to display")
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Comma-separated 15M series tickers (e.g. KXETH15M,KXSOL15M)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Auto-discover all Kalshi 15M series (crypto + commodities)",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Execute only the single best trade per cycle (legacy behavior)",
    )
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

    config = load_config()
    if args.best_only:
        config.v6.execute_all_qualifying = False

    cli_series = [s.strip().upper() for s in args.series.split(",")] if args.series else None
    series_specs = resolve_scan_series(
        config.v6,
        cli_series=cli_series,
        discover=args.discover,
    )
    if not series_specs:
        console.print("[red]No 15M series configured — check --series or config v6.series_tickers[/red]")
        sys.exit(1)

    if args.collect:
        collect_diagnostics(target=args.collect, interval=args.interval or 3.0, series_specs=series_specs)
        return

    if not config.v6.enabled:
        console.print("[red]V6 workflow disabled in config (v6.enabled=false)[/red]")
        sys.exit(1)

    console.print(
        f"Scanning {len(series_specs)} series: "
        + ", ".join(s.asset for s in series_specs)
    )

    if args.loop:
        run_v6_loop(
            max_iterations=args.iterations,
            interval=args.interval,
            execute=args.execute,
            strict_gap=args.strict,
            series_specs=series_specs,
        )
    else:
        run_v6_once(
            allow_synthetic_smile=args.allow_synthetic,
            execute=args.execute,
            strict_gap=args.strict,
            top=args.top,
            series_specs=series_specs,
        )


if __name__ == "__main__":
    main()
