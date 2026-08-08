"""CLI entrypoints for forecasting, scanning, and demos."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.table import Table

from kalshi_bot.config import ForecastGateConfig, SeriesConfig, SmileConfig, load_config
from kalshi_bot.data.ibit_options import load_ibit_smile
from kalshi_bot.main import build_runtime, run_once
from kalshi_bot.models.forecast import ForecastAction
from kalshi_bot.models.probability import options_implied_prob_above
from kalshi_bot.models.smile import synthetic_smile
from kalshi_bot.strategy.decision import DecisionVerdict, evaluate_forecast_market
from kalshi_bot.strategy.fees import quadratic_fee_per_contract
from kalshi_bot.strategy.mispricing import evaluate_market
from kalshi_bot.strategy.scanner import ForecastScanner
from kalshi_bot.utils.logging import setup_logging

console = Console()


def demo_main(argv: list[str] | None = None) -> None:
    """Reproduce the canonical edge example and a live synthetic scan."""
    parser = argparse.ArgumentParser(description="Demo IBIT→BTC digital edge")
    parser.add_argument("--spot", type=float, default=65000.0)
    parser.add_argument("--strike", type=float, default=65200.0)
    parser.add_argument("--kalshi-ask", type=float, default=0.22)
    parser.add_argument("--minutes", type=float, default=55.0)
    parser.add_argument("--atm-iv", type=float, default=1.0)
    args = parser.parse_args(argv)

    setup_logging("INFO")
    smile = synthetic_smile(args.spot, atm_iv=args.atm_iv)
    close = datetime.now(timezone.utc) + timedelta(minutes=args.minutes)
    implied = options_implied_prob_above(
        spot_btc=args.spot,
        strike_btc=args.strike,
        close_time=close,
        smile=smile,
    )
    p = implied.probability
    fee = quadratic_fee_per_contract(args.kalshi_ask)
    edge_pp = (p - args.kalshi_ask) * 100
    edge_after = (p - args.kalshi_ask - fee) * 100

    console.print("[bold]IBIT smile → BTC digital vs Kalshi[/bold]")
    console.print(f"Spot (BRTI proxy): {args.spot:,.2f}")
    console.print(f"Strike:            {args.strike:,.2f}")
    console.print(f"ATM IV:            {args.atm_iv*100:.1f}%  | smile IV@K: {implied.vol_used*100:.1f}%")
    console.print(f"T:                 {args.minutes:.1f} minutes")
    console.print(f"Options prob:      {p*100:.1f}%")
    console.print(f"Kalshi YES ask:    {args.kalshi_ask*100:.1f}%")
    console.print(f"Raw edge:          {edge_pp:.1f} pp")
    console.print(f"Fee @ask:          ${fee:.2f}/contract")
    console.print(f"Edge after fees:   {edge_after:.1f} pp")
    if edge_after >= 8:
        console.print("[green]TAKE TRADE: buy YES[/green]")
    else:
        console.print("[yellow]No trade under default 8pp threshold[/yellow]")

    market = {
        "ticker": "DEMO-KXBTCD-T66000",
        "series_ticker": "KXBTCD",
        "strike": args.strike,
        "close_time": close,
        "yes_ask": 0.22,
        "yes_bid": 0.20,
        "no_ask": 0.80,
        "strike_type": "greater",
    }
    target = 0.378
    best_iv = args.atm_iv
    best_diff = 1.0
    for iv in [i / 100 for i in range(30, 201)]:
        s = synthetic_smile(args.spot, atm_iv=iv)
        pr = options_implied_prob_above(
            spot_btc=args.spot, strike_btc=args.strike, close_time=close, smile=s
        ).probability
        if abs(pr - target) < best_diff:
            best_diff = abs(pr - target)
            best_iv = iv
    smile2 = synthetic_smile(args.spot, atm_iv=best_iv)
    smile2.is_synthetic = False
    mis = evaluate_market(
        market,
        spot=args.spot,
        smile=smile2,
        series_cfg=SeriesConfig(ticker="KXBTCD", min_edge_pp=8.0),
        smile_cfg=SmileConfig(),
    )
    if mis:
        console.print(
            f"\nCanonical-style signal: Kalshi {mis.kalshi_price*100:.1f}% vs "
            f"options {mis.options_prob*100:.1f}% → {mis.edge_pp:.1f}pp "
            f"({mis.side.value.upper()})"
        )


def scan_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan Kalshi BTC markets once (legacy mispricing)")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)

    settings, config, client, scanner, _executor = build_runtime()
    try:
        smile = load_ibit_smile(config.smile, allow_synthetic=True)
        result = scanner.scan(smile)
        table = Table(title=f"Mispricings  spot={result.spot.brti:,.2f} ({result.spot.source})")
        table.add_column("Side")
        table.add_column("Ticker")
        table.add_column("Kalshi")
        table.add_column("Options")
        table.add_column("Edge pp")
        table.add_column("Strike")
        table.add_column("σ")
        table.add_column("t(s)")
        for mis in result.opportunities[: args.top]:
            table.add_row(
                mis.side.value.upper(),
                mis.ticker,
                f"{mis.kalshi_price*100:.1f}%",
                f"{mis.options_prob*100:.1f}%",
                f"{mis.edge_after_fees_pp:.1f}",
                f"{mis.strike:,.2f}",
                f"{mis.vol*100:.1f}%",
                f"{mis.seconds_to_expiry:.0f}",
            )
        console.print(table)
        if not result.opportunities:
            console.print("No edges above threshold on this scan.")
    finally:
        client.close()


def forecast_main(argv: list[str] | None = None) -> None:
    """Institutional 1-hour forecast scan. Default output: NO TRADE."""
    parser = argparse.ArgumentParser(description="Kalshi BTC 1-hour forecasting engine")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)

    settings, config, client, _scanner, _executor = build_runtime()
    forecast_scanner = ForecastScanner(client, config)
    try:
        try:
            smile = load_ibit_smile(config.smile, allow_synthetic=args.allow_synthetic)
        except Exception as exc:
            console.print(f"[yellow]Smile unavailable ({exc}); realized-vol components only[/yellow]")
            smile = None

        result = forecast_scanner.scan(smile)
        official = "official BRTI" if result.spot.is_official else "PROXY — not settlement index"
        console.print(
            f"[bold]KXBTCD 1h forecast[/bold]  spot={result.spot.brti:,.2f} "
            f"({result.spot.source}; {official})  scanned={result.markets_scanned}"
        )

        if result.trades:
            table = Table(title="TRADE candidates (passed all evidence gates)")
            table.add_column("Action")
            table.add_column("Ticker")
            table.add_column("Kalshi")
            table.add_column("Forecast")
            table.add_column("EV $/ctr")
            table.add_column("Edge pp")
            table.add_column("Conf")
            table.add_column("Δpp")
            table.add_column("t(s)")
            for d in result.trades[: args.top]:
                table.add_row(
                    d.action.value,
                    d.ticker,
                    f"{(d.kalshi_price or 0)*100:.1f}%",
                    f"{d.forecast_prob*100:.1f}%",
                    f"{d.expected_value_per_contract:.3f}",
                    f"{d.edge_after_fees_pp:.1f}",
                    f"{d.confidence:.2f}",
                    f"{d.disagreement_pp:.1f}",
                    f"{d.seconds_to_expiry:.0f}",
                )
            console.print(table)
        else:
            console.print("[bold yellow]NO TRADE[/bold yellow] — no market cleared evidence gates.")

        # Near-miss diagnostics
        near = [d for d in result.no_trades if d.edge_after_fees_pp > 0][: args.top]
        if near:
            table = Table(title="Near misses (NO TRADE)")
            table.add_column("Ticker")
            table.add_column("Side")
            table.add_column("Kalshi")
            table.add_column("Forecast")
            table.add_column("Edge pp")
            table.add_column("Conf")
            table.add_column("Top blocker")
            for d in near:
                table.add_row(
                    d.ticker,
                    d.side.value.upper() if d.side else "-",
                    f"{(d.kalshi_price or 0)*100:.1f}%",
                    f"{d.forecast_prob*100:.1f}%",
                    f"{d.edge_after_fees_pp:.1f}",
                    f"{d.confidence:.2f}",
                    (d.blockers[0] if d.blockers else "")[:60],
                )
            console.print(table)

        if result.top_blockers:
            console.print("Blocker frequency:")
            for name, count in result.top_blockers:
                console.print(f"  {count:4d}  {name}")

        # Explicit engine verdict
        if result.trades:
            best = result.trades[0]
            console.print(
                f"\n[green]ENGINE VERDICT: {best.action.value}[/green] {best.ticker} "
                f"@ {best.kalshi_price:.2f} | EV ${best.expected_value_per_contract:.3f}/ctr"
            )
            for r in best.reasons:
                console.print(f"  · {r}")
        else:
            console.print("\n[yellow]ENGINE VERDICT: NO TRADE[/yellow]")
            console.print("Accuracy > frequency. Insufficient statistically favorable edge.")
    finally:
        client.close()


if __name__ == "__main__":
    forecast_main()
