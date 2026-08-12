"""Rich dashboard for 15-minute mispricing opportunities."""

from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kalshi_bot.data.btc_data_engine import BtcMarketSnapshot
from kalshi_bot.strategy.mispricing_engine import MispricingOpportunity, TradeAction
from kalshi_bot.strategy.trade_filter import TradeDecision

console = Console()


def _format_time_remaining(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def print_mispricing_dashboard(
    *,
    series: str,
    btc: BtcMarketSnapshot,
    opp: MispricingOpportunity,
    decision: TradeDecision,
    balance_usd: float | None = None,
) -> None:
    """Single-market dashboard matching the spec format."""
    lines = [
        f"[bold cyan]{series}[/bold cyan]",
        f"BTC: [green]${btc.reference_price:,.2f}[/green]  ({btc.reference_source})",
        f"Strike: [yellow]${opp.strike:,.2f}[/yellow]",
        f"Time remaining: [bold]{_format_time_remaining(opp.seconds_to_expiry)}[/bold]  [{decision.time_bucket}]",
        "",
        f"Model YES: [bold]{opp.model_yes_pct:.1f}%[/bold]",
        f"Kalshi YES ask: {(opp.yes.executable_ask or 0)*100:.0f}¢",
        f"Fair value: {opp.fair_value_yes*100:.0f}¢",
    ]
    best = opp.best_mispricing()
    if best:
        lines.append(f"Net edge: [bold]{best.net_edge_dollars*100:.1f}¢[/bold]")
    lines.extend([
        f"Confidence: {opp.confidence_label}",
        f"Volatility: {opp.volatility_label}",
        f"Order flow: {opp.order_flow_label}",
        f"Liquidity: {opp.liquidity_label}",
    ])
    if balance_usd is not None:
        lines.append(f"Balance: ${balance_usd:.2f}")

    if decision.action in (TradeAction.BUY_YES, TradeAction.BUY_NO):
        border = "green"
        lines.append("")
        lines.append(f"[green bold]{decision.action.value}[/green bold] x{decision.contracts}")
    elif decision.action == TradeAction.WAIT:
        border = "yellow"
        lines.append("")
        lines.append(f"[yellow bold]WAIT[/yellow bold] — {decision.reason}")
    else:
        border = "red"
        lines.append("")
        lines.append(f"[red bold]NO TRADE[/red bold] — {decision.reason}")

    console.print(Panel("\n".join(lines), border_style=border))


def print_scan_summary(
    *,
    asof: datetime,
    markets_scanned: int,
    trades: int,
    btc: BtcMarketSnapshot,
    balance_usd: float | None = None,
) -> None:
    official = "official" if btc.is_official else "proxy"
    stale = " [red]STALE[/red]" if btc.stale else ""
    bal = f"  balance=${balance_usd:.2f}" if balance_usd is not None else ""
    console.print(
        Panel(
            f"[bold]KXBTC15M Mispricing Scanner[/bold]\n"
            f"BTC ${btc.reference_price:,.2f} ({official}{stale})  "
            f"σ={btc.annualized_vol*100:.0f}%  "
            f"feeds={len(btc.feeds)}  "
            f"scanned={markets_scanned}  trades={trades}{bal}",
            border_style="blue",
        )
    )


def print_opportunities_table(rows: list[tuple[str, MispricingOpportunity, TradeDecision]]) -> None:
    if not rows:
        return
    table = Table(title="KXBTC15M Opportunities")
    table.add_column("Ticker")
    table.add_column("T-rem")
    table.add_column("Model")
    table.add_column("YES¢")
    table.add_column("Net¢")
    table.add_column("Conf")
    table.add_column("Bucket")
    table.add_column("Decision")
    for ticker, opp, dec in rows:
        table.add_row(
            ticker[-18:],
            _format_time_remaining(opp.seconds_to_expiry),
            f"{opp.model_yes_pct:.0f}%",
            f"{(opp.yes.executable_ask or 0)*100:.0f}",
            f"{opp.best_net_edge*100:+.1f}",
            opp.confidence_label,
            dec.time_bucket,
            dec.action.value,
        )
    console.print(table)
