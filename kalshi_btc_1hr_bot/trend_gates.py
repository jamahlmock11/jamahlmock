"""Trend alignment and order-flow confirmation gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kalshi_btc_1hr_bot.config import BotConfig
from kalshi_btc_1hr_bot.data_feed import MarketData
from kalshi_btc_1hr_bot.edge import TradeSignal
from kalshi_btc_1hr_bot.evidence import DirectionalEvidence


@dataclass(frozen=True)
class FlowSnapshot:
    yes_volume: int = 0
    no_volume: int = 0
    trade_count: int = 0
    net_side: str = ""

    @property
    def net_volume(self) -> int:
        return abs(self.yes_volume - self.no_volume)


def blended_momentum(data: MarketData) -> float:
    return 0.5 * data.mu_5m + 0.3 * data.mu_15m + 0.2 * data.mu_30m


def check_trend_alignment(
    *,
    side: str,
    spot: float,
    strike: float,
    data: MarketData,
    min_momentum: float,
) -> tuple[bool, str]:
    """Trend-following: momentum and spot vs strike must agree with trade side."""
    mu = blended_momentum(data)
    side = side.lower()
    if side == "yes":
        momentum_ok = mu > min_momentum
        position_ok = spot >= strike
        detail = f"mom {mu:+.5f} (need >{min_momentum:.5f}), spot ${spot:,.0f} vs strike ${strike:,.0f}"
        ok = momentum_ok and position_ok
        if not ok:
            parts = []
            if not momentum_ok:
                parts.append("momentum not up")
            if not position_ok:
                parts.append("spot below strike")
            return False, f"Trend ABOVE: {', '.join(parts)} · {detail}"
        return True, f"Trend ABOVE aligned · {detail}"
    momentum_ok = mu < -min_momentum
    position_ok = spot <= strike
    detail = f"mom {mu:+.5f} (need <{-min_momentum:.5f}), spot ${spot:,.0f} vs strike ${strike:,.0f}"
    ok = momentum_ok and position_ok
    if not ok:
        parts = []
        if not momentum_ok:
            parts.append("momentum not down")
        if not position_ok:
            parts.append("spot above strike")
        return False, f"Trend BELOW: {', '.join(parts)} · {detail}"
    return True, f"Trend BELOW aligned · {detail}"


def check_flow_confirmation(*, side: str, flow: FlowSnapshot | None) -> tuple[bool, str]:
    if flow is None or flow.trade_count == 0:
        return False, "No recent order flow prints"
    side = side.lower()
    if side == "yes":
        ok = flow.yes_volume > flow.no_volume
        net = "YES" if ok else "NO"
    else:
        ok = flow.no_volume > flow.yes_volume
        net = "NO" if ok else "YES"
    detail = f"flow YES {flow.yes_volume} / NO {flow.no_volume} (net {net})"
    if ok:
        return True, f"Flow confirms {side.upper()} · {detail}"
    return False, f"Flow against {side.upper()} · {detail}"


def fetch_flow_snapshot(client: Any, ticker: str, *, limit: int = 80) -> FlowSnapshot:
    """Pull recent public trades for flow confirmation."""
    try:
        raw = client.get_trades(ticker, limit=limit)
    except Exception:
        return FlowSnapshot()
    trades = raw.get("trades") or []
    yes_vol = 0
    no_vol = 0
    for t in trades:
        count = int(float(t.get("count_fp") or t.get("count") or 0))
        if count <= 0:
            continue
        taker = str(t.get("taker_side") or "").lower()
        if taker == "yes":
            yes_vol += count
        elif taker == "no":
            no_vol += count
    net = ""
    if yes_vol > no_vol:
        net = "yes"
    elif no_vol > yes_vol:
        net = "no"
    return FlowSnapshot(
        yes_volume=yes_vol,
        no_volume=no_vol,
        trade_count=len(trades),
        net_side=net,
    )


def apply_confirmation_gates(
    edge: TradeSignal,
    direction: DirectionalEvidence,
    *,
    data: MarketData,
    strike: float,
    flow: FlowSnapshot | None,
    cfg: BotConfig,
) -> tuple[TradeSignal, bool, bool, str, str]:
    """Apply trend + flow gates on top of existing edge signal. Returns edge, trend_ok, flow_ok, trend_detail, flow_detail."""
    trend_ok = True
    flow_ok = True
    trend_detail = "trend gate off"
    flow_detail = "flow gate off"

    if not edge.should_trade:
        return edge, trend_ok, flow_ok, trend_detail, flow_detail

    if cfg.gates.trend_gate_enabled:
        trend_ok, trend_detail = check_trend_alignment(
            side=direction.side,
            spot=data.spot,
            strike=strike,
            data=data,
            min_momentum=cfg.gates.trend_min_momentum,
        )
        if not trend_ok:
            return (
                TradeSignal(
                    False,
                    edge.side,
                    edge.p_fair,
                    edge.market_price,
                    edge.edge_cents,
                    edge.ev_per_contract,
                    trend_detail,
                ),
                trend_ok,
                flow_ok,
                trend_detail,
                flow_detail,
            )

    if cfg.gates.flow_confirm_enabled:
        flow_ok, flow_detail = check_flow_confirmation(side=direction.side, flow=flow)
        if not flow_ok:
            return (
                TradeSignal(
                    False,
                    edge.side,
                    edge.p_fair,
                    edge.market_price,
                    edge.edge_cents,
                    edge.ev_per_contract,
                    flow_detail,
                ),
                trend_ok,
                flow_ok,
                trend_detail,
                flow_detail,
            )

    return edge, trend_ok, flow_ok, trend_detail, flow_detail
