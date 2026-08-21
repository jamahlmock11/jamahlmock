"""Shared dashboard state — file-backed for bot + web server processes."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_btc_1hr_bot import config
from kalshi_btc_1hr_bot.config import BotConfig, ROOT
from kalshi_btc_1hr_bot.dynamic_gates import DynamicThresholds, resolve_dynamic_thresholds
from kalshi_btc_1hr_bot.evidence import MarketCandidate
from kalshi_btc_1hr_bot.forecast import agreement_score_for_gates
from kalshi_btc_1hr_bot.risk import RiskManager

STATE_PATH = ROOT / "data" / "dashboard_state.json"


@dataclass
class CheckItem:
    label: str
    passed: bool
    detail: str
    category: str = "gate"


@dataclass
class DashboardSnapshot:
    updated_at: str = ""
    mode: str = "PAPER"
    env: str = "prod"
    balance_usd: float | None = None
    spot: float = 0.0
    brti_source: str = ""
    brti_official: bool = False
    annualized_vol: float = 0.0
    funding_rate: float = 0.0
    markets_scanned: int = 0
    candidates: int = 0
    cycle_status: str = "WAITING"
    action_light: str = "red"  # green | yellow | red
    action_headline: str = "WAITING FOR SCAN"
    action_detail: str = ""
    readiness_pct: float = 0.0
    checklist: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    best_pick: dict[str, Any] | None = None
    top_markets: list[dict[str, Any]] = field(default_factory=list)
    model_votes: list[dict[str, Any]] = field(default_factory=list)
    risk: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    config_summary: dict[str, Any] = field(default_factory=dict)
    recent_settlements: list[dict[str, Any]] = field(default_factory=list)
    crowd: dict[str, Any] = field(default_factory=dict)
    entry_context: dict[str, Any] | None = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_row(cand: MarketCandidate, *, rank: int, is_pick: bool, action: str, reason: str) -> dict[str, Any]:
    market = cand.market
    return {
        "rank": rank,
        "ticker": cand.ticker,
        "strike": cand.strike,
        "secs_left": round(cand.secs_left),
        "finish": cand.direction.finish_label,
        "side": cand.direction.side,
        "p_fair": round(cand.forecast.p_fair, 4),
        "confidence": round(cand.forecast.confidence, 3),
        "agreement": round(cand.forecast.agreement_score, 3),
        "regime": cand.forecast.vol_regime,
        "edge_cents": round(cand.edge.edge_cents, 2),
        "edge_usd": round(cand.edge.edge_cents / 100.0, 4),
        "evidence_score": round(cand.evidence_score, 4),
        "evidence_margin": round(cand.direction.margin, 4),
        "evidence_above": round(cand.direction.above_score, 4),
        "evidence_below": round(cand.direction.below_score, 4),
        "yes_ask": market.get("yes_ask"),
        "no_ask": market.get("no_ask"),
        "yes_bid": market.get("yes_bid"),
        "no_bid": market.get("no_bid"),
        "price": cand.edge.market_price,
        "yes_price_cents": round((market.get("yes_ask") or 0) * 100),
        "no_price_cents": round((market.get("no_ask") or 0) * 100),
        "buy_price_cents": round(cand.edge.market_price * 100),
        "should_trade": cand.edge.should_trade,
        "is_pick": is_pick,
        "action": action,
        "reason": reason,
        "layers": cand.forecast.layers,
        "votes": [(v.name, round(v.prob_yes, 4), round(v.weight, 3)) for v in cand.direction.top_votes],
    }


def build_checklist(
    *,
    data_ok: bool,
    brti_official: bool,
    markets_scanned: int,
    best: MarketCandidate | None,
    focus: MarketCandidate | None = None,
    is_pick: bool,
    allowed: bool,
    block_reason: str,
    contracts: int,
    cfg: BotConfig,
    thresholds: DynamicThresholds | None = None,
) -> list[CheckItem]:
    items: list[CheckItem] = [
        CheckItem("BRTI spot feed live", data_ok, "CF Benchmarks / fallback", "data"),
        CheckItem(
            "Official BRTI source",
            brti_official,
            "Proxy penalty applied when unofficial",
            "data",
        ),
        CheckItem(
            f"Markets in window ({cfg.risk.min_seconds_to_expiry:.0f}s–{cfg.risk.max_seconds_to_expiry:.0f}s)",
            markets_scanned > 0,
            f"{markets_scanned} quoted hourly markets",
            "scan",
        ),
    ]
    gate_cand = best or focus
    if gate_cand is None:
        items.append(
            CheckItem(
                "Tradeable edge found",
                False,
                "No market cleared min edge + evidence margin",
                "edge",
            )
        )
        return items

    th = thresholds or gate_cand.thresholds
    if th is None:
        th = resolve_dynamic_thresholds(
            gate_cand.secs_left,
            vol_regime=gate_cand.forecast.vol_regime,
            agreement_score=agreement_score_for_gates(
                gate_cand.forecast, use_ensemble=cfg.gates.use_ensemble_agreement
            ),
            edge_cents=gate_cand.edge.edge_cents,
            crowd_side_prob=(
                gate_cand.forecast.crowd.side_prob(gate_cand.direction.side)
                if cfg.gates.crowd_gates_enabled
                else None
            ),
        )

    cf_rng = th.crowd_favorite_range
    ev_rng = th.evidence_margin_range
    edge_rng = th.min_edge_range
    ag_rng = th.agreement_range

    edge_label = "Edge" if not cfg.edge.subtract_fees_from_edge else "Net edge"
    edge_detail = (
        f"{gate_cand.edge.edge_cents:.1f}¢ gross (fees ignored for gates)"
        if not cfg.edge.subtract_fees_from_edge
        else f"{gate_cand.edge.edge_cents:.1f}¢ after {cfg.edge.fee_per_contract_cents:.2f}¢ fee"
    )
    agree_score = agreement_score_for_gates(
        gate_cand.forecast, use_ensemble=cfg.gates.use_ensemble_agreement
    )
    agree_label = "Ensemble agreement" if cfg.gates.use_ensemble_agreement else "Agreement"
    gate_items: list[CheckItem] = [
        CheckItem(
            f"Evidence margin ≥ {th.min_evidence_margin:.3f} ({ev_rng[0]:.3f}–{ev_rng[1]:.3f})",
            gate_cand.direction.margin >= th.min_evidence_margin,
            f"{gate_cand.direction.margin:.3f} ({gate_cand.direction.finish_label}) · {th.bucket_label}",
            "evidence",
        ),
        CheckItem(
            f"{edge_label} ≥ {th.min_edge_cents:.1f}¢ ({edge_rng[0]:.1f}–{edge_rng[1]:.1f}¢)",
            gate_cand.edge.should_trade or gate_cand.edge.edge_cents >= th.min_edge_cents,
            edge_detail,
            "edge",
        ),
        CheckItem(
            f"{agree_label} ≥ {th.min_agreement:.0%} ({ag_rng[0]:.0%}–{ag_rng[1]:.0%})",
            agree_score >= th.min_agreement,
            f"{agree_score:.0%}",
            "model",
        ),
    ]
    if cfg.gates.crowd_gates_enabled:
        gate_items[2:2] = [
            CheckItem(
                f"Crowd quorum ≥ {th.min_quorum}",
                gate_cand.forecast.crowd.quorum_count >= th.min_quorum,
                (
                    f"{gate_cand.forecast.crowd.quorum_count}/{th.min_quorum} "
                    f"on {gate_cand.forecast.crowd.consensus_side.upper()}"
                ),
                "crowd",
            ),
            CheckItem(
                f"Crowd ≥ {th.min_crowd_favorite_pct:.0f}% ({cf_rng[0]*100:.0f}–{cf_rng[1]*100:.0f}%)",
                gate_cand.forecast.crowd.side_met(
                    gate_cand.direction.side, min_favorite=th.min_crowd_favorite
                ),
                (
                    f"{gate_cand.direction.finish_label} "
                    f"{gate_cand.forecast.crowd.side_pct(gate_cand.direction.side):.1f}% "
                    f"(favorite {gate_cand.forecast.crowd.favorite_pct:.1f}%)"
                ),
                "crowd",
            ),
        ]
    items.extend(
        gate_items
        + [
            CheckItem(
                f"Top-{cfg.gates.kalshi_card_picks} Kalshi card pick",
                is_pick and gate_cand.edge.should_trade,
                "Selected as best among Kalshi card strikes" if is_pick else "Not the card winner",
                "selection",
            ),
            CheckItem("Risk gate", allowed, block_reason if not allowed else "ok", "risk"),
            CheckItem(
                f"Kelly sizing > 0 (max ${cfg.sizing.max_trade_usd:.2f})",
                contracts > 0,
                f"{contracts} contracts" if contracts > 0 else "Size capped to zero",
                "sizing",
            ),
        ]
    )
    return items


def _max_entry_price_cents(
    *,
    side: str,
    p_fair: float,
    min_edge_cents: float,
    subtract_fees: bool,
    fee_cents: float,
) -> float | None:
    """Max Kalshi ask (¢) on trade side that still clears min edge."""
    gate_fee = config.gate_fee_cents(fee_cents, subtract=subtract_fees)
    min_edge = min_edge_cents / 100.0
    if side == "yes":
        max_price = p_fair - min_edge - gate_fee / 100.0
    else:
        max_price = (1.0 - p_fair) - min_edge - gate_fee / 100.0
    if max_price <= 0 or max_price >= 1:
        return None
    return round(max_price * 100, 1)


def build_entry_context(
    *,
    focus: MarketCandidate,
    spot: float,
    thresholds: DynamicThresholds,
    cfg: BotConfig,
    risk: RiskManager,
    allowed: bool,
    block_reason: str,
    contracts: int,
    is_pick: bool,
) -> dict[str, Any]:
    """Structured Kalshi-vs-gates view for the dashboard Currently panel."""
    market = focus.market
    side = focus.direction.side
    finish = focus.direction.finish_label
    th = thresholds

    yes_bid = float(market.get("yes_bid") or market.get("yes_ask") or 0)
    yes_ask = float(market.get("yes_ask") or 0)
    no_bid = float(market.get("no_bid") or market.get("no_ask") or 0)
    no_ask = float(market.get("no_ask") or 0)
    yes_bid_c = round(yes_bid * 100)
    yes_ask_c = round(yes_ask * 100)
    no_bid_c = round(no_bid * 100)
    no_ask_c = round(no_ask * 100)

    trade_bid_c = yes_bid_c if side == "yes" else no_bid_c
    trade_ask_c = yes_ask_c if side == "yes" else no_ask_c
    fair_yes_pct = focus.forecast.p_fair * 100.0
    fair_side_pct = fair_yes_pct if side == "yes" else (1.0 - focus.forecast.p_fair) * 100.0
    crowd_side_pct = focus.forecast.crowd.side_pct(side)
    edge_need = th.min_edge_cents
    edge_have = focus.edge.edge_cents
    edge_delta = edge_need - edge_have

    max_entry = _max_entry_price_cents(
        side=side,
        p_fair=focus.forecast.p_fair,
        min_edge_cents=edge_need,
        subtract_fees=cfg.edge.subtract_fees_from_edge,
        fee_cents=cfg.edge.fee_per_contract_cents,
    )
    price_delta_c = (trade_ask_c - max_entry) if max_entry is not None else None

    spot_delta = spot - focus.strike
    if spot_delta >= 0:
        spot_label = f"${abs(spot_delta):,.0f} above strike"
    else:
        spot_label = f"${abs(spot_delta):,.0f} below strike"

    quorum_have = focus.forecast.crowd.quorum_count
    quorum_need = th.min_quorum
    crowd_need_pct = th.min_crowd_favorite_pct
    evidence_have = focus.direction.margin
    evidence_need = th.min_evidence_margin
    agree_have = (
        agreement_score_for_gates(focus.forecast, use_ensemble=cfg.gates.use_ensemble_agreement) * 100.0
    )
    agree_need = th.min_agreement * 100.0
    agree_label = "Ensemble agreement" if cfg.gates.use_ensemble_agreement else "Agreement"

    gates: list[dict[str, Any]] = []
    if cfg.gates.crowd_gates_enabled:
        gates.extend(
            [
                {
                    "key": "quorum",
                    "label": "Crowd quorum",
                    "current": f"{quorum_have}/{quorum_need}",
                    "required": f"≥ {quorum_need}",
                    "delta": None if quorum_have >= quorum_need else f"need {quorum_need - quorum_have} more",
                    "passed": quorum_have >= quorum_need,
                },
                {
                    "key": "crowd",
                    "label": f"Crowd {finish}",
                    "current": f"{crowd_side_pct:.1f}%",
                    "required": f"≥ {crowd_need_pct:.1f}%",
                    "delta": None
                    if crowd_side_pct >= crowd_need_pct
                    else f"+{crowd_need_pct - crowd_side_pct:.1f}pp",
                    "passed": crowd_side_pct >= crowd_need_pct,
                },
            ]
        )
    gates.extend(
        [
            {
                "key": "agreement",
                "label": agree_label,
                "current": f"{agree_have:.1f}%",
                "required": f"≥ {agree_need:.1f}%",
                "delta": None if agree_have >= agree_need else f"+{agree_need - agree_have:.1f}pp",
                "passed": agree_have >= agree_need,
            },
            {
                "key": "evidence",
                "label": "Evidence margin",
                "current": f"{evidence_have:.3f}",
                "required": f"≥ {evidence_need:.3f}",
                "delta": None if evidence_have >= evidence_need else f"+{evidence_need - evidence_have:.3f}",
                "passed": evidence_have >= evidence_need,
            },
            {
                "key": "edge",
                "label": "Edge (gross)" if not cfg.edge.subtract_fees_from_edge else "Net edge",
                "current": f"{edge_have:.1f}¢",
                "required": f"≥ {edge_need:.1f}¢",
                "delta": None if edge_have >= edge_need else f"+{edge_delta:.1f}¢",
                "passed": focus.edge.should_trade or edge_have >= edge_need,
            },
            {
                "key": "risk",
                "label": "Risk gate",
                "current": "ok" if allowed else block_reason,
                "required": "allowed",
                "delta": None if allowed else block_reason,
                "passed": allowed,
            },
            {
                "key": "sizing",
                "label": "Kelly size",
                "current": f"{contracts} contracts",
                "required": "> 0",
                "delta": None if contracts > 0 else "blocked by upstream gates",
                "passed": contracts > 0,
            },
        ]
    )

    binding = next((g for g in gates if not g["passed"]), None)
    if binding:
        binding = {**binding, "binding": True}
        for g in gates:
            g["binding"] = g["key"] == binding["key"]
    else:
        for g in gates:
            g["binding"] = False

    cooldown_remaining = None
    if risk.state.last_trade_ts:
        elapsed = max(0.0, time.time() - risk.state.last_trade_ts)
        remaining = max(0.0, cfg.risk.cooldown_seconds - elapsed)
        cooldown_remaining = round(remaining, 1)

    return {
        "ticker": focus.ticker,
        "side": side,
        "finish": finish,
        "action": focus.edge.reason if not focus.edge.should_trade else f"BUY {side.upper()}",
        "should_trade": focus.edge.should_trade,
        "is_pick": is_pick,
        "bucket_label": th.bucket_label,
        "secs_left": round(focus.secs_left),
        "mins_left": round(focus.secs_left / 60),
        "binding_gate": binding["label"] if binding else None,
        "binding_detail": binding["delta"] or binding["current"] if binding else "All gates pass",
        "spot": round(spot, 2),
        "strike": focus.strike,
        "spot_to_strike_usd": round(spot_delta, 2),
        "spot_to_strike_label": spot_label,
        "regime": focus.forecast.vol_regime,
        "kalshi_book": {
            "yes_bid_cents": yes_bid_c,
            "yes_ask_cents": yes_ask_c,
            "no_bid_cents": no_bid_c,
            "no_ask_cents": no_ask_c,
            "yes_spread_cents": max(0, yes_ask_c - yes_bid_c),
            "no_spread_cents": max(0, no_ask_c - no_bid_c),
            "implied_yes_pct": yes_ask_c,
            "implied_no_pct": no_ask_c,
        },
        "trade_side": {
            "bid_cents": trade_bid_c,
            "ask_cents": trade_ask_c,
            "spread_cents": max(0, trade_ask_c - trade_bid_c),
            "buy_price_cents": trade_ask_c,
        },
        "model": {
            "fair_yes_pct": round(fair_yes_pct, 1),
            "fair_side_pct": round(fair_side_pct, 1),
            "edge_cents": round(edge_have, 2),
            "edge_vs_kalshi_cents": round(fair_side_pct - trade_ask_c, 1),
            "evidence_margin": round(evidence_have, 4),
            "agreement_pct": round(agree_have, 1),
            "ensemble_agreement_pct": round(agree_have, 1),
            "confidence_pct": round(focus.forecast.confidence * 100, 1),
        },
        "requirements": {
            "min_edge_cents": round(edge_need, 2),
            "min_crowd_pct": round(crowd_need_pct, 1) if cfg.gates.crowd_gates_enabled else None,
            "min_ensemble_agreement_pct": round(agree_need, 1),
            "min_evidence_margin": round(evidence_need, 4),
            "min_agreement_pct": round(agree_need, 1),
            "min_quorum": quorum_need if cfg.gates.crowd_gates_enabled else None,
            "max_entry_price_cents": max_entry,
            "price_to_clear_cents": round(price_delta_c, 1) if price_delta_c is not None else None,
        },
        "crowd_gates_enabled": cfg.gates.crowd_gates_enabled,
        "use_ensemble_agreement": cfg.gates.use_ensemble_agreement,
        "gates": gates,
        "risk": {
            "allowed": allowed,
            "block_reason": block_reason,
            "cooldown_remaining_s": cooldown_remaining,
            "open_positions": risk.state.open_positions,
            "max_open_positions": cfg.risk.max_open_positions,
            "already_traded": focus.ticker in risk.state.traded_tickers,
            "daily_pnl_usd": round(risk.state.daily_pnl, 2),
        },
        "sizing": {
            "contracts": contracts,
            "max_trade_usd": round(cfg.sizing.max_trade_usd, 2),
            "cost_per_contract_usd": round(trade_ask_c / 100.0, 2),
        },
    }


def build_snapshot(
    *,
    cfg: BotConfig,
    data: Any,
    candidates: list[MarketCandidate],
    decisions: list[dict],
    best: MarketCandidate | None,
    best_ticker: str | None,
    risk: RiskManager,
    markets_scanned: int,
    balance_usd: float | None = None,
    mode: str,
    recent_settlements: list[dict] | None = None,
) -> DashboardSnapshot:
    selected = next((d for d in decisions if d.get("selected")), None)
    best_decision = next((d for d in decisions if d.get("ticker") == best_ticker), None)

    contracts = 0
    allowed, block_reason = True, "ok"
    is_pick = False
    action = "NO_TRADE"

    # Top 4 by edge — always show prices even when below threshold
    top_by_edge = sorted(candidates, key=lambda c: c.edge.edge_cents, reverse=True)[
        : config.TOP_N_MARKETS
    ]

    focus = best or (top_by_edge[0] if top_by_edge else None)
    focus_th = None
    if focus is not None:
        focus_th = focus.thresholds or resolve_dynamic_thresholds(
            focus.secs_left,
            vol_regime=focus.forecast.vol_regime,
            agreement_score=agreement_score_for_gates(
                focus.forecast, use_ensemble=cfg.gates.use_ensemble_agreement
            ),
            edge_cents=focus.edge.edge_cents,
            crowd_side_prob=(
                focus.forecast.crowd.side_prob(focus.direction.side)
                if cfg.gates.crowd_gates_enabled
                else None
            ),
        )
        allowed, block_reason = risk.allow_trade(
            ticker=focus.ticker, seconds_to_expiry=focus.secs_left
        )

    if best is not None:
        is_pick = best.ticker == best_ticker
        if best_decision:
            contracts = int(best_decision.get("contracts") or 0)
            action = str(best_decision.get("action") or "NO_TRADE")
    elif focus is not None and best_decision and best_decision.get("ticker") == focus.ticker:
        contracts = int(best_decision.get("contracts") or 0)
        action = str(best_decision.get("action") or "NO_TRADE")

    checklist = build_checklist(
        data_ok=data.spot > 0,
        brti_official=bool(getattr(data, "is_official", False)),
        markets_scanned=markets_scanned,
        best=best,
        focus=focus,
        is_pick=is_pick,
        allowed=allowed,
        block_reason=block_reason,
        contracts=contracts,
        cfg=cfg,
        thresholds=focus_th,
    )
    passed = sum(1 for c in checklist if c.passed)
    readiness = round(100.0 * passed / len(checklist), 1) if checklist else 0.0

    blockers = [c.detail for c in checklist if not c.passed]
    if focus and not focus.edge.should_trade:
        blockers.insert(0, focus.edge.reason)

    top_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(top_by_edge, start=1):
        dec = next((d for d in decisions if d["ticker"] == cand.ticker), {})
        top_rows.append(
            _candidate_row(
                cand,
                rank=i,
                is_pick=cand.ticker == best_ticker,
                action=str(dec.get("action") or "NO_TRADE"),
                reason=str(dec.get("reason") or cand.edge.reason),
            )
        )

    crowd_data: dict[str, Any] = {}
    if focus is not None:
        crowd_data = focus.forecast.crowd_summary_at(
            min_favorite=focus_th.min_crowd_favorite if focus_th else None
        )
    if focus is not None:
        dec = best_decision or {}
        best_pick = _candidate_row(
            focus,
            rank=1,
            is_pick=is_pick,
            action=action,
            reason=str(dec.get("reason") or focus.edge.reason),
        )
        best_pick["selected"] = bool(selected)
        best_pick["contracts"] = contracts
    else:
        best_pick = None

    # Traffic-light status for top banner
    min_edge = focus_th.min_edge_cents if focus_th else cfg.edge.min_edge_cents
    top_edge = top_by_edge[0].edge.edge_cents if top_by_edge else -999.0
    close_to_edge = top_edge >= (min_edge - 1.0)

    if selected:
        action_light = "green"
        action_headline = f"TRADE — {action.replace('_', ' ')}"
        action_detail = (
            f"{focus.ticker if focus else ''} · {contracts} contracts · "
            f"edge {top_edge:.1f}¢ · evidence {focus.evidence_score:.3f}"
            if focus
            else "Order executing"
        )
        cycle_status = "TRADE"
    elif focus and focus.edge.should_trade and allowed and contracts > 0:
        action_light = "green"
        action_headline = "READY TO TRADE"
        action_detail = (
            f"{focus.ticker} · BUY {focus.direction.side.upper()} @ "
            f"{focus.edge.market_price * 100:.0f}¢ · edge {top_edge:.1f}¢"
        )
        cycle_status = "READY"
    elif focus and (focus.edge.should_trade or close_to_edge or readiness >= 65):
        action_light = "yellow"
        action_headline = "CLOSE — ALMOST THERE"
        reason_txt = block_reason if not allowed else (focus.edge.reason if not focus.edge.should_trade else "Awaiting pick")
        action_detail = (
            f"Best: {focus.ticker} · edge {top_edge:.1f}¢ (need {min_edge:.1f}¢) · "
            f"readiness {readiness:.0f}% · {reason_txt}"
        )
        cycle_status = "CLOSE"
    else:
        action_light = "red"
        action_headline = "NO TRADE"
        action_detail = (
            blockers[0]
            if blockers
            else f"No edge above {min_edge:.1f}¢ across {len(candidates)} candidates"
        )
        cycle_status = "NO_TRADE"

    entry_context = None
    if focus is not None and focus_th is not None:
        entry_context = build_entry_context(
            focus=focus,
            spot=float(data.spot),
            thresholds=focus_th,
            cfg=cfg,
            risk=risk,
            allowed=allowed,
            block_reason=block_reason,
            contracts=contracts,
            is_pick=is_pick,
        )

    return DashboardSnapshot(
        updated_at=_iso_now(),
        mode=mode,
        env=cfg.kalshi_env,
        balance_usd=balance_usd,
        spot=round(float(data.spot), 2),
        brti_source=str(getattr(data, "source", "")),
        brti_official=bool(getattr(data, "is_official", False)),
        annualized_vol=round(float(getattr(data, "annualized_vol", 0.0)), 4),
        funding_rate=round(float(getattr(data, "funding_rate", 0.0)), 6),
        markets_scanned=markets_scanned,
        candidates=len(candidates),
        cycle_status=cycle_status,
        action_light=action_light,
        action_headline=action_headline,
        action_detail=action_detail,
        readiness_pct=readiness,
        checklist=[asdict(c) for c in checklist],
        blockers=blockers[:8],
        best_pick=best_pick,
        top_markets=top_rows,
        model_votes=[
            {
                "name": v.name,
                "prob_yes": round(v.prob_yes, 4),
                "weight": round(v.weight, 3),
                "confidence": round(v.confidence, 3),
                "side": "ABOVE" if v.prob_yes >= 0.5 else "BELOW",
            }
            for v in (focus.direction.top_votes if focus else [])
        ],
        risk={
            "daily_pnl_usd": round(risk.state.daily_pnl, 4),
            "open_positions": risk.state.open_positions,
            "max_open_positions": cfg.risk.max_open_positions,
            "cooldown_seconds": cfg.risk.cooldown_seconds,
            "last_trade_age_s": round(max(0.0, time.time() - risk.state.last_trade_ts), 1)
            if risk.state.last_trade_ts
            else None,
            "traded_tickers": sorted(risk.state.traded_tickers),
        },
        thresholds={
            "min_edge_cents": focus_th.min_edge_cents if focus_th else cfg.edge.min_edge_cents,
            "min_edge_range": list(focus_th.min_edge_range) if focus_th else None,
            "fee_cents": 0.0 if not cfg.edge.subtract_fees_from_edge else cfg.edge.fee_per_contract_cents,
            "subtract_fees_from_edge": cfg.edge.subtract_fees_from_edge,
            "min_evidence_margin": focus_th.min_evidence_margin if focus_th else config.MIN_EVIDENCE_MARGIN,
            "evidence_margin_range": list(focus_th.evidence_margin_range) if focus_th else None,
            "min_agreement": focus_th.min_agreement if focus_th else config.ENSEMBLE_MIN_AGREEMENT,
            "agreement_range_pct": [round(r * 100, 1) for r in focus_th.agreement_range]
            if focus_th
            else None,
            "min_favorite_pct": focus_th.min_crowd_favorite_pct if focus_th else config.CROWD_MIN_FAVORITE * 100,
            "crowd_favorite_range_pct": [round(r * 100, 1) for r in focus_th.crowd_favorite_range]
            if focus_th
            else None,
            "min_quorum": focus_th.min_quorum if focus_th else config.CROWD_MIN_QUORUM,
            "quorum_range": list(focus_th.quorum_range) if focus_th else None,
            "bucket": focus_th.bucket.value if focus_th else None,
            "bucket_label": focus_th.bucket_label if focus_th else None,
            "dynamic_gates": True,
            "crowd_gates_enabled": cfg.gates.crowd_gates_enabled,
            "use_ensemble_agreement": cfg.gates.use_ensemble_agreement,
            "kalshi_card_only": cfg.gates.kalshi_card_only,
            "kalshi_card_picks": cfg.gates.kalshi_card_picks,
            "top_n_votes": config.TOP_N_VOTES,
            "top_n_markets": config.TOP_N_MARKETS,
            "max_trade_usd": cfg.sizing.max_trade_usd,
        },
        config_summary={
            "bankroll_usd": cfg.sizing.bankroll_usd,
            "max_trade_usd": cfg.sizing.max_trade_usd,
            "cycle_seconds": cfg.cycle_seconds,
            "series": cfg.series_ticker,
        },
        recent_settlements=recent_settlements or [],
        crowd=crowd_data,
        entry_context=entry_context,
    )


def save_snapshot(snapshot: DashboardSnapshot, path: Path | None = None) -> None:
    target = path or STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(snapshot)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(target)


def load_snapshot(path: Path | None = None) -> dict[str, Any]:
    target = path or STATE_PATH
    if not target.exists():
        return asdict(DashboardSnapshot(updated_at=_iso_now()))
    try:
        return json.loads(target.read_text())
    except (json.JSONDecodeError, OSError):
        return asdict(DashboardSnapshot(updated_at=_iso_now()))
