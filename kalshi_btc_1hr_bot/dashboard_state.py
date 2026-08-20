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
from kalshi_btc_1hr_bot.evidence import MarketCandidate
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
    is_pick: bool,
    allowed: bool,
    block_reason: str,
    contracts: int,
    cfg: BotConfig,
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
    if best is None:
        items.append(
            CheckItem(
                "Tradeable edge found",
                False,
                "No market cleared min edge + evidence margin",
                "edge",
            )
        )
        return items

    items.extend(
        [
            CheckItem(
                f"Evidence margin ≥ {config.MIN_EVIDENCE_MARGIN:.3f}",
                best.direction.margin >= config.MIN_EVIDENCE_MARGIN,
                f"{best.direction.margin:.3f} ({best.direction.finish_label})",
                "evidence",
            ),
            CheckItem(
                f"Net edge ≥ {cfg.edge.min_edge_cents:.1f}¢",
                best.edge.should_trade or best.edge.edge_cents >= cfg.edge.min_edge_cents,
                f"{best.edge.edge_cents:.1f}¢ after {cfg.edge.fee_per_contract_cents:.2f}¢ fee",
                "edge",
            ),
            CheckItem(
                f"Ensemble agreement ≥ {config.ENSEMBLE_MIN_AGREEMENT:.0%}",
                best.forecast.agreement_score >= config.ENSEMBLE_MIN_AGREEMENT,
                f"{best.forecast.agreement_score:.0%}",
                "model",
            ),
            CheckItem(
                "Top-4 market by edge + evidence pick",
                is_pick and best.edge.should_trade,
                "Selected as best among top markets" if is_pick else "Not the evidence winner",
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
    if best is not None:
        allowed, block_reason = risk.allow_trade(ticker=best.ticker, seconds_to_expiry=best.secs_left)
        is_pick = best.ticker == best_ticker
        if best_decision:
            contracts = int(best_decision.get("contracts") or 0)
            action = str(best_decision.get("action") or "NO_TRADE")

    checklist = build_checklist(
        data_ok=data.spot > 0,
        brti_official=bool(getattr(data, "is_official", False)),
        markets_scanned=markets_scanned,
        best=best,
        is_pick=is_pick,
        allowed=allowed,
        block_reason=block_reason,
        contracts=contracts,
        cfg=cfg,
    )
    passed = sum(1 for c in checklist if c.passed)
    readiness = round(100.0 * passed / len(checklist), 1) if checklist else 0.0

    blockers = [c.detail for c in checklist if not c.passed]
    if best and not best.edge.should_trade:
        blockers.insert(0, best.edge.reason)

    # Top 4 by edge — always show prices even when below threshold
    top_by_edge = sorted(candidates, key=lambda c: c.edge.edge_cents, reverse=True)[
        : config.TOP_N_MARKETS
    ]
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

    # Best focus = evidence pick, or highest-edge candidate for display
    focus = best or (top_by_edge[0] if top_by_edge else None)
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
    min_edge = cfg.edge.min_edge_cents
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
            "min_edge_cents": cfg.edge.min_edge_cents,
            "fee_cents": cfg.edge.fee_per_contract_cents,
            "min_evidence_margin": config.MIN_EVIDENCE_MARGIN,
            "min_agreement": config.ENSEMBLE_MIN_AGREEMENT,
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
