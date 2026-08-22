"""Late-window crowd-favorite entry — size up when the hour slot was unused."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kalshi_btc_1hr_bot.config import BotConfig, LateCrowdConfig
from kalshi_btc_1hr_bot.dynamic_gates import HourBucket, classify_hour_bucket
from kalshi_btc_1hr_bot.edge import TradeSignal
from kalshi_btc_1hr_bot.evidence import MarketCandidate, evaluate_edge_with_evidence


@dataclass(frozen=True)
class LateCrowdContext:
    """Whether late crowd mode is active this cycle."""

    active: bool
    in_window: bool
    hour_untraded: bool
    slot_free: bool
    bucket: HourBucket | None
    reason: str = ""


@dataclass(frozen=True)
class LateCrowdQualification:
    qualified: bool
    reason: str
    edge: TradeSignal | None = None


def in_late_crowd_window(secs_left: float, cfg: LateCrowdConfig) -> bool:
    return cfg.min_seconds_to_expiry <= secs_left <= cfg.max_seconds_to_expiry


def hour_market_tickers(window_markets: list[dict[str, Any]], hour_close: datetime | None) -> set[str]:
    if hour_close is None:
        return set()
    return {
        str(m["ticker"])
        for m in window_markets
        if m.get("close_time") == hour_close and m.get("ticker")
    }


def traded_current_hour(journal: Any, tickers: set[str]) -> bool:
    if not tickers:
        return False
    for trade in journal.list_trades(limit=200):
        if trade.passed and trade.ticker in tickers:
            return True
    return False


def resolve_late_crowd_context(
    *,
    cfg: BotConfig,
    secs_left: float,
    open_positions: int,
    journal: Any,
    window_markets: list[dict[str, Any]],
    hour_close: datetime | None,
) -> LateCrowdContext:
    late = cfg.late_crowd
    if not late.enabled:
        return LateCrowdContext(False, False, False, False, None, "late crowd off")

    in_window = in_late_crowd_window(secs_left, late)
    bucket = classify_hour_bucket(secs_left) if in_window else None
    slot_free = open_positions < cfg.risk.max_open_positions
    hour_tickers = hour_market_tickers(window_markets, hour_close)
    hour_untraded = not traded_current_hour(journal, hour_tickers)

    if not in_window:
        return LateCrowdContext(False, False, hour_untraded, slot_free, bucket, "outside late window")
    if not slot_free:
        return LateCrowdContext(False, in_window, hour_untraded, False, bucket, "position slot in use")
    if not hour_untraded:
        return LateCrowdContext(False, in_window, False, slot_free, bucket, "already traded this hour")

    return LateCrowdContext(True, in_window, hour_untraded, slot_free, bucket, "late crowd armed")


def crowd_aligns_with_trade(cand: MarketCandidate, min_favorite: float) -> tuple[bool, str]:
    crowd = cand.forecast.crowd
    side = cand.direction.side
    side_pct = crowd.side_pct(side)
    finish = cand.direction.finish_label
    if crowd.consensus_side != side:
        return (
            False,
            f"Crowd favors {crowd.finish_label}, trade wants {finish}",
        )
    if crowd.side_prob(side) < min_favorite:
        return (
            False,
            f"Crowd {finish} {side_pct:.1f}% < {min_favorite * 100:.0f}%",
        )
    return True, f"Crowd {finish} {side_pct:.1f}% (favorite {crowd.favorite_pct:.1f}%)"


def evaluate_late_crowd_edge(
    cand: MarketCandidate,
    *,
    cfg: BotConfig,
    fee_cents: float,
    subtract_fees: bool,
) -> LateCrowdQualification:
    """Re-check edge with crowd gates on and late-specific floors."""
    late = cfg.late_crowd
    aligned_ok, crowd_detail = crowd_aligns_with_trade(cand, late.min_crowd_favorite)
    if not aligned_ok:
        return LateCrowdQualification(False, crowd_detail)

    market = cand.market
    yes_ask = float(market.get("yes_ask") or 1.0)
    no_ask = float(market.get("no_ask") or 1.0)
    yes_bid = float(market.get("yes_bid") or yes_ask)
    no_bid = float(market.get("no_bid") or no_ask)

    edge = evaluate_edge_with_evidence(
        cand.forecast.p_fair,
        yes_ask,
        no_ask,
        yes_bid,
        no_bid,
        cand.direction,
        fee_cents=fee_cents,
        subtract_fees=subtract_fees,
        min_edge=late.min_edge_cents,
        min_evidence_margin=late.min_evidence_margin,
        min_agreement=late.min_agreement,
        min_quorum=late.min_quorum,
        min_crowd_favorite=late.min_crowd_favorite,
        crowd_gates_enabled=True,
        use_ensemble_agreement=cfg.gates.use_ensemble_agreement,
        thresholds=cand.thresholds,
        forecast=cand.forecast,
    )
    if not edge.should_trade:
        return LateCrowdQualification(False, edge.reason, edge=edge)
    return LateCrowdQualification(
        True,
        f"Late crowd: {crowd_detail} · {edge.reason}",
        edge=edge,
    )


def select_late_crowd_pick(
    candidates: list[MarketCandidate],
    *,
    cfg: BotConfig,
    fee_cents: float,
    subtract_fees: bool,
) -> tuple[MarketCandidate | None, LateCrowdQualification | None]:
    """Pick the strongest late crowd favorite among card candidates."""
    if not candidates:
        return None, None

    ranked: list[tuple[MarketCandidate, LateCrowdQualification]] = []
    for cand in candidates:
        qual = evaluate_late_crowd_edge(
            cand,
            cfg=cfg,
            fee_cents=fee_cents,
            subtract_fees=subtract_fees,
        )
        if qual.qualified and qual.edge is not None:
            ranked.append((cand, qual))

    if not ranked:
        return None, None

    best_cand, best_qual = max(
        ranked,
        key=lambda pair: (
            pair[0].forecast.crowd.side_prob(pair[0].direction.side),
            pair[0].evidence_score,
            pair[1].edge.edge_cents if pair[1].edge else 0.0,
        ),
    )
    return best_cand, best_qual
