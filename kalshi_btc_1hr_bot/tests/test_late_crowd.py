"""Tests for late-window crowd-favorite entry."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kalshi_btc_1hr_bot.config import BotConfig, LateCrowdConfig
from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdMember
from kalshi_btc_1hr_bot.dynamic_gates import resolve_dynamic_thresholds
from kalshi_btc_1hr_bot.edge import TradeSignal
from kalshi_btc_1hr_bot.ensemble import EnsembleResult, ModelVote
from kalshi_btc_1hr_bot.evidence import DirectionalEvidence, MarketCandidate
from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput
from kalshi_btc_1hr_bot.late_crowd import (
    evaluate_late_crowd_edge,
    in_late_crowd_window,
    resolve_late_crowd_context,
    select_late_crowd_pick,
    traded_current_hour,
)
from kalshi_btc_1hr_bot.model import ModelOutput
import numpy as np


def _crowd(*, side: str = "yes", side_prob: float = 0.78, quorum: int = 5) -> CrowdForecast:
    prob_yes = side_prob if side == "yes" else 1.0 - side_prob
    members = tuple(
        CrowdMember(f"m{i}", prob_yes if side == "yes" else 1 - prob_yes, 1.0, 0.9, "model")
        for i in range(quorum)
    )
    return CrowdForecast(
        prob_yes=prob_yes,
        prob_no=1.0 - prob_yes,
        consensus_side=side,
        confidence=0.8,
        agreement_score=0.9,
        uncertainty=0.15,
        quorum_count=quorum,
        quorum_required=4,
        quorum_met=True,
        yes_votes=quorum if side == "yes" else 0,
        no_votes=0 if side == "yes" else quorum,
        synthesis="blend",
        members=members,
        top_votes=members,
        disagreeing=(),
    )


def _forecast(*, crowd: CrowdForecast, prob_yes: float = 0.72) -> ForecastEnsembleOutput:
    votes = (ModelVote("five_layer", prob_yes, 0.4, 0.9),)
    mo = ModelOutput(prob_yes, 0.4, prob_yes, prob_yes, prob_yes, prob_yes, prob_yes, 0.5, 0, 0, 0, "med", 0, np.zeros(18))
    ens = EnsembleResult(prob_yes, 1 - prob_yes, 0.8, 0.15, 0.72, votes)
    return ForecastEnsembleOutput(prob_yes, 0.8, 0.9, 0.15, mo, ens, crowd, True)


def _candidate(
    *,
    side: str = "yes",
    crowd_side_prob: float = 0.78,
    edge_blocked: bool = True,
    secs_left: float = 600.0,
) -> MarketCandidate:
    crowd = _crowd(side=side, side_prob=crowd_side_prob)
    forecast = _forecast(crowd=crowd, prob_yes=0.72 if side == "yes" else 0.28)
    direction = DirectionalEvidence(
        side,
        0.30 if side == "yes" else 0.05,
        0.05 if side == "yes" else 0.30,
        0.25,
        forecast.votes,
    )
    th = resolve_dynamic_thresholds(secs_left, vol_regime="med", agreement_score=0.72)
    yes_ask = 0.40
    no_ask = 0.62
    edge = TradeSignal(
        not edge_blocked,
        side,
        forecast.p_fair,
        yes_ask if side == "yes" else no_ask,
        18.0,
        0.18,
        "blocked by flow" if edge_blocked else "ok",
    )
    return MarketCandidate(
        ticker="KXBTCD-TEST-A",
        strike=65000.0,
        secs_left=secs_left,
        forecast=forecast,
        direction=direction,
        edge=edge,
        evidence_score=0.25,
        market={"yes_ask": yes_ask, "no_ask": no_ask, "yes_bid": 0.38, "no_bid": 0.60},
        thresholds=th,
        trend_aligned=False,
        flow_aligned=False,
        trend_detail="blocked",
        flow_detail="blocked",
    )


class _FakeJournal:
    def __init__(self, tickers: set[str] | None = None) -> None:
        self._tickers = tickers or set()

    def list_trades(self, limit: int = 200):
        for t in self._tickers:
            yield type("T", (), {"passed": True, "ticker": t})()


def test_in_late_crowd_window():
    cfg = LateCrowdConfig(min_seconds_to_expiry=120.0, max_seconds_to_expiry=1500.0)
    assert in_late_crowd_window(600, cfg)
    assert in_late_crowd_window(120, cfg)
    assert in_late_crowd_window(1500, cfg)
    assert not in_late_crowd_window(119, cfg)
    assert not in_late_crowd_window(1600, cfg)


def test_traded_current_hour_blocks_late_mode():
    hour_close = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
    tickers = {"KXBTCD-TEST-A", "KXBTCD-TEST-B"}
    journal = _FakeJournal({"KXBTCD-TEST-A"})
    assert traded_current_hour(journal, tickers)

    cfg = BotConfig()
    ctx = resolve_late_crowd_context(
        cfg=cfg,
        secs_left=600,
        open_positions=0,
        journal=journal,
        window_markets=[
            {"ticker": "KXBTCD-TEST-A", "close_time": hour_close},
            {"ticker": "KXBTCD-TEST-B", "close_time": hour_close},
        ],
        hour_close=hour_close,
    )
    assert ctx.in_window
    assert not ctx.hour_untraded
    assert not ctx.active


def test_late_crowd_qualifies_when_crowd_strong():
    cfg = BotConfig()
    cand = _candidate(side="yes", crowd_side_prob=0.78, edge_blocked=True)
    qual = evaluate_late_crowd_edge(cand, cfg=cfg, fee_cents=0.0, subtract_fees=False)
    assert qual.qualified
    assert qual.edge is not None
    assert qual.edge.should_trade


def test_late_crowd_rejects_crowd_mismatch():
    cfg = BotConfig()
    crowd = _crowd(side="yes", side_prob=0.78)
    forecast = _forecast(crowd=crowd, prob_yes=0.28)
    direction = DirectionalEvidence("no", 0.05, 0.30, 0.25, forecast.votes)
    th = resolve_dynamic_thresholds(600, vol_regime="med", agreement_score=0.72)
    cand = MarketCandidate(
        ticker="KXBTCD-MISMATCH",
        strike=65000.0,
        secs_left=600.0,
        forecast=forecast,
        direction=direction,
        edge=TradeSignal(False, "no", forecast.p_fair, 0.62, 10.0, 0.10, "flow blocked"),
        evidence_score=0.25,
        market={"yes_ask": 0.40, "no_ask": 0.62, "yes_bid": 0.38, "no_bid": 0.60},
        thresholds=th,
    )
    qual = evaluate_late_crowd_edge(cand, cfg=cfg, fee_cents=0.0, subtract_fees=False)
    assert not qual.qualified
    assert "Crowd favors" in qual.reason


def test_select_late_crowd_pick_prefers_stronger_crowd():
    cfg = BotConfig()
    weak = _candidate(side="yes", crowd_side_prob=0.73, edge_blocked=True)
    weak = MarketCandidate(
        **{**weak.__dict__, "ticker": "KXBTCD-WEAK", "evidence_score": 0.20}
    )
    strong = _candidate(side="yes", crowd_side_prob=0.82, edge_blocked=True)
    strong = MarketCandidate(
        **{**strong.__dict__, "ticker": "KXBTCD-STRONG", "evidence_score": 0.22}
    )
    pick, qual = select_late_crowd_pick(
        [weak, strong],
        cfg=cfg,
        fee_cents=0.0,
        subtract_fees=False,
    )
    assert pick is not None
    assert pick.ticker == "KXBTCD-STRONG"
    assert qual is not None and qual.qualified


def test_late_context_armed_when_slot_free_and_hour_untraded():
    hour_close = datetime.now(timezone.utc) + timedelta(minutes=10)
    cfg = BotConfig()
    ctx = resolve_late_crowd_context(
        cfg=cfg,
        secs_left=600,
        open_positions=0,
        journal=_FakeJournal(),
        window_markets=[{"ticker": "KXBTCD-TEST-A", "close_time": hour_close}],
        hour_close=hour_close,
    )
    assert ctx.active
    assert ctx.slot_free
    assert ctx.hour_untraded
