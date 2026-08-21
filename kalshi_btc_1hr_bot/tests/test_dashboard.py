"""Dashboard and trade journal tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kalshi_btc_1hr_bot.dashboard import app, build_api_payload
from kalshi_btc_1hr_bot.dashboard_state import (
    CheckItem,
    DashboardSnapshot,
    build_checklist,
    build_entry_context,
    load_snapshot,
    save_snapshot,
)
from kalshi_btc_1hr_bot.trade_journal import TradeJournal


def test_save_and_load_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    snap = DashboardSnapshot(updated_at="2026-01-01T00:00:00+00:00", spot=72000.0, mode="LIVE")
    save_snapshot(snap, path)
    loaded = load_snapshot(path)
    assert loaded["spot"] == 72000.0
    assert loaded["mode"] == "LIVE"


def test_trade_journal_record_and_stats(tmp_path: Path) -> None:
    db = tmp_path / "trades.db"
    journal = TradeJournal(db)
    tid = journal.record_trade(
        ticker="KXBTCD-TEST",
        side="yes",
        contracts=1,
        entry_price=0.45,
        mode="PAPER",
        passed=True,
        edge_cents=3.5,
        evidence_score=0.12,
        p_fair=0.55,
        confidence=0.7,
        strike=72000.0,
        spot=72100.0,
        finish="ABOVE",
    )
    journal.settle_trade(tid, won=True, pnl=0.55, result="yes")
    stats = journal.stats()
    assert stats["executed_trades"] == 1
    assert stats["wins"] == 1
    assert stats["realized_pnl_usd"] == pytest.approx(0.55)


def test_build_checklist_without_best() -> None:
    from kalshi_btc_1hr_bot.config import BotConfig

    items = build_checklist(
        data_ok=True,
        brti_official=True,
        markets_scanned=0,
        best=None,
        is_pick=False,
        allowed=True,
        block_reason="ok",
        contracts=0,
        cfg=BotConfig(),
    )
    assert any(not i.passed for i in items)


def test_api_state_endpoint() -> None:
    client = TestClient(app)
    resp = client.get("/api/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "snapshot" in data
    assert "stats" in data
    assert "trades" in data


def test_build_api_payload_shape() -> None:
    payload = build_api_payload()
    assert isinstance(payload["snapshot"], dict)
    assert "win_rate_pct" in payload["stats"]
    assert "entry_context" in payload["snapshot"] or payload["snapshot"].get("entry_context") is None


def test_build_entry_context_gate_deltas() -> None:
    from kalshi_btc_1hr_bot.config import BotConfig
    from kalshi_btc_1hr_bot.crowd_forecast import CrowdForecast, CrowdMember
    from kalshi_btc_1hr_bot.dynamic_gates import resolve_dynamic_thresholds
    from kalshi_btc_1hr_bot.edge import TradeSignal
    from kalshi_btc_1hr_bot.ensemble import EnsembleResult, ModelVote
    from kalshi_btc_1hr_bot.evidence import DirectionalEvidence, MarketCandidate
    from kalshi_btc_1hr_bot.forecast import ForecastEnsembleOutput
    from kalshi_btc_1hr_bot.model import ModelOutput
    from kalshi_btc_1hr_bot.risk import RiskManager
    import numpy as np

    members = tuple(CrowdMember("m", 0.30, 1.0, 0.9, "model") for _ in range(6))
    crowd = CrowdForecast(
        prob_yes=0.30,
        prob_no=0.70,
        consensus_side="no",
        confidence=0.8,
        agreement_score=0.75,
        uncertainty=0.2,
        quorum_count=6,
        quorum_required=5,
        quorum_met=True,
        yes_votes=0,
        no_votes=6,
        synthesis="blend",
        members=members,
        top_votes=members[:4],
        disagreeing=(),
    )
    votes = (ModelVote("m", 0.30, 0.5, 0.9),)
    direction = DirectionalEvidence("no", 0.0, 0.25, 0.25, votes)
    mo = ModelOutput(0.3, 0.7, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.0, 0.0, 0.0, "med", 0.0, np.zeros(18))
    ens = EnsembleResult(0.3, 0.7, 0.8, 0.2, 0.75, votes)
    forecast = ForecastEnsembleOutput(0.3, 0.8, 0.75, 0.2, mo, ens, crowd, True)
    th = resolve_dynamic_thresholds(1800, vol_regime="med", agreement_score=0.75, crowd_side_prob=0.70)
    edge = TradeSignal(False, "no", 0.3, 0.94, 0.0, -0.64, "Crowd BELOW 70.0% < 71%")
    cand = MarketCandidate(
        ticker="KXBTCD-TEST",
        strike=78000.0,
        secs_left=1800,
        forecast=forecast,
        direction=direction,
        edge=edge,
        evidence_score=0.12,
        market={"yes_ask": 0.08, "no_ask": 0.94, "yes_bid": 0.06, "no_bid": 0.92},
        thresholds=th,
    )
    ctx = build_entry_context(
        focus=cand,
        spot=76700.0,
        thresholds=th,
        cfg=BotConfig(),
        risk=RiskManager(BotConfig()),
        allowed=True,
        block_reason="ok",
        contracts=0,
        is_pick=False,
    )
    assert ctx["kalshi_book"]["no_ask_cents"] == 94
    assert ctx["binding_gate"] is not None
    assert len(ctx["gates"]) >= 5
    assert ctx["spot_to_strike_label"].endswith("below strike")
