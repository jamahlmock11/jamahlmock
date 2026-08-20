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
