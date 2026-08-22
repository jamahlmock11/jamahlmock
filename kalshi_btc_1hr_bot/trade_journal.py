"""SQLite trade journal and settlement tracking for the 1-hour bot."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_btc_1hr_bot.config import ROOT

DEFAULT_DB = ROOT / "data" / "trades.db"
SETTLED_STATUSES = frozenset({"determined", "finalized", "settled", "closed"})


def daily_pnl_today(journal: "TradeJournal") -> float:
    """Realized + open cost for UTC calendar day — used to sync risk daily PnL."""
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    pnl = 0.0
    for trade in journal.list_trades(limit=500):
        if not trade.passed or trade.opened_ts < today_start:
            continue
        if trade.settled and trade.pnl is not None:
            pnl += float(trade.pnl)
        elif not trade.settled:
            pnl -= float(trade.cost_usd)
    return pnl


@dataclass(frozen=True)
class TradeRecord:
    id: int
    opened_ts: float
    ticker: str
    side: str
    contracts: int
    entry_price: float
    cost_usd: float
    mode: str
    order_id: str
    passed: bool
    block_reason: str
    edge_cents: float
    evidence_score: float
    p_fair: float
    confidence: float
    strike: float
    spot: float
    finish: str
    tp_price: float | None
    sl_price: float | None
    exit_price: float | None
    exit_reason: str | None
    exit_order_id: str | None
    closed_early: bool
    settled: bool
    won: bool | None
    pnl: float | None
    result: str | None
    settled_ts: float | None


class TradeJournal:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    opened_ts REAL NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    contracts INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    mode TEXT NOT NULL,
                    order_id TEXT,
                    passed INTEGER NOT NULL,
                    block_reason TEXT,
                    edge_cents REAL,
                    evidence_score REAL,
                    p_fair REAL,
                    confidence REAL,
                    strike REAL,
                    spot REAL,
                    finish TEXT,
                    settled INTEGER DEFAULT 0,
                    won INTEGER,
                    pnl REAL,
                    result TEXT,
                    settled_ts REAL,
                    tp_price REAL,
                    sl_price REAL,
                    exit_price REAL,
                    exit_reason TEXT,
                    exit_order_id TEXT,
                    closed_early INTEGER DEFAULT 0
                )"""
            )
            for col, typ in (
                ("tp_price", "REAL"),
                ("sl_price", "REAL"),
                ("exit_price", "REAL"),
                ("exit_reason", "TEXT"),
                ("exit_order_id", "TEXT"),
                ("closed_early", "INTEGER DEFAULT 0"),
            ):
                try:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                """CREATE TABLE IF NOT EXISTS cycle_log (
                    id INTEGER PRIMARY KEY,
                    ts REAL NOT NULL,
                    mode TEXT,
                    status TEXT,
                    markets_scanned INTEGER,
                    candidates INTEGER,
                    best_ticker TEXT,
                    best_action TEXT,
                    reason TEXT,
                    selected INTEGER,
                    spot REAL,
                    readiness_pct REAL,
                    payload TEXT
                )"""
            )

    def record_trade(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        entry_price: float,
        mode: str,
        order_id: str = "",
        passed: bool = True,
        block_reason: str = "",
        edge_cents: float = 0.0,
        evidence_score: float = 0.0,
        p_fair: float = 0.0,
        confidence: float = 0.0,
        strike: float = 0.0,
        spot: float = 0.0,
        finish: str = "",
        tp_price: float | None = None,
        sl_price: float | None = None,
    ) -> int:
        cost = contracts * entry_price
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                """INSERT INTO trades
                (opened_ts, ticker, side, contracts, entry_price, cost_usd, mode, order_id,
                 passed, block_reason, edge_cents, evidence_score, p_fair, confidence,
                 strike, spot, finish, settled, tp_price, sl_price, closed_early)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,0)""",
                (
                    time.time(),
                    ticker,
                    side,
                    contracts,
                    entry_price,
                    cost,
                    mode,
                    order_id,
                    int(passed),
                    block_reason,
                    edge_cents,
                    evidence_score,
                    p_fair,
                    confidence,
                    strike,
                    spot,
                    finish,
                    tp_price,
                    sl_price,
                ),
            )
            return int(cur.lastrowid or 0)

    def record_cycle(
        self,
        *,
        mode: str,
        status: str,
        markets_scanned: int,
        candidates: int,
        best_ticker: str,
        best_action: str,
        reason: str,
        selected: bool,
        spot: float,
        readiness_pct: float,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO cycle_log
                (ts, mode, status, markets_scanned, candidates, best_ticker, best_action,
                 reason, selected, spot, readiness_pct, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    time.time(),
                    mode,
                    status,
                    markets_scanned,
                    candidates,
                    best_ticker,
                    best_action,
                    reason,
                    int(selected),
                    spot,
                    readiness_pct,
                    json.dumps(payload or {}),
                ),
            )

    def pending_trades(self) -> list[TradeRecord]:
        return self.list_trades(settled=False)

    def list_trades(self, *, settled: bool | None = None, limit: int = 100) -> list[TradeRecord]:
        query = "SELECT * FROM trades"
        params: list[Any] = []
        if settled is not None:
            query += " WHERE settled=?"
            params.append(int(settled))
        query += " ORDER BY opened_ts DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def list_cycles(self, limit: int = 50) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM cycle_log ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            out.append(item)
        return out

    def stats(self) -> dict[str, Any]:
        with sqlite3.connect(self.path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM trades WHERE passed=1").fetchone()[0]
            settled = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE passed=1 AND settled=1"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE passed=1 AND settled=1 AND won=1"
            ).fetchone()[0]
            losses = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE passed=1 AND settled=1 AND won=0"
            ).fetchone()[0]
            pnl_row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE passed=1 AND settled=1"
            ).fetchone()
            cost_row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM trades WHERE passed=1"
            ).fetchone()
            pending = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE passed=1 AND settled=0"
            ).fetchone()[0]
        realized_pnl = float(pnl_row[0] if pnl_row else 0.0)
        total_cost = float(cost_row[0] if cost_row else 0.0)
        win_rate = (wins / settled * 100.0) if settled else 0.0
        return {
            "executed_trades": int(total),
            "settled_trades": int(settled),
            "pending_trades": int(pending),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate_pct": round(win_rate, 1),
            "realized_pnl_usd": round(realized_pnl, 4),
            "total_cost_usd": round(total_cost, 4),
        }

    def settle_trade(self, trade_id: int, *, won: bool, pnl: float, result: str) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """UPDATE trades SET settled=1, won=?, pnl=?, result=?, settled_ts=?
                WHERE id=? AND settled=0""",
                (int(won), pnl, result, time.time(), trade_id),
            )

    def close_trade_early(
        self,
        trade_id: int,
        *,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        exit_order_id: str = "",
    ) -> None:
        """Mark a trade closed via take-profit or stop-loss before settlement."""
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """UPDATE trades SET settled=1, won=?, pnl=?, result=?, settled_ts=?,
                   exit_price=?, exit_reason=?, exit_order_id=?, closed_early=1
                WHERE id=? AND settled=0""",
                (
                    int(pnl >= 0),
                    pnl,
                    exit_reason,
                    time.time(),
                    exit_price,
                    exit_reason,
                    exit_order_id,
                    trade_id,
                ),
            )

    def poll_settlements(self, client: Any) -> list[dict[str, Any]]:
        """Check Kalshi for settled markets and update journal."""
        resolved: list[dict[str, Any]] = []
        for trade in self.pending_trades():
            try:
                raw = client._request("GET", f"/markets/{trade.ticker}")
            except Exception:
                continue
            market = raw.get("market", raw)
            status = str(market.get("status") or "").lower()
            result = str(market.get("result") or "").lower()
            if status not in SETTLED_STATUSES and result not in ("yes", "no"):
                continue
            yes_won = result == "yes"
            side_won = yes_won if trade.side.lower() == "yes" else not yes_won
            payout = 1.0 if side_won else 0.0
            pnl = (payout - trade.entry_price) * trade.contracts
            self.settle_trade(trade.id, won=side_won, pnl=pnl, result=result)
            resolved.append(
                {
                    "ticker": trade.ticker,
                    "side": trade.side,
                    "won": side_won,
                    "pnl": pnl,
                    "cost_usd": trade.cost_usd,
                    "result": result,
                    "settled_ts": time.time(),
                }
            )
        return resolved

    def _row_to_trade(self, row: sqlite3.Row) -> TradeRecord:
        keys = row.keys()
        return TradeRecord(
            id=row["id"],
            opened_ts=row["opened_ts"],
            ticker=row["ticker"],
            side=row["side"],
            contracts=row["contracts"],
            entry_price=row["entry_price"],
            cost_usd=row["cost_usd"],
            mode=row["mode"],
            order_id=row["order_id"] or "",
            passed=bool(row["passed"]),
            block_reason=row["block_reason"] or "",
            edge_cents=row["edge_cents"] or 0.0,
            evidence_score=row["evidence_score"] or 0.0,
            p_fair=row["p_fair"] or 0.0,
            confidence=row["confidence"] or 0.0,
            strike=row["strike"] or 0.0,
            spot=row["spot"] or 0.0,
            finish=row["finish"] or "",
            tp_price=row["tp_price"] if "tp_price" in keys else None,
            sl_price=row["sl_price"] if "sl_price" in keys else None,
            exit_price=row["exit_price"] if "exit_price" in keys else None,
            exit_reason=row["exit_reason"] if "exit_reason" in keys else None,
            exit_order_id=row["exit_order_id"] if "exit_order_id" in keys else None,
            closed_early=bool(row["closed_early"]) if "closed_early" in keys else False,
            settled=bool(row["settled"]),
            won=bool(row["won"]) if row["won"] is not None else None,
            pnl=row["pnl"],
            result=row["result"],
            settled_ts=row["settled_ts"],
        )
