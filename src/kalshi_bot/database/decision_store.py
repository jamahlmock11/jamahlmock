"""Persistent decision snapshots for audit, calibration, and learning."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DecisionSnapshot:
    ts: float
    strategy: str
    series: str
    ticker: str
    signal: str
    reason: str
    model_version: str
    time_remaining_s: float
    model_prob_yes: float
    model_prob_no: float
    market_yes: float | None
    market_no: float | None
    executable_yes: float | None
    executable_no: float | None
    raw_edge: float
    net_edge: float
    confidence: float
    regime: str
    features: dict[str, Any]
    observations: list[dict[str, Any]]
    why_trade: str
    why_not_trade: str


class DecisionStore:
    def __init__(self, path: str = "data/decisions.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY,
                    ts REAL,
                    strategy TEXT,
                    series TEXT,
                    ticker TEXT,
                    signal TEXT,
                    reason TEXT,
                    model_version TEXT,
                    payload TEXT
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS outcomes (
                    id INTEGER PRIMARY KEY,
                    decision_id INTEGER,
                    ticker TEXT,
                    won INTEGER,
                    pnl REAL,
                    actual_outcome TEXT,
                    resolved_ts REAL
                )"""
            )

    def save(self, snap: DecisionSnapshot) -> int:
        payload = {
            "time_remaining_s": snap.time_remaining_s,
            "model_prob_yes": snap.model_prob_yes,
            "model_prob_no": snap.model_prob_no,
            "market_yes": snap.market_yes,
            "market_no": snap.market_no,
            "executable_yes": snap.executable_yes,
            "executable_no": snap.executable_no,
            "raw_edge": snap.raw_edge,
            "net_edge": snap.net_edge,
            "confidence": snap.confidence,
            "regime": snap.regime,
            "features": snap.features,
            "observations": snap.observations,
            "why_trade": snap.why_trade,
            "why_not_trade": snap.why_not_trade,
        }
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                """INSERT INTO decisions
                (ts, strategy, series, ticker, signal, reason, model_version, payload)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    snap.ts,
                    snap.strategy,
                    snap.series,
                    snap.ticker,
                    snap.signal,
                    snap.reason,
                    snap.model_version,
                    json.dumps(payload),
                ),
            )
            return int(cur.lastrowid)

    def recent(self, *, strategy: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q = "SELECT id, ts, strategy, series, ticker, signal, reason, model_version, payload FROM decisions"
        args: list[Any] = []
        if strategy:
            q += " WHERE strategy=?"
            args.append(strategy)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            payload = json.loads(r[8] or "{}")
            out.append(
                {
                    "id": r[0],
                    "ts": r[1],
                    "strategy": r[2],
                    "series": r[3],
                    "ticker": r[4],
                    "signal": r[5],
                    "reason": r[6],
                    "model_version": r[7],
                    **payload,
                }
            )
        return out
