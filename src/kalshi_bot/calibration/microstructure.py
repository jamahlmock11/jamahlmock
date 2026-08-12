"""Microstructure signal calibration — does order-flow improve probability accuracy?"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MicrostructureObservation:
    prediction: float
    won: bool
    bid_ask_imbalance: float
    liquidity_score: float
    trades_per_second: float
    spread: float
    side: str  # yes | no


def _brier(pairs: list[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    return sum((p - float(w)) ** 2 for p, w in pairs) / len(pairs)


def _micro_agrees(obs: MicrostructureObservation) -> bool:
    """Imbalance favors the traded side."""
    if obs.side.lower() == "yes":
        return obs.bid_ask_imbalance > 0
    return obs.bid_ask_imbalance < 0


class MicrostructureCalibrator:
    """Track whether microstructure agreement improves calibration vs baseline."""

    def __init__(self, path: str = "data/calibration/microstructure.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY,
                    ts REAL,
                    ticker TEXT,
                    side TEXT,
                    prediction REAL,
                    won INTEGER,
                    bid_ask_imbalance REAL,
                    liquidity_score REAL,
                    trades_per_second REAL,
                    spread REAL,
                    micro_agrees INTEGER
                )"""
            )

    def record(
        self,
        *,
        ticker: str,
        side: str,
        prediction: float,
        won: bool,
        bid_ask_imbalance: float = 0.0,
        liquidity_score: float = 0.0,
        trades_per_second: float = 0.0,
        spread: float = 0.0,
        ts: float | None = None,
    ) -> None:
        import time

        obs = MicrostructureObservation(
            prediction=prediction,
            won=won,
            bid_ask_imbalance=bid_ask_imbalance,
            liquidity_score=liquidity_score,
            trades_per_second=trades_per_second,
            spread=spread,
            side=side,
        )
        agrees = _micro_agrees(obs)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO observations
                (ts, ticker, side, prediction, won, bid_ask_imbalance, liquidity_score,
                 trades_per_second, spread, micro_agrees)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    ts or time.time(),
                    ticker,
                    side.lower(),
                    prediction,
                    int(won),
                    bid_ask_imbalance,
                    liquidity_score,
                    trades_per_second,
                    spread,
                    int(agrees),
                ),
            )

    def record_from_features(
        self,
        *,
        ticker: str,
        side: str,
        prediction: float,
        won: bool,
        features: dict[str, Any],
        ts: float | None = None,
    ) -> None:
        self.record(
            ticker=ticker,
            side=side,
            prediction=prediction,
            won=won,
            bid_ask_imbalance=float(features.get("bid_ask_imbalance", 0.0)),
            liquidity_score=float(features.get("liquidity_score", features.get("micro_liquidity", 0.0))),
            trades_per_second=float(features.get("trades_per_second", 0.0)),
            spread=float(features.get("spread", 0.0)),
            ts=ts,
        )

    def _load_pairs(self, *, micro_agrees: bool | None = None) -> list[tuple[float, bool]]:
        q = "SELECT prediction, won FROM observations"
        args: list[Any] = []
        if micro_agrees is not None:
            q += " WHERE micro_agrees=?"
            args.append(int(micro_agrees))
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(q, args).fetchall()
        return [(float(p), bool(w)) for p, w in rows]

    def report(self, *, min_samples: int = 5) -> dict[str, Any]:
        all_pairs = self._load_pairs()
        agree_pairs = self._load_pairs(micro_agrees=True)
        disagree_pairs = self._load_pairs(micro_agrees=False)

        baseline_brier = _brier(all_pairs)
        agree_brier = _brier(agree_pairs)
        disagree_brier = _brier(disagree_pairs)

        uplift = None
        if baseline_brier is not None and agree_brier is not None:
            uplift = baseline_brier - agree_brier

        recommend_use = (
            uplift is not None
            and uplift > 0
            and len(agree_pairs) >= min_samples
            and len(disagree_pairs) >= min_samples
        )

        return {
            "n_total": len(all_pairs),
            "n_micro_agrees": len(agree_pairs),
            "n_micro_disagrees": len(disagree_pairs),
            "baseline_brier": baseline_brier,
            "agree_brier": agree_brier,
            "disagree_brier": disagree_brier,
            "brier_uplift_when_agrees": uplift,
            "recommend_use_microstructure": recommend_use,
            "status": "calibrated" if recommend_use else "collecting",
        }

    def export_rows(self, limit: int = 100) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """SELECT ts, ticker, side, prediction, won, bid_ask_imbalance,
                          liquidity_score, micro_agrees
                   FROM observations ORDER BY ts DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            {
                "ts": r[0],
                "ticker": r[1],
                "side": r[2],
                "prediction": r[3],
                "won": bool(r[4]),
                "bid_ask_imbalance": r[5],
                "liquidity_score": r[6],
                "micro_agrees": bool(r[7]),
            }
            for r in rows
        ]
