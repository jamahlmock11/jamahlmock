"""Automatic settlement outcome ingestion for calibration learning."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from kalshi_bot.data.kalshi_client import KalshiClient, normalize_market
from kalshi_bot.strategy.v6_upgrades import V6IntelligenceEngine

logger = logging.getLogger(__name__)

SETTLED_STATUSES = frozenset({"determined", "finalized", "settled", "closed"})


@dataclass(frozen=True)
class PendingEntry:
    id: int
    ticker: str
    side: str
    entry_price: float
    contracts: int
    prediction: float
    confidence: float
    features_json: str
    seconds_to_expiry: float
    volatility: float
    net_edge: float
    reason: str
    opened_ts: float


class SettlementIngestor:
    """Polls Kalshi for settled markets and updates journal + calibrator."""

    def __init__(self, journal_path: str) -> None:
        self.path = Path(journal_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS pending_entries (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT UNIQUE,
                    side TEXT,
                    entry_price REAL,
                    contracts INTEGER,
                    prediction REAL,
                    confidence REAL,
                    features TEXT,
                    seconds_to_expiry REAL,
                    volatility REAL,
                    net_edge REAL,
                    reason TEXT,
                    opened_ts REAL,
                    settled INTEGER DEFAULT 0
                )"""
            )

    def record_entry(
        self,
        *,
        ticker: str,
        side: str,
        entry_price: float,
        contracts: int,
        prediction: float,
        confidence: float,
        features: dict,
        seconds_to_expiry: float,
        volatility: float,
        net_edge: float,
        reason: str,
    ) -> None:
        import json

        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO pending_entries
                (ticker, side, entry_price, contracts, prediction, confidence, features,
                 seconds_to_expiry, volatility, net_edge, reason, opened_ts, settled)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)
                ON CONFLICT(ticker) DO UPDATE SET
                    side=excluded.side, entry_price=excluded.entry_price,
                    contracts=excluded.contracts, prediction=excluded.prediction,
                    confidence=excluded.confidence, features=excluded.features,
                    seconds_to_expiry=excluded.seconds_to_expiry, volatility=excluded.volatility,
                    net_edge=excluded.net_edge, reason=excluded.reason, opened_ts=excluded.opened_ts,
                    settled=0""",
                (
                    ticker,
                    side,
                    entry_price,
                    contracts,
                    prediction,
                    confidence,
                    json.dumps(features),
                    seconds_to_expiry,
                    volatility,
                    net_edge,
                    reason,
                    time.time(),
                ),
            )

    def pending(self) -> list[PendingEntry]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT id, ticker, side, entry_price, contracts, prediction, confidence, "
                "features, seconds_to_expiry, volatility, net_edge, reason, opened_ts "
                "FROM pending_entries WHERE settled=0"
            ).fetchall()
        return [
            PendingEntry(
                id=r[0],
                ticker=r[1],
                side=r[2],
                entry_price=r[3],
                contracts=r[4],
                prediction=r[5],
                confidence=r[6],
                features_json=r[7],
                seconds_to_expiry=r[8],
                volatility=r[9],
                net_edge=r[10],
                reason=r[11],
                opened_ts=r[12],
            )
            for r in rows
        ]

    def ingest(self, client: KalshiClient, engine: V6IntelligenceEngine) -> list[dict]:
        """Check pending entries against Kalshi settlement; update calibrator."""
        import json

        resolved: list[dict] = []
        for entry in self.pending():
            try:
                raw = client.get_market(entry.ticker)
                market = normalize_market(raw)
            except Exception as exc:
                logger.debug("settlement poll %s: %s", entry.ticker, exc)
                continue

            status = str(market.get("status") or raw.get("status") or "").lower()
            result = str(raw.get("result") or market.get("raw", {}).get("result") or "").lower()
            if status not in SETTLED_STATUSES and result not in ("yes", "no"):
                continue

            yes_won = result == "yes"
            side_won = yes_won if entry.side.lower() == "yes" else not yes_won
            payout = 1.0 if side_won else 0.0
            pnl = (payout - entry.entry_price) * entry.contracts

            features = json.loads(entry.features_json or "{}")
            engine.record_settlement(
                ticker=entry.ticker,
                side=entry.side,
                prediction=entry.prediction,
                confidence=entry.confidence,
                features=features,
                won=side_won,
                pnl=pnl,
                reason=entry.reason,
            )

            with sqlite3.connect(self.path) as conn:
                conn.execute("UPDATE pending_entries SET settled=1 WHERE id=?", (entry.id,))

            resolved.append(
                {
                    "ticker": entry.ticker,
                    "side": entry.side,
                    "won": side_won,
                    "pnl": pnl,
                    "result": result,
                    "prediction": entry.prediction,
                }
            )
            logger.info(
                "settlement ingested %s %s won=%s pnl=$%.2f pred=%.0f%%",
                entry.ticker,
                entry.side,
                side_won,
                pnl,
                entry.prediction * 100,
            )
        return resolved

    def calibration_summary(self, calibrator) -> list[dict]:
        buckets = []
        for b in calibrator.buckets():
            buckets.append(
                {
                    "range": f"{b.bucket_lo:.0%}-{b.bucket_hi:.0%}",
                    "n_trades": b.n_trades,
                    "empirical_win_rate": b.empirical_win_rate,
                    "offset": b.calibrated_offset,
                    "calibrated": b.n_trades >= calibrator.min_trades,
                }
            )
        return buckets
