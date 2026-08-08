"""Trade opportunity monitor — logging, dashboard, rejection breakdown."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_bot.strategy.decision_record import MarketEvaluationRecord
from kalshi_bot.strategy.rejection_codes import RejectionCode, REJECTION_DESCRIPTIONS


@dataclass
class RejectionBreakdown:
    total_evaluations: int
    trades: int
    no_trades: int
    by_code: dict[str, int]
    top_3: list[tuple[str, int, float]]  # code, count, pct

    def summary_text(self) -> str:
        lines = [
            f"Total evaluations: {self.total_evaluations}",
            f"TRADES: {self.trades}",
            f"NO_TRADE: {self.no_trades}",
            "",
            "Rejection breakdown:",
        ]
        for code, count in sorted(self.by_code.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / max(self.no_trades, 1)
            desc = REJECTION_DESCRIPTIONS.get(RejectionCode(code), code)
            lines.append(f"  {code}: {count} ({pct:.1f}%) — {desc}")
        lines.append("")
        lines.append("Top 3 rejection reasons:")
        for i, (code, count, pct) in enumerate(self.top_3, 1):
            lines.append(f"  {i}. {code}: {count} ({pct:.1f}%)")
        return "\n".join(lines)


class OpportunityMonitor:
    """Persist and analyze market evaluation records."""

    def __init__(self, db_path: str = "data/diagnostics/evaluations.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    ticker TEXT,
                    verdict TEXT,
                    primary_rejection TEXT,
                    model_prob_up REAL,
                    yes_ask REAL,
                    no_ask REAL,
                    yes_net_edge REAL,
                    no_net_edge REAL,
                    best_net_edge REAL,
                    setup_tier TEXT,
                    opportunity_score REAL,
                    seconds_to_expiry REAL,
                    spot REAL,
                    record_json TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_ts ON evaluations(ts)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_rejection ON evaluations(primary_rejection)"
            )

    def record(self, rec: MarketEvaluationRecord) -> int:
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute(
                """INSERT INTO evaluations
                (ts, ticker, verdict, primary_rejection, model_prob_up, yes_ask, no_ask,
                 yes_net_edge, no_net_edge, best_net_edge, setup_tier, opportunity_score,
                 seconds_to_expiry, spot, record_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.evaluated_at.timestamp(),
                    rec.ticker,
                    rec.verdict,
                    rec.primary_rejection.value,
                    rec.model_prob_up,
                    rec.yes_ask,
                    rec.no_ask,
                    rec.yes_side.net_edge_dollars,
                    rec.no_side.net_edge_dollars,
                    rec.best_net_edge,
                    rec.setup_tier,
                    rec.opportunity_score,
                    rec.seconds_to_expiry,
                    rec.spot,
                    rec.to_json(),
                ),
            )
            return int(cur.lastrowid or 0)

    def rejection_breakdown(self, *, since_ts: float | None = None) -> RejectionBreakdown:
        with sqlite3.connect(self.path) as conn:
            if since_ts:
                rows = conn.execute(
                    "SELECT verdict, primary_rejection FROM evaluations WHERE ts >= ?",
                    (since_ts,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT verdict, primary_rejection FROM evaluations"
                ).fetchall()

        total = len(rows)
        trades = sum(1 for v, _ in rows if v.startswith("TRADE"))
        no_trades = total - trades
        codes: Counter[str] = Counter()
        for verdict, rejection in rows:
            if verdict.startswith("TRADE"):
                continue
            codes[rejection or RejectionCode.EDGE_TOO_SMALL.value] += 1

        top_3 = [
            (code, count, 100.0 * count / max(no_trades, 1))
            for code, count in codes.most_common(3)
        ]
        return RejectionBreakdown(
            total_evaluations=total,
            trades=trades,
            no_trades=no_trades,
            by_code=dict(codes),
            top_3=top_3,
        )

    def filter_attribution(self) -> dict[str, int]:
        """Count how often each filter check failed across all evaluations."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT record_json FROM evaluations").fetchall()
        counts: Counter[str] = Counter()
        for (raw,) in rows:
            import json as _json
            rec = _json.loads(raw)
            for fc in rec.get("filter_checks", []):
                if not fc.get("passed"):
                    counts[fc.get("name", "unknown")] += 1
        return dict(counts)

    def hypothetical_trades_by_tier(self) -> dict[str, int]:
        """Count evals that would qualify per tier ignoring hard model block."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT record_json FROM evaluations").fetchall()
        counts: Counter[str] = Counter()
        for (raw,) in rows:
            rec = json.loads(raw)
            best_net = rec.get("best_net_edge", 0)
            conf = rec.get("model_confidence", 0)
            cents = best_net * 100
            if cents >= 22 and conf >= 0.75:
                counts["A_PLUS"] += 1
            elif cents >= 17 and conf >= 0.65:
                counts["A"] += 1
            elif cents >= 12 and conf >= 0.55:
                counts["B"] += 1
        return dict(counts)

    def tier_hypothetical_breakdown(self) -> dict[str, int]:
        """Count how many evals would qualify per tier (paper analysis)."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT setup_tier, verdict FROM evaluations"
            ).fetchall()
        counts: Counter[str] = Counter()
        for tier, _verdict in rows:
            if tier and tier != "NONE":
                counts[tier] += 1
        return dict(counts)

    def edge_distribution(self) -> dict[str, Any]:
        """Distribution of best net edge for diagnostics."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT best_net_edge, verdict FROM evaluations"
            ).fetchall()
        buckets = {"<5¢": 0, "5-10¢": 0, "10-15¢": 0, "15-20¢": 0, "20¢+": 0}
        for edge, verdict in rows:
            cents = edge * 100
            if cents < 5:
                buckets["<5¢"] += 1
            elif cents < 10:
                buckets["5-10¢"] += 1
            elif cents < 15:
                buckets["10-15¢"] += 1
            elif cents < 20:
                buckets["15-20¢"] += 1
            else:
                buckets["20¢+"] += 1
        return buckets

    def recent(self, limit: int = 10) -> list[MarketEvaluationRecord]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT record_json FROM evaluations ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        # Return raw dicts for display — full deserialization optional
        return rows  # type: ignore[return-value]

    def export_report(self, path: str | None = None) -> str:
        breakdown = self.rejection_breakdown()
        edge_dist = self.edge_distribution()
        tier_counts = self.tier_hypothetical_breakdown()
        tier_hypothetical = self.hypothetical_trades_by_tier()
        filter_attr = self.filter_attribution()
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rejection_breakdown": breakdown.summary_text(),
            "edge_distribution": edge_dist,
            "tier_hypothetical": tier_counts,
            "tier_if_edge_only": tier_hypothetical,
            "filter_attribution": filter_attr,
            "stats": {
                "total": breakdown.total_evaluations,
                "trades": breakdown.trades,
                "no_trades": breakdown.no_trades,
                "trade_rate_pct": round(
                    100.0 * breakdown.trades / max(breakdown.total_evaluations, 1), 1
                ),
            },
        }
        text = json.dumps(report, indent=2)
        if path:
            Path(path).write_text(text)
        return text
