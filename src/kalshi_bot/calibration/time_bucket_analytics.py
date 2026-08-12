"""Performance analytics grouped by time-to-expiration bucket."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from kalshi_bot.strategy.time_buckets import TimeBucket, bucket_policy, classify_time_bucket


def _brier(pairs: Sequence[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    return sum((p - float(w)) ** 2 for p, w in pairs) / len(pairs)


@dataclass(frozen=True)
class SettledTrade:
    ticker: str
    side: str
    prediction: float
    won: bool
    pnl: float
    net_edge: float
    seconds_to_expiry: float
    confidence: float = 0.0


def bucket_label(bucket: TimeBucket) -> str:
    policy = bucket_policy(bucket)
    return policy.label if policy else bucket.value


def time_bucket_performance(
    trades: Sequence[SettledTrade],
    *,
    min_seconds: float = 60.0,
    max_seconds: float = 900.0,
) -> list[dict[str, Any]]:
    """Aggregate realized performance by time bucket (chronological, no shuffle)."""
    groups: dict[str, list[SettledTrade]] = {}
    for t in trades:
        bucket = classify_time_bucket(
            t.seconds_to_expiry, min_seconds=min_seconds, max_seconds=max_seconds
        )
        if bucket in (TimeBucket.TOO_EARLY, TimeBucket.TOO_LATE):
            continue
        key = bucket.value
        groups.setdefault(key, []).append(t)

    order = [
        TimeBucket.BUCKET_15_10.value,
        TimeBucket.BUCKET_10_7.value,
        TimeBucket.BUCKET_7_5.value,
        TimeBucket.BUCKET_5_3.value,
        TimeBucket.BUCKET_3_1.value,
    ]

    rows: list[dict[str, Any]] = []
    for key in order:
        bucket_trades = groups.get(key, [])
        if not bucket_trades:
            rows.append(
                {
                    "bucket": key,
                    "label": bucket_label(TimeBucket(key)),
                    "n_trades": 0,
                    "win_rate": None,
                    "avg_net_edge": None,
                    "total_pnl": 0.0,
                    "brier_score": None,
                    "avg_confidence": None,
                }
            )
            continue
        wins = sum(1 for t in bucket_trades if t.won)
        pairs = [(t.prediction, t.won) for t in bucket_trades]
        rows.append(
            {
                "bucket": key,
                "label": bucket_label(TimeBucket(key)),
                "n_trades": len(bucket_trades),
                "win_rate": wins / len(bucket_trades),
                "avg_net_edge": sum(t.net_edge for t in bucket_trades) / len(bucket_trades),
                "total_pnl": sum(t.pnl for t in bucket_trades),
                "brier_score": _brier(pairs),
                "avg_confidence": sum(t.confidence for t in bucket_trades) / len(bucket_trades),
            }
        )
    return rows


def summarize_time_buckets(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick best/worst buckets by realized edge (requires sample size)."""
    with_data = [r for r in rows if r.get("n_trades", 0) >= 3]
    if not with_data:
        return {"best_bucket": None, "worst_bucket": None, "recommendation": "insufficient_data"}
    best = max(with_data, key=lambda r: r.get("avg_net_edge") or -999)
    worst = min(with_data, key=lambda r: r.get("avg_net_edge") or 999)
    return {
        "best_bucket": best["label"],
        "worst_bucket": worst["label"],
        "recommendation": (
            f"Strongest realized edge in {best['label']}; "
            f"weakest in {worst['label']}"
        ),
    }
