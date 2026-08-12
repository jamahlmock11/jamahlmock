"""Calibration bucket reporting for production monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SPEC_BUCKETS: list[tuple[float, float, str]] = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 0.75, "70-75%"),
    (0.75, 0.80, "75-80%"),
    (0.80, 0.85, "80-85%"),
    (0.85, 0.90, "85-90%"),
    (0.90, 0.95, "90-95%"),
    (0.95, 1.01, "95%+"),
]


@dataclass(frozen=True)
class BucketMetrics:
    range: str
    n_trades: int
    predicted_probability: float | None
    empirical_win_rate: float | None
    brier_score: float | None
    calibrated: bool


def _brier(predictions: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    if not predictions:
        return None
    return sum((p - float(o)) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)


def calibration_table(
    records: Sequence[tuple[float, bool]],
    *,
    min_trades: int = 3,
) -> list[dict]:
    """Build calibration rows from (predicted_prob, won) pairs."""
    rows: list[dict] = []
    for lo, hi, label in SPEC_BUCKETS:
        bucket = [(p, w) for p, w in records if lo <= p < hi]
        n = len(bucket)
        if n == 0:
            rows.append(
                {
                    "range": label,
                    "n_trades": 0,
                    "predicted_probability": (lo + hi) / 2,
                    "empirical_win_rate": None,
                    "brier_score": None,
                    "calibrated": False,
                }
            )
            continue
        preds = [p for p, _ in bucket]
        outs = [w for _, w in bucket]
        empirical = sum(outs) / n
        rows.append(
            {
                "range": label,
                "n_trades": n,
                "predicted_probability": sum(preds) / n,
                "empirical_win_rate": empirical,
                "brier_score": _brier(preds, outs),
                "calibrated": n >= min_trades,
            }
        )
    return rows
