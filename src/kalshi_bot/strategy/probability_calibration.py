"""Bucket-wise probability calibration from realized trade outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationBucket:
    bucket_lo: float
    bucket_hi: float
    n_trades: int
    empirical_win_rate: float
    calibrated_offset: float


class ProbabilityCalibrator:
    """Bucket-wise calibration; requires a minimum trades per bucket."""

    def __init__(self, min_trades_per_bucket: int = 3) -> None:
        self.min_trades = min_trades_per_bucket
        self._buckets: dict[int, list[bool]] = {}

    def _key(self, prob: float) -> int:
        return int(prob * 10)  # decile buckets

    def record(self, predicted_prob: float, won: bool) -> None:
        self._buckets.setdefault(self._key(predicted_prob), []).append(won)

    def calibrate(self, prob: float) -> tuple[float, bool]:
        key = self._key(prob)
        outcomes = self._buckets.get(key, [])
        if len(outcomes) < self.min_trades:
            return prob, False
        empirical = sum(outcomes) / len(outcomes)
        offset = empirical - prob
        return max(0.01, min(0.99, prob + offset)), True

    def buckets(self) -> list[CalibrationBucket]:
        out: list[CalibrationBucket] = []
        for key in sorted(self._buckets):
            outcomes = self._buckets[key]
            if not outcomes:
                continue
            lo = key / 10.0
            hi = (key + 1) / 10.0
            empirical = sum(outcomes) / len(outcomes)
            mid = (lo + hi) / 2
            out.append(
                CalibrationBucket(
                    bucket_lo=lo,
                    bucket_hi=hi,
                    n_trades=len(outcomes),
                    empirical_win_rate=empirical,
                    calibrated_offset=empirical - mid,
                )
            )
        return out
