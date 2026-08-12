"""15-minute time buckets with distinct edge/size behavior."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TimeBucket(str, Enum):
    BUCKET_15_10 = "15_10"  # 10–15 min remaining
    BUCKET_10_7 = "10_7"
    BUCKET_7_5 = "7_5"
    BUCKET_5_3 = "5_3"
    BUCKET_3_1 = "3_1"
    TOO_EARLY = "too_early"
    TOO_LATE = "too_late"


@dataclass(frozen=True)
class BucketPolicy:
    min_net_edge_dollars: float
    min_raw_edge_dollars: float
    size_multiplier: float
    min_confidence: float
    allow_wait: bool
    label: str


# Strategy tightens as expiration approaches: require more edge early, size down late.
DEFAULT_BUCKET_POLICIES: dict[TimeBucket, BucketPolicy] = {
    TimeBucket.BUCKET_15_10: BucketPolicy(
        min_net_edge_dollars=0.14,
        min_raw_edge_dollars=0.18,
        size_multiplier=0.60,
        min_confidence=0.55,
        allow_wait=True,
        label="15→10 min",
    ),
    TimeBucket.BUCKET_10_7: BucketPolicy(
        min_net_edge_dollars=0.12,
        min_raw_edge_dollars=0.16,
        size_multiplier=0.75,
        min_confidence=0.52,
        allow_wait=True,
        label="10→7 min",
    ),
    TimeBucket.BUCKET_7_5: BucketPolicy(
        min_net_edge_dollars=0.10,
        min_raw_edge_dollars=0.14,
        size_multiplier=0.85,
        min_confidence=0.50,
        allow_wait=True,
        label="7→5 min",
    ),
    TimeBucket.BUCKET_5_3: BucketPolicy(
        min_net_edge_dollars=0.08,
        min_raw_edge_dollars=0.12,
        size_multiplier=1.0,
        min_confidence=0.48,
        allow_wait=False,
        label="5→3 min",
    ),
    TimeBucket.BUCKET_3_1: BucketPolicy(
        min_net_edge_dollars=0.06,
        min_raw_edge_dollars=0.10,
        size_multiplier=0.70,
        min_confidence=0.45,
        allow_wait=False,
        label="3→1 min",
    ),
}


def classify_time_bucket(
    seconds_to_expiry: float,
    *,
    min_seconds: float = 60.0,
    max_seconds: float = 900.0,
) -> TimeBucket:
    mins = seconds_to_expiry / 60.0
    if seconds_to_expiry < min_seconds:
        return TimeBucket.TOO_LATE
    if seconds_to_expiry > max_seconds:
        return TimeBucket.TOO_EARLY
    if mins > 10:
        return TimeBucket.BUCKET_15_10
    if mins > 7:
        return TimeBucket.BUCKET_10_7
    if mins > 5:
        return TimeBucket.BUCKET_7_5
    if mins > 3:
        return TimeBucket.BUCKET_5_3
    return TimeBucket.BUCKET_3_1


def bucket_policy(
    bucket: TimeBucket,
    overrides: dict[str, dict] | None = None,
) -> BucketPolicy | None:
    if bucket in (TimeBucket.TOO_EARLY, TimeBucket.TOO_LATE):
        return None
    base = DEFAULT_BUCKET_POLICIES[bucket]
    if not overrides or bucket.value not in overrides:
        return base
    o = overrides[bucket.value]
    return BucketPolicy(
        min_net_edge_dollars=float(o.get("min_net_edge_dollars", base.min_net_edge_dollars)),
        min_raw_edge_dollars=float(o.get("min_raw_edge_dollars", base.min_raw_edge_dollars)),
        size_multiplier=float(o.get("size_multiplier", base.size_multiplier)),
        min_confidence=float(o.get("min_confidence", base.min_confidence)),
        allow_wait=bool(o.get("allow_wait", base.allow_wait)),
        label=str(o.get("label", base.label)),
    )
