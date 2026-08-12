"""Timestamped market observations with data-quality metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DataQuality(str, Enum):
    FRESH = "FRESH"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    MISSING = "MISSING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Observation:
    name: str
    value: Any
    source: str
    observed_at: datetime
    latency_ms: float | None = None
    quality: DataQuality = DataQuality.FRESH
    detail: str = ""


@dataclass
class ObservationBundle:
    """Collected observations for one evaluation cycle."""

    observations: list[Observation] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def add(
        self,
        name: str,
        value: Any,
        *,
        source: str,
        quality: DataQuality = DataQuality.FRESH,
        latency_ms: float | None = None,
        detail: str = "",
    ) -> None:
        self.observations.append(
            Observation(
                name=name,
                value=value,
                source=source,
                observed_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                quality=quality,
                detail=detail,
            )
        )

    def worst_quality(self) -> DataQuality:
        order = [
            DataQuality.ERROR,
            DataQuality.MISSING,
            DataQuality.STALE,
            DataQuality.DEGRADED,
            DataQuality.FRESH,
        ]
        worst = DataQuality.FRESH
        for obs in self.observations:
            if order.index(obs.quality) < order.index(worst):
                worst = obs.quality
        return worst

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "name": o.name,
                "value": o.value,
                "source": o.source,
                "observed_at": o.observed_at.isoformat(),
                "latency_ms": o.latency_ms,
                "quality": o.quality.value,
                "detail": o.detail,
            }
            for o in self.observations
        ]
