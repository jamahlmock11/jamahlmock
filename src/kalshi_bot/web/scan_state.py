"""Shared scan state for CLI and web dashboard."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ScanSnapshot:
    asof: datetime
    spot: float
    spot_source: str
    balance_usd: float | None
    markets_scanned: int
    opportunities: list[dict[str, Any]] = field(default_factory=list)
    tape: dict[str, dict[str, Any]] = field(default_factory=dict)
    settlements: list[dict[str, Any]] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    freshness: dict[str, Any] = field(default_factory=dict)


class ScanState:
    """Thread-safe holder for the latest scan snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: ScanSnapshot | None = None
        self._updated_at: float = 0.0

    def update(self, snapshot: ScanSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot
            self._updated_at = time.time()
        notify_scan_updated()

    def get(self) -> ScanSnapshot | None:
        with self._lock:
            return self._snapshot

    def updated_at(self) -> float:
        with self._lock:
            return self._updated_at

    def to_dict(self) -> dict[str, Any]:
        snap = self.get()
        if snap is None:
            return {"status": "no_data"}
        freshness = dict(snap.freshness)
        freshness.setdefault("scan_age_seconds", round(time.time() - self.updated_at(), 2))
        return {
            "status": "ok",
            "asof": snap.asof.isoformat(),
            "spot": snap.spot,
            "spot_source": snap.spot_source,
            "balance_usd": snap.balance_usd,
            "markets_scanned": snap.markets_scanned,
            "opportunities": snap.opportunities,
            "tape": snap.tape,
            "settlements": snap.settlements,
            "calibration": snap.calibration,
            "freshness": freshness,
        }


GLOBAL_SCAN_STATE = ScanState()


def notify_scan_updated() -> None:
    try:
        from kalshi_bot.web.web_server import request_broadcast

        request_broadcast()
    except Exception:
        pass
