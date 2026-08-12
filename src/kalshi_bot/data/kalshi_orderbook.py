"""Live Kalshi orderbook cache from WebSocket snapshot + delta updates."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _price_dollars(raw: dict) -> float | None:
    if raw.get("price_dollars") is not None:
        try:
            return float(raw["price_dollars"])
        except (TypeError, ValueError):
            return None
    if raw.get("price") is not None:
        try:
            p = float(raw["price"])
            return p / 100.0 if p > 1.0 else p
        except (TypeError, ValueError):
            return None
    return None


def _qty(raw_level) -> float:
    if isinstance(raw_level, (list, tuple)) and len(raw_level) >= 2:
        try:
            return float(raw_level[1])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _price_from_level(raw_level) -> float | None:
    if isinstance(raw_level, (list, tuple)) and len(raw_level) >= 1:
        try:
            p = float(raw_level[0])
            return p / 100.0 if p > 1.0 else p
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class OrderbookQuote:
    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None
    spread: float
    last_update: float
    stale: bool
    source: str


class OrderbookCache:
    """Per-ticker YES/NO level books maintained from WS snapshots and deltas."""

    def __init__(self, *, stale_after: float = 8.0) -> None:
        self.stale_after = stale_after
        self._yes: dict[str, dict[float, float]] = {}
        self._no: dict[str, dict[float, float]] = {}
        self._last_update: dict[str, float] = {}
        self._lock = threading.Lock()

    def _parse_levels(self, levels: list) -> dict[float, float]:
        out: dict[float, float] = {}
        for lvl in levels or []:
            price = _price_from_level(lvl)
            qty = _qty(lvl)
            if price is None or qty <= 0:
                continue
            out[price] = qty
        return out

    def apply_snapshot(self, ticker: str, body: dict) -> None:
        yes = self._parse_levels(body.get("yes") or [])
        no = self._parse_levels(body.get("no") or [])
        with self._lock:
            self._yes[ticker] = yes
            self._no[ticker] = no
            self._last_update[ticker] = time.time()

    def apply_delta(self, ticker: str, body: dict) -> None:
        side = str(body.get("side") or "").lower()
        price = _price_dollars(body)
        if price is None:
            return
        try:
            delta = float(body.get("delta_fp") or body.get("delta") or 0)
        except (TypeError, ValueError):
            delta = 0.0
        with self._lock:
            book = self._yes if side == "yes" else self._no
            levels = book.setdefault(ticker, {})
            if delta <= 0:
                levels.pop(price, None)
            else:
                levels[price] = delta
            self._last_update[ticker] = time.time()

    def quote(self, ticker: str) -> OrderbookQuote | None:
        with self._lock:
            yes_levels = self._yes.get(ticker) or {}
            no_levels = self._no.get(ticker) or {}
            last_up = self._last_update.get(ticker, 0.0)

        if not yes_levels and not no_levels:
            return None

        yes_bid = max(yes_levels) if yes_levels else None
        no_bid = max(no_levels) if no_levels else None
        yes_ask = (1.0 - no_bid) if no_bid is not None else None
        no_ask = (1.0 - yes_bid) if yes_bid is not None else None
        spread = (yes_ask - yes_bid) if yes_bid is not None and yes_ask is not None else 0.0
        age = time.time() - last_up
        return OrderbookQuote(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            spread=spread,
            last_update=last_up,
            stale=age > self.stale_after,
            source="ws",
        )

    def ticker_count(self) -> int:
        with self._lock:
            return len(self._last_update)

    def to_orderbook_dict(self, ticker: str) -> dict | None:
        """REST-shaped orderbook for compute_microstructure."""
        with self._lock:
            yes_levels = self._yes.get(ticker) or {}
            no_levels = self._no.get(ticker) or {}
        if not yes_levels and not no_levels:
            return None
        yes = [[int(round(p * 100)), q] for p, q in sorted(yes_levels.items())]
        no = [[int(round(p * 100)), q] for p, q in sorted(no_levels.items())]
        return {"orderbook": {"yes": yes, "no": no}}
