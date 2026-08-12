"""Kalshi trade tape — REST backfill + WebSocket live prints for microstructure."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kalshi_bot.data.kalshi_client import KalshiClient, dollars
from kalshi_bot.data.kalshi_orderbook import OrderbookCache, OrderbookQuote

logger = logging.getLogger(__name__)


def _parse_trade_ts(raw: dict) -> float:
    created = raw.get("created_time") or raw.get("ts")
    if isinstance(created, (int, float)):
        return float(created)
    if isinstance(created, str):
        if created.endswith("Z"):
            created = created[:-1] + "+00:00"
        return datetime.fromisoformat(created).timestamp()
    return time.time()


def normalize_trade(raw: dict) -> dict[str, Any]:
    """Normalize Kalshi trade payload for microstructure."""
    yes_px = dollars(raw.get("yes_price_dollars"))
    if yes_px is None and raw.get("yes_price") is not None:
        yes_px = float(raw["yes_price"]) / 100.0
    side = raw.get("taker_outcome_side") or raw.get("taker_side") or "yes"
    count = raw.get("count_fp") or raw.get("count") or "1"
    try:
        qty = float(count)
    except (TypeError, ValueError):
        qty = 1.0
    return {
        "ts": _parse_trade_ts(raw),
        "ticker": raw.get("ticker") or "",
        "yes_price": yes_px or 0.0,
        "side": side,
        "quantity": qty,
        "is_block": bool(raw.get("is_block_trade")),
        "raw": raw,
    }


@dataclass
class TapeStats:
    trades_per_second: float
    buy_pressure: float  # -1..+1 yes-side taker flow
    volume_1m: float
    last_price: float | None
    stale: bool
    source: str


class TradeTapeBuffer:
    """Rolling per-ticker trade buffer."""

    def __init__(self, *, maxlen: int = 500) -> None:
        self._trades: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=maxlen))
        self._last_update: dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, ticker: str, trade: dict[str, Any]) -> None:
        with self._lock:
            self._trades[ticker].append(trade)
            self._last_update[ticker] = time.time()

    def extend(self, ticker: str, trades: list[dict[str, Any]]) -> None:
        with self._lock:
            buf = self._trades[ticker]
            for t in sorted(trades, key=lambda x: x.get("ts", 0)):
                buf.append(t)
            if trades:
                self._last_update[ticker] = time.time()

    def recent(self, ticker: str, *, max_age_seconds: float = 120.0) -> list[dict[str, Any]]:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return [t for t in self._trades[ticker] if t.get("ts", 0) >= cutoff]

    def stats(self, ticker: str, *, stale_after: float = 45.0) -> TapeStats:
        trades = self.recent(ticker)
        if len(trades) < 2:
            last_up = self._last_update.get(ticker, 0)
            return TapeStats(0.0, 0.0, 0.0, trades[-1]["yes_price"] if trades else None, time.time() - last_up > stale_after, "rest")
        ts0 = trades[0]["ts"]
        ts1 = trades[-1]["ts"]
        dt = max(ts1 - ts0, 0.001)
        tps = len(trades) / dt
        yes_vol = sum(t["quantity"] for t in trades if t.get("side") == "yes")
        no_vol = sum(t["quantity"] for t in trades if t.get("side") == "no")
        total = yes_vol + no_vol
        pressure = (yes_vol - no_vol) / total if total > 0 else 0.0
        cutoff_1m = time.time() - 60.0
        vol_1m = sum(t["quantity"] for t in trades if t.get("ts", 0) >= cutoff_1m)
        last_up = self._last_update.get(ticker, 0)
        return TapeStats(
            tps,
            pressure,
            vol_1m,
            trades[-1].get("yes_price"),
            time.time() - last_up > stale_after,
            "ws" if tps > 0 else "rest",
        )

    def tickers(self) -> list[str]:
        with self._lock:
            return list(self._trades.keys())


class KalshiTradeTapeService:
    """Fetches REST trade history and streams live trades + orderbook via WebSocket."""

    def __init__(
        self,
        client: KalshiClient,
        *,
        buffer: TradeTapeBuffer | None = None,
        orderbook: OrderbookCache | None = None,
    ) -> None:
        self.client = client
        self.buffer = buffer or TradeTapeBuffer()
        self.orderbook = orderbook or OrderbookCache()
        self._ws = None
        self._subscribed: set[str] = set()

    def refresh_ticker(self, ticker: str, *, limit: int = 50) -> list[dict[str, Any]]:
        try:
            raw_trades = list(self.client.iter_trades(ticker=ticker, limit=limit, max_pages=1))
            normalized = [normalize_trade(t) for t in raw_trades]
            self.buffer.extend(ticker, normalized)
            return normalized
        except Exception as exc:
            logger.debug("trade tape REST refresh failed %s: %s", ticker, exc)
            return self.buffer.recent(ticker)

    def ensure_subscription(self, tickers: list[str]) -> None:
        new = [t for t in tickers if t and t not in self._subscribed]
        if not new:
            return
        for t in new:
            self.refresh_ticker(t)
        self._subscribed.update(new)
        self._start_ws_if_needed()

    def recent_trades(self, ticker: str) -> list[dict[str, Any]]:
        if ticker not in self._subscribed:
            self.ensure_subscription([ticker])
        return self.buffer.recent(ticker)

    def tape_stats(self, ticker: str) -> TapeStats:
        return self.buffer.stats(ticker)

    def orderbook_quote(self, ticker: str) -> OrderbookQuote | None:
        return self.orderbook.quote(ticker)

    def orderbook_dict(self, ticker: str) -> dict | None:
        return self.orderbook.to_orderbook_dict(ticker)

    def feed_status(self) -> dict[str, Any]:
        ws_status = self._ws.status() if self._ws is not None else {
            "connected": False,
            "last_message_at": None,
            "last_error": None,
            "subscribed_tickers": len(self._subscribed),
        }
        return {
            **ws_status,
            "orderbook_tickers": self.orderbook.ticker_count(),
        }

    def _start_ws_if_needed(self) -> None:
        if self._ws is not None:
            self._ws.subscribe(list(self._subscribed))
            return
        try:
            from kalshi_bot.data.kalshi_websocket import KalshiWebSocketFeed

            self._ws = KalshiWebSocketFeed(self.client, self.buffer, self.orderbook)
            self._ws.start()
            self._ws.subscribe(list(self._subscribed))
        except Exception as exc:
            logger.warning("WebSocket feed unavailable: %s (REST only)", exc)

    def close(self) -> None:
        if self._ws is not None:
            self._ws.stop()
            self._ws = None
