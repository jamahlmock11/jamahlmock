"""Kalshi Pro real-time depth + order-flow stream for the dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from kalshi_btc_1hr_bot.config import BotConfig, load_config
from kalshi_btc_1hr_bot.dashboard_state import load_snapshot
from kalshi_btc_1hr_bot.kalshi_client import KalshiClient
from kalshi_btc_1hr_bot.orderbook_analytics import (
    ProMarketAnalytics,
    TradePrint,
    build_pro_analytics,
    parse_orderbook_levels,
    pro_analytics_to_dict,
)

logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None  # type: ignore


@dataclass
class ProStreamStatus:
    enabled: bool = False
    connected: bool = False
    mode: str = "off"  # websocket | rest | off
    subscribed_tickers: list[str] = field(default_factory=list)
    last_message_ts: float | None = None
    last_error: str = ""
    reconnects: int = 0


class KalshiProHub:
    """Background hub: WebSocket orderbook + trade flow with REST fallback."""

    def __init__(self, cfg: BotConfig | None = None) -> None:
        self.cfg = cfg or load_config()
        self._lock = threading.Lock()
        self._analytics: dict[str, ProMarketAnalytics] = {}
        self._status = ProStreamStatus(enabled=self._should_enable())
        self._trade_tape: dict[str, deque[TradePrint]] = {}
        self._books: dict[str, dict[str, list[tuple[float, int]]]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: KalshiClient | None = None
        self._depth_limit = int(__import__("os").getenv("KALSHI_PRO_DEPTH", "10"))
        self._flow_window = float(__import__("os").getenv("KALSHI_PRO_FLOW_WINDOW", "120"))

    def _should_enable(self) -> bool:
        flag = __import__("os").getenv("KALSHI_PRO_ENABLED", "true").lower()
        if flag in ("false", "0", "no"):
            return False
        return bool(self.cfg.kalshi_api_key_id and self.cfg.kalshi_private_key_pem)

    def start(self) -> None:
        if not self._status.enabled:
            logger.info("Kalshi Pro stream disabled (no credentials or KALSHI_PRO_ENABLED=false)")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="kalshi-pro-hub", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get_payload(self, *, focus_ticker: str | None = None, focus_side: str = "yes") -> dict[str, Any]:
        with self._lock:
            status = {
                "enabled": self._status.enabled,
                "connected": self._status.connected,
                "mode": self._status.mode,
                "subscribed_tickers": list(self._status.subscribed_tickers),
                "last_message_age_ms": round((time.time() - self._status.last_message_ts) * 1000, 0)
                if self._status.last_message_ts
                else None,
                "reconnects": self._status.reconnects,
                "last_error": self._status.last_error,
            }
            markets = {t: pro_analytics_to_dict(p) for t, p in self._analytics.items()}
            focus = None
            if focus_ticker and focus_ticker in self._analytics:
                focus = pro_analytics_to_dict(self._analytics[focus_ticker])
            elif focus_ticker:
                focus = markets.get(focus_ticker)
        return {"status": status, "focus_ticker": focus_ticker, "focus_side": focus_side, "focus": focus, "markets": markets}

    def _run_loop(self) -> None:
        if websockets is None:
            logger.warning("websockets package missing — Kalshi Pro using REST polling only")
            self._rest_poll_loop()
            return
        while not self._stop.is_set():
            try:
                asyncio.run(self._ws_main())
            except Exception as exc:
                with self._lock:
                    self._status.connected = False
                    self._status.last_error = str(exc)[:200]
                    self._status.reconnects += 1
                logger.exception("Kalshi Pro websocket loop failed")
            if self._stop.wait(3.0):
                break

    def _rest_poll_loop(self) -> None:
        self._client = KalshiClient(self.cfg)
        with self._lock:
            self._status.mode = "rest"
        while not self._stop.is_set():
            tickers = self._target_tickers()
            with self._lock:
                self._status.subscribed_tickers = tickers
            for ticker in tickers:
                if self._stop.is_set():
                    break
                self._refresh_rest(ticker)
            self._stop.wait(1.0)
        if self._client:
            self._client.close()

    async def _ws_main(self) -> None:
        ws_url = self._ws_url()
        sign_path = "/trade-api/ws/v2"
        client = KalshiClient(self.cfg)
        self._client = client
        try:
            ts = str(int(time.time() * 1000))
            headers = {
                "KALSHI-ACCESS-KEY": self.cfg.kalshi_api_key_id,
                "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": client._sign(ts, "GET", sign_path),
            }
            async with websockets.connect(ws_url, additional_headers=headers, ping_interval=20) as ws:
                with self._lock:
                    self._status.connected = True
                    self._status.mode = "websocket"
                    self._status.last_error = ""
                await self._subscribe_tickers(ws, self._target_tickers())
                refresh_task = asyncio.create_task(self._refresh_subscriptions(ws))
                try:
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        self._handle_ws_message(message)
                finally:
                    refresh_task.cancel()
                    try:
                        await refresh_task
                    except asyncio.CancelledError:
                        pass
        finally:
            with self._lock:
                self._status.connected = False
            client.close()

    async def _refresh_subscriptions(self, ws: Any) -> None:
        last: list[str] = []
        msg_id = 100
        while not self._stop.is_set():
            tickers = self._target_tickers()
            if tickers != last:
                await self._subscribe_tickers(ws, tickers, msg_id=msg_id)
                msg_id += 1
                last = list(tickers)
            await asyncio.sleep(2.0)

    async def _subscribe_tickers(self, ws: Any, tickers: list[str], *, msg_id: int = 1) -> None:
        if not tickers:
            return
        for ticker in tickers:
            if ticker not in self._books:
                self._bootstrap_book(ticker)
        sub = {
            "id": msg_id,
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_delta", "trade"], "market_tickers": tickers},
        }
        await ws.send(json.dumps(sub))
        with self._lock:
            self._status.subscribed_tickers = list(tickers)

    def _handle_ws_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return
        msg_type = data.get("type")
        msg = data.get("msg") or {}
        with self._lock:
            self._status.last_message_ts = time.time()
        if msg_type == "orderbook_snapshot":
            self._apply_snapshot(msg)
        elif msg_type == "orderbook_delta":
            self._apply_delta(msg)
        elif msg_type == "trade":
            self._apply_trade(msg)

    def _apply_snapshot(self, msg: dict[str, Any]) -> None:
        ticker = str(msg.get("market_ticker") or msg.get("ticker") or "")
        if not ticker:
            return
        yes, no = parse_orderbook_levels(msg.get("orderbook_fp") or msg)
        self._books[ticker] = {"yes": yes, "no": no}
        self._publish(ticker, source="websocket")

    def _apply_delta(self, msg: dict[str, Any]) -> None:
        ticker = str(msg.get("market_ticker") or msg.get("ticker") or "")
        if not ticker:
            return
        book = self._books.setdefault(ticker, {"yes": [], "no": []})
        side = str(msg.get("side") or "yes").lower()
        price = _price(msg.get("price_dollars") or msg.get("price"))
        delta = int(float(msg.get("delta_fp") or msg.get("delta") or 0))
        if price is None:
            return
        key = "yes" if side == "yes" else "no"
        levels = dict(book[key])
        levels[price] = max(0, int(levels.get(price, 0)) + delta)
        if levels[price] <= 0:
            levels.pop(price, None)
        book[key] = sorted(levels.items(), key=lambda x: -x[0])
        self._publish(ticker, source="websocket")

    def _apply_trade(self, msg: dict[str, Any]) -> None:
        ticker = str(msg.get("market_ticker") or msg.get("ticker") or "")
        if not ticker:
            return
        side = str(msg.get("taker_side") or msg.get("side") or "yes").lower()
        yes_price = _price(msg.get("yes_price_dollars") or msg.get("yes_price"))
        count = int(float(msg.get("count_fp") or msg.get("count") or 0))
        if yes_price is None or count <= 0:
            return
        if side == "yes":
            price = yes_price
            price_cents = int(round(yes_price * 100))
        else:
            price = round(1.0 - yes_price, 4)
            price_cents = int(round(price * 100))
        tape = self._trade_tape.setdefault(ticker, deque(maxlen=200))
        tape.append(TradePrint(ts=time.time(), side=side, price=price, count=count, price_cents=price_cents))
        self._publish(ticker, source="websocket")

    def _bootstrap_book(self, ticker: str) -> None:
        if not self._client:
            self._client = KalshiClient(self.cfg)
        try:
            raw = self._client.get_orderbook(ticker)
            yes, no = parse_orderbook_levels(raw)
            self._books[ticker] = {"yes": yes, "no": no}
        except Exception:
            logger.debug("orderbook bootstrap failed for %s", ticker, exc_info=True)

    def _refresh_rest(self, ticker: str) -> None:
        if not self._client:
            self._client = KalshiClient(self.cfg)
        t0 = time.time()
        try:
            raw = self._client.get_orderbook(ticker)
            yes, no = parse_orderbook_levels(raw)
            self._books[ticker] = {"yes": yes, "no": no}
            latency = (time.time() - t0) * 1000
            self._publish(ticker, source="rest", latency_ms=latency)
            with self._lock:
                self._status.last_message_ts = time.time()
        except Exception as exc:
            with self._lock:
                self._status.last_error = str(exc)[:200]

    def _publish(self, ticker: str, *, source: str, latency_ms: float | None = None) -> None:
        book = self._books.get(ticker) or {"yes": [], "no": []}
        side = self._side_for(ticker)
        trades = self._recent_trades(ticker)
        raw = {
            "orderbook_fp": {
                "yes_dollars": [[f"{p:.4f}", str(q)] for p, q in book["yes"]],
                "no_dollars": [[f"{p:.4f}", str(q)] for p, q in book["no"]],
            }
        }
        pro = build_pro_analytics(
            ticker,
            raw,
            side=side,
            recent_trades=trades,
            source=source,
            latency_ms=latency_ms,
            depth_limit=self._depth_limit,
        )
        with self._lock:
            self._analytics[ticker] = pro

    def _recent_trades(self, ticker: str) -> list[TradePrint]:
        tape = self._trade_tape.get(ticker)
        if not tape:
            return []
        cutoff = time.time() - self._flow_window
        return [t for t in tape if t.ts >= cutoff]

    def _side_for(self, ticker: str) -> str:
        snap = load_snapshot()
        best = snap.get("best_pick") or {}
        if best.get("ticker") == ticker:
            return str(best.get("side") or "yes")
        for pos in snap.get("open_positions") or []:
            if pos.get("ticker") == ticker:
                return str(pos.get("side") or "yes")
        return "yes"

    def _target_tickers(self) -> list[str]:
        snap = load_snapshot()
        tickers: list[str] = []
        best = snap.get("best_pick") or {}
        if best.get("ticker"):
            tickers.append(str(best["ticker"]))
        for row in snap.get("top_markets") or []:
            t = row.get("ticker")
            if t and t not in tickers:
                tickers.append(str(t))
        for pos in snap.get("open_positions") or []:
            t = pos.get("ticker")
            if t and t not in tickers:
                tickers.append(str(t))
        return tickers[:5]

    def _ws_url(self) -> str:
        env = self.cfg.kalshi_env.lower()
        if env == "demo":
            return "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
        return "wss://external-api-ws.kalshi.com/trade-api/ws/v2"


def _price(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        v = float(val)
        return v / 100.0 if v > 1.0 else v
    except (TypeError, ValueError):
        return None


_hub: KalshiProHub | None = None


def get_pro_hub() -> KalshiProHub:
    global _hub
    if _hub is None:
        _hub = KalshiProHub()
    return _hub
