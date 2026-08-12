"""Kalshi WebSocket client for public trade + orderbook channels."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any
from urllib.parse import urlparse

from kalshi_bot.data.kalshi_client import KalshiClient
from kalshi_bot.data.kalshi_trade_tape import TradeTapeBuffer, normalize_trade

logger = logging.getLogger(__name__)


class KalshiWebSocketFeed:
    """Background asyncio WebSocket subscriber for Kalshi trade prints."""

    def __init__(self, client: KalshiClient, buffer: TradeTapeBuffer) -> None:
        self.client = client
        self.buffer = buffer
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tickers: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._msg_id = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kalshi-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)

    def subscribe(self, tickers: list[str]) -> None:
        self._tickers.update(tickers)
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._send_subscribe(list(tickers)), self._loop)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:
            logger.warning("Kalshi WS thread exited: %s", exc)
        finally:
            self._loop.close()

    async def _main(self) -> None:
        import websockets

        url = self.client.websocket_url()
        headers = self._ws_headers(url)
        while not self._stop.is_set():
            try:
                async with websockets.connect(url, additional_headers=headers, ping_interval=20) as ws:
                    logger.info("Kalshi WebSocket connected: %s", url)
                    if self._tickers:
                        await self._send_subscribe(list(self._tickers), ws=ws)
                    while not self._stop.is_set():
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        except asyncio.TimeoutError:
                            continue
                        self._handle_message(json.loads(raw))
            except Exception as exc:
                logger.warning("Kalshi WS reconnect in 5s: %s", exc)
                await asyncio.sleep(5)

    def _ws_headers(self, url: str) -> dict[str, str]:
        if not self.client.authenticated:
            return {}
        path = urlparse(url).path or "/trade-api/ws/v2"
        ts = str(int(time.time() * 1000))
        sig = self.client._sign(ts, "GET", path)
        return {
            "KALSHI-ACCESS-KEY": self.client.api_key_id or "",
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    async def _send_subscribe(self, tickers: list[str], ws: Any | None = None) -> None:
        if not tickers:
            return
        self._msg_id += 1
        msg = {
            "id": self._msg_id,
            "cmd": "subscribe",
            "params": {"channels": ["trade"], "market_tickers": tickers},
        }
        if ws is not None:
            await ws.send(json.dumps(msg))
        else:
            import websockets

            url = self.client.websocket_url()
            headers = self._ws_headers(url)
            async with websockets.connect(url, additional_headers=headers) as conn:
                await conn.send(json.dumps(msg))

    def _handle_message(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type") or msg.get("msg", {}).get("type")
        payload = msg.get("msg") if "msg" in msg else msg
        if mtype == "trade" or payload.get("type") == "trade":
            trade_body = payload.get("trade") or payload
            ticker = trade_body.get("market_ticker") or trade_body.get("ticker")
            if ticker:
                self.buffer.add(ticker, normalize_trade(trade_body))
