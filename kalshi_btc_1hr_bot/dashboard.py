"""FastAPI web dashboard for the KXBTCD 1-hour bot."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kalshi_btc_1hr_bot.config import load_config
from kalshi_btc_1hr_bot.dashboard_state import STATE_PATH, load_snapshot
from kalshi_btc_1hr_bot.kalshi_client import KalshiClient
from kalshi_btc_1hr_bot.trade_journal import TradeJournal

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"
_ws_clients: set[WebSocket] = set()
_settlement_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _trade_to_dict(trade: Any) -> dict[str, Any]:
    from datetime import datetime, timezone

    opened = datetime.fromtimestamp(trade.opened_ts, tz=timezone.utc).isoformat()
    settled = (
        datetime.fromtimestamp(trade.settled_ts, tz=timezone.utc).isoformat()
        if trade.settled_ts
        else None
    )
    return {
        "id": trade.id,
        "opened_at": opened,
        "settled_at": settled,
        "ticker": trade.ticker,
        "side": trade.side.upper(),
        "finish": trade.finish,
        "contracts": trade.contracts,
        "entry_price": trade.entry_price,
        "cost_usd": round(trade.cost_usd, 4),
        "mode": trade.mode,
        "order_id": trade.order_id,
        "passed": trade.passed,
        "block_reason": trade.block_reason,
        "edge_cents": round(trade.edge_cents, 2),
        "evidence_score": round(trade.evidence_score, 4),
        "p_fair": round(trade.p_fair, 4),
        "confidence": round(trade.confidence, 3),
        "strike": trade.strike,
        "spot": trade.spot,
        "settled": trade.settled,
        "won": trade.won,
        "pnl_usd": round(trade.pnl, 4) if trade.pnl is not None else None,
        "result": trade.result,
        "status": _trade_status(trade),
    }


def _trade_status(trade: Any) -> str:
    if not trade.passed:
        return "BLOCKED"
    if not trade.settled:
        return "OPEN"
    if trade.won:
        return "WIN"
    return "LOSS"


def build_api_payload() -> dict[str, Any]:
    journal = TradeJournal()
    snapshot = load_snapshot()
    trades = [_trade_to_dict(t) for t in journal.list_trades(limit=200)]
    cycles = journal.list_cycles(limit=30)
    stats = journal.stats()
    return {
        "snapshot": snapshot,
        "stats": stats,
        "trades": trades,
        "cycles": [
            {
                "ts": c["ts"],
                "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(c["ts"])),
                "mode": c["mode"],
                "status": c["status"],
                "markets_scanned": c["markets_scanned"],
                "candidates": c["candidates"],
                "best_ticker": c["best_ticker"],
                "best_action": c["best_action"],
                "reason": c["reason"],
                "selected": bool(c["selected"]),
                "spot": c["spot"],
                "readiness_pct": c["readiness_pct"],
            }
            for c in cycles
        ],
        "state_age_s": _state_age(snapshot),
    }


def _state_age(snapshot: dict[str, Any]) -> float | None:
    updated = snapshot.get("updated_at")
    if not updated:
        return None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        return round(time.time() - dt.timestamp(), 1)
    except ValueError:
        return None


def _settlement_loop() -> None:
    cfg = load_config()
    journal = TradeJournal()
    while not _stop_event.is_set():
        try:
            if cfg.kalshi_api_key_id and cfg.kalshi_private_key_pem:
                client = KalshiClient(cfg)
                try:
                    resolved = journal.poll_settlements(client)
                    if resolved:
                        snap = load_snapshot()
                        snap["recent_settlements"] = resolved + snap.get("recent_settlements", [])[:10]
                        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                        STATE_PATH.write_text(json.dumps(snap, indent=2))
                finally:
                    client.close()
        except Exception:
            logger.exception("settlement poll failed")
        _stop_event.wait(30.0)


async def _broadcast(payload: dict[str, Any]) -> None:
    dead: list[WebSocket] = []
    data = json.dumps(payload)
    for ws in list(_ws_clients):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _push_loop() -> None:
    while True:
        await _broadcast(build_api_payload())
        await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _settlement_thread
    _stop_event.clear()
    _settlement_thread = threading.Thread(target=_settlement_loop, daemon=True)
    _settlement_thread.start()
    push_task = asyncio.create_task(_push_loop())
    yield
    _stop_event.set()
    push_task.cancel()
    try:
        await push_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="KXBTCD 1-Hour Command Center", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def api_state() -> dict[str, Any]:
    return build_api_payload()


@app.get("/api/health")
async def api_health() -> dict[str, str]:
    snap = load_snapshot()
    age = _state_age(snap)
    stale = age is None or age > 30
    return {
        "status": "ok" if not stale else "stale",
        "state_age_s": str(age) if age is not None else "unknown",
    }


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_text(json.dumps(build_api_payload()))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(__import__("os").getenv("DASHBOARD_PORT", "8090"))
    logger.info("KXBTCD dashboard → http://0.0.0.0:%d", port)
    uvicorn.run(
        "kalshi_btc_1hr_bot.dashboard:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
