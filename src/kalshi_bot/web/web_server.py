"""FastAPI web dashboard for KXBTC15M mispricing bot."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kalshi_bot.web.scan_state import GLOBAL_SCAN_STATE

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="KXBTC15M Mispricing Dashboard", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_ws_clients: list[WebSocket] = []


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def api_state():
    return GLOBAL_SCAN_STATE.to_dict()


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        await websocket.send_json(GLOBAL_SCAN_STATE.to_dict())
        while True:
            await asyncio.sleep(2)
            await websocket.send_json(GLOBAL_SCAN_STATE.to_dict())
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def broadcast_state() -> None:
    payload = GLOBAL_SCAN_STATE.to_dict()
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def run_server(*, host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
