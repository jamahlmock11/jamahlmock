"""FastAPI web dashboard for KXBTC15M mispricing bot."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kalshi_bot.web.scan_state import GLOBAL_SCAN_STATE

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

_ws_clients: list[WebSocket] = []
_scan_stop: threading.Event | None = None
_scan_cleanup: list = []
_app_loop: asyncio.AbstractEventLoop | None = None


def request_broadcast() -> None:
    """Schedule an immediate WebSocket push from a sync scan thread."""
    loop = _app_loop
    if loop is None or not loop.is_running():
        return
    loop.call_soon_threadsafe(lambda: asyncio.create_task(broadcast_state()))


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "run.py").is_file() and (parent / "src" / "kalshi_bot").is_dir():
            return parent
    cwd = Path.cwd()
    if (cwd / "run.py").is_file():
        return cwd
    return None


def _load_run_module(root: Path):
    spec = importlib.util.spec_from_file_location("kalshi_run", root / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load run.py from {root}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _start_platform_scan() -> None:
    """Start production platform scan loop for dashboard data."""
    global _scan_stop, _scan_cleanup

    root = _find_repo_root()
    if root is None:
        return

    try:
        os.chdir(root)
        from kalshi_bot.platform.runner import ProductionPlatform
    except Exception as exc:
        logger.warning("could not import ProductionPlatform: %s", exc)
        return

    _scan_stop = threading.Event()
    resources: dict = {}

    def scan_loop() -> None:
        try:
            platform = ProductionPlatform()
            resources["platform"] = platform
            interval = platform.config.scan_interval_seconds
            while _scan_stop is not None and not _scan_stop.is_set():
                try:
                    platform.run_cycle(execute=False)
                except Exception:
                    logger.exception("platform background scan failed")
                if _scan_stop.wait(interval):
                    break
        except Exception:
            logger.exception("platform scan thread failed to start")

    thread = threading.Thread(target=scan_loop, name="platform-web-scan", daemon=True)
    thread.start()
    _scan_cleanup = [resources]
    logger.info("Background production platform scan started (repo: %s)", root)


def _start_background_scan() -> None:
    """Start scan loop so the dashboard has live data (platform preferred)."""
    global _scan_stop, _scan_cleanup

    root = _find_repo_root()
    if root is None:
        logger.warning(
            "run.py not found — start the dashboard with: python3 platform_run.py --web "
            "(from the repository root)"
        )
        return

    use_platform = os.environ.get("KALSHI_USE_PLATFORM", "1") != "0"
    if use_platform and (root / "platform_run.py").is_file():
        try:
            _start_platform_scan()
            return
        except Exception as exc:
            logger.warning("platform scan unavailable, falling back to run.py: %s", exc)

    try:
        run_mod = _load_run_module(root)
        os.chdir(root)
    except Exception as exc:
        logger.warning("could not start background scan: %s", exc)
        return

    _scan_stop = threading.Event()
    resources: dict = {}

    def scan_loop() -> None:
        try:
            (
                _settings,
                config,
                client,
                _engine,
                scanner,
                executor,
                _btc,
                trade_tape,
                settlement,
            ) = run_mod.build_v6_runtime()
            resources["client"] = client
            resources["trade_tape"] = trade_tape
            sleep_s = config.scan_interval_seconds
            smile = None
            while _scan_stop is not None and not _scan_stop.is_set():
                try:
                    if smile is None:
                        try:
                            from kalshi_bot.data.ibit_options import load_ibit_smile

                            smile = load_ibit_smile(config.smile, allow_synthetic=False)
                        except Exception:
                            smile = None
                    scanner.scan(smile)
                except Exception:
                    logger.exception("background scan failed")
                if _scan_stop.wait(sleep_s):
                    break
        except Exception:
            logger.exception("background scan thread failed to start")

    thread = threading.Thread(target=scan_loop, name="web-scan", daemon=True)
    thread.start()
    _scan_cleanup = [resources]
    logger.info("Background mispricing scan started (repo: %s)", root)


def _stop_background_scan() -> None:
    global _scan_stop, _scan_cleanup
    if _scan_stop is not None:
        _scan_stop.set()
    for resources in _scan_cleanup:
        platform = resources.get("platform")
        tape = resources.get("trade_tape")
        client = resources.get("client")
        if platform is not None:
            try:
                platform.close()
            except Exception:
                pass
        if tape is not None:
            try:
                tape.close()
            except Exception:
                pass
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    _scan_stop = None
    _scan_cleanup = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_loop
    _app_loop = asyncio.get_running_loop()
    if getattr(app.state, "with_scan", True):
        _start_background_scan()
    yield
    _stop_background_scan()
    _app_loop = None


app = FastAPI(title="Kalshi Production Platform", version="2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
async def api_state():
    return GLOBAL_SCAN_STATE.to_dict()


@app.get("/api/health")
async def health():
    snap = GLOBAL_SCAN_STATE.get()
    safety = snap.safety if snap else {}
    freshness = snap.freshness if snap else {}
    status_label = safety.get("status_label") or "DISABLED"
    live = status_label == "LIVE"
    return {
        "status": "ok",
        "has_data": snap is not None,
        "markets_scanned": snap.markets_scanned if snap else 0,
        "live_trading_enabled": live,
        "status_label": status_label,
        "block_reason": safety.get("block_reason"),
        "balance_usd": snap.balance_usd if snap else None,
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        await websocket.send_json(GLOBAL_SCAN_STATE.to_dict())
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
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


def run_server(*, host: str = "0.0.0.0", port: int = 8080, with_scan: bool = True) -> None:
    import uvicorn

    root = _find_repo_root()
    if root is not None:
        os.chdir(root)

    logger.info(
        "Production dashboard listening on http://%s:%d — "
        "in Cloud Agent, open the forwarded port URL from the Ports panel",
        host if host != "0.0.0.0" else "localhost",
        port,
    )
    app.state.with_scan = with_scan
    uvicorn.run(app, host=host, port=port, log_level="info")
