"""Tests for web dashboard API."""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from kalshi_bot.web.scan_state import GLOBAL_SCAN_STATE, ScanSnapshot
from kalshi_bot.web.web_server import app


def test_api_state_no_data():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_state_with_snapshot():
    GLOBAL_SCAN_STATE.update(
        ScanSnapshot(
            asof=datetime(2026, 1, 1, tzinfo=timezone.utc),
            spot=65000.0,
            spot_source="test",
            balance_usd=1000.0,
            markets_scanned=3,
            opportunities=[
                {
                    "ticker": "KXBTC15M-X",
                    "strategy": "KXBTC15M",
                    "model_yes": 72.0,
                    "decision": "BUY YES",
                    "gates": {
                        "gates": [
                            {"name": "Time to expiry", "status": "pass", "detail": "ok", "side": None}
                        ],
                        "ready_side": "YES",
                        "position_detail": "YES x1",
                        "crowd_yes_pct": 65.0,
                        "crowd_direction": "UP",
                    },
                }
            ],
            calibration=[{"range": "70%-80%", "n_trades": 5, "empirical_win_rate": 0.8, "calibrated": True}],
            freshness={"scan_duration_ms": 120.0, "brti_official": True, "kalshi_ws_connected": True},
        )
    )
    client = TestClient(app)
    data = client.get("/api/state").json()
    assert data["status"] == "ok"
    assert data["spot"] == 65000.0
    assert len(data["opportunities"]) == 1
    assert data["opportunities"][0]["gates"]["ready_side"] == "YES"
    assert data["calibration"][0]["n_trades"] == 5
    assert data["freshness"]["kalshi_ws_connected"] is True
