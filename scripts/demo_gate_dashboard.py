#!/usr/bin/env python3
"""Seed dashboard with sample gate data and start web server (no live scan)."""

from datetime import datetime, timezone

from kalshi_bot.strategy.trade_gates import evaluate_trade_gates
from kalshi_bot.web.scan_state import GLOBAL_SCAN_STATE, ScanSnapshot
from kalshi_bot.web.web_server import run_server

gates = evaluate_trade_gates(
    model_prob_yes=0.42,
    yes_net_ev=-0.327,
    no_net_ev=0.277,
    yes_ask=0.73,
    yes_bid=0.71,
    no_ask=0.27,
    seconds_to_expiry=7.1 * 60,
    uncertainty_pct=12.2,
    bucket_overrides={"7_5": {"min_net_edge_dollars": 0.122}},
)

GLOBAL_SCAN_STATE.update(
    ScanSnapshot(
        asof=datetime.now(timezone.utc),
        spot=71926.40,
        spot_source="demo",
        balance_usd=4.0,
        markets_scanned=1,
        opportunities_15m=[
            {
                "strategy": "KXBTC15M",
                "ticker": "KXBTC15M-DEMO",
                "btc": 71926.40,
                "strike": 71879.04,
                "seconds_to_expiry": 7.1 * 60,
                "model_yes": 42.0,
                "model_no": 58.0,
                "yes_ask": 0.73,
                "no_ask": 0.27,
                "net_edge": 0.277,
                "confidence": "52%",
                "price_pattern": "Drift",
                "decision": "WAIT",
                "reason": "gates not cleared",
                "gates": gates.to_dict(),
            }
        ],
        freshness={"scan_age_seconds": 1, "kalshi_ws_connected": False, "brti_official": True},
        safety={"status_label": "DISABLED", "api_connected": False, "market_data_connected": False},
    )
)

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8090, with_scan=False)
