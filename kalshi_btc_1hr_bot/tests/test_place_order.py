"""Tests for Kalshi order placement."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from kalshi_btc_1hr_bot.config import BotConfig, load_config
from kalshi_btc_1hr_bot.kalshi_client import KalshiClient


def _mock_response(status: int, json_data: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.text = text or str(json_data)
    resp.raise_for_status.side_effect = None if status < 400 else httpx.HTTPStatusError(
        "error", request=MagicMock(), response=resp
    )
    return resp


def test_place_order_uses_v2_events_api():
    cfg = BotConfig(
        kalshi_env="prod",
        kalshi_api_key_id="test-key",
        kalshi_private_key_pem="",
    )
    client = KalshiClient(cfg)
    client._private_key = None  # unsigned for unit test

    v2_resp = _mock_response(200, {"order": {"order_id": "ord-123"}})

    with patch.object(client._client, "request", return_value=v2_resp) as mock_req:
        result = client.place_order(
            ticker="KXBTCD-TEST",
            side="no",
            action="buy",
            count=1,
            price_cents=8,
        )

    assert result["order"]["order_id"] == "ord-123"
    call_args = mock_req.call_args
    assert "/portfolio/events/orders" in call_args[0][1]
    body = call_args[1]["json"]
    assert body["side"] == "ask"
    assert body["price"] == "0.9200"  # complementary YES for NO @ 8¢
    assert body["count"] == "1.00"


def test_place_order_falls_back_to_legacy():
    cfg = BotConfig(kalshi_env="prod", kalshi_api_key_id="test-key", kalshi_private_key_pem="")
    client = KalshiClient(cfg)
    client._private_key = None

    v2_fail = _mock_response(400, text='{"error":"bad"}')
    legacy_ok = _mock_response(200, {"order_id": "legacy-1"})

    with patch.object(client._client, "request", side_effect=[v2_fail, legacy_ok]) as mock_req:
        result = client.place_order(
            ticker="KXBTCD-TEST",
            side="yes",
            action="buy",
            count=1,
            price_cents=40,
        )

    assert result["order_id"] == "legacy-1"
    assert mock_req.call_count == 2
    legacy_body = mock_req.call_args_list[1][1]["json"]
    assert legacy_body["yes_price"] == 40
