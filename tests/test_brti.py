"""Tests for CF Benchmarks BRTI resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kalshi_bot.config import BrtiConfig
from kalshi_bot.data.brti import resolve_spot
from kalshi_bot.data.cfbenchmarks import BrtiQuote, fetch_brti_public_summary, parse_brti_payload


def test_parse_brti_payload_kalshi_envelope():
    data = {
        "data": {
            "serverTime": "2026-08-09T13:00:00.000Z",
            "payload": [
                {"time": "2026-08-09T12:59:59.000Z", "value": 64900.0},
                {"time": "2026-08-09T13:00:00.000Z", "value": 65041.59},
            ],
        }
    }
    assert parse_brti_payload(data) == 65041.59


def test_parse_brti_payload_public_summary():
    assert parse_brti_payload({"value": "65041.59"}) == 65041.59


def test_resolve_spot_prefers_public_cf_benchmarks():
    client = MagicMock()
    client.authenticated = False
    client.get_brti.return_value = None
    quote = BrtiQuote(value=65041.59, source="cfbenchmarks_public_rti", is_official=True)

    with patch("kalshi_bot.data.brti.fetch_brti_public_summary", return_value=quote):
        snap = resolve_spot(client, brti_cfg=BrtiConfig())

    assert snap.brti == 65041.59
    assert snap.is_official is True
    assert snap.source == "cfbenchmarks_public_rti"


def test_fetch_brti_public_summary_live():
    quote = fetch_brti_public_summary()
    assert quote is not None
    assert quote.value > 1000
    assert quote.is_official is True
