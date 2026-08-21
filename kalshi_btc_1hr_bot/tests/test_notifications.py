"""SMS notification tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from kalshi_btc_1hr_bot.config import NotifyConfig
from kalshi_btc_1hr_bot.notifications import PhoneNotifier, normalize_phone


def test_normalize_phone_us_10_digit():
    assert normalize_phone("5551234567") == "+15551234567"
    assert normalize_phone("(555) 123-4567") == "+15551234567"
    assert normalize_phone("+15551234567") == "+15551234567"


def test_notifier_skips_when_not_configured():
    notifier = PhoneNotifier(NotifyConfig(enabled=False))
    assert notifier.send("hello") is False


def test_notifier_sends_via_twilio():
    cfg = NotifyConfig(
        enabled=True,
        phone_to="+15551234567",
        twilio_account_sid="ACtest",
        twilio_auth_token="secret",
        twilio_from="+15559876543",
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"sid": "SM123"}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    notifier = PhoneNotifier(cfg, client=mock_client)
    assert notifier.notify_trade(
        mode="LIVE",
        ticker="KXBTCD-TEST",
        side="yes",
        contracts=1,
        price=0.45,
        finish="ABOVE",
        edge_cents=3.5,
        order_id="ord-123",
    )
    mock_client.post.assert_called_once()
    call = mock_client.post.call_args
    assert "Accounts/ACtest/Messages.json" in call.args[0]
    assert call.kwargs["data"]["To"] == "+15551234567"
    assert "LIVE TRADE" in call.kwargs["data"]["Body"]


def test_notify_settlement_message():
    cfg = NotifyConfig(
        enabled=True,
        phone_to="+15551234567",
        twilio_account_sid="ACtest",
        twilio_auth_token="secret",
        twilio_from="+15559876543",
    )
    with patch.object(PhoneNotifier, "send", return_value=True) as send:
        notifier = PhoneNotifier(cfg, client=MagicMock())
        notifier.notify_settlement(ticker="T", side="no", won=True, pnl=0.37, result="no")
    assert send.called
    body = send.call_args.args[0]
    assert "WIN" in body
    assert "+$0.37" in body
