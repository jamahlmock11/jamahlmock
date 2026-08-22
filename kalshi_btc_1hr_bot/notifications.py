"""SMS notifications for live trading events (Twilio)."""

from __future__ import annotations

import base64
import logging
import re

import httpx

from kalshi_btc_1hr_bot.config import NotifyConfig

log = logging.getLogger(__name__)

_E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def normalize_phone(value: str) -> str:
    """Normalize US-style numbers to E.164 (+1XXXXXXXXXX)."""
    raw = (value or "").strip()
    if not raw:
        return ""
    if raw.startswith("+"):
        return raw
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}" if digits else ""


class PhoneNotifier:
    """Send SMS alerts via Twilio REST API."""

    def __init__(self, config: NotifyConfig, *, client: httpx.Client | None = None) -> None:
        self.config = config
        self._client = client
        self._own_client = client is None

    def close(self) -> None:
        if self._own_client and self._client is not None:
            self._client.close()
            self._client = None

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=15.0)
        return self._client

    def send(self, body: str) -> bool:
        if not self.config.configured:
            log.debug("SMS skipped — notifications not configured")
            return False
        to_num = normalize_phone(self.config.phone_to)
        from_num = normalize_phone(self.config.twilio_from)
        if not _E164_RE.match(to_num):
            log.warning("Invalid NOTIFY_PHONE_NUMBER: %s", self.config.phone_to)
            return False
        if not _E164_RE.match(from_num):
            log.warning("Invalid TWILIO_FROM_NUMBER: %s", self.config.twilio_from)
            return False

        text = body.strip()
        if self.config.twilio_trial_template:
            log.info("Twilio trial template SMS (intended: %s)", text[:120])
            text = self.config.twilio_trial_template
        elif len(text) > 1500:
            text = text[:1497] + "..."

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.config.twilio_account_sid}/Messages.json"
        auth = base64.b64encode(
            f"{self.config.twilio_account_sid}:{self.config.twilio_auth_token}".encode()
        ).decode()
        try:
            resp = self._http().post(
                url,
                headers={"Authorization": f"Basic {auth}"},
                data={"To": to_num, "From": from_num, "Body": text},
            )
            if resp.status_code >= 400:
                err = resp.text[:300]
                if not self.config.twilio_trial_template and "572006" in err:
                    log.warning("Twilio trial account — retry with TWILIO_TRIAL_TEMPLATE=sms_appointment_reminders")
                log.error("Twilio SMS failed %s: %s", resp.status_code, err)
                return False
            sid = resp.json().get("sid", "ok")
            log.info("SMS sent to %s sid=%s", to_num[-4:].rjust(len(to_num), "*"), sid)
            return True
        except Exception:
            log.exception("SMS send failed")
            return False

    def notify_trade(
        self,
        *,
        mode: str,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
        finish: str = "",
        edge_cents: float = 0.0,
        order_id: str = "",
    ) -> bool:
        if mode.upper() == "PAPER" and not self.config.notify_on_paper:
            return False
        if not self.config.notify_on_trade:
            return False
        label = "LIVE TRADE" if mode.upper() == "LIVE" else "PAPER TRADE"
        msg = (
            f"KXBTCD 1hr · {label}\n"
            f"BUY {side.upper()} x{contracts} @ {price * 100:.0f}¢\n"
            f"{ticker}\n"
            f"Finish {finish or side.upper()} · edge {edge_cents:.1f}¢"
        )
        if order_id:
            msg += f"\norder {order_id[:12]}"
        return self.send(msg)

    def notify_settlement(
        self,
        *,
        ticker: str,
        side: str,
        won: bool,
        pnl: float,
        result: str,
    ) -> bool:
        if not self.config.notify_on_settlement:
            return False
        outcome = "WIN" if won else "LOSS"
        sign = "+" if pnl >= 0 else ""
        return self.send(
            f"KXBTCD 1hr · SETTLED {outcome}\n"
            f"{ticker} · {side.upper()}\n"
            f"Result {result.upper()} · PnL {sign}${pnl:.2f}"
        )

    def notify_exit(
        self,
        *,
        mode: str,
        ticker: str,
        side: str,
        contracts: int,
        entry_price: float,
        exit_price: float,
        reason: str,
        pnl: float,
        order_id: str = "",
    ) -> bool:
        if mode.upper() == "PAPER" and not self.config.notify_on_paper:
            return False
        if not self.config.notify_on_exit:
            return False
        label = "TAKE PROFIT" if reason == "take_profit" else "STOP LOSS"
        sign = "+" if pnl >= 0 else ""
        msg = (
            f"KXBTCD 1hr · {label}\n"
            f"SELL {side.upper()} x{contracts} @ {exit_price * 100:.0f}¢\n"
            f"{ticker}\n"
            f"Entry {entry_price * 100:.0f}¢ · PnL {sign}${pnl:.2f}"
        )
        if order_id:
            msg += f"\norder {order_id[:12]}"
        return self.send(msg)

    def notify_order_failed(
        self,
        *,
        ticker: str,
        side: str,
        contracts: int,
        price: float,
    ) -> bool:
        if not self.config.notify_on_order_failed:
            return False
        return self.send(
            f"KXBTCD 1hr · ORDER FAILED\n"
            f"BUY {side.upper()} x{contracts} @ {price * 100:.0f}¢\n"
            f"{ticker}"
        )

    def notify_startup(self, *, mode: str, max_trade_usd: float) -> bool:
        return self.send(
            f"KXBTCD 1hr bot online\n"
            f"Mode {mode} · max ${max_trade_usd:.2f}/trade\n"
            f"SMS alerts enabled"
        )
