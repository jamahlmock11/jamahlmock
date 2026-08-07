"""Deribit BTC options smile — 24/7 fallback when IBIT tape is dark."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone

import httpx

from kalshi_bot.models.smile import SmilePoint, VolSmile

logger = logging.getLogger(__name__)

DERIBIT_SUMMARY = (
    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
)


def load_deribit_btc_smile(spot_btc: float | None = None) -> VolSmile:
    """Build a BTC-space smile from Deribit instrument mark IVs."""
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(DERIBIT_SUMMARY, params={"currency": "BTC", "kind": "option"})
        resp.raise_for_status()
        payload = resp.json()
    rows = payload.get("result") or []
    if not rows:
        raise RuntimeError("empty Deribit option summary")

    by_expiry: dict[int, list[dict]] = {}
    for row in rows:
        name = row.get("instrument_name") or ""
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            exp_date = datetime.strptime(parts[1], "%d%b%y").replace(
                hour=8, minute=0, second=0, tzinfo=timezone.utc
            )
        except ValueError:
            continue
        by_expiry.setdefault(int(exp_date.timestamp()), []).append(row)

    now = time.time()
    best_exp = None
    best_rows: list[dict] = []
    for exp_ts, group in sorted(by_expiry.items()):
        if exp_ts < now + 6 * 3600:
            continue
        if len(group) >= 10:
            best_exp = exp_ts
            best_rows = group
            break
    if not best_rows:
        best_exp, best_rows = max(by_expiry.items(), key=lambda kv: len(kv[1]))

    spots = [float(r["underlying_price"]) for r in best_rows if r.get("underlying_price")]
    spot = float(spot_btc) if spot_btc else (sum(spots) / len(spots) if spots else 0.0)
    if spot <= 0:
        raise RuntimeError("could not infer BTC spot from Deribit")

    points: list[SmilePoint] = []
    for row in best_rows:
        name = row.get("instrument_name") or ""
        parts = name.split("-")
        try:
            strike = float(parts[2])
            option_type = parts[-1]
        except (IndexError, ValueError):
            continue
        mark_iv = row.get("mark_iv")
        if mark_iv is None:
            continue
        iv = float(mark_iv) / 100.0
        if iv < 0.15 or iv > 2.5:
            continue
        if option_type == "C" and strike < spot * 0.98:
            continue
        if option_type == "P" and strike > spot * 1.02:
            continue
        points.append(
            SmilePoint(
                strike_btc=strike,
                log_moneyness=math.log(strike / spot),
                iv=iv,
                weight=1.0 + float(row.get("open_interest") or 0),
            )
        )

    if len(points) < 4:
        raise RuntimeError(f"insufficient Deribit smile points: {len(points)}")

    by_k: dict[float, list[SmilePoint]] = {}
    for p in points:
        by_k.setdefault(round(p.strike_btc, 2), []).append(p)
    merged: list[SmilePoint] = []
    for _, group in sorted(by_k.items()):
        iv = sum(p.iv for p in group) / len(group)
        p0 = group[0]
        merged.append(
            SmilePoint(
                strike_btc=p0.strike_btc,
                log_moneyness=math.log(p0.strike_btc / spot),
                iv=iv,
                weight=float(len(group)),
            )
        )

    t_years = max((best_exp or now) - now, 3600) / (365.25 * 24 * 3600)
    expiry = datetime.fromtimestamp(best_exp or now, tz=timezone.utc).strftime("%Y-%m-%d")
    smile = VolSmile(
        asof_ts=time.time(),
        spot_btc=spot,
        spot_ibit=spot * 0.00056,
        btc_per_share=0.00056,
        expiry=f"deribit:{expiry}",
        t_years=t_years,
        points=merged,
        is_synthetic=False,
    )
    smile.atm_iv = smile.iv_at_strike(spot)
    logger.info(
        "Deribit smile expiry=%s points=%d atm_iv=%.1f%% spot=%.2f",
        smile.expiry,
        len(smile.points),
        smile.atm_iv * 100,
        spot,
    )
    return smile
