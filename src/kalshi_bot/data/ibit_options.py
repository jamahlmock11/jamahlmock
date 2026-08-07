"""IBIT ETF option chain → BTC-space vol smile."""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from kalshi_bot.config import SmileConfig
from kalshi_bot.models.black_scholes import implied_vol_from_price
from kalshi_bot.models.smile import VolSmile, build_smile_from_ibit_chain, synthetic_smile

logger = logging.getLogger(__name__)


def _mid(bid: float, ask: float, last: float) -> float | None:
    if bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (bid + ask)
    if last > 0:
        return last
    return None


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def _years_to_expiry_date(expiry: str, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    exp = datetime.strptime(expiry, "%Y-%m-%d").replace(
        hour=20, minute=0, second=0, tzinfo=timezone.utc
    )
    return max((exp - now).total_seconds(), 60.0) / (365.25 * 24 * 3600)


def fetch_spot_pair(symbol: str = "IBIT") -> tuple[float, float]:
    ibit = yf.Ticker(symbol)
    btc = yf.Ticker("BTC-USD")
    ibit_px = (
        ibit.info.get("regularMarketPrice")
        or ibit.info.get("previousClose")
        or float(ibit.fast_info["lastPrice"])
    )
    btc_px = (
        btc.info.get("regularMarketPrice")
        or btc.info.get("previousClose")
        or float(btc.fast_info["last_price"])
    )
    return float(ibit_px), float(btc_px)


def _extract_points_from_chain(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot_ibit: float,
    t_years: float,
    cfg: SmileConfig,
) -> tuple[list[float], list[float], list[float]]:
    strikes: list[float] = []
    ivs: list[float] = []
    weights: list[float] = []

    def consider(row: pd.Series, is_call: bool) -> None:
        strike = _finite(row.get("strike"))
        if strike <= 0:
            return
        bid = _finite(row.get("bid"))
        ask = _finite(row.get("ask"))
        last = _finite(row.get("lastPrice"))
        oi = int(_finite(row.get("openInterest")))
        vol_n = int(_finite(row.get("volume")))
        mid = _mid(bid, ask, last)
        quoted_iv = _finite(row.get("impliedVolatility"))

        # Prefer OTM options
        if is_call and strike < spot_ibit * 0.98:
            return
        if (not is_call) and strike > spot_ibit * 1.02:
            return

        # Penny last prints on far OTM options produce garbage IVs after hours
        moneyness = abs(math.log(strike / spot_ibit))
        if (bid <= 0 or ask <= 0) and last <= 0.02 and moneyness > 0.04:
            return

        iv = None
        if mid is not None and mid > 0.02:
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / max(mid, 1e-6)
                if spread_pct > cfg.max_spread_pct:
                    return
            # Invert from mid/last — more trustworthy than Yahoo placeholders
            iv = implied_vol_from_price(
                mid,
                spot_ibit,
                strike,
                t_years,
                rate=cfg.risk_free_rate,
                dividend=cfg.dividend_yield,
                is_call=is_call,
            )

        # Only accept quoted IV if inversion failed and quote looks real
        if (iv is None or iv < 0.15) and 0.20 <= quoted_iv <= 1.80:
            if oi >= cfg.min_oi or vol_n >= 1:
                iv = quoted_iv

        if iv is None or iv < 0.15 or iv > 2.5:
            return

        weight = math.sqrt(max(oi, 1) + max(vol_n, 0))
        # Down-weight pure last-trade inversions without a live book
        if bid <= 0 or ask <= 0:
            weight *= 0.5

        strikes.append(strike)
        ivs.append(float(iv))
        weights.append(weight)

    for _, row in calls.iterrows():
        consider(row, True)
    for _, row in puts.iterrows():
        consider(row, False)
    return strikes, ivs, weights


def smile_quality_ok(smile: VolSmile) -> bool:
    if len(smile.points) < 4:
        return False
    if not (0.25 <= smile.atm_iv <= 1.80):
        return False
    # Reject placeholder-dominated smiles (many identical IVs)
    rounded = [round(p.iv, 3) for p in smile.points]
    most_common_n = Counter(rounded).most_common(1)[0][1]
    if most_common_n / len(rounded) > 0.55:
        return False
    return True


def load_ibit_smile(cfg: SmileConfig, allow_synthetic: bool = False) -> VolSmile:
    """Build BTC-space smile from live IBIT options, falling back to cache."""
    cache = Path(cfg.cache_path)
    try:
        spot_ibit, spot_btc = fetch_spot_pair(cfg.symbol)
        ticker = yf.Ticker(cfg.symbol)
        expiries = list(ticker.options or [])
        if not expiries:
            raise RuntimeError("no IBIT expiries available")

        best: VolSmile | None = None
        best_score = -1.0
        for expiry in expiries[: cfg.prefer_expiries]:
            t_years = _years_to_expiry_date(expiry)
            # Skip near-expired 0DTE outside RTH when T is tiny — noisy tape
            if t_years < 0.2 / 365:
                continue
            chain = ticker.option_chain(expiry)
            strikes, ivs, weights = _extract_points_from_chain(
                chain.calls, chain.puts, spot_ibit, t_years, cfg
            )
            if len(strikes) < 4:
                continue
            smile = build_smile_from_ibit_chain(
                strikes_ibit=strikes,
                ivs=ivs,
                weights=weights,
                spot_ibit=spot_ibit,
                spot_btc=spot_btc,
                expiry=expiry,
                t_years=t_years,
            )
            if not smile_quality_ok(smile):
                logger.debug("rejecting low-quality smile expiry=%s atm=%.3f", expiry, smile.atm_iv)
                continue
            # Prefer nearer expiries with more points
            score = len(smile.points) / math.sqrt(max(t_years, 1e-4)) + smile.atm_iv
            if score > best_score:
                best_score = score
                best = smile

        if best is None:
            raise RuntimeError("could not build quality IBIT smile from available chains")

        best.save(cache)
        logger.info(
            "IBIT smile built expiry=%s points=%d atm_iv=%.1f%% spot_btc=%.2f ratio=%.6g",
            best.expiry,
            len(best.points),
            best.atm_iv * 100,
            best.spot_btc,
            best.btc_per_share,
        )
        return best
    except Exception as exc:
        logger.warning("live IBIT smile failed: %s", exc)
        if cache.exists():
            cached = VolSmile.load(cache)
            if smile_quality_ok(cached) or cached.atm_iv >= 0.25:
                logger.warning(
                    "using cached smile age=%.0fs atm_iv=%.1f%%",
                    cached.age_seconds,
                    cached.atm_iv * 100,
                )
                try:
                    spot_ibit, spot_btc = fetch_spot_pair(cfg.symbol)
                    cached.spot_btc = spot_btc
                    cached.spot_ibit = spot_ibit
                    cached.btc_per_share = spot_ibit / spot_btc
                except Exception:
                    pass
                return cached
        # Optional Deribit 24/7 BTC smile when IBIT tape is dark
        try:
            from kalshi_bot.data.deribit_options import load_deribit_btc_smile

            deribit = load_deribit_btc_smile(spot_btc=None)
            logger.warning(
                "using Deribit BTC smile (IBIT unavailable) atm_iv=%.1f%%",
                deribit.atm_iv * 100,
            )
            return deribit
        except Exception as der_exc:
            logger.warning("Deribit fallback failed: %s", der_exc)
        if allow_synthetic:
            spot_btc = 65000.0
            try:
                _, spot_btc = fetch_spot_pair(cfg.symbol)
            except Exception:
                pass
            logger.warning("using synthetic smile for demo/testing only")
            return synthetic_smile(spot_btc, atm_iv=0.55)
        raise
