"""Canonical trade decision states for both strategies."""

from __future__ import annotations

from enum import Enum


class TradeSignal(str, Enum):
    STRONG_BUY_YES = "STRONG BUY YES"
    BUY_YES = "BUY YES"
    WAIT = "WAIT"
    NO_TRADE = "NO TRADE"
    BUY_NO = "BUY NO"
    STRONG_BUY_NO = "STRONG BUY NO"
    EXIT = "EXIT"
    DATA_ERROR = "DATA ERROR"


def signal_from_action(action: str, *, net_edge: float, strong_edge: float = 0.15) -> TradeSignal:
    """Map legacy action strings to canonical signals."""
    a = (action or "").upper().replace("_", " ")
    if "DATA" in a and "ERROR" in a:
        return TradeSignal.DATA_ERROR
    if a in ("EXIT",):
        return TradeSignal.EXIT
    if "BUY YES" in a or a == "TRADE YES":
        return TradeSignal.STRONG_BUY_YES if net_edge >= strong_edge else TradeSignal.BUY_YES
    if "BUY NO" in a or a == "TRADE NO":
        return TradeSignal.STRONG_BUY_NO if net_edge >= strong_edge else TradeSignal.BUY_NO
    if a == "WAIT":
        return TradeSignal.WAIT
    return TradeSignal.NO_TRADE
