"""Integration tests for TP/SL position management."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from kalshi_btc_1hr_bot.bot import HourlyBot
from kalshi_btc_1hr_bot.config import BotConfig, ExitConfig
from kalshi_btc_1hr_bot.trade_journal import TradeJournal


def test_close_trade_early_marks_settled(tmp_path):
    journal = TradeJournal(path=tmp_path / "trades.db")
    trade_id = journal.record_trade(
        ticker="KXBTCD-TEST",
        side="yes",
        contracts=1,
        entry_price=0.40,
        mode="PAPER",
        tp_price=0.60,
        sl_price=0.24,
    )
    journal.close_trade_early(
        trade_id,
        exit_price=0.62,
        exit_reason="take_profit",
        pnl=0.22,
        exit_order_id="sell-1",
    )
    trades = journal.list_trades(settled=True, limit=1)
    assert len(trades) == 1
    assert trades[0].closed_early
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].pnl == 0.22


def test_manage_open_positions_triggers_take_profit(tmp_path):
    cfg = BotConfig(paper=True)
    cfg.exit = ExitConfig(enabled=True, take_profit_pct=0.50, stop_loss_pct=0.40, min_hold_seconds=0)
    journal = TradeJournal(path=tmp_path / "trades.db")
    trade_id = journal.record_trade(
        ticker="KXBTCD-TP",
        side="yes",
        contracts=1,
        entry_price=0.40,
        mode="PAPER",
        tp_price=0.60,
        sl_price=0.24,
    )
    bot = HourlyBot(cfg)
    bot.journal = journal
    bot.risk.register_trade("KXBTCD-TP", 0.40)
    bot.client = MagicMock()
    bot.client._request.return_value = {
        "market": {
            "ticker": "KXBTCD-TP",
            "yes_bid_dollars": "0.6500",
            "yes_ask_dollars": "0.6800",
        }
    }
    bot.notifier = MagicMock()
    exits = bot._manage_open_positions()
    assert len(exits) == 1
    assert exits[0]["reason"] == "take_profit"
    settled = journal.list_trades(settled=True, limit=1)[0]
    assert settled.id == trade_id
    assert settled.closed_early
    assert bot.risk.state.open_positions == 0
