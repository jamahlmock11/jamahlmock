# Kalshi BTC 1-Hour Forecasting Bot

A Python framework for forecasting Kalshi's `KXBTCD` hourly binary contracts —
"Will BTC close above the strike at the top of the hour?" — using a 5-layer
ensemble probability model with multi-timeframe momentum, funding rates,
mean reversion, and volatility regime detection.

Same settlement mechanism as the 15-min bot (60-second BRTI average), but the
longer window enables a richer signal set.

---

## Quickstart (Cursor IDE)

1. **Open in Cursor**: File → Open Folder → select `kalshi_btc_1hr_bot/`
2. **Set up environment**: Open Cursor terminal (Ctrl+`) and run:
   ```bash
   make setup
   source .venv/bin/activate
   ```
3. **Run the synthetic backtest** (no API keys needed):
   - Press F5 → pick "Backtest (synthetic)" — OR — `make backtest`
4. **Paper trade** (single cycle): `make paper-once`
5. **Paper trade continuously**: `make paper`

### Cursor Features
- `.cursorrules` gives Cursor's AI full project context
- `.vscode/launch.json` has 5 debug configurations
- `.vscode/settings.json` auto-configures Python + formatting

### Command line (any editor)
```bash
pip install -r requirements.txt
python backtest.py --synthetic --n-markets 100   # synthetic backtest
python bot.py --paper --once                       # single paper trade cycle
python bot.py --paper                              # continuous paper trading
```

---

## How It Differs From the 15-Minute Bot

| Feature | 15-min bot | 1-hour bot |
|---------|-----------|------------|
| Series | KXBTC15M | KXBTCD |
| Window | 900s | 3600s |
| Model | GBM + OBI + momentum | 5-layer ensemble (see below) |
| Momentum | Single 90s | Multi-TF: 5m + 15m + 30m |
| Funding rate | No | Yes (Binance perp) |
| Mean reversion | No | Yes (VWAP pullback) |
| Vol regime | No | Yes (low/med/high) |
| Kelly fraction | 25% | 20% (more conservative) |
| Cycle | 2s | 5s (less latency-critical) |
| Settlement | 60s BRTI avg | 60s BRTI avg (same) |

---

## The 5-Layer Model

1. **GBM core** — lognormal probability with averaging adjustment
2. **Multi-timeframe momentum** — weighted blend of 5m/15m/30m drift
3. **Funding rate signal** — BTC perp funding as sentiment proxy
4. **Mean reversion** — pullback toward VWAP
5. **Volatility regime** — confidence adjustment based on vol classification

Final probability is calibrated via logistic regression on 18 features.

See `STRATEGY_GUIDE.md` for full details.

---

## Files

```
kalshi_btc_1hr_bot/
├── README.md
├── STRATEGY_GUIDE.md
├── requirements.txt
├── config.py          ← KXBTCD, 3600s window, hourly-tuned params
├── kalshi_client.py   ← Kalshi API (auth, markets, orders)
├── data_feed.py       ← Binance WS + funding rate + BRTI
├── model.py           ← 5-layer ensemble forecast model
├── edge.py            ← EV calculation
├── sizing.py          ← Fractional Kelly
├── risk.py            ← Bankroll caps, drawdown stop
├── backtest.py        ← Synthetic + real data backtester
├── bot.py             ← Main paper/live loop
├── utils.py
├── Makefile
├── .cursorrules
├── .env.example
└── .vscode/
    ├── settings.json
    ├── launch.json
    └── extensions.json
```

---

## Honest Expectations

75-80% win rate is not realistic. The target is positive EV with calibrated
probabilities. Trade with money you can afford to lose. Start in paper mode.
This is educational engineering, not financial advice.
