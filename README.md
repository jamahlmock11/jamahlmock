# Kalshi BTC Forecasting Engine

Institutional-grade expected-value engine for Kalshi **KXBTCD** (Bitcoin 1-hour) and **KXBTC15M** (15-minute) markets.

**Objective:** maximize long-term EV.  
**Constraint:** accuracy > trade frequency.  
**Default action:** **NO TRADE** when evidence is insufficient.

---

## Arbitrary policy (both bots)

Independent judgment — does **not** blindly follow the market favorite:

- Evaluates **both YES and NO** every cycle
- **Fades overpriced favorites** (NO TRADE on favorite; can trade the other side)
- Buys **underpriced underdogs** when calibrated edge clears the bar
- **Time-weighted edge** — more time remaining requires a larger edge
- **Edge chase guard** — won't enter after edge decays or price runs away
- **Probability calibration** — uncalibrated model output is shrunk toward 50%; conditional entries require calibration history

Configured in `config/default.yaml` under `arbitrary:`.

---

The **V6 workflow** (`run.py`) targets **KXBTC15M** with microstructure, multi-model ensemble, Monte Carlo (5000 sims), regime detection, manipulation detection, probability calibration, pattern matching, and confidence-scaled Kelly sizing.

### STRICT EDGE RULE (hard filter)

Only recommend **BUY** when the market price is **≥20–25¢ below** the model probability. No exceptions for A/B setups.

| Model prob | Max market YES | Gap |
|---|---|---|
| 60% | ≤35¢ | 25¢ |
| 60% | ≤40¢ | 20¢ |
| 60% | 45¢+ | **NO TRADE** |

```bash
python run.py                  # single V6 scan
python run.py --loop -n 30     # 30-cycle loop
python run.py --strict 0.25    # extra-strict 25¢ floor
kalshi-intelligence            # same via entrypoint
```

V6 features: live bid/ask imbalance, order book depth (top 10), whale detection, spread dynamics, trade velocity, liquidity score, VWAP distance, short-term momentum, volatility expansion, support/resistance, breakout/fake-breakout detection, time features, multi-model agreement (gradient boosting, logistic, neural net, time-series), Do Not Trade score, historical pattern matching, kill switch, institutional flow, strike gravity, explainability score, and continuous learning journal.

Calibration requires **≥3 trades per probability bucket** before adjustments apply.

---

## Decision policy

For each open hourly contract “YES if BRTI settles ≥ K”:

1. Build an **ensemble probability** from:
   - IBIT options smile → BTC log-moneyness → short-horizon digital (blended with realized vol)
   - Physical-measure digital from **recent realized vol** (Kraken 1m OHLC)
   - Conservative vol-anchor digital (μ = 0)
2. Measure **disagreement** and **confidence**
3. Classify the **raw gap** `gap_pp = (model − market) × 100` into a bot action tier:
   - **≥20pp** → Strong BUY candidate
   - **≥15pp** → Only if other signals confirm (elevated confidence / low disagreement)
   - **<15pp** → No trade
4. Compute **conservative post-fee EV** (uses forecast uncertainty band, not the mean alone)
5. Apply hard gates (spread, volume, **last-20-minute entry window**, confidence, disagreement, min edge)
6. If any gate fails → **NO TRADE**

**Timing:** the bot does **not** enter during the first ~40 minutes of an hourly contract. Entries are allowed only when **≤20 minutes** remain until expiry (and still above the 2-minute floor).

Reference matrix at model probability = 60%:

| Market YES | Gap | Bot action |
|---|---|---|
| 35¢ | 25pp | Strong BUY candidate |
| 40¢ | 20pp | Strong BUY candidate |
| 45¢ | 15pp | Only if other signals confirm |
| 50¢ | 10pp | No trade |
| 55¢ | 5pp | No trade |

Settlement reference is **CF Benchmarks BRTI** (60-second average into expiry). Both bots resolve spot from the official index page first, then fall back:

1. Public CF Benchmarks BRTI summary ([cfbenchmarks.com/data/indices/BRTI](https://www.cfbenchmarks.com/data/indices/BRTI)) — **primary**
2. Kalshi authenticated `/cfbenchmarks` passthrough (if API keys set)
3. Licensed CF Benchmarks REST API (`CF_BENCHMARKS_API_USERNAME` / `CF_BENCHMARKS_API_KEY`)
4. Kraken/Coinbase proxy (scanning only)

Kalshi quadratic fees are deducted before the threshold check:

`fee ≈ ceil_cent(0.07 × P × (1−P))` per contract.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Demo the classic options digital math
kalshi-demo

# Institutional 1h forecast scan (default NO TRADE)
kalshi-forecast

# Paper trading loop (forecast gates)
kalshi-bot --once
kalshi-bot --iterations 30
```

### Live trading

1. Create Kalshi API keys (demo or prod)
2. Copy `.env.example` → `.env` and set:

```bash
KALSHI_API_KEY_ID=...
KALSHI_PRIVATE_KEY_PEM="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"
KALSHI_ENV=demo   # or prod
```

3. In `config/default.yaml` set:

```yaml
execution:
  mode: live
  dry_run: false
```

4. Run `kalshi-forecast` / `kalshi-bot --once` and confirm fills / risk limits before leaving a loop running.

---

## Architecture

```
src/kalshi_bot/
  models/          Black-Scholes digitals, smile, ensemble forecast
  data/            Kalshi client, BRTI resolver, IBIT options, realized vol
  strategy/        Fees, evidence gates, forecast scanner, legacy mispricing
  execution/       Kelly sizing, risk gates, paper/live executor
  backtest/        Snapshot replay harness
```

Config knobs that matter for PnL / accuracy:

| Knob | Role |
|------|------|
| `bot_action.strong_buy_min_gap_pp` | Raw gap (≥20pp) → Strong BUY candidate |
| `bot_action.conditional_min_gap_pp` | Raw gap (≥15pp) → conditional; below → No trade |
| `bot_action.conditional_min_confidence` | Extra confirmation floor for conditional tier |
| `forecast_gates.max_seconds_to_expiry` | Last-N-minute entry window (default **1200s = 20m**) |
| `forecast_gates.min_seconds_to_expiry` | Too-close floor (default 120s) |
| `forecast_gates.min_edge_pp` | Minimum **conservative** post-fee edge (pp) |
| `forecast_gates.min_confidence` | Ensemble confidence floor |
| `forecast_gates.max_disagreement_pp` | Max model disagreement before NO TRADE |
| `forecast_gates.max_spread` | Reject wide Kalshi books |
| `risk.kelly_fraction` | Fractional Kelly (default 0.15) |

Legacy options-only mispricing remains available via `kalshi-scan` / `kalshi-bot --legacy-mispricing`.

---

## Design stance

- **Fail closed:** missing data, synthetic smiles, stale smiles, thin books, or model conflict → NO TRADE
- **Hourly only:** daily KXBTCD buckets are filtered out when `hourly_only: true`
- **Last 20 minutes only:** no entries until ≤20m remain on the hourly contract
- **Conservative EV:** trades must clear the edge using the uncertainty band, not just the point estimate
