# Kalshi BTC 1-Hour Forecasting Engine

Institutional-grade expected-value engine for Kalshi **KXBTCD** (Bitcoin 1-hour) markets.

**Objective:** maximize long-term EV.  
**Constraint:** accuracy > trade frequency.  
**Default action:** **NO TRADE** when evidence is insufficient.

This is **not** a price-following bot. A trade is emitted only when an ensemble forecast clears fee-aware edge floors **and** statistical evidence gates.

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

Settlement reference is **CF Benchmarks BRTI** (60-second average into expiry). Authenticated Kalshi credentials unlock the official BRTI passthrough; without them the engine scans on a public BTC proxy (paper mode only).

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
- **Conservative EV:** trades must clear the edge using the uncertainty band, not just the point estimate
