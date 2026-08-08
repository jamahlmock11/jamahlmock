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
3. Compute **conservative post-fee EV** (uses forecast uncertainty band, not the mean alone)
4. Apply hard gates (spread, volume, horizon, confidence, disagreement, min edge)
5. If any gate fails → **NO TRADE**

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
