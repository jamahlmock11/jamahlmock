# Kalshi BTC Mispricing Bot

Profitability-first trading system for Kalshi **KXBTC15M** (15-minute) and **KXBTCD** (hourly) Bitcoin markets.

The edge is **not** “follow price.” It is a cross-venue probability mismatch:

1. Build the **IBIT ETF options volatility smile**
2. Translate that smile into **BTC spot space** (log-moneyness preserved)
3. Price a Black–Scholes **digital** probability for the Kalshi strike / horizon
4. Compare to Kalshi’s traded implied probability
5. Trade only when edge clears fees + risk gates

Canonical example:

> Kalshi prices a strike at **22%**, options imply **37.8%** → **+15.8pp** edge → buy YES.

Settlement reference is **CF Benchmarks BRTI** (60-second average into expiry). Authenticated Kalshi credentials unlock the official BRTI passthrough; without them the bot scans on a public BTC proxy (paper mode only).

---

## Signal math

For a Kalshi binary that pays $1 if \(S_T \ge K\):

\[
p_{\text{options}} = N(d_2),\quad
d_2=\frac{\ln(S/K)+(r-q-\sigma(K)^2/2)T}{\sigma(K)\sqrt{T}}
\]

- \(S\): live BRTI (or proxy)
- \(K\): Kalshi `floor_strike` (open BRTI for KXBTC15M; level strike for KXBTCD)
- \(\sigma(K)\): IV from the IBIT smile at the strike’s log-moneyness
- Edge (YES): \(p_{\text{options}} - \text{yes\_ask} - \text{fee}(ask)\)

IBIT→BTC translation uses \(K_{\text{btc}} = K_{\text{ibit}} \cdot S_{\text{btc}} / S_{\text{ibit}}\). Relative moves match, so the smile in log-moneyness space transfers directly.

Kalshi quadratic fees are deducted before the threshold check:

`fee ≈ ceil_cent(0.07 × P × (1−P))` per contract.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Demo the edge math (no API keys)
kalshi-demo

# Scan live Kalshi books once (public market data)
kalshi-scan

# Paper trading loop
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

4. Run `kalshi-bot --once` and confirm fills / risk limits before leaving a loop running.

---

## Architecture

```
src/kalshi_bot/
  models/          Black-Scholes digitals, smile, IBIT→BTC probability
  data/            Kalshi client, BRTI resolver, IBIT options loader
  strategy/        Fee model, mispricing detector, scanner
  execution/       Kelly sizing, risk gates, paper/live executor
  backtest/        Snapshot replay harness
```

Config knobs that actually matter for PnL:

| Knob | Role |
|------|------|
| `series[].min_edge_pp` | Minimum post-fee edge (pp) to trade |
| `smile.stale_edge_multiplier` | Widen threshold when IBIT smile is old (nights/weekends) |
| `risk.kelly_fraction` | Fractional Kelly (default 0.25) |
| `risk.max_exposure_usd` | Hard portfolio cap |
| `risk.min_seconds_to_expiry` | Avoid last-second microstructure |

---

## What this will / won’t do

**Will**

- Detect genuine probability mispricing vs an options-implied distribution
- Size with fractional Kelly and hard loss/exposure caps
- Paper-trade safely with public Kalshi market data
- Cache the last good IBIT smile when the equity options tape is dark
- Fall back to Deribit BTC mark IVs (already in BTC space) when IBIT quality fails

**Won’t guarantee**

- Risk-neutral options probs ≠ real-world measure; residual drift / jump risk remains
- IBIT options close nights/weekends — overnight edges use Deribit/cache with stricter stale thresholds
- Latency vs other arb bots on very tight edges

Tune `min_edge_pp` up until live expectancy is positive; profitability is the only acceptance criterion.

---

## Tests

```bash
pytest -q
```
