function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtUsd(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return `$${Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function badge(text, level = "") {
  return `<span class="badge ${level}">${text}</span>`;
}

function signalClass(decision) {
  const d = (decision || "").toUpperCase();
  if (d.includes("STRONG BUY")) return "strong-buy";
  if (d.includes("BUY")) return "buy";
  if (d === "WAIT") return "wait";
  if (d === "DATA ERROR") return "error";
  return "no-trade";
}

function renderLiveBanner(safety, freshness) {
  const el = document.getElementById("live-banner");
  const live = safety?.status_label === "LIVE" || freshness?.live_trading_enabled;
  el.textContent = live ? "LIVE TRADING" : "DISABLED";
  el.className = `live-banner ${live ? "live" : "disabled"}`;
}

function renderSafety(safety, freshness) {
  const el = document.getElementById("safety-detail");
  if (!safety && !freshness) {
    el.textContent = "No safety data yet.";
    return;
  }
  const s = safety || {};
  const f = freshness || {};
  el.innerHTML = `
<div><span class="label">Status</span> ${s.status_label || (f.live_trading_enabled ? "LIVE" : "DISABLED")}</div>
<div><span class="label">Balance</span> ${fmtUsd(s.balance_usd ?? f.balance_usd)}</div>
<div><span class="label">Daily P&amp;L</span> ${fmtUsd(s.daily_pnl_usd)}</div>
<div><span class="label">Loss limit</span> ${fmtUsd(s.daily_loss_limit_usd)}</div>
<div><span class="label">API</span> ${s.api_connected ? "connected" : "down"}</div>
<div><span class="label">Market data</span> ${s.market_data_connected ? "connected" : "down"}</div>
<div><span class="label">Last update</span> ${s.last_update_age_s != null ? `${s.last_update_age_s}s ago` : "—"}</div>
<div><span class="label">Model</span> ${s.model_version || "—"}</div>
${s.block_reason ? `<div class="block-reason">${s.block_reason}</div>` : ""}`;
}

function renderPerformance(perf) {
  const el = document.getElementById("performance-detail");
  if (!perf || Object.keys(perf).length === 0) {
    el.textContent = "No performance data yet.";
    return;
  }
  const lines = [];
  for (const [strategy, data] of Object.entries(perf)) {
    const signals = data.signals || {};
    const sigText = Object.entries(signals).map(([k, v]) => `${k}: ${v}`).join(", ") || "—";
    lines.push(`${strategy}: ${data.trade_count ?? 0} markets · ${sigText}`);
  }
  el.innerHTML = lines.map((l) => `<div>${l}</div>`).join("");
}

function renderFreshness(freshness) {
  const el = document.getElementById("badges");
  if (!freshness) {
    el.innerHTML = "";
    return;
  }
  const items = [];
  items.push(badge(`BRTI ${freshness.brti_source || "—"}`, freshness.brti_official ? "ok" : "warn"));
  items.push(badge(`scan ${freshness.scan_duration_ms || "?"}ms`, "ok"));
  items.push(
    badge(
      `age ${freshness.scan_age_seconds ?? "?"}s`,
      (freshness.scan_age_seconds || 0) <= 5 ? "ok" : "warn"
    )
  );
  if (freshness.kalshi_ws_connected) {
    items.push(badge(`WS ${freshness.kalshi_ws_last_message_age ?? "?"}s`, "ok"));
  } else {
    items.push(badge("WS offline", "bad"));
  }
  if (freshness.btc_stale) items.push(badge("BTC stale", "warn"));
  el.innerHTML = items.join("");
}

function renderBestCard(elId, opp, strategyLabel) {
  const el = document.getElementById(elId);
  if (!opp) {
    el.textContent = `No ${strategyLabel} markets scanned yet.`;
    return;
  }
  const cls = signalClass(opp.decision);
  el.innerHTML = `
<strong>${opp.ticker || strategyLabel}</strong>
BTC: $${Number(opp.btc || 0).toLocaleString()} · Strike: $${Number(opp.strike || 0).toLocaleString()}
Time remaining: ${fmtTime(opp.seconds_to_expiry || 0)}

Model YES: ${Number(opp.model_yes || 0).toFixed(1)}% · NO: ${Number(opp.model_no || 0).toFixed(1)}%
Executable YES: ${Math.round((opp.yes_ask || 0) * 100)}¢ · NO: ${Math.round((opp.no_ask || 0) * 100)}¢
Fair YES: ${Math.round((opp.fair_value || 0) * 100)}¢ · Net edge: ${((opp.net_edge || 0) * 100).toFixed(1)}¢
Confidence: ${opp.confidence || "—"} · Regime: ${opp.regime || "—"}

<span class="${cls}">${opp.decision || "NO TRADE"}</span>
${opp.reason ? " — " + opp.reason : ""}

${opp.why_trade ? `<div class="why-box why-yes"><strong>WHY TRADE?</strong>\n${opp.why_trade}</div>` : ""}
${opp.why_not_trade ? `<div class="why-box why-no"><strong>WHY NOT?</strong>\n${opp.why_not_trade}</div>` : ""}`;
}

function renderTable(tableId, opps, onSelect) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";
  for (const o of opps || []) {
    const tr = document.createElement("tr");
    tr.className = signalClass(o.decision);
    tr.innerHTML = `
      <td>${o.ticker || ""}</td>
      <td>${fmtTime(o.seconds_to_expiry || 0)}</td>
      <td>$${Number(o.strike || 0).toLocaleString()}</td>
      <td>$${Number(o.btc || 0).toLocaleString()}</td>
      <td>${Number(o.model_yes || 0).toFixed(0)}%</td>
      <td>${Math.round((o.yes_ask || 0) * 100)}</td>
      <td>${Math.round((o.no_ask || 0) * 100)}</td>
      <td>${((o.net_edge || 0) * 100).toFixed(1)}</td>
      <td>${o.confidence || ""}</td>
      <td>${o.regime || ""}</td>
      <td>${o.decision || ""}</td>`;
    tr.addEventListener("click", () => onSelect(o));
    tbody.appendChild(tr);
  }
}

function renderWhyPanel(opp) {
  const el = document.getElementById("why-panel");
  if (!opp) {
    el.textContent = "No market selected.";
    return;
  }
  el.innerHTML = `
<strong>${opp.ticker}</strong> (${opp.strategy || "—"})
Signal: <span class="${signalClass(opp.decision)}">${opp.decision}</span>

${opp.why_trade ? `<div class="why-box why-yes"><strong>WHY THIS TRADE?</strong>\n${opp.why_trade}</div>` : ""}
${opp.why_not_trade ? `<div class="why-box why-no"><strong>WHY NOT TRADE?</strong>\n${opp.why_not_trade}</div>` : ""}
${!opp.why_trade && !opp.why_not_trade ? `<div>${opp.reason || "No detail available."}</div>` : ""}`;
}

function renderCalibration(rows) {
  const tbody = document.querySelector("#calibration-table tbody");
  tbody.innerHTML = "";
  for (const r of rows || []) {
    const tr = document.createElement("tr");
    const pred = r.predicted_probability;
    const empirical = r.empirical_win_rate;
    tr.innerHTML = `
      <td>${r.range || ""}</td>
      <td>${r.n_trades ?? 0}</td>
      <td>${pred == null ? "—" : `${(pred * 100).toFixed(0)}%`}</td>
      <td>${empirical == null ? "—" : `${(empirical * 100).toFixed(0)}%`}</td>
      <td>${r.brier_score == null ? "—" : r.brier_score.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  }
}

function renderSettlements(items) {
  const ul = document.getElementById("settlements");
  ul.innerHTML = "";
  for (const s of items || []) {
    const li = document.createElement("li");
    li.textContent = `${s.ticker} ${s.side} → ${s.result} won=${s.won} pnl=${fmtUsd(s.pnl)} (pred ${((s.prediction || 0) * 100).toFixed(0)}%)`;
    ul.appendChild(li);
  }
}

let selectedOpp = null;

function applyState(data) {
  const meta = document.getElementById("meta");
  if (!data || data.status === "no_data") {
    meta.textContent = "Waiting for first scan… Run: python3 platform_run.py --web";
    renderLiveBanner(null, null);
    renderSafety(null, null);
    renderPerformance(null);
    renderBestCard("best-15m-detail", null, "KXBTC15M");
    renderBestCard("best-1h-detail", null, "KXBTCD");
    renderTable("opp-table-15m", [], () => {});
    renderTable("opp-table-1h", [], () => {});
    renderWhyPanel(null);
    renderCalibration([]);
    renderSettlements([]);
    renderFreshness(null);
    return;
  }

  const opps15 = data.opportunities_15m || [];
  const opps1h = data.opportunities_1h || [];
  meta.textContent = `BTC $${Number(data.spot).toLocaleString()} · ${data.markets_scanned} markets · ${data.asof || ""}`;

  renderLiveBanner(data.safety, data.freshness);
  renderSafety(data.safety, data.freshness);
  renderPerformance(data.performance);
  renderFreshness(data.freshness);
  renderBestCard("best-15m-detail", opps15[0], "KXBTC15M");
  renderBestCard("best-1h-detail", opps1h[0], "KXBTCD");
  renderTable("opp-table-15m", opps15, (o) => {
    selectedOpp = o;
    renderWhyPanel(o);
  });
  renderTable("opp-table-1h", opps1h, (o) => {
    selectedOpp = o;
    renderWhyPanel(o);
  });
  if (!selectedOpp) {
    renderWhyPanel(opps15[0] || opps1h[0] || null);
  }
  renderCalibration(data.calibration || []);
  renderSettlements(data.settlements || []);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onmessage = (ev) => {
    try {
      applyState(JSON.parse(ev.data));
    } catch (err) {
      console.error("bad websocket payload", err);
    }
  };
  ws.onerror = () => {
    document.getElementById("meta").textContent = "WebSocket error — retrying…";
  };
  ws.onclose = () => setTimeout(connect, 3000);
}

fetch("/api/state")
  .then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  })
  .then(applyState)
  .catch(() => {
    document.getElementById("meta").textContent =
      "Cannot reach API — is the server running? Try: python3 platform_run.py --web";
  });
connect();
