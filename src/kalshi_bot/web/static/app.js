function gatesBadge(opp) {
  const g = opp?.gates;
  if (!g) return "—";
  if (g.ready_side === "YES") return `<span class="gates-badge ready-yes">YES</span>`;
  if (g.ready_side === "NO") return `<span class="gates-badge ready-no">NO</span>`;
  const passed = (g.gates || []).filter((x) => x.status === "pass").length;
  const total = (g.gates || []).filter((x) => x.name !== "Position size").length;
  return `<span class="gates-badge wait">${passed}/${total}</span>`;
}

function gateIcon(status) {
  if (status === "pass") return "✓";
  if (status === "warn") return "⚠";
  return "✗";
}

function renderGateDashboard(opp) {
  const listEl = document.getElementById("gate-list");
  const labelEl = document.getElementById("gate-market-label");
  const ctxEl = document.getElementById("gate-context");
  const summaryEl = document.getElementById("gate-summary");

  if (!opp || !opp.gates) {
    labelEl.textContent = opp?.ticker || "No KXBTC15M market selected";
    ctxEl.innerHTML = "";
    listEl.innerHTML = `<div class="gate-empty">${opp ? "Gate data not available for this market." : "Select a 15m market row below."}</div>`;
    summaryEl.textContent = "";
    summaryEl.className = "gate-summary";
    return;
  }

  const g = opp.gates;
  labelEl.textContent = opp.ticker || "—";
  ctxEl.innerHTML = `
    <span>Model YES <strong>${Number(opp.model_yes || 0).toFixed(0)}%</strong></span>
    <span>Crowd <strong>${g.crowd_direction} ${Number(g.crowd_yes_pct || 0).toFixed(0)}%</strong></span>
    <span>Uncertainty <strong>${Number(g.uncertainty_pct || 0).toFixed(1)}%</strong></span>
    <span>EV floor <strong>${Number(g.min_net_ev || 0).toFixed(3)}</strong></span>
    <span>Bucket <strong>${g.time_bucket || "—"}</strong></span>`;

  listEl.innerHTML = (g.gates || [])
    .map(
      (gate) => `
    <div class="gate-item ${gate.status}">
      <div class="gate-icon">${gateIcon(gate.status)}</div>
      <div class="gate-body">
        <div class="gate-name">${gate.name}</div>
        <div class="gate-detail">${gate.detail}</div>
      </div>
    </div>`
    )
    .join("");

  if (g.ready_side) {
    summaryEl.textContent = `Ready to trade: BUY ${g.ready_side}`;
    summaryEl.className = "gate-summary ready";
  } else {
    summaryEl.textContent = g.position_detail || "Waiting for a side to clear all gates.";
    summaryEl.className = "gate-summary";
  }
}

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

function renderTimeBuckets(rows, summary) {
  const tbody = document.querySelector("#time-bucket-table tbody");
  tbody.innerHTML = "";
  for (const r of rows || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.label || r.bucket || ""}</td>
      <td>${r.n_trades ?? 0}</td>
      <td>${r.win_rate == null ? "—" : `${(r.win_rate * 100).toFixed(0)}%`}</td>
      <td>${r.avg_net_edge == null ? "—" : `${(r.avg_net_edge * 100).toFixed(1)}¢`}</td>
      <td>${fmtUsd(r.total_pnl)}</td>
      <td>${r.brier_score == null ? "—" : r.brier_score.toFixed(3)}</td>`;
    tbody.appendChild(tr);
  }
  const el = document.getElementById("time-bucket-summary");
  if (summary?.recommendation) {
    el.textContent = summary.recommendation;
  } else {
    el.textContent = "Need ≥3 settled trades per bucket for recommendations.";
  }
}

function renderMicroCalibration(micro, perfMicro) {
  const data = micro || perfMicro || {};
  const el = document.getElementById("micro-calibration");
  if (!data || !data.n_total) {
    el.textContent = "No settled trades with microstructure features yet.";
    return;
  }
  const uplift = data.brier_uplift_when_agrees;
  el.innerHTML = `
Samples: ${data.n_total} (${data.n_micro_agrees} agree / ${data.n_micro_disagrees} disagree)
Baseline Brier: ${data.baseline_brier?.toFixed(3) ?? "—"}
Agree Brier: ${data.agree_brier?.toFixed(3) ?? "—"}
Disagree Brier: ${data.disagree_brier?.toFixed(3) ?? "—"}
Uplift when micro agrees: ${uplift == null ? "—" : uplift.toFixed(3)}
Status: ${data.status || "collecting"}
${data.recommend_use_microstructure ? "✓ Microstructure improves calibration" : "— Still collecting evidence"}`;
}

function renderPerformance(perf) {
  const el = document.getElementById("performance-detail");
  if (!perf || Object.keys(perf).length === 0) {
    el.textContent = "No performance data yet.";
    return;
  }
  const lines = [];
  for (const [strategy, data] of Object.entries(perf)) {
    if (strategy === "time_buckets" || strategy === "time_bucket_summary" || strategy === "microstructure") {
      continue;
    }
    const signals = data.signals || {};
    const sigText = Object.entries(signals).map(([k, v]) => `${k}: ${v}`).join(", ") || "—";
    let line = `${strategy}: ${data.trade_count ?? 0} markets · ${sigText}`;
    if (data.settled_trades != null) {
      line += ` · settled ${data.settled_trades}`;
    }
    if (data.win_rate != null) {
      line += ` · win ${(data.win_rate * 100).toFixed(0)}%`;
    }
    if (data.total_pnl != null) {
      line += ` · P&L ${fmtUsd(data.total_pnl)}`;
    }
    lines.push(line);
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

function renderTable(tableId, opps, onSelect, { showGates = false } = {}) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  tbody.innerHTML = "";
  for (const o of opps || []) {
    const tr = document.createElement("tr");
    tr.className = signalClass(o.decision);
    const gatesCol = showGates ? `<td>${gatesBadge(o)}</td>` : "";
    const regimeCol = showGates ? "" : `<td>${o.regime || ""}</td>`;
    tr.innerHTML = `
      <td>${o.ticker || ""}</td>
      <td>${fmtTime(o.seconds_to_expiry || 0)}</td>
      <td>$${Number(o.strike || 0).toLocaleString()}</td>
      <td>$${Number(o.btc || 0).toLocaleString()}</td>
      <td>${Number(o.model_yes || 0).toFixed(0)}%</td>
      <td>${Math.round((o.yes_ask || 0) * 100)}</td>
      <td>${Math.round((o.no_ask || 0) * 100)}</td>
      <td>${((o.yes_net_edge ?? o.net_edge ?? 0) * 100).toFixed(1)}</td>
      <td>${((o.no_net_edge ?? 0) * 100).toFixed(1)}</td>
      ${gatesCol}
      <td>${o.price_pattern || "—"}</td>
      <td>${o.confidence || ""}</td>
      ${regimeCol}
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

function selectOpp(o) {
  selectedOpp = o;
  renderWhyPanel(o);
  if (o?.strategy === "KXBTC15M" || (o?.ticker || "").includes("KXBTC15M")) {
    renderGateDashboard(o);
  }
}

function applyState(data) {
  const meta = document.getElementById("meta");
  if (!data || data.status === "no_data") {
    meta.textContent = "Waiting for first scan… Run: python3 platform_run.py --web";
    renderLiveBanner(null, null);
    renderSafety(null, null);
    renderPerformance(null);
    renderTimeBuckets([], null);
    renderMicroCalibration(null, null);
    renderBestCard("best-15m-detail", null, "KXBTC15M");
    renderBestCard("best-1h-detail", null, "KXBTCD");
    renderTable("opp-table-15m", [], () => {});
    renderTable("opp-table-1h", [], () => {});
    renderWhyPanel(null);
    renderCalibration([]);
    renderSettlements([]);
    renderFreshness(null);
    renderGateDashboard(null);
    return;
  }

  const opps15 = data.opportunities_15m || [];
  const opps1h = data.opportunities_1h || [];
  meta.textContent = `BTC $${Number(data.spot).toLocaleString()} · ${data.markets_scanned} markets · ${data.asof || ""}`;

  renderLiveBanner(data.safety, data.freshness);
  renderSafety(data.safety, data.freshness);
  renderPerformance(data.performance);
  renderTimeBuckets(
    data.time_bucket_performance || data.performance?.time_buckets,
    data.performance?.time_bucket_summary
  );
  renderMicroCalibration(data.microstructure_calibration, data.performance?.microstructure);
  renderFreshness(data.freshness);
  renderBestCard("best-15m-detail", opps15[0], "KXBTC15M");
  renderBestCard("best-1h-detail", opps1h[0], "KXBTCD");
  renderTable("opp-table-15m", opps15, selectOpp, { showGates: true });
  renderTable("opp-table-1h", opps1h, selectOpp);
  const gateOpp =
    selectedOpp && (selectedOpp.strategy === "KXBTC15M" || (selectedOpp.ticker || "").includes("KXBTC15M"))
      ? selectedOpp
      : opps15[0] || null;
  renderGateDashboard(gateOpp);
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
