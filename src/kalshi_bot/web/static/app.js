function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function renderBest(opp) {
  const el = document.getElementById("best-detail");
  if (!opp) {
    el.textContent = "No opportunities scanned yet.";
    return;
  }
  const cls = opp.decision.includes("BUY") ? "buy" : opp.decision === "WAIT" ? "wait" : "no-trade";
  el.innerHTML = `
<strong>KXBTC15M</strong>
BTC: $${Number(opp.btc || 0).toLocaleString()}
Strike: $${Number(opp.strike || 0).toLocaleString()}
Time remaining: ${fmtTime(opp.seconds_to_expiry || 0)}

Model YES: ${Number(opp.model_yes || 0).toFixed(1)}%
Kalshi YES ask: ${Math.round((opp.yes_ask || 0) * 100)}¢
Fair value: ${Math.round((opp.fair_value || 0) * 100)}¢
Net edge: ${((opp.net_edge || 0) * 100).toFixed(1)}¢
Confidence: ${opp.confidence || "—"}
Volatility: ${opp.volatility || "—"}
Order flow: ${opp.order_flow || "—"}
Liquidity: ${opp.liquidity || "—"}
Tape TPS: ${(opp.tape_tps || 0).toFixed(2)}

<span class="${cls}">${opp.decision}</span>
${opp.reason ? "— " + opp.reason : ""}`;
}

function renderTable(opps) {
  const tbody = document.querySelector("#opp-table tbody");
  tbody.innerHTML = "";
  for (const o of opps || []) {
    const tr = document.createElement("tr");
    tr.className = (o.decision || "").includes("BUY") ? "buy-row" : "no-row";
    tr.innerHTML = `
      <td>${o.ticker || ""}</td>
      <td>${fmtTime(o.seconds_to_expiry || 0)}</td>
      <td>${Number(o.model_yes || 0).toFixed(0)}%</td>
      <td>${Math.round((o.yes_ask || 0) * 100)}</td>
      <td>${((o.net_edge || 0) * 100).toFixed(1)}</td>
      <td>${o.confidence || ""}</td>
      <td>${o.order_flow || ""}</td>
      <td>${(o.tape_tps || 0).toFixed(2)}</td>
      <td>${o.decision || ""}</td>`;
    tbody.appendChild(tr);
  }
}

function renderCalibration(rows) {
  const tbody = document.querySelector("#calibration-table tbody");
  tbody.innerHTML = "";
  for (const r of rows || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.range}</td>
      <td>${r.n_trades}</td>
      <td>${(r.empirical_win_rate * 100).toFixed(0)}%</td>
      <td>${r.calibrated ? "✓ calibrated" : "collecting"}</td>`;
    tbody.appendChild(tr);
  }
}

function renderSettlements(items) {
  const ul = document.getElementById("settlements");
  ul.innerHTML = "";
  for (const s of items || []) {
    const li = document.createElement("li");
    li.textContent = `${s.ticker} ${s.side} → ${s.result} won=${s.won} pnl=$${Number(s.pnl).toFixed(2)} (pred ${(s.prediction * 100).toFixed(0)}%)`;
    ul.appendChild(li);
  }
}

function applyState(data) {
  document.getElementById("meta").textContent =
    data.status === "ok"
      ? `BTC $${Number(data.spot).toLocaleString()} · scanned ${data.markets_scanned} · ${data.asof || ""}`
      : "Waiting for first scan…";
  const opps = data.opportunities || [];
  renderBest(opps[0] || null);
  renderTable(opps);
  renderCalibration(data.calibration || []);
  renderSettlements(data.settlements || []);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onmessage = (ev) => applyState(JSON.parse(ev.data));
  ws.onclose = () => setTimeout(connect, 3000);
}

fetch("/api/state").then((r) => r.json()).then(applyState).catch(() => {});
connect();
