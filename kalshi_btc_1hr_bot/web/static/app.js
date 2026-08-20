(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  function fmtUsd(v) {
    if (v == null || Number.isNaN(v)) return "—";
    const n = Number(v);
    return (n >= 0 ? "$" : "-$") + Math.abs(n).toFixed(2);
  }

  function fmtPct(v) {
    if (v == null) return "—";
    return Number(v).toFixed(1) + "%";
  }

  function fmtCents(v) {
    if (v == null) return "—";
    return Number(v).toFixed(1) + "¢";
  }

  function fmtTime(isoOrTs) {
    if (!isoOrTs) return "—";
    const d = typeof isoOrTs === "number" ? new Date(isoOrTs * 1000) : new Date(isoOrTs);
    return d.toISOString().replace("T", " ").slice(0, 19);
  }

  function pnlClass(v) {
    if (v == null || v === 0) return "pnl-neutral";
    return v > 0 ? "pnl-pos" : "pnl-neg";
  }

  function statusClass(status) {
    const s = (status || "").toUpperCase();
    if (s === "WIN") return "status-win";
    if (s === "LOSS") return "status-loss";
    if (s === "OPEN") return "status-open";
    return "status-blocked";
  }

  function cardLight(row, snap) {
    if (row.is_pick && (row.action || "").startsWith("BUY")) return "green";
    if (row.should_trade) return "green";
    const th = snap.thresholds || {};
    const minEdge = th.min_edge_cents || 2.5;
    if (row.edge_cents >= minEdge - 1.0) return "yellow";
    return "red";
  }

  function renderStatusBanner(snap) {
    const banner = $("status-banner");
    const light = snap.action_light || "red";
    banner.className = "status-banner status-" + light + " flash";
    $("status-headline").textContent = snap.action_headline || snap.cycle_status || "WAITING";
    $("status-detail").textContent = snap.action_detail || (snap.blockers && snap.blockers[0]) || "Scanning markets…";
    $("status-readiness").textContent = fmtPct(snap.readiness_pct || 0);
  }

  function renderTop4(snap) {
    const grid = $("top4-grid");
    const rows = snap.top_markets || [];
    $("top4-updated").textContent = snap.updated_at ? "Updated " + snap.updated_at.slice(11, 19) + " UTC" : "—";

    if (!rows.length) {
      grid.innerHTML = '<div class="top4-card empty">No markets in window — waiting for bot scan…</div>';
      return;
    }

    grid.innerHTML = rows.map(function (r) {
      const light = cardLight(r, snap);
      const finishClass = r.finish === "ABOVE" ? "above" : "below";
      const edgeClass = r.edge_cents >= 0 ? "pos" : "neg";
      const yesActive = r.side === "yes" ? " active" : "";
      const noActive = r.side === "no" ? " active" : "";
      const pickClass = r.is_pick ? " card-pick flash-pick" : "";

      return (
        '<div class="top4-card card-' + light + pickClass + '">' +
        '<div class="top4-rank">#' + r.rank + " · " + Math.round(r.secs_left / 60) + "m left</div>" +
        '<div class="top4-strike">$' + Number(r.strike).toLocaleString() + "</div>" +
        '<div class="top4-finish ' + finishClass + '">' + r.finish + " · " + r.side.toUpperCase() + "</div>" +
        '<div class="top4-prices">' +
        '<div class="top4-price-box' + yesActive + '"><div class="top4-price-label">YES</div><div class="top4-price-val">' +
        (r.yes_price_cents != null ? r.yes_price_cents + "¢" : "—") + "</div></div>" +
        '<div class="top4-price-box' + noActive + '"><div class="top4-price-label">NO</div><div class="top4-price-val">' +
        (r.no_price_cents != null ? r.no_price_cents + "¢" : "—") + "</div></div>" +
        "</div>" +
        '<div class="top4-edge ' + edgeClass + '">Edge ' + fmtCents(r.edge_cents) +
        " · Ev " + Number(r.evidence_score).toFixed(3) + "</div>" +
        '<div style="font-size:0.62rem;color:var(--muted);margin-top:0.3rem;overflow:hidden;text-overflow:ellipsis">' +
        r.ticker + "</div>" +
        "</div>"
      );
    }).join("");
  }

  function renderSnapshot(snap, stats) {
    $("stat-spot").textContent = snap.spot ? "$" + Number(snap.spot).toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—";
    $("stat-brti").textContent = (snap.brti_source || "—") + (snap.brti_official ? " · official" : " · proxy");
    $("stat-balance").textContent = snap.balance_usd != null ? fmtUsd(snap.balance_usd) : "—";
    $("stat-bankroll").textContent = "max " + fmtUsd(snap.config_summary?.max_trade_usd || 1) + "/trade";
    $("stat-readiness").textContent = fmtPct(snap.readiness_pct || 0);
    $("readiness-fill").style.width = Math.min(100, snap.readiness_pct || 0) + "%";
    $("stat-winrate").textContent = stats.settled_trades ? fmtPct(stats.win_rate_pct) : "—";
    $("stat-record").textContent = stats.wins + "W / " + stats.losses + "L · " + stats.pending_trades + " open";
    const pnlEl = $("stat-pnl");
    pnlEl.textContent = fmtUsd(stats.realized_pnl_usd || 0);
    pnlEl.className = "stat-value " + pnlClass(stats.realized_pnl_usd || 0);
    $("stat-trades").textContent = stats.executed_trades + " executed";
    $("stat-markets").textContent = String(snap.markets_scanned || 0);
    $("stat-vol").textContent = "vol " + (snap.annualized_vol ? (snap.annualized_vol * 100).toFixed(0) + "%" : "—");

    const mode = (snap.mode || "PAPER").toUpperCase();
    const modeEl = $("mode-badge");
    modeEl.textContent = mode;
    modeEl.className = "mode-badge " + (mode === "LIVE" ? "live" : "paper");

    const cycleEl = $("cycle-badge");
    const cs = snap.cycle_status || "WAITING";
    cycleEl.textContent = cs;
    cycleEl.className = "cycle-badge" + (cs === "TRADE" || cs === "READY" ? " trade" : cs === "CLOSE" ? " close" : "");

    const th = snap.thresholds || {};
    $("thresholds-line").textContent =
      "Edge ≥ " + (th.min_edge_cents || "?") + "¢ · Evidence ≥ " + (th.min_evidence_margin || "?") +
      " · Agreement ≥ " + Math.round((th.min_agreement || 0) * 100) + "% · Top " + (th.top_n_markets || 4) + " markets";
    $("updated-line").textContent = snap.updated_at ? "Updated " + snap.updated_at : "No scan yet";
  }

  function renderBest(snap) {
    const best = snap.best_pick;
    const actionEl = $("best-action");
    const detailEl = $("best-detail");
    const blockersEl = $("blockers");

    if (!best) {
      actionEl.textContent = "NO TRADE";
      actionEl.className = "pill pill-idle";
      detailEl.innerHTML = '<div class="empty-state">No qualifying market in current window.</div>';
      blockersEl.innerHTML = (snap.blockers || []).map((b) => '<span class="blocker">' + b + "</span>").join("");
      return;
    }

    const action = best.action || "NO_TRADE";
    actionEl.textContent = action.replace("_", " ");
    actionEl.className = "pill " + (action.startsWith("BUY") ? "pill-buy" : best.should_trade ? "pill-block" : "pill-idle");

    detailEl.innerHTML = [
      kv("Ticker", best.ticker),
      kv("Strike", "$" + Number(best.strike).toLocaleString()),
      kv("Time left", Math.round(best.secs_left / 60) + " min"),
      kv("Finish", best.finish + " (" + best.side.toUpperCase() + ")"),
      kv("Fair YES", (best.p_fair * 100).toFixed(1) + "%", true),
      kv("Edge", fmtCents(best.edge_cents)),
      kv("Evidence", Number(best.evidence_score).toFixed(3)),
      kv("Confidence", Math.round(best.confidence * 100) + "%"),
      kv("Ask", Math.round((best.price || 0) * 100) + "¢"),
      kv("Contracts", best.contracts != null ? best.contracts : "0"),
    ].join("");

    blockersEl.innerHTML = (snap.blockers || []).map((b) => '<span class="blocker">' + b + "</span>").join("");
  }

  function kv(label, value, big) {
    return '<div class="kv"><span class="k">' + label + '</span><span class="v' + (big ? " big" : "") + '">' + value + "</span></div>";
  }

  function renderChecklist(snap) {
    const ul = $("checklist");
    const items = snap.checklist || [];
    if (!items.length) {
      ul.innerHTML = '<li class="empty-state">Waiting for scan data…</li>';
      return;
    }
    ul.innerHTML = items.map(function (item) {
      return (
        '<li class="check-item">' +
        '<div class="check-icon ' + (item.passed ? "pass" : "fail") + '">' + (item.passed ? "✓" : "✕") + "</div>" +
        '<div><div class="check-label">' + item.label + '</div><div class="check-detail">' + item.detail + "</div></div>" +
        "</li>"
      );
    }).join("");
  }

  function renderCrowd(snap) {
    const grid = $("votes-grid");
    const crowd = snap.crowd || {};
    const members = crowd.members || snap.model_votes || [];
    $("crowd-quorum").textContent =
      crowd.quorum ? "quorum " + crowd.quorum + (crowd.quorum_met ? " ✓" : " ✕") : "quorum —";

    const summaryEl = $("crowd-summary");
    if (crowd.prob_yes != null) {
      const sideClass = crowd.consensus === "YES" ? "yes" : "no";
      summaryEl.innerHTML =
        '<div class="crowd-stat"><div class="k">Crowd P(YES)</div><div class="v">' +
        (crowd.prob_yes * 100).toFixed(1) + "%</div></div>" +
        '<div class="crowd-stat"><div class="k">Consensus</div><div class="v ' + sideClass + '">' +
        (crowd.finish || crowd.consensus) + "</div></div>" +
        '<div class="crowd-stat"><div class="k">Synthesis</div><div class="v">' + (crowd.synthesis || "—") + "</div></div>" +
        '<div class="crowd-stat"><div class="k">Agreement</div><div class="v">' +
        (crowd.agreement != null ? Math.round(crowd.agreement * 100) + "%" : "—") + "</div></div>" +
        '<div class="crowd-stat"><div class="k">Votes</div><div class="v">' +
        (crowd.yes_votes != null ? crowd.yes_votes + "Y / " + crowd.no_votes + "N" : "—") + "</div></div>";
    } else {
      summaryEl.innerHTML = '<span class="empty-state">Crowd data loading…</span>';
    }

    if (!members.length) {
      grid.innerHTML = '<div class="empty-state">No crowd voters yet.</div>';
      return;
    }
    const topNames = new Set((crowd.top_votes || []).map(function (v) { return v.name; }));
    grid.innerHTML = members.map(function (v) {
      const side = (v.side || (v.prob_yes >= 0.5 ? "ABOVE" : "BELOW")).toUpperCase();
      const sideClass = side === "YES" || side === "ABOVE" ? "above" : "below";
      const qClass = side === "YES" || side === "ABOVE" ? "quorum-yes" : "quorum-no";
      const topClass = topNames.has(v.name) ? " top-vote" : "";
      return (
        '<div class="vote-card ' + qClass + topClass + '">' +
        '<div class="vote-name">' + v.name + "</div>" +
        '<div class="vote-group">' + (v.group || "model") + "</div>" +
        '<div class="vote-side ' + sideClass + '">' + (side === "ABOVE" ? "ABOVE" : side === "BELOW" ? "BELOW" : side) + "</div>" +
        '<div class="vote-meta">P(YES) ' + ((v.prob_yes || 0) * 100).toFixed(1) + "% · w=" + (v.weight || 0) + "</div>" +
        "</div>"
      );
    }).join("");
  }

  function renderMarkets(snap) {
    const tbody = $("markets-table").querySelector("tbody");
    const rows = snap.top_markets || [];
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No markets in window this cycle.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(function (r) {
      const ask = r.side === "yes" ? r.yes_ask : r.no_ask;
      let status = "NO EDGE";
      if (r.is_pick && (r.action || "").startsWith("BUY")) status = "▶ TRADE";
      else if (r.is_pick) status = "★ PICK";
      else if (r.should_trade) status = "EDGE OK";
      else if (r.edge_cents >= ((snap.thresholds || {}).min_edge_cents || 2.5) - 1) status = "CLOSE";
      return (
        '<tr class="' + (r.is_pick ? "pick" : "") + '">' +
        "<td>" + r.rank + "</td>" +
        "<td>" + r.ticker + "</td>" +
        "<td>" + Math.round(r.secs_left / 60) + "m</td>" +
        "<td>$" + Number(r.strike).toLocaleString() + "</td>" +
        "<td>" + r.finish + "</td>" +
        "<td>" + fmtCents(r.edge_cents) + "</td>" +
        "<td>" + Number(r.evidence_score).toFixed(3) + "</td>" +
        "<td>" + Math.round(r.p_fair * 100) + "%</td>" +
        "<td>YES " + (r.yes_price_cents != null ? r.yes_price_cents + "¢" : "—") +
        " / NO " + (r.no_price_cents != null ? r.no_price_cents + "¢" : "—") + "</td>" +
        "<td>" + status + "</td>" +
        "</tr>"
      );
    }).join("");
  }

  function renderTrades(trades) {
    const tbody = $("trades-table").querySelector("tbody");
    if (!trades.length) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty-state">No trades recorded yet.</td></tr>';
      return;
    }
    tbody.innerHTML = trades.map(function (t) {
      return (
        "<tr>" +
        "<td>" + fmtTime(t.opened_at) + "</td>" +
        "<td>" + t.ticker + "</td>" +
        "<td>" + t.side + "</td>" +
        "<td>" + t.contracts + "</td>" +
        "<td>" + Math.round(t.entry_price * 100) + "¢</td>" +
        "<td>" + fmtUsd(t.cost_usd) + "</td>" +
        "<td>" + fmtCents(t.edge_cents) + "</td>" +
        "<td>" + (t.passed ? "YES" : "NO") + "</td>" +
        '<td class="' + statusClass(t.status) + '">' + t.status + "</td>" +
        '<td class="' + pnlClass(t.pnl_usd) + '">' + (t.pnl_usd != null ? fmtUsd(t.pnl_usd) : "—") + "</td>" +
        "</tr>"
      );
    }).join("");
  }

  function renderCycles(cycles) {
    const tbody = $("cycles-table").querySelector("tbody");
    if (!cycles.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No cycle history yet.</td></tr>';
      return;
    }
    tbody.innerHTML = cycles.map(function (c) {
      return (
        "<tr>" +
        "<td>" + c.time + "</td>" +
        "<td>" + c.status + "</td>" +
        "<td>" + c.markets_scanned + "</td>" +
        "<td>" + (c.best_ticker || "—") + "</td>" +
        "<td>" + (c.best_action || "—") + "</td>" +
        "<td>" + fmtPct(c.readiness_pct) + "</td>" +
        "<td>" + (c.reason || "—") + "</td>" +
        "</tr>"
      );
    }).join("");
  }

  function render(payload) {
    const snap = payload.snapshot || {};
    const stats = payload.stats || {};
    renderStatusBanner(snap);
    renderTop4(snap);
    renderSnapshot(snap, stats);
    renderBest(snap);
    renderChecklist(snap);
    renderCrowd(snap);
    renderMarkets(snap);
    renderTrades(payload.trades || []);
    renderCycles(payload.cycles || []);

    const age = payload.state_age_s;
    const stale = age != null && age > 30;
    $("conn-dot").className = "conn-dot " + (stale ? "offline" : "online");
    $("meta-line").textContent =
      (stale ? "Bot scan stale (" + age + "s ago) — is the bot running?" : "Live · scan " + (age != null ? age + "s ago" : "—"));
  }

  function connectWs() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(proto + "://" + location.host + "/ws/live");
    ws.onmessage = function (ev) {
      try {
        render(JSON.parse(ev.data));
      } catch (e) {
        console.error(e);
      }
    };
    ws.onclose = function () {
      $("conn-dot").className = "conn-dot offline";
      setTimeout(connectWs, 2000);
    };
  }

  fetch("/api/state")
    .then(function (r) { return r.json(); })
    .then(render)
    .catch(function () {
      $("meta-line").textContent = "Failed to load state";
    });

  connectWs();
})();
