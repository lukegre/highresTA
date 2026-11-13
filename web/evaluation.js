/* Evaluation page: held-out scores of the gridded-salinity model (model "E" in comparison-data.json)
 * by validation region, ocean basin, salinity bin and 5-year period, with a locator map of each group's samples.
 * Regions, labels and group order come from the Experiments page (content.json "comparison"); text from "evaluation". */
"use strict";
(async function startEvaluation() {
  const content = await (await fetch("content.json", { cache: "no-store" })).json();
  const COPY = content.comparison, EV = content.evaluation;
  const D = await (await fetch("comparison-data.json")).json();
  const MODEL = "E";
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
  const fmt = (x, d = 1) => x == null || isNaN(x) ? "–" : Number(x).toFixed(d);
  const signed = x => { const t = Math.abs(x).toFixed(1); return (+t === 0 ? "" : x > 0 ? "+" : "−") + t; };

  document.title = EV.title;
  $("heading").textContent = EV.heading;
  $("hero").textContent = EV.hero;
  $("intro").textContent = EV.intro;
  $("evTableTitle").textContent = EV.table_title;
  $("evMapTitle").textContent = EV.map_title;
  $("evMapNote").textContent = EV.map_note;
  $("evFootnote").textContent = EV.footnote;
  $("evGroupLabel").textContent = COPY.ui.group_by;

  /* ---------------- groups: same definitions as the Experiments page ---------------- */
  // Validation regions are built from switches: depth (deep / coastal) x latitude (high / mid and low), or the Mediterranean.
  // Neither or both switches of a pair mean "no restriction", so every combination is one of 10 selections (keys below).
  const SWITCHES = [["deep", "Deep ocean"], ["coastal", "Coastal ocean"], ["highlat", "High lat"], ["lowlat", "Mid and low lat"], ["med", "Mediterranean"], ["nomed", "Excl. Mediterranean"]];
  const SW_LABEL = Object.fromEntries(SWITCHES);
  const sw = { deep: false, coastal: false, highlat: false, lowlat: false, med: false, nomed: false };
  const selKey = (s = sw) => {
    if (s.med) return "med";
    const depth = s.deep === s.coastal ? "" : s.deep ? "deep" : "coastal", lat = s.highlat === s.lowlat ? "" : s.highlat ? "highlat" : "lowlat";
    const base = [depth, lat].filter(Boolean).join(",") || "all";
    return !s.nomed ? base : base === "all" ? "nomed" : `${base},nomed`;
  };
  const REGIONS = ["all", "deep", "coastal", "highlat", "lowlat", "deep,highlat", "deep,lowlat", "coastal,highlat", "coastal,lowlat", "med", "nomed"];
  // Rows always shown for the validation regions; any other selection made with the pills is added on top.
  const TABLE_REGIONS = ["deep,lowlat", "all", "highlat", "med"];
  const GROUP_ORDER = {
    valregion: TABLE_REGIONS,
    basin: ["Pacific", "Atlantic", "Indian", "Southern Ocean", "Arctic / subpolar N", "Mediterranean"],
    salbin: ["<30", "30–33", "33–34", "34–35", "35–36", "36–37", "≥37"],
    year: ["1980", "1985", "1990", "1995", "2000", "2005", "2010", "2015", "2020"] };
  // Short names for the table (the pills keep the full names)
  const SHORT = { deep: "Deep", coastal: "Coastal", highlat: "High lat", lowlat: "Mid/low lat", med: "Med", nomed: "excl. Med" };
  const regLabel = key => key === "all" ? "All" : key === "highlat" ? "All high lat" : key === "med" ? "Mediterranean" : key.split(",").map(k => SHORT[k]).join(" · ");
  const groupLabel = (by, g) => by === "valregion" ? regLabel(g) : by === "year" ? `${g}–${String(+g + 4).slice(2)}` : COPY.group_labels[by]?.[g] ?? g;
  const P = D.pts;
  function regMember(i, key) {
    if (key === "all") return true;
    if (key === "med") return P.basin[i] === "Mediterranean";
    if (key === "nomed") return P.basin[i] !== "Mediterranean";
    const parts = key.split(","), hi = Math.abs(P.lat[i]) > 60, co = P.region[i] === "coastal";
    return parts.every(k => k === "nomed" ? P.basin[i] !== "Mediterranean" : k === "deep" ? !co : k === "coastal" ? co : k === "highlat" ? hi : !hi);
  }
  const member = (i, g) => by() === "valregion" ? regMember(i, g) : String(P[by()][i]) === String(g);

  /* ---------------- scores per group (pooled test block + out-of-fold) ---------------- */
  // Region scores exist per row set (test, out-of-fold): pool them exactly from n, MAE, RMSE and bias.
  function regionScores() {
    const rows = D.valscores[MODEL].filter(x => x.salinity_product === "stacked");
    const pool = names => {
      const parts = rows.filter(x => names.includes(x.region)), n = parts.reduce((t, x) => t + x.n, 0);
      if (!n) return null;
      const mean = k => parts.reduce((t, x) => t + x.n * x[k], 0) / n;
      return { n, mae: mean("mean_absolute_error"), rmse: Math.sqrt(parts.reduce((t, x) => t + x.n * x.root_mean_squared_error ** 2, 0) / n), bias: mean("mean_bias") };
    };
    // The four depth x latitude cells are disjoint, so any selection is an exact pool of them.
    const DEPTH = { deep: 1, coastal: 1 }, LAT = { highlat: 1, lowlat: 1 };
    // Mediterranean scores exist only as a whole (basin table), so "excluding" it pools the five other basins exactly.
    const basins = D.heldout.all.basin[MODEL], others = Object.entries(basins).filter(([b]) => b !== "Mediterranean").map(([, v]) => v);
    const nOther = others.reduce((t, v) => t + v.n, 0), meanOther = k => others.reduce((t, v) => t + v.n * v[k], 0) / nOther;
    const out = { all: pool(["all"]), med: basins.Mediterranean,
      nomed: nOther ? { n: nOther, mae: meanOther("mae"), rmse: Math.sqrt(others.reduce((t, v) => t + v.n * v.rmse ** 2, 0) / nOther), bias: meanOther("bias") } : null };
    for (const key of REGIONS) {
      if (key === "all" || key === "med" || key === "nomed") continue;
      const parts = key.split(","), d = parts.filter(k => DEPTH[k]), l = parts.filter(k => LAT[k]);
      const depths = d.length ? [d[0] === "deep" ? "oceanic" : "coastal"] : ["oceanic", "coastal"];
      const lats = l.length ? [l[0] === "highlat" ? "> 60deg" : "< 60deg"] : ["< 60deg", "> 60deg"];
      out[key] = pool(depths.flatMap(dp => lats.map(la => `${dp} & ${la}`)));
    }
    for (const k of Object.keys(out)) if (!out[k]) delete out[k];
    return out;
  }
  const REGION_SCORES = regionScores();
  // "Excl. Mediterranean" combined with depth / latitude: no scores exist for the Mediterranean by depth or latitude, so its share
  // of the selection is estimated (rows: Mediterranean rows x the fraction of sampled Mediterranean points inside the selection;
  // scores: the Mediterranean's overall scores) and subtracted from the pooled selection. Approximate; marked in the table.
  const MED_PTS = P.basin.reduce((t, b) => t + (b === "Mediterranean"), 0);
  function keyScores(key) {
    if (REGION_SCORES[key]) return { ...REGION_SCORES[key], approx: false };
    const parts = key.split(","), base = parts.filter(k => k !== "nomed").join(",");
    if (parts.at(-1) !== "nomed" || !REGION_SCORES[base]) return null;
    let inSel = 0;
    for (let i = 0; i < P.lat.length; i++) if (P.basin[i] === "Mediterranean" && regMember(i, base)) inSel++;
    const b = REGION_SCORES[base], med = REGION_SCORES.med, nm = Math.min(b.n - 1, Math.round(med.n * inSel / MED_PTS)), n = b.n - nm;
    const rest = k => (b.n * b[k] - nm * med[k]) / n;
    return { n, mae: rest("mae"), rmse: Math.sqrt((b.n * b.rmse ** 2 - nm * med.rmse ** 2) / n), bias: rest("bias"), approx: true };
  }
  const scoresOf = grouping => grouping === "valregion" ? REGION_SCORES : D.heldout.all[grouping][MODEL];
  const overall = D.heldout.all.overall[MODEL];

  /* ---------------- controls ---------------- */
  const groupings = EV.groupings.filter(g => g === "valregion" ? Object.keys(REGION_SCORES).length : D.heldout.all[g]?.[MODEL]);
  const evGroup = $("evGroup");
  evGroup.classList.add("seg");
  evGroup.innerHTML = groupings.map((g, i) => `<button type="button" data-v="${g}" aria-pressed="${i === 0}">${COPY.groupings[g]}</button>`).join("");
  const by = () => evGroup.querySelector('[aria-pressed="true"]').dataset.v;
  evGroup.onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    evGroup.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    render();
  };

  /* ---------------- region switches (validation regions), below the map title: any combination ---------------- */
  const evRegion = $("evRegion");
  evRegion.classList.add("chips");
  evRegion.innerHTML = SWITCHES.map(([id, label]) => `<button type="button" class="chip" data-s="${id}" aria-pressed="false">${label}</button>`).join("");
  const syncSwitches = () => {
    evRegion.querySelectorAll("button").forEach(b => {
      b.setAttribute("aria-pressed", !!sw[b.dataset.s]);
      b.disabled = sw.med && b.dataset.s !== "med";
    });
  };
  const loadSwitches = key => {
    for (const k in sw) sw[k] = key.split(",").includes(k);
    syncSwitches();
  };
  evRegion.onclick = e => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    const id = b.dataset.s;
    sw[id] = !sw[id];
    // Scores of the Mediterranean exist only as a whole (no depth or latitude split), so it excludes the other switches.
    if (id === "med") for (const k in sw) if (k !== id) sw[k] = false;
    if (id === "nomed") sw.med = false;
    syncSwitches();
    setGroup(selKey());
  };

  /* ---------------- locator map: where a group's rows are (as on the Salinity sensitivity tab) ---------------- */
  const st = { sel: null, hover: null, groups: [] };
  function geoLayout() {
    return { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false,
      geo: { projection: { type: "natural earth" }, showland: true, landcolor: css("--map-land"), showocean: true, oceancolor: css("--map-ocean"),
        showcoastlines: false, showcountries: false, showframe: false, bgcolor: "rgba(0,0,0,0)", lataxis: { range: [-80, 90] } } };
  }
  function drawMap() {
    const g = st.hover ?? st.sel, on = [], off = [];
    for (let i = 0; i < P.lat.length; i++) (g != null && member(i, g) ? on : off).push(i);
    const trace = (idx, color, size, name) => ({ type: "scattergeo", mode: "markers", lat: idx.map(i => P.lat[i]), lon: idx.map(i => P.lon[i]),
      marker: { size, color, opacity: 0.9, line: { width: 0 } }, customdata: idx, hovertemplate: "%{lat:.1f}°, %{lon:.1f}°<extra>" + name + "</extra>", name });
    Plotly.react("evMap", [trace(off, css("--muted-pt"), 3, "other"), trace(on, "#f2b134", 4.5, g != null ? groupLabel(by(), g) : "")], geoLayout(), { displayModeBar: false, responsive: true });
    $("evChips").querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", c.dataset.g === String(g)));
    document.querySelectorAll("#evTable tr[data-g]").forEach(tr => tr.classList.toggle("sel", tr.dataset.g === String(st.sel)));
  }
  const setGroup = (g, hover) => {
    hover ? st.hover = g : (st.sel = g, st.hover = null);
    // A valregion selection may need its own table row (combinations that are not listed), so rebuild the table.
    if (!hover && by() === "valregion") render(); else drawMap();
  };
  for (const host of [$("evChips"), $("evTable")]) {
    const pick = e => e.target.closest("[data-g]");
    host.addEventListener("click", e => { const x = pick(e); if (x) setGroup(x.dataset.g); });
    host.addEventListener("mouseover", e => { const x = pick(e); if (x) setGroup(x.dataset.g, true); });
    host.addEventListener("mouseleave", () => setGroup(null, true));
  }
  let mapBound = false;
  function bindMapClick() {
    if (mapBound) return; mapBound = true;
    $("evMap").on("plotly_click", ev => {
      const g = st.groups.find(x => member(ev.points[0].customdata, x));
      if (g != null) setGroup(g);
    });
  }

  /* ---------------- table ---------------- */
  const METRIC = COPY.metrics;
  const tip = k => `data-tip="${esc(`<b>${METRIC[k].label}</b>${METRIC[k].tip}`)}" tabindex="0"`;
  const tipEl = document.body.appendChild(Object.assign(document.createElement("div"), { className: "tip", hidden: true }));
  tipEl.setAttribute("role", "tooltip");
  function showTip(target) {
    tipEl.innerHTML = target.dataset.tip; tipEl.hidden = false;
    const r = target.getBoundingClientRect(), w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    tipEl.style.left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8) + "px";
    tipEl.style.top = (r.bottom + h + 8 < innerHeight ? r.bottom + 8 : Math.max(8, r.top - h - 8)) + "px";
  }
  const onPoint = e => { const t = e.target.closest?.("[data-tip]"); if (t) showTip(t); else tipEl.hidden = true; };
  document.addEventListener("mouseover", onPoint);
  document.addEventListener("focusin", onPoint);
  document.addEventListener("focusout", () => { tipEl.hidden = true; });
  window.addEventListener("scroll", () => { tipEl.hidden = true; }, { passive: true });

  function render() {
    const g = by(), scores = g === "valregion" ? Object.fromEntries(REGIONS.map(k => [k, REGION_SCORES[k]])) : scoresOf(g);
    const groups = GROUP_ORDER[g].filter(x => scores[x]?.n > 0);
    // A valregion selection can be a combination that is not a table row (e.g. deep ocean excl. Mediterranean): add its row on top.
    const extra = g === "valregion" && st.sel && !groups.includes(st.sel) && keyScores(st.sel) ? st.sel : null;
    const row = (label, s, attrs = "") => `<tr ${attrs}><td class="name">${label}</td><td class="num">${s.n.toLocaleString("en")}</td>` +
      `<td class="num">${fmt(s.mae)}</td><td class="num">${fmt(s.rmse)}</td><td class="num">${signed(s.bias)}</td></tr>`;
    $("evTable").innerHTML = `<thead><tr><th>${COPY.groupings[g]}</th><th class="num" ${tip("n")}>${METRIC.n.label}</th>` +
      ["mae", "rmse", "bias"].map(k => `<th class="num" ${tip(k)}>${METRIC[k].label}</th>`).join("") + `</tr></thead><tbody>` +
      (g === "valregion" ? "" : row(EV.all_label, overall, 'class="total"')) +
      (extra ? row(esc(groupLabel(g, extra)) + (keyScores(extra).approx ? " (est.)" : ""), keyScores(extra), `data-g="${esc(extra)}"`) : "") +
      groups.map(x => row(esc(groupLabel(g, x)), scores[x], `data-g="${esc(x)}"${g === "valregion" && x === "all" ? ' class="total"' : ""}`)).join("") + "</tbody>";
    $("evGroupNote").textContent = EV.group_note[g];
    st.groups = groups;
    if (!groups.includes(st.sel) && !extra) st.sel = g === "valregion" ? "all" : groups[0] ?? null;
    st.hover = null;
    $("evRegion").hidden = g !== "valregion";
    $("evChips").hidden = g === "valregion";
    if (g === "valregion") loadSwitches(st.sel);
    $("evChips").innerHTML = groups.map(x => `<button class="chip" data-g="${esc(x)}">${esc(groupLabel(g, x))}</button>`).join("");
    drawMap();
    bindMapClick();
  }

  render();
  const rerender = () => render();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
  new MutationObserver(rerender).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})().catch(e => { console.error(e); document.body.prepend(Object.assign(document.createElement("p"), { textContent: "Could not load the evaluation data. Serve the web folder over HTTP and reload.", role: "alert" })); });
