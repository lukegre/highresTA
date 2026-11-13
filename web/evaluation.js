/* Evaluation page: held-out scores of the gridded-salinity model (model "E" in comparison-data.json)
 * by validation region, ocean basin, salinity bin and 5-year period, with a locator map of each group's samples.
 * Regions, labels and group order come from the Experiments page (content.json "comparison"); text from "evaluation". */
"use strict";
(async function startEvaluation() {
  const content = await (await fetch("content.json", { cache: "no-store" })).json();
  const COPY = content.comparison, EV = content.evaluation;
  const D = await (await fetch("comparison-data.json")).json();
  // Exact pooled scores for every combination of the region switches, from per-sample predictions (scripts/export_region_scores.py).
  const REGION_SCORES = (await (await fetch("evaluation-regions.json")).json()).scores;
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
  // Neither or both switches of a pair mean "no restriction"; the selection key lists depth, latitude, then Mediterranean (as in evaluation-regions.json).
  const SWITCHES = [["deep", "Deep ocean"], ["coastal", "Coastal ocean"], ["highlat", "High lat"], ["lowlat", "Mid and low lat"], ["med", "Mediterranean"], ["nomed", "Excl. Mediterranean"]];
  const SW_LABEL = Object.fromEntries(SWITCHES);
  const sw = { deep: false, coastal: false, highlat: false, lowlat: false, med: false, nomed: false };
  const selKey = (s = sw) => {
    const depth = s.deep === s.coastal ? "" : s.deep ? "deep" : "coastal", lat = s.highlat === s.lowlat ? "" : s.highlat ? "highlat" : "lowlat";
    return [depth, lat, s.med ? "med" : s.nomed ? "nomed" : ""].filter(Boolean).join(",") || "all";
  };
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
    const parts = key.split(","), hi = Math.abs(P.lat[i]) >= 60, co = P.region[i] === "coastal", med = P.basin[i] === "Mediterranean";
    return parts.every(k => k === "med" ? med : k === "nomed" ? !med : k === "deep" ? !co : k === "coastal" ? co : k === "highlat" ? hi : !hi);
  }
  const member = (i, g) => by() === "valregion" ? regMember(i, g) : String(P[by()][i]) === String(g);

  /* ---------------- scores per group (pooled test block + out-of-fold) ---------------- */
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
    });
  };
  const loadSwitches = key => {
    for (const k in sw) sw[k] = key.split(",").includes(k);
    syncSwitches();
  };
  evRegion.onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    const id = b.dataset.s;
    sw[id] = !sw[id];
    // Mediterranean and Excl. Mediterranean are opposites, so switching one on switches the other off.
    if (sw[id] && id === "med") sw.nomed = false;
    if (sw[id] && id === "nomed") sw.med = false;
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
    const g = by(), scores = scoresOf(g);
    const groups = GROUP_ORDER[g].filter(x => scores[x]?.n > 0);
    // A valregion selection can be a combination that is not a table row (e.g. deep ocean excl. Mediterranean): add its row on top.
    const extra = g === "valregion" && st.sel && !groups.includes(st.sel) && REGION_SCORES[st.sel] ? st.sel : null;
    const row = (label, s, attrs = "") => `<tr ${attrs}><td class="name">${label}</td><td class="num">${s.n.toLocaleString("en")}</td>` +
      `<td class="num">${fmt(s.mae)}</td><td class="num">${fmt(s.rmse)}</td><td class="num">${signed(s.bias)}</td></tr>`;
    $("evTable").innerHTML = `<thead><tr><th>${COPY.groupings[g]}</th><th class="num" ${tip("n")}>${METRIC.n.label}</th>` +
      ["mae", "rmse", "bias"].map(k => `<th class="num" ${tip(k)}>${METRIC[k].label}</th>`).join("") + `</tr></thead><tbody>` +
      (g === "valregion" ? "" : row(EV.all_label, overall, 'class="total"')) +
      (extra ? row(esc(groupLabel(g, extra)), REGION_SCORES[extra], `data-g="${esc(extra)}"`) : "") +
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
