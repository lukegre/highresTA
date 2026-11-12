/* Uncertainty explorer and related widgets.
   Registers block renderers for the app.js contract.
   Classes prefixed with ux- for isolation.
*/

window.BLOCKS = window.BLOCKS || {};

// ============================================================================
// Utilities
// ============================================================================

/** Simple seeded PRNG for reproducible random subset selection */
class SeededRandom {
  constructor(seed) {
    this.seed = seed;
  }
  next() {
    this.seed = (this.seed * 9301 + 49297) % 233280;
    return this.seed / 233280;
  }
}

/** Clamp value to [min, max] */
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/** Format number with given decimals */
function fmt(value, decimals = 2) {
  if (value === null || value === undefined) return '—';
  if (decimals === 'int') return Math.round(value).toString();
  return value.toFixed(decimals);
}

/** Get CSS variable value (e.g., --accent) and parse as color */
function getCSSVar(varName) {
  const root = document.documentElement;
  return getComputedStyle(root).getPropertyValue(varName).trim();
}

/** Get RdYlBu color via CSS variable */
function getRdylbuColor(t, reverse = false) {
  const clamped = clamp(t, 0, 1);
  const idx = Math.round(clamped * 10);
  const varName = reverse ? `--rdylbu-${idx}` : `--rdylbu-${10 - idx}`;
  return getCSSVar(varName);
}

/** Calculate percentile of array */
function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil(sorted.length * p) - 1;
  return sorted[Math.max(0, idx)];
}

/** Generate nice (1-2-5) tick positions inside [min, max] */
function niceTicks(min, max, targetCount = 5) {
  if (!(max > min)) return [];
  const raw = (max - min) / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let val = Math.ceil(min / step) * step; val <= max + step * 1e-6; val += step) {
    ticks.push(Math.round(val / step) * step);
  }
  return ticks.map((v) => Math.round(v * 1e9) / 1e9);
}

/** Compact tick label: no trailing zeros */
function tickLabel(v) {
  return String(Math.round(v * 1e6) / 1e6);
}

/** Render axis ticks and labels on SVG */
function renderAxisTicks(svg, scale, min, max, isVertical, isX, margin, plotW, plotH, ticks, formatter = tickLabel) {
  const tickSize = 4;
  const textOffset = 12;

  ticks.forEach((tickVal) => {
    if (isX) {
      const x = scale(tickVal);
      // Tick line
      const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tickLine.setAttribute('x1', x);
      tickLine.setAttribute('x2', x);
      tickLine.setAttribute('y1', margin.top + plotH);
      tickLine.setAttribute('y2', margin.top + plotH + tickSize);
      tickLine.setAttribute('stroke', getCSSVar('--text-muted'));
      tickLine.setAttribute('stroke-width', '0.5');
      svg.appendChild(tickLine);

      // Label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', x);
      label.setAttribute('y', margin.top + plotH + textOffset);
      label.setAttribute('text-anchor', 'middle');
      label.setAttribute('font-size', '11');
      label.setAttribute('fill', getCSSVar('--text-muted'));
      label.setAttribute('font-family', getCSSVar('--font-mono'));
      label.textContent = formatter(tickVal);
      svg.appendChild(label);
    } else {
      const y = scale(tickVal);
      // Tick line
      const tickLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      tickLine.setAttribute('x1', margin.left - tickSize);
      tickLine.setAttribute('x2', margin.left);
      tickLine.setAttribute('y1', y);
      tickLine.setAttribute('y2', y);
      tickLine.setAttribute('stroke', getCSSVar('--text-muted'));
      tickLine.setAttribute('stroke-width', '0.5');
      svg.appendChild(tickLine);

      // Label
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', margin.left - textOffset);
      label.setAttribute('y', y + 3);
      label.setAttribute('text-anchor', 'end');
      label.setAttribute('font-size', '11');
      label.setAttribute('fill', getCSSVar('--text-muted'));
      label.setAttribute('font-family', getCSSVar('--font-mono'));
      label.textContent = formatter(tickVal);
      svg.appendChild(label);
    }
  });
}

/** Mean and population variance */
function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

function variance(arr) {
  const m = mean(arr);
  return mean(arr.map((v) => (v - m) ** 2));
}

const SVG_NS = 'http://www.w3.org/2000/svg';

/** Create an SVG element with attributes and optional text */
function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Redraw an SVG on every container resize and return a manual redraw.
    A hidden container (e.g. a closed <details>) is skipped and drawn once it gets a size. */
function setupSVGResize(container, svg, renderFn) {
  const redraw = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    renderFn(w, h);
  };
  new ResizeObserver(redraw).observe(container);
  return redraw;
}

// ============================================================================
// Data computations
// ============================================================================

// Q90 - Q10 of a normal distribution spans 2 x 1.2816 sigma
const QUANTILE_WIDTH_PER_SIGMA = 2.563;

// The three terms of the total uncertainty, in stacking order
const TERMS = [
  { key: 'split', field: 'splitVar', uiKey: 'term_split', fallback: 'Data split' },
  { key: 'member', field: 'memberVar', uiKey: 'term_member', fallback: 'Member (HPO + seed)' },
  { key: 'quantile', field: 'quantVar', uiKey: 'term_quantile', fallback: 'Quantile width' },
];

/** Ensemble statistics of one sample for the selected members.
    sigma = sqrt(splitVar + memberVar + quantVar): the first two decompose the variance of the
    members' medians (law of total variance, as decompose_variance in quantile_ensemble.py),
    the last is the ensemble Q10-Q90 width expressed as a standard deviation. */
function computeEnsembleStats(sample, memberIndices, members) {
  const q1 = sample.pred.map((p) => p[0]);
  const q2 = sample.pred.map((p) => p[1]);
  const q3 = sample.pred.map((p) => p[2]);
  const pick = (q) => mean(memberIndices.map((i) => q[i]));
  const ensQ1 = pick(q1);
  const ensQ2 = pick(q2);
  const ensQ3 = pick(q3);

  const mediansBySplit = new Map();
  memberIndices.forEach((i) => {
    const split = members[i].split;
    if (!mediansBySplit.has(split)) mediansBySplit.set(split, []);
    mediansBySplit.get(split).push(q2[i]);
  });
  const groups = [...mediansBySplit.values()];
  const splitVar = variance(groups.map(mean));
  const memberVar = mean(groups.map(variance));
  const quantVar = ((ensQ3 - ensQ1) / QUANTILE_WIDTH_PER_SIGMA) ** 2;

  return {
    observed: sample.observed,
    ensQ1,
    ensQ2,
    ensQ3,
    spread: Math.sqrt(variance(memberIndices.map((i) => q2[i]))),
    splitVar,
    memberVar,
    quantVar,
    sigma: Math.sqrt(splitVar + memberVar + quantVar),
    width: ensQ3 - ensQ1,
    covered: sample.observed >= ensQ1 && sample.observed <= ensQ3 ? 1 : 0,
    error: Math.abs(ensQ2 - sample.observed),
    q1,
    q2,
    q3,
  };
}

/** Headline scores over per-sample stats */
function summarize(rows) {
  const avg = (f) => mean(rows.map(f));
  return {
    mae: avg((r) => r.error),
    rmse: Math.sqrt(avg((r) => r.error ** 2)),
    coverage: avg((r) => r.covered),
    width: avg((r) => r.width),
    sigma: avg((r) => r.sigma),
  };
}

// ============================================================================
// 1. UNCERTAINTY EXPLORER
// ============================================================================

window.BLOCKS['uncertainty-explorer'] = (host, block, ctx) => {
  const { data, content, md } = ctx;
  const { explorer } = content;
  const ui = content.ui;
  const labels = explorer.labels;
  const { members, test_sample } = data;
  const nSplits = data.run.n_splits;
  const nPerSplit = data.run.n_replicates;
  const copy = (key, fallback) => ui[key] || fallback;

  // === State ===
  let selectedMembers = new Set(members.map((_, i) => i));
  let selectedSampleIdx = 0;
  let randomN = 10;
  let currentPreset = 'all';
  let rows = []; // per-sample stats for the current member selection
  let colours = []; // per-sample colour (by sigma_total)
  let colourRange = { min: 0, max: 1 };

  function recompute() {
    const indices = Array.from(selectedMembers);
    rows = test_sample.map((s) => computeEnsembleStats(s, indices, members));
    const sigmas = rows.map((r) => r.sigma);
    colourRange = { min: percentile(sigmas, 0.05), max: percentile(sigmas, 0.95) };
    const span = colourRange.max - colourRange.min || 1;
    colours = sigmas.map((v) => getRdylbuColor(clamp((v - colourRange.min) / span, 0, 1)));
  }

  // === Member grid ===
  function updateMemberGrid() {
    const gridContainer = host.querySelector('.ux-member-grid');
    gridContainer.innerHTML = '';

    // Corner + caption: the number in each cell is that member's test MAE
    const corner = document.createElement('div');
    corner.className = 'ux-member-header ux-member-corner';
    corner.textContent = copy('member_cell_unit', 'MAE');
    corner.title = 'Test MAE of each member (µmol/kg)';
    corner.style.gridColumn = 1;
    corner.style.gridRow = 1;
    gridContainer.appendChild(corner);

    // Column headers: member index within a split
    for (let col = 0; col < nPerSplit; col++) {
      const h = document.createElement('div');
      h.className = 'ux-member-header';
      h.textContent = `Model ${col}`;
      h.style.gridColumn = 2 + col;
      h.style.gridRow = 1;
      h.setAttribute('aria-label', `Member index ${col}`);
      h.addEventListener('click', () => toggleReplicate(col));
      gridContainer.appendChild(h);
    }

    // Row headers: splits
    for (let row = 0; row < nSplits; row++) {
      const h = document.createElement('div');
      h.className = 'ux-member-header';
      h.textContent = `Split ${row + 1}`;
      h.style.gridColumn = 1;
      h.style.gridRow = row + 2;
      h.setAttribute('aria-label', `Split ${row + 1}`);
      h.addEventListener('click', () => toggleSplit(row + 1));
      gridContainer.appendChild(h);
    }

    // Member cells
    for (let split = 1; split <= nSplits; split++) {
      for (let rep = 0; rep < nPerSplit; rep++) {
        const memberIdx = members.findIndex((m) => m.split === split && m.replicate === rep);
        if (memberIdx < 0) continue;
        const cell = document.createElement('button');
        cell.className = 'ux-member-cell';
        const m = members[memberIdx];
        cell.textContent = fmt(m.test.mae, 1);
        cell.title = `Split ${split} · model ${rep}\nTrial ${m.trial_number}, loss rank ${m.loss_rank}\nDropped: ${m.dropped_features.join(', ') || 'none'}\nTest MAE ${fmt(m.test.mae, 1)} µmol/kg · 80 % coverage ${fmt(m.test.coverage80 * 100, 0)} %`;
        cell.setAttribute('aria-pressed', selectedMembers.has(memberIdx) ? 'true' : 'false');
        cell.setAttribute('aria-label', `Split ${split}, member index ${rep}`);
        cell.style.gridColumn = rep + 2;
        cell.style.gridRow = split + 1;
        cell.addEventListener('click', () => toggleMember(memberIdx));
        gridContainer.appendChild(cell);
      }
    }
  }

  function toggleMember(idx) {
    if (selectedMembers.has(idx)) {
      if (selectedMembers.size > 1) selectedMembers.delete(idx);
    } else {
      selectedMembers.add(idx);
    }
    updateSelection();
  }

  function toggleGroup(indices) {
    const allSelected = indices.every((i) => selectedMembers.has(i));
    if (allSelected && selectedMembers.size > indices.length) {
      indices.forEach((i) => selectedMembers.delete(i));
    } else {
      indices.forEach((i) => selectedMembers.add(i));
    }
    updateSelection();
  }

  const toggleSplit = (split) => toggleGroup(members.flatMap((m, i) => (m.split === split ? [i] : [])));
  const toggleReplicate = (rep) => toggleGroup(members.flatMap((m, i) => (m.replicate === rep ? [i] : [])));

  function updateSelection() {
    recompute();
    updateMemberGrid();
    updateMetrics();
    updateColorbar();
    redrawSelection();
  }

  /** Everything that shows the selected sample */
  function redrawSelection() {
    redrawScatter();
    redrawMap();
    updateInspector();
    updateLiveRegion();
  }

  // === Preset handling ===
  function applyPreset(presetId) {
    currentPreset = presetId;
    selectedMembers.clear();

    if (presetId === 'all') {
      members.forEach((_, i) => selectedMembers.add(i));
    } else if (presetId === 'one-split') {
      const split = (parseInt(currentPreset.split) || 1);
      members.forEach((member, i) => { if (member.split === split) selectedMembers.add(i); });
    } else if (presetId === 'one-replicate') {
      const rep = (parseInt(currentPreset.replicate) || 0);
      members.forEach((member, i) => { if (member.replicate === rep) selectedMembers.add(i); });
    } else if (presetId === 'random') {
      const rng = new SeededRandom(42);
      members.map((_, i) => i)
        .sort(() => rng.next() - 0.5)
        .slice(0, randomN)
        .forEach((i) => selectedMembers.add(i));
    } else if (presetId === 'single') {
      selectedMembers.add(parseInt(currentPreset.single) || 0);
    }

    updateSelection();
  }

  // === Metrics ===
  const tiles = [
    { key: 'mae', label: copy('metric_mae', 'MAE'), unit: 'µmol/kg', value: (s) => fmt(s.mae, 2), sub: (s) => `RMSE ${fmt(s.rmse, 2)}` },
    { key: 'coverage', label: labels.coverage, unit: '%', value: (s) => fmt(s.coverage * 100, 1) },
    { key: 'sigma', label: copy('metric_sigma', 'Mean total uncertainty'), unit: 'µmol/kg', value: (s) => fmt(s.sigma, 2) },
  ];

  function updateMetrics() {
    const stats = summarize(rows);
    tiles.forEach((tile) => {
      const elem = host.querySelector(`[data-metric="${tile.key}"]`);
      elem.querySelector('.ux-metric-value').innerHTML = `${tile.value(stats)}<span class="ux-metric-unit">${tile.unit}</span>`;
      if (tile.sub) elem.querySelector('.ux-metric-sub').textContent = tile.sub(stats);
    });
    host.querySelector('.ux-selection-label').textContent = ui.member_count.replace('{count}', selectedMembers.size);
  }

  function updateColorbar() {
    const { min, max } = colourRange;
    const ramp = Array.from({ length: 11 }, (_, i) => `<div style="flex:1;background:${getRdylbuColor(i / 10)}"></div>`).join('');
    host.querySelector('.ux-colorbar').innerHTML = `
      <div class="ux-colorbar-label">${copy('sigma_color', 'Total uncertainty σ_total (µmol/kg)')}</div>
      <div class="ux-colorbar-ramp">${ramp}</div>
      <div class="ux-colorbar-tick"><span>${fmt(min, 2)}</span><span>${fmt((min + max) / 2, 2)}</span><span>${fmt(max, 2)}</span></div>`;
  }

  // === Markers shared by scatter and map ===
  /** Draw all samples (xy(i) gives [x, y]); the selected one last, enlarged with a contrasting ring */
  function drawMarkers(svg, xy, radius) {
    const marker = (i, r, extra = '') => {
      const [x, y] = xy(i);
      const circle = el('circle', { cx: x, cy: y, r, fill: colours[i], class: `ux-point ${extra}` });
      circle.addEventListener('click', () => selectSample(i));
      circle.addEventListener('mouseenter', (e) => showTooltip(e, i));
      circle.addEventListener('mouseleave', hideTooltip);
      return circle;
    };
    rows.forEach((_, i) => {
      if (i !== selectedSampleIdx) svg.appendChild(marker(i, radius));
    });
    const [x, y] = xy(selectedSampleIdx);
    svg.appendChild(el('circle', { cx: x, cy: y, r: radius + 6, class: 'ux-point-halo' }));
    svg.appendChild(marker(selectedSampleIdx, radius + 3, 'ux-point-selected'));
  }

  // === Scatter plot ===
  function drawScatter(width, height) {
    const svg = scatterSvg;
    const margin = { top: 12, right: 16, bottom: 46, left: 56 };
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;

    // same range on both axes so the dashed line is 1:1
    const values = test_sample.map((s) => s.observed).concat(rows.map((r) => r.ensQ2));
    const pad = (Math.max(...values) - Math.min(...values)) * 0.05;
    const axisMin = Math.min(...values) - pad;
    const axisMax = Math.max(...values) + pad;
    const xScale = (v) => margin.left + ((v - axisMin) / (axisMax - axisMin)) * plotW;
    const yScale = (v) => margin.top + (1 - (v - axisMin) / (axisMax - axisMin)) * plotH;
    const muted = getCSSVar('--text-muted');
    const border = getCSSVar('--border');

    svg.innerHTML = '';
    const ticks = niceTicks(axisMin, axisMax, 5);
    ticks.forEach((t) => {
      svg.appendChild(el('line', { x1: xScale(t), x2: xScale(t), y1: margin.top, y2: margin.top + plotH, stroke: border, 'stroke-width': 0.5, opacity: 0.5 }));
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: yScale(t), y2: yScale(t), stroke: border, 'stroke-width': 0.5, opacity: 0.5 }));
    });
    svg.appendChild(el('line', { x1: xScale(axisMin), y1: yScale(axisMin), x2: xScale(axisMax), y2: yScale(axisMax), stroke: border, 'stroke-width': 1.5, 'stroke-dasharray': '4,4' }));
    svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH, stroke: muted }));
    svg.appendChild(el('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotH, stroke: muted }));
    renderAxisTicks(svg, xScale, axisMin, axisMax, false, true, margin, plotW, plotH, ticks);
    renderAxisTicks(svg, yScale, axisMin, axisMax, true, false, margin, plotW, plotH, ticks);
    svg.appendChild(el('text', { x: margin.left + plotW / 2, y: height - 8, 'text-anchor': 'middle', 'font-size': 11, fill: muted }, ui.scatter_x));
    svg.appendChild(el('text', {
      x: 12, y: margin.top + plotH / 2, 'text-anchor': 'middle', 'font-size': 11, fill: muted,
      transform: `rotate(-90 12 ${margin.top + plotH / 2})`,
    }, ui.scatter_y));

    drawMarkers(svg, (i) => [xScale(test_sample[i].observed), yScale(rows[i].ensQ2)], 3.5);
  }

  // === Map ===
  // The SVG fills a 2:1 container, so lon -180..180 / lat -90..90 map linearly to x / y.
  function drawMap(width, height) {
    mapSvg.innerHTML = '';
    mapSvg.appendChild(el('image', { href: 'maps/land.png', x: 0, y: 0, width, height, preserveAspectRatio: 'none' }));
    drawMarkers(
      mapSvg,
      (i) => [((test_sample[i].lon + 180) / 360) * width, ((90 - test_sample[i].lat) / 180) * height],
      3.6,
    );
  }

  // === Sample inspector ===
  function updateInspector() {
    const sample = test_sample[selectedSampleIdx];
    const row = rows[selectedSampleIdx];
    const cell = (label, value, cls = '') => `<div class="ux-insp-cell ${cls}"><span>${label}</span><b>${value}</b></div>`;

    host.querySelector('.ux-insp-line').textContent =
      `${sample.cruise} · ${sample.date} · ${fmt(sample.lat, 2)}°, ${fmt(sample.lon, 2)}° · ${fmt(sample.depth, 0)} m`;

    host.querySelector('.ux-insp-stats').innerHTML = [
      cell(copy('inspector_observed', 'Observed'), fmt(sample.observed, 1)),
      cell(copy('inspector_q50', 'Ensemble Q50'), fmt(row.ensQ2, 1)),
      cell(copy('inspector_error', 'Error'), fmt(row.ensQ2 - sample.observed, 1)),
      cell(copy('inspector_interval', 'Q10–Q90'), `${fmt(row.ensQ1, 0)}–${fmt(row.ensQ3, 0)}`),
    ].join('');

    host.querySelector('.ux-insp-sigma').innerHTML =
      cell(copy('inspector_sigma_total', 'σ_total'), fmt(row.sigma, 2), 'ux-insp-total') +
      TERMS.map((t) => cell(copy(t.uiKey, t.fallback), fmt(Math.sqrt(row[t.field]), 2), `ux-term-${t.key}`)).join('');

    renderWhiskerPlot(host.querySelector('.ux-whisker-svg'), sample, row);
  }

  /** Per-member median and Q10-Q90 whiskers, grouped by split, with observed and ensemble lines */
  function renderWhiskerPlot(svg, sample, stats) {
    const container = svg.parentElement;

    const draw = () => {
      const { sample: smp, stats: s } = container.__whisker;
      const cs = getComputedStyle(container);
      const width = Math.max(
        240,
        container.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0),
      );
      const rowH = 4.5;
      const groupGap = 6;
      const margin = { left: 44, right: 8, top: 22, bottom: 34 };
      const ids = Array.from(selectedMembers).sort((x, y) => x - y);
      const groups = new Map();
      ids.forEach((m) => {
        const split = members[m].split;
        if (!groups.has(split)) groups.set(split, []);
        groups.get(split).push(m);
      });
      const plotW = width - margin.left - margin.right;
      const plotH = ids.length * rowH + (groups.size - 1) * groupGap + 8;
      const height = margin.top + plotH + margin.bottom;

      // x range fitted to the selected members' bands + observed + ensemble
      let lo = Math.min(smp.observed, s.ensQ1);
      let hi = Math.max(smp.observed, s.ensQ3);
      ids.forEach((m) => {
        lo = Math.min(lo, s.q1[m]);
        hi = Math.max(hi, s.q3[m]);
      });
      const pad = (hi - lo || 1) * 0.08;
      lo -= pad;
      hi += pad;
      const xScale = (v) => margin.left + ((v - lo) / (hi - lo)) * plotW;
      const muted = getCSSVar('--text-muted');
      const border = getCSSVar('--border');

      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      svg.style.height = `${height}px`;
      svg.innerHTML = '';

      // legend above the plot
      let lx = 4;
      [
        ['Observed', getCSSVar('--c-observed'), 'line'],
        ['Ensemble', getCSSVar('--c-ensemble'), 'line'],
        ['Q10–Q90', getCSSVar('--c-band'), 'box'],
        ['Members', getCSSVar('--c-member'), 'dot'],
      ].forEach(([label, color, kind]) => {
        if (kind === 'line') svg.appendChild(el('line', { x1: lx, x2: lx + 12, y1: 9, y2: 9, stroke: color, 'stroke-width': 2 }));
        else if (kind === 'box') svg.appendChild(el('rect', { x: lx, y: 4, width: 12, height: 10, fill: color }));
        else svg.appendChild(el('circle', { cx: lx + 6, cy: 9, r: 3, fill: color }));
        svg.appendChild(el('text', { x: lx + 16, y: 13, 'font-size': 10, fill: muted }, label));
        lx += 16 + label.length * 5.4 + 10;
      });

      // grid + x axis
      const ticks = niceTicks(lo, hi, 4);
      ticks.forEach((v) => {
        svg.appendChild(el('line', { x1: xScale(v), x2: xScale(v), y1: margin.top, y2: margin.top + plotH, stroke: border, 'stroke-width': 0.5, opacity: 0.7 }));
      });
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH, stroke: muted }));
      renderAxisTicks(svg, xScale, lo, hi, false, true, margin, plotW, plotH, ticks);
      svg.appendChild(el('text', { x: margin.left + plotW / 2, y: height - 4, 'text-anchor': 'middle', 'font-size': 10, fill: muted }, 'TA (µmol/kg)'));

      // ensemble band behind everything
      svg.appendChild(el('rect', {
        x: xScale(s.ensQ1), y: margin.top, width: Math.max(1, xScale(s.ensQ3) - xScale(s.ensQ1)), height: plotH,
        class: 'ux-whisker-ensemble-band',
      }));

      // member rows grouped by split
      let y = margin.top + 8;
      groups.forEach((groupIds, split) => {
        const y0 = y;
        groupIds.forEach((m) => {
          const rowGroup = el('g', { class: 'ux-whisker-row' });
          rowGroup.appendChild(el('title', {}, `Split ${split}, member ${members[m].replicate}: median ${fmt(s.q2[m], 1)} (Q10 ${fmt(s.q1[m], 1)} to Q90 ${fmt(s.q3[m], 1)})`));
          rowGroup.appendChild(el('rect', {
            x: xScale(s.q1[m]), y: y - 1.2, width: Math.max(1, xScale(s.q3[m]) - xScale(s.q1[m])), height: 2.4, class: 'ux-whisker-band',
          }));
          rowGroup.appendChild(el('circle', { cx: xScale(s.q2[m]), cy: y, r: 1.8, class: 'ux-whisker-dot' }));
          svg.appendChild(rowGroup);
          y += rowH;
        });
        svg.appendChild(el('text', {
          x: margin.left - 6, y: (y0 + y - rowH) / 2 + 3, 'text-anchor': 'end', 'font-size': 10, 'font-weight': 600, fill: muted,
        }, `Split ${split}`));
        y += groupGap;
      });

      // observed + ensemble median lines on top
      svg.appendChild(el('line', { x1: xScale(smp.observed), x2: xScale(smp.observed), y1: margin.top, y2: margin.top + plotH, class: 'ux-whisker-observed' }));
      svg.appendChild(el('line', { x1: xScale(s.ensQ2), x2: xScale(s.ensQ2), y1: margin.top, y2: margin.top + plotH, class: 'ux-whisker-ensemble' }));
    };

    container.__whisker = { sample, stats };
    if (!container.__whiskerObserver) {
      let lastWidth = 0;
      container.__whiskerObserver = new ResizeObserver(() => {
        if (container.clientWidth && container.clientWidth !== lastWidth) {
          lastWidth = container.clientWidth;
          draw();
        }
      });
      container.__whiskerObserver.observe(container);
    }
    draw();
  }

  // === Sample selection ===
  function selectSample(idx) {
    selectedSampleIdx = (idx + test_sample.length) % test_sample.length;
    redrawSelection();
  }

  // === Tooltip ===
  let tooltipElem = null;

  function showTooltip(e, idx) {
    const sample = test_sample[idx];
    const row = rows[idx];
    if (!tooltipElem) {
      tooltipElem = document.createElement('div');
      tooltipElem.className = 'ux-scatter-tooltip';
      host.appendChild(tooltipElem);
    }
    const line = (label, value) => `<div class="ux-tooltip-row"><div class="ux-tooltip-label">${label}</div><div class="ux-tooltip-value">${value}</div></div>`;
    tooltipElem.innerHTML = [
      line(ui.tooltip_cruise, sample.cruise),
      line(ui.tooltip_date, sample.date),
      line(ui.tooltip_position, `${fmt(sample.lat, 1)}°, ${fmt(sample.lon, 1)}°`),
      line(ui.tooltip_observed, fmt(sample.observed, 2)),
      line(ui.tooltip_predicted, fmt(row.ensQ2, 2)),
      line(copy('tooltip_sigma', 'σ_total:'), fmt(row.sigma, 2)),
    ].join('');
    tooltipElem.style.display = '';

    const rect = e.target.getBoundingClientRect();
    const parentRect = host.getBoundingClientRect();
    tooltipElem.style.left = rect.left - parentRect.left + 10 + 'px';
    tooltipElem.style.top = rect.top - parentRect.top + 10 + 'px';
  }

  function hideTooltip() {
    if (tooltipElem) tooltipElem.style.display = 'none';
  }

  // === Live region for accessibility ===
  function updateLiveRegion() {
    const sample = test_sample[selectedSampleIdx];
    const row = rows[selectedSampleIdx];
    host.querySelector('.ux-live-region').textContent = ui.live_summary
      .replace('{lat}', sample.lat.toFixed(1))
      .replace('{lon}', sample.lon.toFixed(1))
      .replace('{observed}', sample.observed.toFixed(2))
      .replace('{predicted}', row.ensQ2.toFixed(2))
      .replace('{spread}', row.spread.toFixed(2))
      .replace('{count}', selectedMembers.size);
  }

  // === Initial render ===
  host.classList.add('ux-host');
  host.innerHTML = `
    <div class="ux-explorer-heading"><h3>${explorer.title}</h3><p>${md(explorer.intro, ctx)}</p></div>
    <div class="ux-live-region" role="status" aria-live="polite" aria-atomic="true"></div>

    <div class="ux-explorer">
      <div class="ux-controls">
        <div class="ux-member-grid-controls">
          ${explorer.presets.map((preset, index) => `<button type="button" class="ux-preset-chip" aria-pressed="${index === 0}">${preset.label}</button>`).join('')}
          <div class="ux-selection-label"></div>
        </div>

        <div class="ux-random-controls ux-hidden">
          <label aria-label="${ui.random_count_label}">
            ${ui.random_count_label}:
            <input type="range" class="ux-random-slider" min="2" max="${members.length}" value="10">
            <span class="ux-random-value">10</span>
          </label>
          <button type="button" class="ux-reshuffle-btn">${ui.reshuffle}</button>
        </div>

        <div class="ux-member-grid"></div>
        <p class="ux-member-note">${copy('member_cell_note', 'Number in each cell: test MAE of that member (µmol/kg). Dark cells are included in the ensemble.')}</p>

        <div class="ux-metrics-grid">
          ${tiles.map((t) => `
            <div class="ux-metric-tile" data-metric="${t.key}">
              <div class="ux-metric-label">${t.label}</div>
              <div class="ux-metric-value"></div>
              ${t.sub ? '<div class="ux-metric-sub"></div>' : ''}
            </div>`).join('')}
        </div>
      </div>

      <div class="ux-chart-card">
        <div class="ux-scatter-wrapper"><svg class="ux-scatter-svg"></svg></div>
        <div class="ux-colorbar"></div>
      </div>
    </div>

    <div class="ux-map-row">
      <div class="ux-chart-card ux-map-card">
        <div class="ux-map-container"><svg class="ux-map-svg"></svg></div>
      </div>

      <div class="ux-sample-inspector">
        <div class="ux-inspector-header">
          <strong>${ui.sample_inspector}</strong>
          <div class="ux-sample-nav">
            <button type="button" class="ux-nav-btn" data-step="-1" aria-label="${ui.previous_sample}" title="${ui.previous_sample}">←</button>
            <button type="button" class="ux-nav-btn" data-step="1" aria-label="${ui.next_sample}" title="${ui.next_sample}">→</button>
            <select class="ux-sort-select" aria-label="${ui.jump_to}">
              <option value="">${ui.jump_to}</option>
              <option value="uncertain">${ui.sort_options[0]}</option>
              <option value="least">${ui.sort_options[1]}</option>
              <option value="error">${ui.sort_options[2]}</option>
              <option value="random">${ui.sort_options[3]}</option>
            </select>
          </div>
        </div>
        <div class="ux-insp-line"></div>
        <div class="ux-insp-stats"></div>
        <div class="ux-insp-sigma"></div>
        <div class="ux-whisker-plot-container"><svg class="ux-whisker-svg"></svg></div>
      </div>
    </div>
  `;

  const scatterSvg = host.querySelector('.ux-scatter-svg');
  const mapSvg = host.querySelector('.ux-map-svg');
  const redrawScatter = setupSVGResize(host.querySelector('.ux-scatter-wrapper'), scatterSvg, drawScatter);
  const redrawMap = setupSVGResize(host.querySelector('.ux-map-container'), mapSvg, drawMap);

  // === Event handlers ===
  const presetChips = host.querySelectorAll('.ux-member-grid-controls .ux-preset-chip');
  presetChips.forEach((chip, idx) => {
    const presetIds = ['all', 'one-split', 'one-replicate', 'random', 'single'];
    chip.addEventListener('click', () => {
      // Toggle if same preset, or switch
      if (currentPreset === presetIds[idx]) {
        // Cycle to next variant
        if (presetIds[idx] === 'one-split') {
          currentPreset = 'one-split-2';
        } else if (presetIds[idx] === 'one-replicate') {
          currentPreset = 'one-replicate-1';
        } else if (presetIds[idx] === 'single') {
          currentPreset = 'single-1';
        }
      } else {
        currentPreset = presetIds[idx];
      }
      applyPreset(currentPreset);

      presetChips.forEach((c, i) => {
        c.setAttribute('aria-pressed', i === idx ? 'true' : 'false');
      });
    });
  });

  host.querySelector('.ux-random-slider').addEventListener('input', (e) => {
    randomN = parseInt(e.target.value);
    host.querySelector('.ux-random-value').textContent = randomN;
    if (currentPreset === 'random') applyPreset('random');
  });
  host.querySelector('.ux-reshuffle-btn').addEventListener('click', () => applyPreset('random'));

  host.querySelectorAll('.ux-nav-btn').forEach((btn) => {
    btn.addEventListener('click', () => selectSample(selectedSampleIdx + parseInt(btn.dataset.step)));
  });

  // Jump to the sample with the highest score
  const argBest = (score) => rows.reduce((best, r, i) => (score(r) > score(rows[best]) ? i : best), 0);
  const jumpTo = {
    uncertain: () => argBest((r) => r.sigma),
    least: () => argBest((r) => -r.sigma),
    error: () => argBest((r) => r.error),
    random: () => Math.floor(Math.random() * rows.length),
  };
  host.querySelector('.ux-sort-select').addEventListener('change', (e) => {
    if (jumpTo[e.target.value]) selectSample(jumpTo[e.target.value]());
    e.target.value = '';
  });

  // Initial update: start on the most uncertain of the first 100 samples
  recompute();
  selectedSampleIdx = rows.slice(0, 100).reduce((best, r, i) => (r.sigma > rows[best].sigma ? i : best), 0);
  updateMemberGrid();
  updateMetrics();
  updateColorbar();
  updateInspector();
  updateLiveRegion();
};

// ============================================================================
// 2. UNCERTAINTY BUDGET
// ============================================================================

window.BLOCKS['uncertainty-budget'] = (host, block, ctx) => {
  const { data, content } = ctx;
  const copy = (key, fallback) => content.ui[key] || fallback;
  const all = data.members.map((_, i) => i);
  const rows = data.test_sample.map((s) => computeEnsembleStats(s, all, data.members));
  const budget = (subset) => {
    const terms = TERMS.map((t) => ({ ...t, name: copy(t.uiKey, t.fallback), variance: mean(subset.map((r) => r[t.field])) }));
    return { terms, total: terms.reduce((s, t) => s + t.variance, 0) };
  };
  const overall = budget(rows);
  const pct = (v, total) => `${fmt((v / total) * 100, 1)} %`;

  // Bins of equal sample count, ordered by predicted uncertainty
  const N_BINS = 8;
  const sorted = [...rows].sort((a, b) => a.sigma - b.sigma);
  const bins = Array.from({ length: N_BINS }, (_, i) => {
    const subset = sorted.slice(Math.round((i * sorted.length) / N_BINS), Math.round(((i + 1) * sorted.length) / N_BINS));
    const b = budget(subset);
    return { ...b, n: subset.length, predicted: Math.sqrt(b.total), actual: Math.sqrt(mean(subset.map((r) => r.error ** 2))) };
  });

  // Square log-log plot. Each bin is one row at its actual error: the predicted uncertainty
  // grows from left to right as the variance terms are added, ending at sigma_total.
  bins.forEach((b) => {
    let sum = 0;
    b.steps = b.terms.map((t) => ({ key: t.key, sigma: Math.sqrt(sum += t.variance) }));
  });
  const SIZE = 460, m = { l: 58, r: 16, t: 16, b: 52 }, plot = SIZE - m.l - m.r;
  const values = bins.flatMap((b) => [b.steps[0].sigma, b.predicted, b.actual]);
  const lo = Math.min(...values) / 1.5, hi = Math.max(...values) * 1.4;
  const pos = (v) => (Math.log(v / lo) / Math.log(hi / lo)) * plot;
  const X = (v) => m.l + pos(v), Y = (v) => m.t + plot - pos(v);
  const ticks = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500].filter((t) => t > lo && t < hi);
  const row = (b) => `<line class="ux-cal-link" x1="${X(b.steps[0].sigma)}" x2="${X(b.predicted)}" y1="${Y(b.actual)}" y2="${Y(b.actual)}"/>`
    + b.steps.map((st, k) => `<circle class="ux-term-${st.key}" cx="${X(st.sigma)}" cy="${Y(b.actual)}" r="${k === b.steps.length - 1 ? 6.5 : 4}"/>`).join('');
  const tip = (b) => `Predicted σ_total ${fmt(b.predicted, 1)}, actual RMSE ${fmt(b.actual, 1)} µmol/kg (n = ${b.n})\n`
    + b.terms.map((t) => `${t.name}: ${pct(t.variance, b.total)}`).join('\n');
  const svg = `
    <svg class="ux-cal-plot" viewBox="0 0 ${SIZE} ${SIZE}" role="img" aria-label="${copy('cal_title', 'Predicted total uncertainty against actual error')}">
      ${ticks.map((t) => `
        <line class="ux-cal-grid" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${m.t + plot}"/>
        <line class="ux-cal-grid" x1="${m.l}" x2="${m.l + plot}" y1="${Y(t)}" y2="${Y(t)}"/>
        <text class="ux-cal-tick" x="${X(t)}" y="${m.t + plot + 16}" text-anchor="middle">${t}</text>
        <text class="ux-cal-tick" x="${m.l - 8}" y="${Y(t) + 4}" text-anchor="end">${t}</text>`).join('')}
      <rect class="ux-cal-frame" x="${m.l}" y="${m.t}" width="${plot}" height="${plot}"/>
      <line class="ux-cal-identity" x1="${X(lo)}" y1="${Y(lo)}" x2="${X(hi)}" y2="${Y(hi)}"/>
      <text class="ux-cal-note" x="${m.l + 10}" y="${m.t + 18}">${copy('cal_above', 'above the line: actual error larger than predicted (over-confident)')}</text>
      <text class="ux-cal-note" x="${m.l + plot - 10}" y="${m.t + plot - 10}" text-anchor="end">${copy('cal_below', 'below the line: uncertainty over-estimated')}</text>
      ${bins.map((b) => `<g class="ux-cal-row"><title>${tip(b)}</title>${row(b)}</g>`).join('')}
      <text class="ux-cal-axis" x="${m.l + plot / 2}" y="${SIZE - 10}" text-anchor="middle">${copy('cal_axis_x', 'Predicted uncertainty σ (µmol kg⁻¹), log scale')}</text>
      <text class="ux-cal-axis" transform="translate(16 ${m.t + plot / 2}) rotate(-90)" text-anchor="middle">${copy('cal_axis_y', 'Actual error, RMSE (µmol kg⁻¹), log scale')}</text>
    </svg>`;

  // append: the host already holds the block title
  host.insertAdjacentHTML('beforeend', `
    <div class="ux-cal">
      <div>${svg}</div>
      <div class="ux-cal-side">
        <p>${copy('cal_guide', 'Test samples are sorted by predicted σ_total and cut into {bins} equal groups, one row each, placed at the group\'s actual error. Along a row the predicted uncertainty grows as terms are added: data split only, then + member, then + quantile width, which gives σ_total (large dot). A large dot on the dashed line means σ_total matches the actual error.').replace('{bins}', N_BINS)}</p>
        <table class="ux-budget-table">
          <thead>
            <tr><th>${copy('budget_term', 'Term')}</th><th>${copy('budget_sigma', 'σ (µmol/kg)')}</th><th>${copy('budget_share', 'Share of variance')}</th></tr>
          </thead>
          <tbody>
            ${overall.terms.map((t) => `
              <tr><td><span class="ux-budget-swatch ux-term-${t.key}"></span>${t.name}</td><td>${fmt(Math.sqrt(t.variance), 1)}</td><td>${pct(t.variance, overall.total)}</td></tr>`).join('')}
          </tbody>
          <tfoot>
            <tr><td>${copy('budget_total', 'Total (σ_total)')}</td><td>${fmt(Math.sqrt(overall.total), 1)}</td><td>100 %</td></tr>
          </tfoot>
        </table>
        <p class="ux-budget-caption">${copy('budget_caption', 'σ is the square root of the mean variance over the {samples} displayed test samples, all {members} members.').replace('{samples}', rows.length).replace('{members}', all.length)}</p>
      </div>
    </div>`);
};


// ============================================================================
// 4. ENSEMBLE GROWTH
// ============================================================================

window.BLOCKS['ensemble-growth'] = (host, block, ctx) => {
  const { data, content } = ctx;
  const ui = content.ui;
  const { growth } = data;

  host.innerHTML = `
    <div class="ux-growth-panels">
      <div class="ux-growth-panel">
        <div class="ux-growth-panel-title">${ui.growth_rmse}</div>
        <div class="ux-growth-plot" data-kind="rmse"><svg class="ux-growth-svg"></svg></div>
      </div>
      <div class="ux-growth-panel">
        <div class="ux-growth-panel-title">${ui.growth_coverage}</div>
        <div class="ux-growth-plot" data-kind="coverage"><svg class="ux-growth-svg"></svg></div>
      </div>
    </div>
    <div class="ux-growth-caption"></div>
  `;

  // kind: key in growth rows, yLabel, optional reference value (dashed) with label, extra y-range to include
  const specs = {
    rmse: { key: 'rmse', yLabel: ui.chart_axis_rmse, include: [], pad: 0.15 },
    coverage: { key: 'coverage80', yLabel: ui.chart_axis_coverage, ref: { value: 0.8, label: ui.nominal_coverage }, include: [0.8], pad: 0.1 },
  };
  const xTicks = growth.map((d) => d.n).filter((n) => [1, 7, 14, 21, 28, 42].includes(n));

  host.querySelectorAll('.ux-growth-plot').forEach((plot) => {
    const spec = specs[plot.dataset.kind];
    const svg = plot.querySelector('svg');

    const render = (width, height) => {
      const margin = { top: 16, right: 16, bottom: 40, left: 52 };
      const plotW = width - margin.left - margin.right;
      const plotH = height - margin.top - margin.bottom;
      const ys = growth.map((d) => d[spec.key]).concat(spec.include);
      let yMin = Math.min(...ys);
      let yMax = Math.max(...ys);
      const pad = (yMax - yMin || 1) * spec.pad;
      yMin -= pad;
      yMax += pad;
      const nMin = growth[0].n;
      const nMax = growth[growth.length - 1].n;
      const xScale = (v) => margin.left + ((v - nMin) / (nMax - nMin)) * plotW;
      const yScale = (v) => margin.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;
      const muted = getCSSVar('--text-muted');
      const border = getCSSVar('--border');
      const accent = getCSSVar('--accent');

      svg.innerHTML = '';
      const yTicks = niceTicks(yMin, yMax, 4);
      yTicks.forEach((v) => {
        svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: yScale(v), y2: yScale(v), stroke: border, 'stroke-width': 0.5, opacity: 0.6 }));
      });
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: margin.top + plotH, y2: margin.top + plotH, stroke: muted }));
      svg.appendChild(el('line', { x1: margin.left, x2: margin.left, y1: margin.top, y2: margin.top + plotH, stroke: muted }));
      renderAxisTicks(svg, xScale, nMin, nMax, false, true, margin, plotW, plotH, xTicks);
      renderAxisTicks(svg, yScale, yMin, yMax, true, false, margin, plotW, plotH, yTicks,
        spec.key === 'coverage80' ? (v) => `${Math.round(v * 100)} %` : tickLabel);

      if (spec.ref) {
        const y = yScale(spec.ref.value);
        svg.appendChild(el('line', { x1: margin.left, x2: margin.left + plotW, y1: y, y2: y, stroke: getCSSVar('--c-warn'), 'stroke-width': 1.5, 'stroke-dasharray': '4,3' }));
        svg.appendChild(el('text', { x: margin.left + plotW, y: y - 5, 'text-anchor': 'end', 'font-size': 11, fill: getCSSVar('--c-warn') }, spec.ref.label));
      }

      const d = growth.map((row, i) => `${i ? 'L' : 'M'} ${xScale(row.n)} ${yScale(row[spec.key])}`).join(' ');
      svg.appendChild(el('path', { d, stroke: accent, 'stroke-width': 2, fill: 'none' }));
      growth.forEach((row) => {
        const c = el('circle', { cx: xScale(row.n), cy: yScale(row[spec.key]), r: 3.5, fill: accent });
        const value = spec.key === 'coverage80' ? (row[spec.key] * 100).toFixed(1) + ' %' : row[spec.key].toFixed(2);
        c.appendChild(el('title', {}, ui.growth_point_title.replace('{members}', row.n).replace('{value}', value)));
        svg.appendChild(c);
      });

      svg.appendChild(el('text', { x: margin.left + plotW / 2, y: height - 6, 'text-anchor': 'middle', 'font-size': 11, fill: muted }, ui.growth_x));
      svg.appendChild(el('text', { x: 12, y: margin.top + plotH / 2, 'text-anchor': 'middle', 'font-size': 11, fill: muted, transform: `rotate(-90 12 ${margin.top + plotH / 2})` }, spec.yLabel));
    };
    setupSVGResize(plot, svg, render);
  });

  const finalRmse = growth[growth.length - 1].rmse;
  const flat = growth.find((d) => d.rmse <= finalRmse + 0.05);
  const finalCov = growth[growth.length - 1].coverage80;
  host.querySelector('.ux-growth-caption').textContent = ui.growth_caption.replace('{members}', flat ? flat.n : 'N/A').replace('{coverage}', (finalCov * 100).toFixed(1));
};
