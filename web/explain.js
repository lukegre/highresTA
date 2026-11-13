/* SHAP contributions and interactions: beeswarm, contribution maps, interaction matrix and pair viewer.
 * Data: explain-data.json, written by web/build_explain.py (packing described there); fetched once.
 * Registers window.BLOCKS['shap-summary'], ['shap-maps'] and ['interactions']; load after sensitivity.js
 * (shared helpers in window.SX) and before app.js. Each card holds exactly one figure. */
window.BLOCKS = window.BLOCKS || {};

(() => {
  const SX = window.SX;
  if (!SX) {
    console.error('explain.js needs sensitivity.js (window.SX) loaded first');
    return;
  }
  const { bytes, cssVar, hexRgb, esc, makeRamp, lazy, fetchJson, mountMaps } = SX;
  const NS = 'http://www.w3.org/2000/svg';
  const NO_S = -32768;
  const MINUS = '−';

  const signed = (v, d = 1) => (v > 0 ? '+' : v < 0 ? MINUS : '') + Math.abs(v).toFixed(d);
  const plain = (v) => (v < 0 ? MINUS : '') + Math.abs(v);
  const sig3 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : String(Number(v.toPrecision(3)))).replace('-', MINUS);
  const line = (k, v) => `<div class="ux-tooltip-row"><div class="ux-tooltip-label">${esc(k)}</div><div class="ux-tooltip-value">${esc(v)}</div></div>`;
  const withUnit = (v, unit) => `${sig3(v)}${unit ? ` ${unit}` : ''}`;
  const percentile = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))))];

  /** Round tick values covering [lo, hi] with about `count` ticks. */
  function niceTicks(lo, hi, count) {
    const raw = (hi - lo) / Math.max(1, count);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
    const first = Math.ceil(lo / step - 1e-9) * step;
    const ticks = [];
    for (let t = first; t <= hi + 1e-9; t += step) ticks.push(Math.abs(t) < step * 1e-9 ? 0 : Number(t.toPrecision(12)));
    return { ticks, step };
  }

  /** Piecewise-linear ramp through the site's RdYlBu tokens (indices into --rdylbu-0..10). */
  function rampFrom(indices) {
    const stops = indices.map((i) => hexRgb(cssVar(`--rdylbu-${i}`) || '#888888'));
    const colour = (t) => {
      const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(x));
      const f = x - i;
      return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
    };
    const css = `linear-gradient(90deg, ${stops.map((c, i) => `rgb(${c}) ${(i / (stops.length - 1)) * 100}%`).join(', ')})`;
    return { colour, css };
  }
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
  const textOn = (c) => ((0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255 > 0.58 ? '#17302f' : '#ffffff');

  /** Call draw(width) now and whenever the element's width changes. */
  function fitWidth(el, draw) {
    let last = 0;
    const run = () => {
      const w = Math.floor(el.clientWidth);
      if (w > 0 && w !== last) { last = w; draw(w); }
    };
    if ('ResizeObserver' in window) new ResizeObserver(run).observe(el);
    else window.addEventListener('resize', run);
    run();
    return () => { last = 0; run(); };
  }

  /** Place a tooltip next to a pointer inside its positioned container. */
  function placeTip(tip, container, evt) {
    const box = container.getBoundingClientRect();
    const x = evt.clientX - box.left;
    const y = evt.clientY - box.top;
    const left = x + 12 + tip.offsetWidth > box.width ? x - 12 - tip.offsetWidth : x + 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, Math.min(y + 12, box.height - tip.offsetHeight))}px`;
  }

  const cache = {};
  const loadData = (url) => (cache[url] = cache[url] || fetchJson(url));
  const failed = (el, name, ui) => (err) => {
    console.error(`${name}:`, err);
    el.innerHTML = `<p class="sx-loading">${esc((ui || {}).error || 'Could not load the figure.')}</p>`;
  };

  function card(host, title, sub) {
    const el = document.createElement('figure');
    el.className = 'sx-card ex-card';
    el.innerHTML = `<figcaption class="sx-head"><span class="sx-title">${esc(title)}</span><span class="sx-sub"></span></figcaption>`;
    const subEl = el.querySelector('.sx-sub');
    const setSub = (t) => { subEl.textContent = t; subEl.hidden = !t; };
    setSub(sub);
    host.appendChild(el);
    return { el, setSub };
  }

  function note(host, text) {
    if (!text) return;
    const p = document.createElement('p');
    p.className = 'sx-note';
    p.textContent = text;
    host.appendChild(p);
  }

  function segmented(label, options, current, onPick) {
    const wrap = document.createElement('div');
    wrap.className = 'segmented-control';
    wrap.innerHTML = `<label>${esc(label)}</label>`;
    const buttons = options.map((o) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = o.label;
      b.addEventListener('click', () => onPick(o.value));
      wrap.appendChild(b);
      return { b, value: o.value };
    });
    const set = (value) => buttons.forEach(({ b, value: v }) => {
      b.classList.toggle('active', v === value);
      b.setAttribute('aria-pressed', String(v === value));
    });
    set(current);
    return { el: wrap, set };
  }

  /* ---------------------------------------------------------------- beeswarm */

  window.BLOCKS['shap-summary'] = (host, block) => {
    const ui = block.ui || {};
    const labels = block.labels || {};
    const units = block.units || {};
    const c = card(host, ui.title || 'Contribution of each predictor', '');
    c.el.insertAdjacentHTML('beforeend', `<p class="sx-loading">${esc(ui.loading || 'Loading…')}</p>`);
    note(host, block.note);

    const build = (data) => {
      const b = data.bottles;
      const feats = b.features;
      const n = feats.length;
      c.el.querySelector('.sx-loading').remove();
      c.setSub((ui.sub || '{shown} of {total} bottles, {y0}–{y1} · base value {base} µmol kg⁻¹')
        .replace('{shown}', b.n_shown).replace('{total}', b.n_total.toLocaleString('en-US').replace(/,/g, ' '))
        .replace('{y0}', b.years[0]).replace('{y1}', b.years[1]).replace('{base}', b.base.toFixed(1)));

      // deterministic jitter, one random number per point
      const points = feats.map((f, j) => {
        let s = 0x9e3779b9 ^ (j * 2654435761);
        const rnd = () => { s = (s + 0x6d2b79f5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
        return f.shap.map((x, i) => ({ x, pct: f.pct[i], val: f.value[i], u: rnd() * 2 - 1 }));
      });
      const all = points.flat().map((p) => p.x).sort((x, y) => x - y);
      const rest = points.slice(1).flat().map((p) => p.x).sort((x, y) => x - y);
      const domains = {
        full: [percentile(all, 0.1), percentile(all, 99.9)],
        zoom: [percentile(rest, 0.5), percentile(rest, 99.5)],
      };
      const state = { zoom: 'full' };

      const controls = document.createElement('div');
      controls.className = 'ex-controls';
      const axis = segmented(ui.axis_label || 'Axis', [
        { value: 'full', label: ui.axis_full || 'Full range' },
        { value: 'zoom', label: ui.axis_zoom || 'Zoom in' },
      ], state.zoom, (v) => { state.zoom = v; axis.set(v); redraw(); });
      controls.appendChild(axis.el);
      c.el.appendChild(controls);

      const stage = document.createElement('div');
      stage.className = 'ex-stage';
      stage.innerHTML = '<div class="sx-tip" hidden></div>';
      c.el.appendChild(stage);
      const tip = stage.querySelector('.sx-tip');

      const ramp = makeRamp();
      const legend = document.createElement('div');
      legend.className = 'ex-legend';
      legend.innerHTML = `
        <span class="ex-legend-title">${esc(ui.legend || 'Feature value')}</span>
        <span>${esc(ui.low || 'low')}</span>
        <span class="ex-legend-ramp" style="background:${ramp.css}"></span>
        <span>${esc(ui.high || 'high')}</span>
        <span class="ex-legend-grey"><i></i>${esc(ui.grey || 'group or missing value')}</span>`;
      c.el.appendChild(legend);

      let rows = [];
      let geom = null;
      const draw = (width) => {
        const compact = width < 540;
        const L = compact ? 6 : 118;
        const R = compact ? 6 : 64;
        const top = compact ? 4 : 24;
        const rh = compact ? 60 : 46;
        const labH = compact ? 20 : 0;
        const axisH = 46;
        const pw = width - L - R;
        const [lo, hi] = domains[state.zoom];
        const count = Math.max(3, Math.floor(pw / (compact ? 60 : 80)));
        const { step } = niceTicks(lo, hi, count);
        const d0 = Math.floor(lo / step) * step;
        const d1 = Math.ceil(hi / step) * step;
        const X = (v) => L + ((Math.min(d1, Math.max(d0, v)) - d0) / (d1 - d0)) * pw;
        const H = top + n * rh + axisH;
        const allTicks = niceTicks(d0, d1, count).ticks;
        const r = compact ? 2.4 : 2.2;
        const colours = Array.from({ length: 100 }, (_, i) => rgb(ramp.colour(i / 99)));
        const out = [];
        rows = [];

        allTicks.forEach((t) => {
          const x = X(t).toFixed(1);
          out.push(`<line class="${t === 0 ? 'ex-zero' : 'ex-grid'}" x1="${x}" x2="${x}" y1="${top}" y2="${top + n * rh}"/>`);
          out.push(`<text class="ex-tick" x="${x}" y="${top + n * rh + 16}" text-anchor="middle">${plain(t)}</text>`);
        });
        out.push(`<line class="ex-axis" x1="${L}" x2="${L + pw}" y1="${top + n * rh}" y2="${top + n * rh}"/>`);
        out.push(`<text class="ex-axis-label" x="${L + pw / 2}" y="${top + n * rh + 36}" text-anchor="middle">${esc(ui.x_label || 'Contribution to median TA (µmol kg⁻¹)')}</text>`);
        if (!compact) out.push(`<text class="ex-head" x="${width - 4}" y="${top - 8}" text-anchor="end">${esc(ui.mean_abs_head || 'mean |SHAP|')}</text>`);

        feats.forEach((f, j) => {
          const y0 = top + j * rh;
          const yc = y0 + labH + (rh - labH) / 2;
          const half = (rh - labH) / 2 - 4;
          const label = labels[f.id] || f.id;
          out.push(`<line class="ex-rowline" x1="${L}" x2="${L + pw}" y1="${yc}" y2="${yc}"/>`);
          if (j > 0) out.push(`<line class="ex-sep" x1="0" x2="${width}" y1="${y0}" y2="${y0}"/>`);
          if (compact) {
            out.push(`<text class="ex-lab" x="${L + 2}" y="${y0 + 15}">${esc(label)}</text>`);
            out.push(`<text class="ex-mean" x="${width - R - 2}" y="${y0 + 15}" text-anchor="end">${esc(ui.mean_abs_head || 'mean |SHAP|')} ${f.mean_abs.toFixed(1)}</text>`);
          } else {
            out.push(`<text class="ex-lab" x="${L - 10}" y="${yc + 4}" text-anchor="end">${esc(label)}</text>`);
            out.push(`<text class="ex-mean" x="${width - 4}" y="${yc + 4}" text-anchor="end">${f.mean_abs.toFixed(1)}</text>`);
          }
          // local density: points within ±5 px, relative to the busiest spot in the row
          const px = points[j].map((p) => X(p.x));
          const order = px.map((_, i) => i).sort((p, q) => px[p] - px[q]);
          const near = new Float32Array(px.length);
          let lo = 0;
          let hi = 0;
          order.forEach((i) => {
            while (px[order[lo]] < px[i] - 5) lo++;
            while (hi < order.length && px[order[hi]] <= px[i] + 5) hi++;
            near[i] = hi - lo;
          });
          const maxNear = Math.max(...near);
          const row = { px: new Float32Array(px.length), py: new Float32Array(px.length) };
          const dots = points[j].map((p, i) => {
            const dens = near[i] / maxNear;
            const x = px[i];
            const y = yc + p.u * half * dens;
            row.px[i] = x;
            row.py[i] = y;
            const fill = p.pct === null ? 'var(--text-muted)' : colours[p.pct];
            return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r}" fill="${fill}"${p.pct === null ? ' class="ex-grey"' : ''}/>`;
          });
          out.push(`<g class="ex-dots">${dots.join('')}</g>`);
          rows.push(row);
        });
        out.push('<circle class="ex-ring" r="5.5" hidden/>');
        stage.querySelector('svg')?.remove();
        stage.insertAdjacentHTML('afterbegin', `<svg width="${width}" height="${H}" viewBox="0 0 ${width} ${H}" role="img" aria-label="${esc(ui.aria || 'Beeswarm of SHAP contributions per predictor')}">${out.join('')}</svg>`);
        geom = { width, top, rh, n, X, d0, d1 };
      };

      const ring = () => stage.querySelector('.ex-ring');
      const hide = () => { ring()?.setAttribute('hidden', ''); tip.hidden = true; };
      const show = (evt) => {
        const svg = stage.querySelector('svg');
        if (!svg || !geom) return;
        const rect = svg.getBoundingClientRect();
        const x = evt.clientX - rect.left;
        const y = evt.clientY - rect.top;
        const j = Math.floor((y - geom.top) / geom.rh);
        if (j < 0 || j >= geom.n) return hide();
        const row = rows[j];
        let best = -1;
        let bd = 16 * 16;
        for (let i = 0; i < row.px.length; i++) {
          const d = (row.px[i] - x) ** 2 + (row.py[i] - y) ** 2;
          if (d < bd) { bd = d; best = i; }
        }
        if (best < 0) return hide();
        const f = feats[j];
        const p = points[j][best];
        const rg = ring();
        rg.setAttribute('cx', row.px[best]);
        rg.setAttribute('cy', row.py[best]);
        rg.removeAttribute('hidden');
        const hasValue = p.val !== null;
        const valueText = hasValue ? withUnit(p.val, units[f.id]) : (p.pct === null && f.n_missing === null ? (ui.group || 'sum of its coordinates') : (ui.missing || 'missing'));
        tip.innerHTML = [
          line(ui.feature || 'Predictor', labels[f.id] || f.id),
          line(ui.value || 'Value', valueText),
          line(ui.contribution || 'Contribution', `${signed(p.x, 1)} µmol/kg`),
        ].join('');
        tip.hidden = false;
        placeTip(tip, stage, evt);
      };
      stage.addEventListener('pointermove', show);
      stage.addEventListener('pointerdown', show);
      stage.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hide(); });
      document.addEventListener('pointerdown', (e) => { if (!c.el.contains(e.target)) hide(); });

      const redraw = fitWidth(stage, draw);
    };

    lazy(c.el, () => loadData(block.data || 'explain-data.json').then(build).catch(failed(c.el, 'shap-summary', ui)));
    return host;
  };

  /* -------------------------------------------------------------------- maps */

  async function inflate(b64) {
    const stream = new Blob([bytes(b64)]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  /** Undo build_explain.pack_map: low/high byte planes -> int16 deltas -> running sum. */
  async function unpackMap(b64) {
    const raw = await inflate(b64);
    const n = raw.length / 2;
    const out = new Int16Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      acc = (acc + (raw[i] | (raw[n + i] << 8))) << 16 >> 16;
      out[i] = acc;
    }
    return out;
  }

  window.BLOCKS['shap-maps'] = (host, block) => {
    const ui = block.ui || {};
    const labels = block.labels || {};
    return mountMaps(host, block, {
      name: 'shap-maps',
      load: () => loadData(block.data || 'explain-data.json').then(async (data) => {
        const m = data.maps;
        const { nlat, nlon } = m.grid;
        const maskBytes = bytes(m.mask);
        const index = new Int32Array(nlat * nlon).fill(-1);
        let k = 0;
        for (let i = 0; i < nlat * nlon; i++) if ((maskBytes[i >> 3] >> (i & 7)) & 1) index[i] = k++;
        const ramp = makeRamp();
        const decoded = await Promise.all(m.features.map((f) => unpackMap(f.s)));
        const cards = m.features.map((f, fi) => {
          const s = decoded[fi];
          const canvas = document.createElement('canvas');
          canvas.width = nlon;
          canvas.height = nlat;
          const ctx2d = canvas.getContext('2d');
          const img = ctx2d.createImageData(nlon, nlat);
          for (let i = 0; i < nlat * nlon; i++) {
            const q = index[i];
            if (q < 0 || s[q] === NO_S) continue;
            img.data.set([...ramp.colour((s[q] / m.scale / f.limit + 1) / 2), 255], i * 4);
          }
          ctx2d.putImageData(img, 0, 0);
          const name = labels[f.id] || f.id;
          return {
            id: f.id,
            title: name,
            sub: (ui.sub || 'mean |contribution| {v} µmol kg⁻¹').replace('{v}', f.mean_abs.toFixed(1)),
            limit: f.limit,
            image: canvas.toDataURL('image/png'),
            aria: (ui.aria || 'Contribution of {name} to the estimated alkalinity').replace('{name}', name),
            rows: (i, ln) => {
              const q = index[i];
              if (q < 0 || s[q] === NO_S) return [ln(ui.no_value || 'Value', ui.land || 'land / no data')];
              return [ln(ui.contribution || 'Contribution', `${signed(s[q] / m.scale, 1)} µmol/kg`)];
            },
          };
        });
        return { grid: m.grid, index, ramp, cards };
      }),
    });
  };

  /* ------------------------------------------------------------ interactions */

  window.BLOCKS.interactions = (host, block) => {
    const ui = block.ui || {};
    const labels = block.labels || {};
    const units = block.units || {};
    const name = (id) => labels[id] || id;
    const mount = document.createElement('div');
    mount.className = 'ex-interactions';
    host.appendChild(mount);

    const mat = card(mount, ui.matrix_title || 'Interaction strength', '');
    mat.el.insertAdjacentHTML('beforeend', `<p class="sx-loading">${esc(ui.loading || 'Loading…')}</p>`);
    const view = card(mount, ui.pair_title || 'Joint effect of a pair', '');
    note(host, block.note);

    const build = (data) => {
      mat.el.querySelector('.sx-loading').remove();
      const I = data.interactions;
      const ids = I.features;
      const n = ids.length;
      const pairs = data.pdp2d;
      const pairIndex = (a, b) => pairs.findIndex((p) => (p.a === a && p.b === b) || (p.a === b && p.b === a));
      const state = { pair: 0, mode: 'joint' };

      /* ---- matrix */
      const offdiag = I.strength.flat().filter((v) => v !== null);
      const maxStrength = Math.max(...offdiag);
      mat.setSub((ui.matrix_sub || 'mean |interaction| per pair, µmol kg⁻¹; {n} bottles').replace('{n}', I.n_rows));
      const stage = document.createElement('div');
      stage.className = 'ex-stage';
      const grid = document.createElement('div');
      grid.className = 'ex-matrix';
      grid.style.setProperty('--n', n);
      const tip = document.createElement('div');
      tip.className = 'sx-tip';
      tip.hidden = true;
      stage.append(grid, tip);
      mat.el.appendChild(stage);

      const ramp = rampFrom([5, 4, 3, 2, 1, 0]);
      const t = (v) => Math.sqrt(v / maxStrength);
      const cells = [];
      grid.appendChild(document.createElement('div'));
      ids.forEach((id) => {
        const h = document.createElement('div');
        h.className = 'ex-col';
        h.textContent = name(id);
        grid.appendChild(h);
      });
      ids.forEach((ra, i) => {
        const h = document.createElement('div');
        h.className = 'ex-row';
        h.textContent = name(ra);
        grid.appendChild(h);
        ids.forEach((cb, j) => {
          const diag = i === j;
          const v = diag ? I.main[i] : I.strength[i][j];
          const pi = diag ? -1 : pairIndex(ra, cb);
          const el = document.createElement(pi >= 0 ? 'button' : 'div');
          el.className = `ex-cell${diag ? ' ex-diag' : ''}${pi >= 0 ? ' ex-link' : ''}`;
          if (pi >= 0) el.type = 'button';
          const col = diag ? null : ramp.colour(t(v));
          if (col) { el.style.background = rgb(col); el.style.color = textOn(col); }
          el.textContent = diag && v >= 10 ? v.toFixed(0) : v.toFixed(1);
          el.setAttribute('aria-label', diag ? `${name(ra)}: ${v.toFixed(1)}` : `${name(ra)} × ${name(cb)}: ${v.toFixed(1)}${pi >= 0 ? ' (show surface)' : ''}`);
          grid.appendChild(el);
          cells.push({ el, i, j, pi, v, diag });
        });
      });

      const markSelected = () => cells.forEach((c) => c.el.classList.toggle('ex-selected', c.pi >= 0 && c.pi === state.pair));
      const hideTip = () => { tip.hidden = true; };
      const showTip = (c, evt) => {
        const rows = c.diag
          ? [line(ui.predictor || 'Predictor', name(ids[c.i])), line(ui.main || 'Main effect', `${c.v.toFixed(1)} µmol/kg`)]
          : [line(ui.pair || 'Pair', `${name(ids[c.i])} × ${name(ids[c.j])}`), line(ui.strength || 'Interaction', `${c.v.toFixed(1)} µmol/kg`)];
        if (c.pi >= 0) rows.push(line(ui.surface || 'Surface', `H² ${pairs[c.pi].h2.toFixed(3)} · ${ui.click || 'click to show'}`));
        tip.innerHTML = rows.join('');
        tip.hidden = false;
        placeTip(tip, stage, evt);
      };
      cells.forEach((c) => {
        c.el.addEventListener('pointermove', (e) => showTip(c, e));
        c.el.addEventListener('pointerdown', (e) => showTip(c, e));
        c.el.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hideTip(); });
        if (c.pi >= 0) {
          c.el.addEventListener('click', () => {
            selectPair(c.pi);
            const r = view.el.getBoundingClientRect();
            if (r.top > window.innerHeight - 160) view.el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
          });
        }
      });
      document.addEventListener('pointerdown', (e) => { if (!stage.contains(e.target)) hideTip(); });

      const bar = document.createElement('div');
      bar.className = 'sx-bar ex-bar';
      const marks = [0, 1, 2, 5, 10].filter((v) => v <= maxStrength);
      bar.innerHTML = `
        <div class="sx-ramp" style="background:${ramp.css}"></div>
        <div class="ex-bar-ticks">${marks.map((v) => `<span style="left:${(t(v) * 100).toFixed(1)}%">${v}</span>`).join('')}</div>
        <div class="ex-bar-caption">${esc(ui.matrix_legend || 'Off-diagonal: interaction strength, µmol kg⁻¹ (square-root colour scale). Diagonal: main effect. Outlined cells open a surface below.')}</div>`;
      mat.el.appendChild(bar);

      /* ---- pair viewer */
      const controls = document.createElement('div');
      controls.className = 'ex-controls';
      const pairSeg = segmented(ui.pair_label || 'Pair', pairs.map((p, i) => ({ value: i, label: `${name(p.a)} × ${name(p.b)}` })), state.pair, (v) => selectPair(v));
      const modeSeg = segmented(ui.mode_label || 'Show', [
        { value: 'joint', label: ui.mode_joint || 'Joint effect' },
        { value: 'interaction', label: ui.mode_interaction || 'Interaction only' },
      ], state.mode, (v) => { state.mode = v; modeSeg.set(v); drawView(); });
      controls.append(pairSeg.el, modeSeg.el);
      view.el.appendChild(controls);

      const vstage = document.createElement('div');
      vstage.className = 'ex-stage';
      vstage.innerHTML = '<div class="sx-tip" hidden></div>';
      view.el.appendChild(vstage);
      const vtip = vstage.querySelector('.sx-tip');
      const vbar = document.createElement('div');
      vbar.className = 'sx-bar';
      view.el.appendChild(vbar);

      let vgeom = null;
      const axisTitle = (id) => `${name(id)}${units[id] ? ` (${units[id]})` : ''}`;
      const surface = (p, i, j) => (state.mode === 'joint' ? p.pd[i][j] : p.interaction[i][j]);
      let width = 0;

      const drawView = () => {
        if (!width) return;
        const p = pairs[state.pair];
        const compact = width < 540;
        const L = compact ? 44 : 60;
        const R = 6;
        const top = 6;
        const B = 44;
        const pw = width - L - R;
        const ph = Math.round(Math.min(380, Math.max(220, pw * 0.62)));
        const H = top + ph + B;
        const na = p.a_value.length;
        const nb = p.b_value.length;
        const flat = p.pd.flat();
        const joint = state.mode === 'joint';
        const vals = joint ? flat : p.interaction.flat();
        const lo = Math.min(...vals);
        const hi = Math.max(...vals);
        const lim = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
        const rmp = joint ? rampFrom([5, 6, 7, 8, 9, 10]) : rampFrom([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        const tOf = (v) => (joint ? (v - lo) / (hi - lo || 1) : (v / lim + 1) / 2);
        const [a0, a1] = [p.a_value[0], p.a_value[na - 1]];
        const [b0, b1] = [p.b_value[0], p.b_value[nb - 1]];
        const da = (a1 - a0) / (na - 1);
        const db = (b1 - b0) / (nb - 1);
        const xs = (v) => L + ((v - (a0 - da / 2)) / (na * da)) * pw;
        const ys = (v) => top + ph - ((v - (b0 - db / 2)) / (nb * db)) * ph;
        const cw = pw / na;
        const chh = ph / nb;
        const out = ['<g shape-rendering="crispEdges">'];
        for (let i = 0; i < na; i++) {
          for (let j = 0; j < nb; j++) {
            out.push(`<rect x="${(L + i * cw).toFixed(2)}" y="${(top + ph - (j + 1) * chh).toFixed(2)}" width="${(cw + 0.4).toFixed(2)}" height="${(chh + 0.4).toFixed(2)}" fill="${rgb(rmp.colour(tOf(surface(p, i, j))))}"/>`);
          }
        }
        out.push('</g>');
        out.push(`<rect class="ex-frame" x="${L}" y="${top}" width="${pw}" height="${ph}"/>`);
        const xt = niceTicks(a0, a1, Math.max(3, Math.floor(pw / 70))).ticks;
        const yt = niceTicks(b0, b1, Math.max(3, Math.floor(ph / 50))).ticks;
        xt.forEach((v) => out.push(`<line class="ex-tickmark" x1="${xs(v)}" x2="${xs(v)}" y1="${top + ph}" y2="${top + ph + 4}"/><text class="ex-tick" x="${xs(v)}" y="${top + ph + 17}" text-anchor="middle">${plain(v)}</text>`));
        yt.forEach((v) => out.push(`<line class="ex-tickmark" x1="${L - 4}" x2="${L}" y1="${ys(v)}" y2="${ys(v)}"/><text class="ex-tick" x="${L - 7}" y="${ys(v) + 4}" text-anchor="end">${plain(v)}</text>`));
        out.push(`<text class="ex-axis-label" x="${L + pw / 2}" y="${top + ph + 38}" text-anchor="middle">${esc(axisTitle(p.a))}</text>`);
        out.push(`<text class="ex-axis-label" transform="translate(12 ${top + ph / 2}) rotate(-90)" text-anchor="middle">${esc(axisTitle(p.b))}</text>`);
        out.push('<rect class="ex-ring ex-cellring" hidden/>');
        vstage.querySelector('svg')?.remove();
        vstage.insertAdjacentHTML('afterbegin', `<svg width="${width}" height="${H}" viewBox="0 0 ${width} ${H}" role="img" aria-label="${esc((ui.pair_aria || '{mode} of {a} and {b}').replace('{mode}', joint ? (ui.mode_joint || 'Joint effect') : (ui.mode_interaction || 'Interaction only')).replace('{a}', name(p.a)).replace('{b}', name(p.b)))}">${out.join('')}</svg>`);
        vgeom = { L, top, pw, ph, cw, chh, na, nb, p };

        const fmt = (v) => (joint ? v.toFixed(0) : signed(v, 0));
        const caption = joint ? (ui.joint_caption || 'Median alkalinity, µmol kg⁻¹') : (ui.interaction_caption || 'Pure interaction, µmol kg⁻¹ (blue and red: below and above the sum of the single effects)');
        const mid = joint ? (lo + hi) / 2 : 0;
        vbar.innerHTML = `
          <div class="sx-ramp" style="background:${rmp.css}"></div>
          <div class="sx-ticks"><span>${fmt(joint ? lo : -lim)}</span><span>${fmt(mid)}</span><span>${fmt(joint ? hi : lim)}</span></div>
          <div class="ex-bar-caption">${esc(caption)}</div>`;
        view.setSub((ui.pair_sub || 'H² = {h2}: {pct}% of the joint effect is not the sum of the two single effects')
          .replace('{h2}', p.h2.toFixed(3)).replace('{pct}', (p.h2 * 100).toFixed(p.h2 < 0.1 ? 1 : 0)));
      };

      const hideV = () => { vstage.querySelector('.ex-cellring')?.setAttribute('hidden', ''); vtip.hidden = true; };
      const showV = (evt) => {
        const svg = vstage.querySelector('svg');
        if (!svg || !vgeom) return;
        const rect = svg.getBoundingClientRect();
        const x = evt.clientX - rect.left - vgeom.L;
        const y = evt.clientY - rect.top - vgeom.top;
        const i = Math.floor(x / vgeom.cw);
        const jj = vgeom.nb - 1 - Math.floor(y / vgeom.chh);
        if (x < 0 || y < 0 || i < 0 || i >= vgeom.na || jj < 0 || jj >= vgeom.nb) return hideV();
        const p = vgeom.p;
        const r = svg.querySelector('.ex-cellring');
        r.setAttribute('x', vgeom.L + i * vgeom.cw);
        r.setAttribute('y', vgeom.top + (vgeom.nb - 1 - jj) * vgeom.chh);
        r.setAttribute('width', vgeom.cw);
        r.setAttribute('height', vgeom.chh);
        r.removeAttribute('hidden');
        const v = surface(p, i, jj);
        vtip.innerHTML = [
          line(name(p.a), withUnit(p.a_value[i], units[p.a])),
          line(name(p.b), withUnit(p.b_value[jj], units[p.b])),
          state.mode === 'joint' ? line(ui.joint_row || 'Median TA', `${v.toFixed(1)} µmol/kg`) : line(ui.interaction_row || 'Interaction', `${signed(v, 1)} µmol/kg`),
        ].join('');
        vtip.hidden = false;
        placeTip(vtip, vstage, evt);
      };
      vstage.addEventListener('pointermove', showV);
      vstage.addEventListener('pointerdown', showV);
      vstage.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hideV(); });
      document.addEventListener('pointerdown', (e) => { if (!view.el.contains(e.target)) hideV(); });

      function selectPair(i) {
        state.pair = i;
        pairSeg.set(i);
        markSelected();
        hideV();
        drawView();
      }

      markSelected();
      fitWidth(vstage, (w) => { width = w; drawView(); });
    };

    lazy(mount, () => loadData(block.data || 'explain-data.json').then(build).catch(failed(mat.el, 'interactions', ui)));
    return host;
  };
})();
