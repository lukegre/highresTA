/* Spatial sensitivity maps: one card per predictor, hover/tap for values.
 * Data: sensitivity-data.json, written by web/build_sensitivity.py (packing described there).
 * Registers window.BLOCKS['sensitivity-maps'] and exports shared helpers (colour ramp, lazy loading,
 * the map-card grid) as window.SX for explain.js; load before explain.js and app.js. */
window.BLOCKS = window.BLOCKS || {};

(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const NO_S = -32768;
  const NO_Q = 255;
  const WEAK_AGREEMENT = 0.8;

  const bytes = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const hexRgb = (hex) => {
    const h = hex.replace('#', '');
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  };
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  const num = (v, d = 1) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const tick = (v) => {
    const d = Math.abs(v) >= 10 ? 0 : Math.abs(v) >= 1 ? 1 : 2;
    return num(v, d);
  };
  const round3 = (v) => Number(v.toPrecision(3)).toString();

  /** Diverging ramp from the site's RdYlBu tokens: blue (negative) .. pale (0) .. red (positive). */
  function makeRamp() {
    const stops = Array.from({ length: 11 }, (_, i) => hexRgb(cssVar(`--rdylbu-${10 - i}`) || '#888888'));
    const colour = (t) => {
      const x = Math.min(1, Math.max(0, t)) * 10;
      const i = Math.min(9, Math.floor(x));
      const f = x - i;
      return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
    };
    const css = `linear-gradient(90deg, ${stops.map((c, i) => `rgb(${c}) ${i * 10}%`).join(', ')})`;
    return { colour, css };
  }

  function decode(data) {
    const { nlat, nlon } = data.grid;
    const maskBytes = bytes(data.mask);
    const index = new Int32Array(nlat * nlon).fill(-1);
    let k = 0;
    for (let i = 0; i < nlat * nlon; i++) {
      if ((maskBytes[i >> 3] >> (i & 7)) & 1) index[i] = k++;
    }
    const features = data.features.map((f) => {
      const sb = bytes(f.s);
      return { ...f, s: new Int16Array(sb.buffer, sb.byteOffset, sb.byteLength / 2), q: bytes(f.q) };
    });
    return { ...data, index, features };
  }

  /** Cell value at grid cell i, or null. */
  function cellValue(data, f, i) {
    const k = data.index[i];
    if (k < 0 || f.s[k] === NO_S) return null;
    const q = f.q[k];
    return {
      value: f.s[k] / data.scale,
      agreement: q === NO_Q ? null : Math.floor(q / 5) / 50,
      outside: q === NO_Q ? null : (q % 5) / 4,
    };
  }

  function paint(data, f, ramp) {
    const { nlat, nlon } = data.grid;
    const canvas = document.createElement('canvas');
    canvas.width = nlon;
    canvas.height = nlat;
    const ctx2d = canvas.getContext('2d');
    const img = ctx2d.createImageData(nlon, nlat);
    for (let i = 0; i < nlat * nlon; i++) {
      const c = cellValue(data, f, i);
      if (!c) continue;
      const [r, g, b] = ramp.colour((c.value / f.limit + 1) / 2);
      const weak = c.agreement !== null && c.agreement < WEAK_AGREEMENT;
      img.data.set([r, g, b, weak ? 110 : 255], i * 4);
    }
    ctx2d.putImageData(img, 0, 0);
    return canvas.toDataURL('image/png');
  }

  /** Run fn once when el comes (almost) into view. */
  function lazy(el, fn) {
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        if (entries.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
      }, { rootMargin: '600px 0px' });
      io.observe(el);
    } else {
      fn();
    }
  }

  const fetchJson = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });

  /** One map card per feature, shared hover ring and tooltip. Used by 'sensitivity-maps' and 'shap-maps'.
   * source = { name, load(): Promise<{ grid, index, ramp, cards }> } where each card is
   * { id, title, sub, limit, image (data URL), aria, rows(i, line) -> html rows (without position) }
   * and index maps grid cell -> ocean cell (-1 = land). */
  function mountMaps(host, block, source) {
    const ui = block.ui || {};
    const grid = document.createElement('div');
    grid.className = 'sx-grid';
    grid.innerHTML = `<p class="sx-loading">${esc(ui.loading || 'Loading maps…')}</p>`;
    host.appendChild(grid);
    if (block.note) {
      const note = document.createElement('p');
      note.className = 'sx-note';
      note.textContent = block.note;
      host.appendChild(note);
    }

    const build = ({ grid: g, index, ramp, cards: specs }) => {
      const { nlat, nlon, lat0, dlat, lon0, dlon } = g;
      grid.innerHTML = '';
      const cards = specs.map((f) => {
        const card = document.createElement('figure');
        card.className = 'sx-card';
        card.innerHTML = `
          <figcaption class="sx-head">
            <span class="sx-title">${esc(f.title)}</span>
            <span class="sx-sub">${esc(f.sub)}</span>
          </figcaption>
          <div class="sx-map">
            <svg viewBox="-180 -90 360 180" role="img" aria-label="${esc(f.aria)}">
              <image href="${f.image}" x="-180" y="-90" width="360" height="180" preserveAspectRatio="none"/>
              <image href="maps/land.png" x="-180" y="-90" width="360" height="180" preserveAspectRatio="none"/>
              <circle class="sx-marker" r="3" hidden/>
            </svg>
            <div class="sx-tip" hidden></div>
          </div>
          <div class="sx-bar">
            <div class="sx-ramp" style="background:${ramp.css}"></div>
            <div class="sx-ticks"><span>${tick(-f.limit)}</span><span>0</span><span>${tick(f.limit)}</span></div>
          </div>`;
        grid.appendChild(card);
        return { f, card, svg: card.querySelector('svg'), marker: card.querySelector('.sx-marker'), tip: card.querySelector('.sx-tip') };
      });

      const hide = () => cards.forEach((c) => { c.marker.setAttribute('hidden', ''); c.tip.hidden = true; });
      const show = (active, evt) => {
        const rect = active.svg.getBoundingClientRect();
        const lon = -180 + ((evt.clientX - rect.left) / rect.width) * 360;
        const lat = 90 - ((evt.clientY - rect.top) / rect.height) * 180;
        const row = Math.round((lat - lat0) / dlat);
        const col = Math.round((lon - lon0) / dlon);
        if (row < 0 || row >= nlat || col < 0 || col >= nlon) return hide();
        const i = row * nlon + col;
        const cx = lon0 + col * dlon;
        const cy = -(lat0 + row * dlat);
        cards.forEach((c) => {
          c.marker.setAttribute('cx', cx);
          c.marker.setAttribute('cy', cy);
          c.marker.removeAttribute('hidden');
          if (c !== active) c.tip.hidden = true;
        });
        const cellLat = lat0 + row * dlat;
        const pos = `${Math.abs(cellLat).toFixed(1)}°${cellLat >= 0 ? 'N' : 'S'}, ${Math.abs(cx).toFixed(1)}°${cx >= 0 ? 'E' : 'W'}`;
        const line = (k, val) => `<div class="ux-tooltip-row"><div class="ux-tooltip-label">${esc(k)}</div><div class="ux-tooltip-value">${esc(val)}</div></div>`;
        const rows = [line(ui.position || 'Position', pos), ...active.f.rows(i, line)];
        const tip = active.tip;
        tip.innerHTML = rows.join('');
        tip.hidden = false;
        const box = active.svg.parentElement.getBoundingClientRect();
        const x = evt.clientX - box.left;
        const y = evt.clientY - box.top;
        const left = x + 12 + tip.offsetWidth > box.width ? x - 12 - tip.offsetWidth : x + 12;
        tip.style.left = `${Math.max(0, left)}px`;
        tip.style.top = `${Math.max(0, Math.min(y + 12, box.height - tip.offsetHeight))}px`;
      };

      cards.forEach((c) => {
        c.svg.addEventListener('pointermove', (e) => show(c, e));
        c.svg.addEventListener('pointerdown', (e) => show(c, e));
        c.svg.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') hide(); });
      });
      document.addEventListener('pointerdown', (e) => { if (!grid.contains(e.target)) hide(); });
    };

    lazy(grid, () => source.load()
      .then(build)
      .catch((err) => {
        console.error(`${source.name}:`, err);
        grid.innerHTML = `<p class="sx-loading">${esc(ui.error || 'Could not load the maps.')}</p>`;
      }));
    return host;
  }

  window.SX = { bytes, cssVar, hexRgb, esc, num, tick, round3, makeRamp, lazy, fetchJson, mountMaps };

  window.BLOCKS['sensitivity-maps'] = (host, block) => {
    const ui = block.ui || {};
    const labels = block.labels || {};
    const units = block.units || {};
    return mountMaps(host, block, {
      name: 'sensitivity-maps',
      load: () => fetchJson(block.data || 'sensitivity-data.json').then((raw) => {
        const data = decode(raw);
        const ramp = makeRamp();
        const cards = data.features.map((f) => {
          const unit = units[f.id] ? ` ${units[f.id]}` : '';
          const used = f.n_members < data.n_members_total
            ? (ui.members_used || '{n} of {total} models use it').replace('{n}', f.n_members).replace('{total}', data.n_members_total)
            : (ui.members_all || 'all {total} models').replace('{total}', data.n_members_total);
          return {
            id: f.id,
            title: labels[f.id] || f.id,
            sub: `+δ = ${round3(f.delta)}${unit} · ${used}`,
            limit: f.limit,
            image: paint(data, f, ramp),
            aria: (ui.aria || 'Sensitivity of alkalinity to {name}').replace('{name}', labels[f.id] || f.id),
            rows: (i, line) => {
              const v = cellValue(data, f, i);
              if (!v) return [line(ui.no_value || 'No value', data.index[i] < 0 ? (ui.land || 'land / no data') : '—')];
              const rows = [line(ui.change || 'ΔTA per +δ', `${num(v.value, 1)} µmol/kg`)];
              if (v.agreement !== null) rows.push(line(ui.agreement || 'Models agree', `${Math.round(v.agreement * 100)}%`));
              if (v.outside) rows.push(line(ui.outside || 'Outside training', `${Math.round(v.outside * data.dates.length)} of ${data.dates.length} dates`));
              return rows;
            },
          };
        });
        return { grid: data.grid, index: data.index, ramp, cards };
      }),
    });
  };
})();
