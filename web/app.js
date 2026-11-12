// High-resolution TA methods page renderer
// Fetches content.json and data.json, renders the page with block handlers

const app = {
  data: null,
  content: null,
  animateOnScroll: new Map(),

  async init() {
    try {
      const [dataResp, contentResp] = await Promise.all([
        fetch('data.json', { cache: 'no-store' }),
        fetch('content.json', { cache: 'no-store' })
      ]);

      if (!dataResp.ok || !contentResp.ok) throw new Error('Failed to fetch');

      this.data = await dataResp.json();
      this.content = await contentResp.json();

      this.render();
      this.setupScrollSpy();
      this.setupAnimations();
    } catch (err) {
      console.error('Failed to load data:', err);
      const errorEl = document.getElementById('load-error');
      errorEl.innerHTML = `
        <strong>Could not load the page.</strong>
        <p>The app needs to be served over HTTP (file:// doesn't work with fetch):</p>
        <p><code>cd web && python3 -m http.server 8000</code></p>
        <p>Then open <code>http://localhost:8000</code></p>
      `;
      errorEl.hidden = false;
    }
  },

  render() {
    // Create helper context
    const ctx = {
      data: this.data,
      content: this.content,
      tpl: this.tpl.bind(this),
      md: this.md.bind(this),
      rdylbu: this.rdylbu.bind(this)
    };

    // Brand
    const brand = document.getElementById('brand');
    if (brand) brand.textContent = this.content.site.short_title;

    // Nav
    const nav = document.getElementById('nav');
    this.content.site.nav.forEach(item => {
      const a = document.createElement('a');
      a.href = `#${item.id}`;
      a.textContent = item.label;
      a.className = 'nav-link';
      nav.appendChild(a);
    });

    // Hero: lead and headline numbers (no title)
    const hero = document.getElementById('hero');
    hero.innerHTML = `
      <div class="section__inner">
        <div class="hero-content">
          <p class="hero-lead">${this.md(this.content.hero.lead, ctx)}</p>
          <div class="hero-stats">
            ${this.content.hero.stats.map(stat => `
              <div class="stat-card">
                <div class="stat-label">${this.escape(stat.label)}</div>
                <div class="stat-value">${this.tpl(stat.value, ctx)}</div>
                <div class="stat-note">${this.tpl(this.escape(stat.note), ctx)}</div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>
    `;

    // Content sections
    const contentDiv = document.getElementById('content');
    this.content.sections.forEach(section => {
      const sectionEl = document.createElement('section');
      sectionEl.className = 'section';
      sectionEl.id = section.id;

      const inner = document.createElement('div');
      inner.className = 'section__inner';

      const sectionHeader = document.createElement('div');
      sectionHeader.className = 'section-header';
      sectionHeader.innerHTML = `
        <div class="kicker">${this.escape(section.kicker)}</div>
        <h2 class="section-title">${this.escape(section.title)}</h2>
        <p class="section-lead">${this.md(section.lead, ctx)}</p>
      `;
      inner.appendChild(sectionHeader);

      this.renderBlocks(section.blocks, inner, ctx);

      sectionEl.appendChild(inner);
      contentDiv.appendChild(sectionEl);
    });

    // Footer
    const footer = document.getElementById('footer');
    footer.innerHTML = `
      <div class="section__inner">
        <p>${this.tpl(this.escape(this.content.site.footer), ctx)}</p>
      </div>
    `;
  },

  renderBlocks(blocks, parent, ctx) {
    blocks.forEach(block => {
      try {
        const blockEl = this.renderBlock(block, ctx);
        if (blockEl) {
          parent.appendChild(blockEl);
          this.animateOnScroll.set(blockEl, true);
        }
      } catch (err) {
        console.error(`Error rendering block type "${block.type}":`, err);
        const errorNote = document.createElement('div');
        errorNote.className = 'block-error';
        errorNote.textContent = `Error rendering ${block.type} block`;
        parent.appendChild(errorNote);
      }
    });
  },

  renderBlock(block, ctx) {
    const container = document.createElement('div');
    container.className = `block block--${block.type}`;

    // Add title if present (a details block shows it in its summary)
    if (block.title && block.type !== 'details') {
      const title = document.createElement('h3');
      title.className = 'block-title';
      title.textContent = block.title;
      container.appendChild(title);
    }

    // Route to renderer
    switch (block.type) {
      case 'text':
        return this.renderText(container, block, ctx);
      case 'cards':
        return this.renderCards(container, block, ctx);
      case 'steps':
        return this.renderSteps(container, block, ctx);
      case 'table':
        return this.renderTable(container, block, ctx);
      case 'callout':
        return this.renderCallout(container, block, ctx);
      case 'equation':
        return this.renderEquation(container, block, ctx);
      case 'details':
        return this.renderDetails(container, block, ctx);
      case 'cleaning-steps':
        return this.renderCleaningSteps(container, block, ctx);
      case 'split-explorer':
        return this.renderSplitExplorer(container, block, ctx);
      case 'member-grid':
        return this.renderMemberGrid(container, block, ctx);
      default:
        // Try to call external renderer
        if (window.BLOCKS && window.BLOCKS[block.type]) {
          window.BLOCKS[block.type](container, block, ctx);
          return container;
        }
        console.warn(`Unknown block type: ${block.type}`);
        return null;
    }
  },

  renderText(container, block, ctx) {
    const p = document.createElement('p');
    p.className = 'block-text';
    p.innerHTML = this.md(block.body, ctx);
    container.appendChild(p);
    return container;
  },

  renderCards(container, block, ctx) {
    const grid = document.createElement('div');
    grid.className = `cards-grid cards-${block.columns}col`;

    block.items.forEach(item => {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        ${item.tag ? `<div class="card-tag">${this.escape(item.tag)}</div>` : ''}
        <h4 class="card-title">${this.escape(item.title)}</h4>
        <p class="card-body">${this.md(item.body, ctx)}</p>
      `;
      grid.appendChild(card);
    });

    container.appendChild(grid);
    return container;
  },

  renderSteps(container, block, ctx) {
    const timeline = document.createElement('ol');
    timeline.className = 'steps-timeline';

    block.items.forEach((item, idx) => {
      const li = document.createElement('li');
      li.className = 'step-item';
      li.innerHTML = `
        <div class="step-number">${idx + 1}</div>
        <div class="step-content">
          <h4 class="step-title">${this.escape(item.title)}</h4>
          <p class="step-body">${this.md(item.body, ctx)}</p>
        </div>
      `;
      timeline.appendChild(li);
    });

    container.appendChild(timeline);
    return container;
  },

  renderTable(container, block, ctx) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';

    const table = document.createElement('table');
    table.className = 'data-table';

    // Header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    block.columns.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    block.rows.forEach((row, idx) => {
      const tr = document.createElement('tr');
      if (idx % 2 === 1) tr.className = 'zebra';
      row.forEach(cell => {
        const td = document.createElement('td');
        td.innerHTML = this.md(cell, ctx);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    wrapper.appendChild(table);
    container.appendChild(wrapper);
    return container;
  },

  renderCallout(container, block, ctx) {
    container.className += ` callout callout--${block.tone}`;
    container.innerHTML = `
      <div class="callout-bar"></div>
      <div class="callout-content">
        ${block.title ? `<h4 class="callout-title">${this.escape(block.title)}</h4>` : ''}
        <p class="callout-body">${this.md(block.body, ctx)}</p>
      </div>
    `;
    return container;
  },

  renderDetails(container, block, ctx) {
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = block.title;
    const body = document.createElement('div');
    body.className = 'details-body';
    this.renderBlocks(block.blocks, body, ctx);
    details.append(summary, body);
    container.appendChild(details);
    return container;
  },

  renderEquation(container, block, ctx) {
    // Items are LaTeX; fall back to the raw string if KaTeX did not load
    block.items.forEach(item => {
      const line = document.createElement('div');
      line.className = 'equation';
      if (window.katex) katex.render(item, line, { displayMode: true, throwOnError: false });
      else line.textContent = item;
      container.appendChild(line);
    });
    return container;
  },

  renderCleaningSteps(container, block, ctx) {
    // One row per cleaning step: text from content, numbers from data.funnel (same order)
    const funnel = this.data.funnel;
    const columns = this.content.ui.cleaning_columns;
    const rows = block.items.map((item, idx) => {
      const step = funnel[idx];
      const pct = (step.rows / funnel[0].rows * 100).toFixed(0);
      return `
        <tr>
          <th scope="row">${this.escape(item.name)}</th>
          <td>${this.md(item.body, ctx)}</td>
          <td class="cleaning-rows">
            <span>${this.formatInt(step.rows)} <small>${pct} %</small></span>
            <span class="cleaning-bar"><span style="width: ${pct}%"></span></span>
          </td>
          <td class="cleaning-cruises">${step.cruises}</td>
        </tr>`;
    });
    container.innerHTML = `
      <div class="table-wrapper">
        <table class="data-table cleaning-table">
          <thead><tr>${columns.map(col => `<th>${this.escape(col)}</th>`).join('')}</tr></thead>
          <tbody>${rows.join('')}</tbody>
        </table>
      </div>`;
    return container;
  },

  renderSplitExplorer(container, block, ctx) {
    const blocks = this.data.blocks;
    const ui = this.content.ui;
    const nMembers = this.data.run.n_members;

    const explorer = document.createElement('div');
    explorer.className = 'split-explorer';

    // Split selector and member popup button
    const splitControl = document.createElement('div');
    splitControl.className = 'segmented-control';
    splitControl.innerHTML = `<label>${ui.split_label}</label>`;
    for (let i = 1; i <= this.data.run.n_splits; i++) {
      const btn = document.createElement('button');
      btn.textContent = `${i}`;
      btn.className = i === 1 ? 'active' : '';
      btn.setAttribute('aria-pressed', i === 1 ? 'true' : 'false');
      btn.addEventListener('click', () => this.updateSplitExplorer(i, explorer));
      splitControl.appendChild(btn);
    }
    const membersBtn = document.createElement('button');
    membersBtn.className = 'members-button';
    membersBtn.textContent = ui.members_button;
    membersBtn.addEventListener('click', () => this.showSplitMembers(Number(explorer.dataset.split)));
    splitControl.appendChild(membersBtn);
    explorer.appendChild(splitControl);
    const modelsNote = document.createElement('p');
    modelsNote.className = 'split-models-note';
    explorer.appendChild(modelsNote);

    // Blocks: the test block stands apart and never changes with the selected split
    const visualization = document.createElement('div');
    visualization.className = 'split-visualization';

    const layout = document.createElement('div');
    layout.className = 'split-layout';
    const testGroup = document.createElement('div');
    testGroup.className = 'split-group split-group--test';
    testGroup.innerHTML = `
      <div class="group-label">${this.escape(ui.test_label.replace('{n}', nMembers))}</div>
      <div class="block-tile block-tile--test">
        <div class="block-label">${ui.block} 0</div>
        <div class="block-info">
          <div>${this.formatInt(blocks[0].observations)} ${ui.observations}</div>
          <div>${blocks[0].cruises} ${ui.cruises}</div>
        </div>
      </div>
      <div class="group-note">${this.escape(ui.test_note)}</div>
    `;
    const otherGroup = document.createElement('div');
    otherGroup.className = 'split-group';
    otherGroup.innerHTML = `<div class="group-label">${this.escape(ui.other_blocks_label.replace('{last}', blocks.length - 1))}</div>`;
    const blockRow = document.createElement('div');
    blockRow.className = 'blocks-row';
    blockRow.id = 'split-blocks';
    otherGroup.appendChild(blockRow);
    layout.append(testGroup, otherGroup);
    // Blocks stacked on the left, map of the same split on the right
    const main = document.createElement('div');
    main.className = 'split-main';
    main.appendChild(layout);
    visualization.appendChild(main);

    explorer.appendChild(visualization);

    // Map of sample locations coloured by their role in the selected split
    const map = this.data.split_map;
    if (map) {
      const figure = document.createElement('figure');
      figure.className = 'split-map';
      const points = map.lat.map((lat, i) => `<circle data-block="${map.block[i] ?? ''}" cx="${map.lon[i]}" cy="${-lat}" r="0.9"/>`);
      // test locations last so they sit on top
      const test = points.filter((_, i) => map.block[i] === 0), rest = points.filter((_, i) => map.block[i] !== 0);
      figure.innerHTML = `<svg viewBox="-180 -90 360 180" role="img" aria-label="${this.escape(ui.split_map_title)}">
          <image href="maps/land.png" x="-180" y="-90" width="360" height="180" preserveAspectRatio="none"/>${rest.join('')}${test.join('')}</svg>
        <figcaption></figcaption>
        <p class="split-map__note">${this.md(ui.split_map_note, ctx)}</p>`;
      main.appendChild(figure);
    }
    container.appendChild(explorer);

    // Initial render
    this.updateSplitExplorer(1, explorer);

    return container;
  },

  updateSplitExplorer(split, container) {
    const ui = this.content.ui;
    container.dataset.split = split;
    const blocks = this.data.blocks;
    const testBlock = 0;
    const validationBlock = split;

    // Update split buttons
    const splitBtns = container.querySelectorAll('.segmented-control')[0].querySelectorAll('button:not(.members-button)');
    splitBtns.forEach((btn, idx) => {
      if (idx + 1 === split) {
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');
      } else {
        btn.classList.remove('active');
        btn.setAttribute('aria-pressed', 'false');
      }
    });

    // Render validation and training blocks (the test block is drawn once, apart)
    const blockRow = container.querySelector('#split-blocks');
    blockRow.innerHTML = '';
    blocks.forEach((block, idx) => {
      if (idx === testBlock) return;
      const role = idx === validationBlock ? ui.validation_role : ui.train_role;
      const tile = document.createElement('div');
      tile.className = `block-tile block-tile--${role}`;
      tile.innerHTML = `
        <div class="block-label">${ui.block} ${idx}</div>
        <div class="block-info">
          <div>${this.formatInt(block.observations)} ${ui.observations}</div>
          <div>${block.cruises} ${ui.cruises}</div>
        </div>
      `;
      blockRow.appendChild(tile);
    });

    container.querySelector('.split-models-note').textContent = ui.split_models_note.replace('{k}', split);

    // Recolour the map: test never changes, the validation block follows the split
    const figure = container.querySelector('.split-map');
    if (figure) {
      const hasBlocks = this.data.split_map.block.some(b => b > 0);
      figure.querySelectorAll('circle').forEach(c => {
        c.setAttribute('class', c.dataset.block === '0' ? 'is-test' : Number(c.dataset.block) === split ? 'is-validation' : 'is-train');
      });
      const key = (cls, text) => `<span class="split-map__key ${cls}"></span>${this.escape(text)}`;
      figure.querySelector('figcaption').innerHTML = [
        key('is-test', ui.split_map_test.replace('{n}', this.data.run.n_members)),
        hasBlocks ? key('is-validation', ui.split_map_validation.replace('{k}', split)) : '',
        key('is-train', hasBlocks ? ui.split_map_train : ui.split_map_pool),
      ].join(' ');
    }
  },

  renderMemberGrid(container, block, ctx) {
    const members = this.data.members;
    const ui = this.content.ui;
    const splits = this.data.run.n_splits;
    const replicates = this.data.run.n_replicates;

    const gridContainer = document.createElement('div');
    gridContainer.className = 'member-grid-container';

    // The two axes are the two sources of diversity
    const axes = document.createElement('dl');
    axes.className = 'grid-axes';
    axes.innerHTML = `
      <dt>↓ ${this.escape(block.rows_label)}</dt><dd>${this.md(block.rows_note, ctx)}</dd>
      <dt>→ ${this.escape(block.cols_label)}</dt><dd>${this.md(block.cols_note, ctx)}</dd>
    `;
    gridContainer.appendChild(axes);

    // Cells are coloured by test MAE
    const maes = members.map(m => m.test.mae);
    const minMae = Math.min(...maes);
    const maxMae = Math.max(...maes);
    const legend = document.createElement('div');
    legend.className = 'grid-legend';
    legend.innerHTML = `
      <div class="legend-label">${this.escape(ui.mae_legend)}</div>
      <div class="legend-gradient">
        <div class="legend-tick" style="left: 0%">${minMae.toFixed(1)}</div>
        <div class="legend-tick" style="left: 100%">${maxMae.toFixed(1)}</div>
      </div>
    `;
    gridContainer.appendChild(legend);

    const gridWrapper = document.createElement('div');
    gridWrapper.className = 'member-grid-wrapper';
    const gridContent = document.createElement('div');
    gridContent.className = 'member-grid';

    // Column headers
    const headerRowTop = document.createElement('div');
    headerRowTop.className = 'grid-row header-row';
    const cornerCell = document.createElement('div');
    cornerCell.className = 'grid-cell corner';
    headerRowTop.appendChild(cornerCell);
    for (let r = 0; r < replicates; r++) {
      const cell = document.createElement('div');
      cell.className = 'grid-cell header';
      cell.textContent = ui.member_header.replace('{index}', r);
      headerRowTop.appendChild(cell);
    }
    gridContent.appendChild(headerRowTop);

    // One row per split
    for (let s = 1; s <= splits; s++) {
      const row = document.createElement('div');
      row.className = 'grid-row';

      const rowHeader = document.createElement('button');
      rowHeader.className = 'grid-cell row-header';
      rowHeader.textContent = `Split ${s}`;
      rowHeader.addEventListener('click', () => this.showSplitMembers(s));
      row.appendChild(rowHeader);

      for (let r = 0; r < replicates; r++) {
        const member = members.find(m => m.split === s && m.replicate === r);
        if (!member) continue;
        const cell = document.createElement('button');
        cell.className = 'grid-cell data-cell';
        cell.style.setProperty('--cell-color', this.rdylbu((member.test.mae - minMae) / (maxMae - minMae), true));
        cell.textContent = member.test.mae.toFixed(1);
        cell.addEventListener('click', () => this.showMembers(
          [member], ui.member_dialog.member_title.replace('{split}', s).replace('{index}', r)));
        row.appendChild(cell);
      }
      gridContent.appendChild(row);
    }

    gridWrapper.appendChild(gridContent);
    gridContainer.appendChild(gridWrapper);

    const hint = document.createElement('p');
    hint.className = 'grid-hint';
    hint.textContent = block.hint;
    gridContainer.appendChild(hint);

    container.appendChild(gridContainer);
    return container;
  },

  showSplitMembers(split) {
    const members = this.data.members.filter(m => m.split === split);
    this.showMembers(members, this.content.ui.member_dialog.split_title.replace('{split}', split));
  },

  showMembers(members, title) {
    // One shared popup; shows a table with one row per member (MAE only, no RMSE)
    const copy = this.content.ui.member_dialog;
    const fixed = (value, decimals) => value == null ? '–' : Number(value).toFixed(decimals);
    const columns = [
      ['member', m => m.replicate],
      ['trial', m => m.trial_number ?? '–'],
      ['loss_rank', m => m.loss_rank ?? '–'],
      ['dropped', m => this.escape((m.dropped_features || []).join(', ') || copy.none)],
      ['depth', m => fixed(m.params.depth, 0)],
      ['learning_rate', m => fixed(m.params.learning_rate, 3)],
      ['l2_leaf_reg', m => fixed(m.params.l2_leaf_reg, 2)],
      ['iterations', m => m.iterations],
      ['inner_cv_loss', m => fixed(m.inner_cv_loss, 3)],
      ...(members.some(m => m.validation.mae != null) ? [['validation_mae', m => fixed(m.validation.mae, 1)]] : []),
      ['test_mae', m => fixed(m.test.mae, 1)],
      ['coverage', m => fixed(m.test.coverage80 * 100, 1) + ' %']
    ];

    if (!this.dialog) {
      this.dialog = document.createElement('dialog');
      this.dialog.className = 'member-dialog';
      // A click on the backdrop (the dialog element itself) closes it
      this.dialog.addEventListener('click', e => { if (e.target === this.dialog) this.dialog.close(); });
      document.body.appendChild(this.dialog);
    }
    this.dialog.innerHTML = `
      <div class="dialog-header">
        <h4>${this.escape(title)}</h4>
        <form method="dialog"><button class="dialog-close">${this.escape(copy.close)}</button></form>
      </div>
      <div class="table-wrapper">
        <table class="data-table">
          <thead><tr>${columns.map(([key]) => `<th>${this.escape(copy.columns[key])}</th>`).join('')}</tr></thead>
          <tbody>${members.map(m => `<tr>${columns.map(([, value]) => `<td>${value(m)}</td>`).join('')}</tr>`).join('')}</tbody>
        </table>
      </div>
    `;
    this.dialog.showModal();
  },

  // Helper functions
  tpl(str, ctx) {
    // Resolve {{path.to.value|fmt}} placeholders
    return str.replace(/\{\{([^|]+)(?:\|([^}]+))?\}\}/g, (match, path, fmt) => {
      const keys = path.trim().split('.');
      let value = ctx.data;

      for (const key of keys) {
        if (value && typeof value === 'object' && key in value) {
          value = value[key];
        } else {
          return 'n/a';
        }
      }

      if (fmt) {
        fmt = fmt.trim();
        if (fmt === 'int') return this.formatInt(value);
        if (fmt === 'pct') return (value * 100).toFixed(1) + ' %';
        // Numeric format: N decimals
        const decimals = parseInt(fmt);
        if (!isNaN(decimals)) return value.toFixed(decimals);
      }

      return String(value);
    });
  },

  md(str, ctx) {
    // First apply template substitutions
    let html = this.tpl(str, ctx);
    // HTML escape
    html = this.escape(html);
    // Then parse markdown-like syntax
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // [label](https://...) links, http(s) only, opened in a new tab
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s"]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>');
    return html;
  },

  rdylbu(t, reverse = false) {
    // Interpolate across 11 RdYlBu stops
    const stops = [];
    for (let i = 0; i <= 10; i++) {
      const varName = `--rdylbu-${i}`;
      const color = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      stops.push(color);
    }

    if (reverse) t = 1 - t;
    t = Math.max(0, Math.min(1, t));
    const idx = t * 10;
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const frac = idx - lower;

    if (lower === upper) return stops[lower];

    // Interpolate colors
    const c1 = this.parseColor(stops[lower]);
    const c2 = this.parseColor(stops[upper]);
    return this.mixColors(c1, c2, frac);
  },

  parseColor(color) {
    // Parse hex color
    const hex = color.trim();
    const r = parseInt(hex.substr(1, 2), 16);
    const g = parseInt(hex.substr(3, 2), 16);
    const b = parseInt(hex.substr(5, 2), 16);
    return { r, g, b };
  },

  mixColors(c1, c2, t) {
    const r = Math.round(c1.r * (1 - t) + c2.r * t);
    const g = Math.round(c1.g * (1 - t) + c2.g * t);
    const b = Math.round(c1.b * (1 - t) + c2.b * t);
    return `rgb(${r}, ${g}, ${b})`;
  },

  escape(str) {
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
  },

  formatInt(num) {
    // Swiss apostrophe thousands separator
    return Math.round(num).toString().replace(/\B(?=(\d{3})+(?!\d))/g, "'");
  },

  setupScrollSpy() {
    const sections = [...document.querySelectorAll('.section')];
    const navLinks = document.querySelectorAll('.nav-link');
    // Active = the last section whose top has passed under the sticky header
    const update = () => {
      const current = sections.filter(s => s.getBoundingClientRect().top <= 180).pop();
      navLinks.forEach(link => link.classList.toggle('active', !!current && link.getAttribute('href') === `#${current.id}`));
    };
    window.addEventListener('scroll', update, { passive: true });
    update();
  },

  setupAnimations() {
    // Animate elements on scroll with fade/slide-up
    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting && this.animateOnScroll.has(entry.target)) {
          if (window.matchMedia('(prefers-reduced-motion: no-preference)').matches) {
            entry.target.style.animation = 'fadeSlideUp 0.5s ease-out backwards';
          }
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    this.animateOnScroll.forEach((_, el) => {
      observer.observe(el);
    });
  }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => app.init());
} else {
  app.init();
}
