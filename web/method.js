// The original methods renderer loads the custom block registry before app.js.
window.BLOCKS = window.BLOCKS || {};
window.BLOCKS['worked-bottle'] = (host, block, ctx) => {
  const samples = ctx.data.test_sample || [];
  const copy = ctx.content.ui;
  if (!samples.length) {
    host.textContent = copy.no_bottle_data;
    return;
  }

  const widget = document.createElement('div');
  widget.className = 'worked-bottle';
  const controls = document.createElement('div');
  controls.className = 'worked-bottle__controls';
  const label = document.createElement('label');
  label.textContent = copy.worked_bottle_label;
  label.htmlFor = 'worked-bottle-select';
  const select = document.createElement('select');
  select.id = label.htmlFor;
  samples.forEach((sample, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `Bottle ${sample.id} · ${sample.cruise} · ${sample.date}`;
    select.appendChild(option);
  });
  controls.append(label, select);

  const result = document.createElement('div');
  function render(sample) {
    const members = sample.pred || [];
    const means = [0, 1, 2].map(q => members.reduce((sum, row) => sum + row[q], 0) / members.length);
    const sd = Math.sqrt(members.reduce((sum, row) => sum + (row[1] - means[1]) ** 2, 0) / members.length);
    result.replaceChildren();
    const meta = document.createElement('p');
    meta.className = 'worked-bottle__meta';
    meta.textContent = `${sample.date} · ${Math.abs(sample.lat)}°${sample.lat < 0 ? 'S' : 'N'}, ${Math.abs(sample.lon)}°${sample.lon < 0 ? 'W' : 'E'} · depth ${sample.depth} m · ${sample.cruise}`;
    result.appendChild(meta);

    const values = document.createElement('div');
    values.className = 'worked-bottle__values';
    const entries = [
      [copy.worked_observed, sample.observed],
      [copy.worked_q10, means[0]],
      [copy.worked_q50, means[1]],
      [copy.worked_q90, means[2]]
    ];
    entries.forEach(([name, number]) => {
      const cell = document.createElement('div');
      cell.className = 'worked-bottle__value';
      const caption = document.createElement('span'); caption.textContent = name;
      const value = document.createElement('strong'); value.textContent = Number(number).toFixed(1);
      cell.append(caption, value); values.appendChild(cell);
    });
    result.appendChild(values);

    const note = document.createElement('p');
    note.className = 'worked-bottle__note';
    if (Number.isFinite(sample.salinity) && Number.isFinite(sample.baseline)) {
      const residual = means[1] - sample.baseline;
      note.textContent = copy.worked_with_baseline
        .replace('{salinity}', Number(sample.salinity).toFixed(2)).replace('{baseline}', Number(sample.baseline).toFixed(1))
        .replace('{residual}', residual.toFixed(1)).replace('{q50}', means[1].toFixed(1))
        .replace('{observed}', Number(sample.observed).toFixed(1)).replace('{error}', (means[1] - sample.observed).toFixed(1))
        .replace('{members}', members.length).replace('{spread}', sd.toFixed(1));
    } else {
      note.textContent = copy.worked_without_baseline.replace('{q50}', means[1].toFixed(1))
        .replace('{observed}', Number(sample.observed).toFixed(1)).replace('{error}', (means[1] - sample.observed).toFixed(1))
        .replace('{members}', members.length).replace('{spread}', sd.toFixed(1));
    }
    result.appendChild(note);
  }
  select.addEventListener('change', () => render(samples[Number(select.value)]));
  widget.append(controls, result);
  host.appendChild(widget);
  render(samples[0]);
};
