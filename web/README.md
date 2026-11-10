# highres-TA web pages

Serve the checked-in site over HTTP, or publish the contents of this directory at a static host. The landing page, interactive method, run comparison and data explorer are all addressable directly.

    python3 web/viewer/serve.py web --port 8780     # then open http://localhost:8780

The observation map uses 42,134 collocated GLODAPv2.2023 bottle records and the local ETOPO2022 land mask. The dated Q50 image and its full scale come from the saved figure in [the output-check notebook](../notebooks/05_check_outputs.ipynb).

Set the public dataset URL in [the viewer TOML](viewer/explorer.toml) and [rebuild the viewer](viewer/README.md) when the pyramid is ready. The blank setting leaves the store picker available.

| File | Purpose | Edit it? |
|---|---|---|
| `content.json` | Site navigation, overview, figures, method, interactive labels, comparison narrative and run lineage, viewer introduction | **Yes** |
| `data.json` | Numbers and member predictions from the model run | No: regenerate with `uv run python web/build_data.py` |
| `tokens.css` | RdYlBu palette and light/dark tokens | Yes, for colours |
| `style.css`, `app.js` | Layout and page renderer | Rarely |
| `uncertainty.css`, `uncertainty.js` | Interactive uncertainty explorer and charts | Rarely |
| `method.html`, `method.css`, `method.js` | Production method and interactive held-out bottle example | Yes |

## Editing content

Edit `project.links` for the shared primary navigation and `overview.sectionNav` for landing-page anchors. `overview.maps` pairs each source image and optional full-resolution link with its caption, legend, and alt text. Edit `comparison.lineage` to change each run’s parent and description of what changed. Page copy updates on reload. Measured scores and saved predictions are kept in their source data, and upstream viewer configuration lives in TOML.

Each section in `content.json` has `blocks`, rendered in order. Text supports `**bold**`, `` `code` ``
and `{{path.to.value|fmt}}` placeholders that read from `data.json` (`fmt`: `int`, `pct`, or a number of decimals).

Block types: `text`, `cards`, `steps`, `table`, `callout` (`tone`: info/warn/key), `equation`, `funnel`,
`split-explorer`, `member-grid`, `uncertainty-explorer`, `variance-decomposition`, `spread-vs-error`,
`ensemble-growth`. Reorder, remove or add blocks freely.

## Current methods and result provenance

The method page describes the split-wise Rashomon design: one Optuna
study per split, feature omissions, and seven diverse near-optimal trials
refitted per split. `data.json` currently contains the completed
`highresTA-rashomon-20261106` run (1982–2021). The method page identifies the
source run and computes its worked bottle estimates from saved held-out
predictions. The salinity baseline is reproduced by fitting degree-one OLS on each split’s training observations and weighting those fits by member count.

After a new run has finished training **and assembly**, regenerate the results:

    uv run python web/build_data.py

The exporter defaults to `paths.output_dir`, reads that run's saved `configs.json`,
and handles the current member metadata (`params`, selected trial, loss rank,
Rashomon membership and dropped features). It refuses incomplete runs before
replacing `data.json`. The explorer requires mean aggregation and Q10/Q50/Q90.
To export a different completed run, pass `--run models/<run-folder>`.
The historical independent-HPO run can be regenerated explicitly with:

    uv run python web/build_data.py --run models/quantile_ensemble_split_hpo_1993-2022

That historical compatibility path restores its SSH completeness requirement and
10-feature input set; other runs require their own `configs.json` snapshot.
Member indices denote selection order within each split, not matched trials
across splits. The saved `hpo_var` field is shown as within-split variance: in a
Rashomon run it includes selected model choices and refit randomness.

## Regenerate the landing-page figure

The existing figure stays in place until you generate another one. Read an inference store and choose a May 2021 field with:

```sh
uv run python scripts/build_demo_map.py --store /path/to/highresTA.zarr --date 2021-05-20 --nearest --update-site
```

The CLI writes a full PNG and a WebP map card, and prints the actual selected date. Omit `--update-site` to generate images without changing the card. Use `--group 0` for a grouped store. Defaults resolve relative to the repository.
