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
| `sensitivity.js`, `explain.js` | Sensitivity maps; SHAP summary, maps and interactions (styles at the end of `method.css`; `explain.js` needs `sensitivity.js` loaded first) | Rarely |
| `method.html`, `diagnostics.html`, `uncertainty.html`, `method.css` | Shells for the Method, Uncertainty and Drivers pages; `content.json` sections and nav items carry a `"page"` (default `method`) that picks the shell | Yes |
| `access.html`, `access-repr.html` | Access Data page; `access-repr.html` is the saved `ds._repr_html_()` of the published store (`xr.open_zarr(url, group="0")`), regenerate it when the store changes | Yes |

## Hosting on RenkuLab

`web/` is a self-contained RenkuLab app: `Procfile` starts `viewer/serve.py` on `$RENKU_SESSION_PORT`, and `requirements.txt` (empty, stdlib only) marks the build as Python. Create an **app** launcher that builds from this repository with `context_dir: "web"`, `builder_variant: "python"`, `frontend_variant: "none"`, so the heavy model dependencies in the root `pyproject.toml` are not installed. The project and the git repository must both be public.

The data store sends no CORS headers, so the explorer reads it through the server's `proxy/` relay. The Procfile limits the relay to `data.up.ethz.ch` (`--relay-host`); add another `--relay-host` if the store moves. All links in the site are relative, so it also works under a path prefix.

Test the app command locally:

    cd web && RENKU_SESSION_PORT=8000 sh -c "$(sed 's/^web: //' Procfile)"

## Regenerate the sensitivity maps

The Drivers section on the diagnostics page reads `sensitivity-data.json`. After running `scripts/spatial_pdp.py` for a run, export it with:

    uv run python web/build_sensitivity.py --run models/<run-folder>

## Regenerate the SHAP contributions and interactions

The Contributions section reads `explain-data.json` (one fetch shared by its three blocks). After running `scripts/explain_model.py` for a run, export it with:

    uv run python web/build_explain.py --run models/<run-folder>

The file holds a random subset of 1,500 explained bottles, the SHAP maps (zlib-compressed int16, inflated in the browser with `DecompressionStream`), the interaction matrix and the 2-D partial-dependence pairs (about 0.6 MB).

## Editing content

Edit `project.links` for the shared primary navigation and `overview.sectionNav` for landing-page anchors. `overview.maps` pairs each source image and optional full-resolution link with its caption, legend, and alt text. Edit `comparison.lineage` to change each run’s parent and description of what changed. Page copy updates on reload. Measured scores and saved predictions are kept in their source data, and upstream viewer configuration lives in TOML.

Each section in `content.json` has `blocks`, rendered in order. Text supports `**bold**`, `` `code` ``, `[label](https://…)` links (http/https only, new tab)
and `{{path.to.value|fmt}}` placeholders that read from `data.json` (`fmt`: `int`, `pct`, or a number of decimals).
`equation` items are LaTeX, rendered with KaTeX from cdnjs (the raw string is shown if it fails to load).

Block types: `text`, `cards`, `steps`, `table`, `callout` (`tone`: info/warn/key), `equation`, `sensitivity-maps` (`data`, `labels`, `units`, `note`, `ui`; one interactive map per predictor from `sensitivity-data.json`),
`shap-summary` (`labels`, `units`, `note`, `ui`; beeswarm of bottle contributions from `explain-data.json`),
`shap-maps` (same fields as `sensitivity-maps`; one map per predictor with its contribution, including Location),
`interactions` (`labels`, `units`, `note`, `ui`; interaction-strength matrix plus a pair viewer with a joint/interaction toggle),
`details` (collapsed `<details>`; `title` + nested `blocks`), `cleaning-steps` (`items`: `name`, `body`; numbers
come from `data.funnel` in the same order), `split-explorer`, `member-grid` (`rows_label`, `rows_note`,
`cols_label`, `cols_note`, `hint`),
`uncertainty-explorer`, `uncertainty-budget`, `spread-vs-error`, `ensemble-growth`. Reorder, remove or add blocks freely.

Clicking a member cell, a split label or "Members of this split" opens a popup table of those members (MAE, never RMSE).
The hero's open-ocean MAE comes from `data.json` → `regional_scores.open_ocean` (bottom depth > 500 m and |lat| < 60°).

## Current methods and result provenance

The method page describes the split-wise Rashomon design: one Optuna
study per split, feature omissions, and seven diverse near-optimal trials
refitted per split. `data.json` currently contains the completed
`highresTA-stacked-salinity-20261109` run (1982–2021), using GLODAP and CARIMED
observations and a gridded salinity stack. The method page draws its model schematic from saved held-out
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
