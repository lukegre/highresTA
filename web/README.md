# highres-TA methods page

Static page, no build step. Because it loads JSON with `fetch`, serve the folder:

    cd web && python -m http.server 8000     # then open http://localhost:8000

| File | Purpose | Edit it? |
|---|---|---|
| `content.json` | **All text**, section order and block layout | **Yes** |
| `data.json` | Numbers and member predictions from the model run | No: regenerate with `uv run python web/build_data.py` |
| `tokens.css` | RdYlBu palette and light/dark tokens | Yes, for colours |
| `style.css`, `app.js` | Layout and page renderer | Rarely |
| `uncertainty.css`, `uncertainty.js` | Interactive uncertainty explorer and charts | Rarely |

## Editing content

Each section in `content.json` has `blocks`, rendered in order. Text supports `**bold**`, `` `code` ``
and `{{path.to.value|fmt}}` placeholders that read from `data.json` (`fmt`: `int`, `pct`, or a number of decimals).

Block types: `text`, `cards`, `steps`, `table`, `callout` (`tone`: info/warn/key), `equation`, `funnel`,
`split-explorer`, `member-grid`, `uncertainty-explorer`, `variance-decomposition`, `spread-vs-error`,
`ensemble-growth`. Reorder, remove or add blocks freely.

## Current methods and result provenance

The text describes the split-wise Rashomon design: one Optuna study per split,
feature omissions, and seven diverse near-optimal trials refitted per split.
`data.json` currently contains the earlier 1993–2021 independent-HPO results.
The page identifies that source run above the methods; its scores and charts must
not be interpreted as evaluation of the new design.

After the new run has finished training **and assembly**, regenerate the results:

    uv run python web/build_data.py

The exporter defaults to `paths.output_dir`, reads that run's saved `configs.json`,
and handles the current member metadata (`params`, selected trial, loss rank,
Rashomon membership and dropped features). It refuses incomplete runs before
replacing `data.json`. The explorer requires mean aggregation and Q10/Q50/Q90.
To export a different completed run, pass `--run models/<run-folder>`.
The previously published run can be regenerated explicitly with:

    uv run python web/build_data.py --run models/quantile_ensemble_split_hpo_1993-2022

That historical compatibility path restores its SSH completeness requirement and
10-feature input set; other runs require their own `configs.json` snapshot.
Member indices denote selection order within each split, not matched trials
across splits. The saved `hpo_var` field is shown as within-split variance: in a
Rashomon run it includes selected model choices and refit randomness.
