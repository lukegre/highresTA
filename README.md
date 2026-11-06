# High-res Total Alkalinity

Machine-learning reconstruction of surface ocean total alkalinity (TA) at 8-day, 0.25° resolution.
GLODAPv2 bottle measurements are collocated with gridded predictors (salinity, temperature,
sea-surface height, chlorophyll, mixed-layer depth, nutrients, bathymetry) and used to train an
ensemble of quantile regressors that give TA with an uncertainty estimate (10th, 50th and 90th percentiles).

## Method in brief

- **Model:** a linear salinity baseline plus a CatBoost model fitted to the residuals
  (`highres_ta.CatBoostResidualRegressor`), trained with a multi-quantile loss.
- **Splits:** the data are divided once into blocks, grouped by cruise and stratified by salinity.
  Block 0 is a held-out test set shared by all ensemble members.
- **Ensemble:** each remaining block serves in turn as the validation set, and for each such split
  one Optuna study (hyperparameters plus which features to drop) is run, and several diverse
  near-optimal trials (a Rashomon set) are refitted as members.
  The spread across members is decomposed into a data-split and a within-split (Rashomon/HPO) component.

Schematics are in [`docs/`](docs/), and a full methods page is in [`web/`](web/README.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```

Training data (parquet), the GLODAP adjustment table and the bathymetry file are expected in
`data/` (not tracked in git).

## Configuration

All settings live in [`configs/`](configs/), one YAML file per concern:

| File | Contents |
|---|---|
| `paths.yaml` | Every file location: training data, adjustment table, output directory, gridded predictors (remote OceanSODA-ETHZv2 store), bathymetry |
| `data.yaml` | Target, columns, quality filters, feature names |
| `training.yaml` | Model, cross-validation, hyperparameter search, ensemble and evaluation settings |
| `inference.yaml` | Model path, output zarr, date range, parallelism, and which gridded variables feed each feature |
| `validation.yaml` | Member aggregation, CPU limit, coastal depth threshold and alternate salinity products |

Relative paths resolve against the repository root. `ta.load_config()` merges all files in the
folder; scripts take `--config` to point at another folder or a single file.

## Usage

Train the ensemble (Optuna studies are stored in SQLite and resume where they stopped):

```bash
uv run python scripts/train_quantile_ensemble.py --dry-run      # check the data splits only
uv run python scripts/train_quantile_ensemble.py --stage all    # train all members and assemble the ensemble
scripts/train_splits.sh                                          # same, but every split in parallel (see optimization: in configs/training.yaml)
```

Run inference over the gridded predictors:

```bash
uv run python scripts/inference.py                       # everything from configs/inference.yaml
uv run python scripts/inference.py --start-date 2020-01-01 --end-date 2020-12-31
```

Gridded predictors are read over HTTP from the public OceanSODA-ETHZv2 store, so inference runs
anywhere with network access. The output zarr is created on the first run; rerunning the same
command on an existing store resumes it (dates are rewritten). Transient network errors are retried,
and the script exits with code 1 if any date still fails.

Extract each saved member's CatBoost feature importance and plot the mean ± standard deviation:

```bash
uv run python scripts/feature_importance.py
uv run python scripts/feature_importance.py --run-dir models/<run> --show
```

Outputs go to `<run>/feature_importance/`: `per_member.csv`, `summary.csv`, and a PNG/PDF
bar plot. Zero importances are masked (blank in the per-member CSV) and excluded from
each feature's statistics; `summary.csv` includes the contributing member count.
Importances describe the
boosted residual jointly across quantiles, excluding the separate linear salinity baseline.

Validate a saved model and generate PDF and interactive HTML reports:

```bash
uv run python scripts/validate_model.py
uv run python scripts/validate_model.py --model-path models/highresTA-rashomon-20261106/models/quantile_ensemble.joblib
```

The default model is `<paths.output_dir>/models/quantile_ensemble.joblib`.
Outputs go to `<run>/validation/`: `validation.pdf`, `validation.html`,
`predictions.parquet`, `scores.csv` and `metadata.json`. Use `--output-dir`,
`--max-cpus` or `--aggregation mean` to override defaults. The HTML is standalone
and works offline, with dataset/product selectors and interactive charts. Maps use bundled
public-domain Natural Earth 1:110m coastlines; chart widths are capped at 1,200 px
and geographic axes preserve their aspect ratio.

This formalises notebook 04: all members predict the common held-out test set;
each validation block is predicted only by members trained for that split.
The default is the median of member q50 predictions, as in the notebook (the saved
ensemble uses a mean). Data preparation uses the run's saved `configs.json` and
checks its fingerprint against `models/manifest.json` before using saved positional
split IDs. Regional scores include sample counts, RMSE, MAE, median absolute error,
Huber loss (delta 1.35), R² and mean/median bias (predicted minus observed).
Coastal means absolute bottom depth ≤500 m; high latitude means |latitude| ≥60°.
Alternate salinities retain original row positions and exclude missing or out-of-range
replacement values. Each product's scores use its available observations, so sample
counts can differ and comparisons are not paired. No gridded inference data are needed.

View the methods page:

```bash
cd web && python -m http.server 8000
```

## Repository layout

| Path | Contents |
|---|---|
| `src/highres_ta/` | Package: data loading, features, estimator, ensemble, training, splits, inference |
| `scripts/` | Command-line entry points for training, inference and station time series |
| `notebooks/` | Numbered analysis steps: collocation, EDA, training experiments, validation |
| `web/` | Static methods page and the script that builds its data |
| `docs/` | Train/test split and training schematics |
| `tests/` | Unit tests |

## Tests

```bash
uv run python -m unittest discover -s tests
```
