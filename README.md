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
  several independent Optuna hyperparameter searches each produce one member.
  The spread across members is decomposed into a data-split and a hyperparameter (HPO) component.

Schematics are in [`docs/`](docs/), and a full methods page is in [`web/`](web/README.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync
```

Training data are expected as parquet files in `data/training/` (not tracked in git).
Paths, features, quantiles and cross-validation settings are set in
[`scripts/quantile_ensemble_config.yaml`](scripts/quantile_ensemble_config.yaml).

## Usage

Train the ensemble (Optuna studies are stored in SQLite and resume where they stopped):

```bash
uv run python scripts/train_quantile_ensemble.py --dry-run      # check the data splits only
uv run python scripts/train_quantile_ensemble.py --stage all    # train all members and assemble the ensemble
```

Run inference over the gridded predictors:

```bash
uv run python scripts/inference.py models/<run>/models/quantile_ensemble.joblib <output.zarr>
```

The input data paths default to the ETH `/net/sea` file system. To run elsewhere, copy
`.env.example` to `.env` and set the `HIGHRES_TA_FNAME_*` paths.

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
