# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Machine-learning reconstruction of high-resolution (8-day, 0.25°) surface ocean total alkalinity (TA, column `talk`) from GLODAP bottle data collocated with gridded satellite/reanalysis predictors. The model is an ensemble of quantile regressors (q10/q50/q90) whose spread is decomposed into data-split and hyperparameter-optimisation (HPO) uncertainty. `TODO.md` tracks the current state of the pipeline (uncertainty calibration, inference, comparison with BATS/HOT).

## Commands

Environment is managed with `uv` (Python 3.11, `src/` layout, package `highres_ta`, imported as `ta`).

```bash
uv sync                                                   # install
uv run python -m unittest discover -s tests               # all tests (unittest, no pytest dependency)
uv run python -m unittest tests.test_quantiles            # one module
uv run python -m unittest tests.test_split_hpo_ensemble.<TestClass>.<test_method>
```

Training (`scripts/train_quantile_ensemble.py`, typer CLI, config `scripts/quantile_ensemble_config.yaml`):

```bash
uv run python scripts/train_quantile_ensemble.py --dry-run                     # load data + validate splits only
uv run python scripts/train_quantile_ensemble.py --stage train --split 1 --hpo-replicate 0 --n-trials 2 --optuna-jobs 1   # smoke test one member
uv run python scripts/train_quantile_ensemble.py --stage all                   # train/resume all members, then assemble
uv run python scripts/train_quantile_ensemble.py --stage assemble              # assemble from existing members
```

Inference: `uv run python scripts/inference.py <model_path> <zarr_out> [--max-cpus N] [--aggregation mean|median]` (writes per-date zarr of member predictions + variance decomposition; aggregation defaults to what the ensemble was saved with).

Methods web page: `cd web && python -m http.server 8000`; regenerate `web/data.json` with `uv run python web/build_data.py` (see `web/README.md` — edit text in `content.json`, never hand-edit `data.json`).

## Architecture

**Pipeline:** collocation (`colloc/`, `glodap/`, `notebooks/00_collocate.ipynb`) → training parquet in `data/training/*.pq` → `train_quantile_ensemble.py` → `QuantileRegressionEnsemble` joblib → `scripts/inference.py` over the gridded predictor zarr.

**Where code lives:** scripts are thin typer CLIs. Training logic (data prep, Optuna objective, member training, ensemble assembly, `load_config`/`prepare_data`/`resolve_path`) is in `highres_ta/training.py`; the outer block/split/tuning-fold logic is in `highres_ta/train_test_split.py`. Notebooks and `web/build_data.py` use these via `import highres_ta as ta`.

**Model (`estimators.CatBoostResidualRegressor`):** a polynomial linear baseline on `linear_features` (salinity — TA is dominated by salinity) plus a CatBoost model fitted to the residuals. CatBoost kwargs are passed through `**catboost_kwargs` and exposed via custom `get_params`/`set_params` so it works with sklearn cloning/search; `loss_function` (e.g. `MultiQuantile:alpha=...`) determines `quantiles_`. Use `predict_quantiles` for the (n, n_quantiles) output.

**Ensemble design (the non-obvious part, see docstring of `train_quantile_ensemble.py` and `docs/split_*schematic*`):**
- Data are partitioned once into `n_parts` blocks with `StratifiedGroupKFold` — grouped by cruise (`expocode`) to avoid leakage between spatially/temporally correlated samples, stratified by salinity bins (`train_test_split.py`). `split_random_state` is independent of member seeds.
- Block 0 is the test set for **every** member and is never trained on.
- Split `k` (1..n_parts-1) validates on block `k` and trains on the remaining blocks. Within each split, `n_hpo_replicates` independent Optuna studies (seed `100*k + r`) each produce one member. HPO uses leave-one-block-out CV inside the training blocks only.
- `quantile_ensemble.decompose_variance` splits member variance into split + HPO terms via the law of total variance (predictions shaped `(splits, replicates, obs, quantiles)`).
- Optuna studies are SQLite-backed and resumable; a data fingerprint is stored in metadata to detect changed inputs.

**Outputs** go to `config.output_dir` (e.g. `models/highresTA-<date>/`, gitignored): `studies/*.sqlite3`, `splits/split_k.parquet`, `predictions/<member>_{validation,test}.parquet`, `models/quantile_split_k_hpo_r.{joblib,json}`, `models/quantile_ensemble.joblib`, `models/manifest.json`.

**Features:** configured in the YAML. `engineered_feature_names` (spherical `ncoord_x/y/z` from lat/lon, cyclical day-of-year) are computed in `features.py` and must not be read from the parquet. Data prep in `prepare_data` also applies salinity bounds, TA QC flag filtering (`target_filtering.py`) and GLODAP cruise adjustments (`dataio.add_talk_adjustment`, CSV in `data/`).

**Inference data paths** in `src/highres_ta/inference.py` default to the ETH `/net/sea/work/...` filesystem; override with `HIGHRES_TA_FNAME_PROCESSED` / `_BATHYMETRY` / `_NUTRIENTS` env vars or a repo-root `.env` (see `.env.example`).

Notebooks `01`–`05` are numbered analysis steps (EDA, training variants, validation, output checks). `scripts/get_timeseries_stations.py` fetches station time series (BATS, HOT, ESTOC, …) for evaluation.

## Notes

- Notebook-only packages are in the `dev` dependency group (installed by default with `uv sync`).
- `glodap/bottle_data.py` contains legacy functions importing a `utils` module that no longer exists; they are not used by the current pipeline.
- `data/`, `models/`, `*.nc`, `*.pkl`, `*.db` are gitignored; `ta.resolve_path` resolves relative paths against the repo root (found via `pyproject.toml`).
