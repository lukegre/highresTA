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

Training (`scripts/train_quantile_ensemble.py`, typer CLI, config from `configs/`):

```bash
uv run python scripts/train_quantile_ensemble.py --dry-run                     # load data + validate splits only
uv run python scripts/train_quantile_ensemble.py --stage train --split 1 --n-trials 8 --maximum-iterations 20 --parallel-trials 1 --threads-per-trial 2   # smoke test one split
uv run python scripts/train_quantile_ensemble.py --stage all                   # train/resume all members, then assemble
scripts/train_splits.sh                                                        # every split as its own parallel process (cores = n_splits × parallel_trials × threads_per_trial), then assemble; `SPLITS="1 2"` runs only those splits and skips assembly (run `--stage assemble` afterwards)
uv run python scripts/train_quantile_ensemble.py --stage assemble              # assemble from existing members
```

Inference: `uv run python scripts/inference.py` — all settings from `configs/inference.yaml`; CLI options (`--model-path`, `--zarr-out`, `--start-date`, `--end-date`, `--n-workers`, `--max-cpus`, `--aggregation`) override. `ta.create_output_template` first creates the output zarr (NaN-filled, full grid, all requested time steps, layout taken from one real prediction); each date is then written with `to_zarr(region="auto")` by concurrent workers. An existing store is reused (resume) if it covers the requested times on the same grid, else `ValueError`. Transient HTTP errors from the remote store are retried per date (`N_ATTEMPTS`); any date that still fails makes the script exit 1. `aggregation: null` uses what the ensemble was saved with.

## Configuration

`configs/*.yaml`, one file per concern, each owning distinct top-level keys: `paths.yaml` (`paths:` — every file location and URL), `data.yaml` (`data:`), `training.yaml` (`model:`, `cross_validation:`, `optimization:`, `evaluation:`, `ensemble:`), `inference.yaml` (`inference:`). `ta.load_config()` merges the whole folder (error on duplicate top-level keys) or loads one file; scripts take `--config`. Relative paths resolve against the repo root, URLs pass through. Keep paths in `paths.yaml` — no env vars or hard-coded paths in code.

Methods web page: `cd web && python -m http.server 8000`; regenerate `web/data.json` with `uv run python web/build_data.py` (see `web/README.md` — edit text in `content.json`, never hand-edit `data.json`).

## Architecture

**Pipeline:** collocation (`colloc/`, `glodap/`, `notebooks/00_collocate.ipynb`) → training parquet in `data/training/*.pq` → `train_quantile_ensemble.py` → `QuantileRegressionEnsemble` joblib → `scripts/inference.py` over the gridded predictor zarr.

**Where code lives:** scripts are thin typer CLIs. Training logic (data prep, Optuna objective, member training, ensemble assembly, `load_config`/`prepare_data`/`resolve_path`) is in `highres_ta/training.py`; the outer block/split/tuning-fold logic is in `highres_ta/train_test_split.py`. Notebooks and `web/build_data.py` use these via `import highres_ta as ta`.

**Model (`estimators.CatBoostResidualRegressor`):** a polynomial linear baseline on `linear_features` (salinity — TA is dominated by salinity) plus a CatBoost model fitted to the residuals. CatBoost kwargs are passed through `**catboost_kwargs` and exposed via custom `get_params`/`set_params` so it works with sklearn cloning/search; `loss_function` (e.g. `MultiQuantile:alpha=...`) determines `quantiles_`. Use `predict_quantiles` for the (n, n_quantiles) output.

**Ensemble design (the non-obvious part, see docstring of `train_quantile_ensemble.py` and `docs/split_*schematic*`; the schematic still shows the previous per-replicate-HPO design):**
- Data are partitioned once into `n_parts` blocks with `StratifiedGroupKFold` — grouped by cruise (`expocode`) to avoid leakage between spatially/temporally correlated samples, stratified by salinity bins (`train_test_split.py`). `split_random_state` is independent of member seeds.
- Block 0 is the test set for **every** member and is never trained on.
- Split `k` (1..n_parts-1) validates on block `k` and trains on the remaining blocks. Each split runs **one** Optuna study (sampler seed `100*k`; TPE + percentile pruning) over CatBoost hyperparameters plus a categorical `dropped_features` (0..`max_dropped_features` non-protected features passed as `ignored_features`). HPO uses leave-one-block-out CV inside the training blocks only.
- Members are chosen per split by Rashomon selection (`select_rashomon_members`): trials within `rashomon_epsilon` of the best inner-CV loss, picked greedily for diversity (dropped-feature Jaccard + normalised hyperparameter distance), `n_members_per_split` of them, each refitted with seed `100*k + r`.
- `quantile_ensemble.decompose_variance` splits member variance into split + HPO terms via the law of total variance (predictions shaped `(splits, replicates, obs, quantiles)`; the replicate axis is the Rashomon member index within a split).
- Optuna studies are SQLite-backed and resumable (heartbeat fails trials left RUNNING by a killed process); member selection is deterministic given the stored study; the study itself is only reproducible with `parallel_trials: 1`; a data fingerprint is stored in metadata to detect changed inputs.

**Outputs** go to `paths.output_dir` (e.g. `models/highresTA-<date>/`, gitignored): `studies/split_k.sqlite3`, `studies/split_k_rashomon.csv`, `splits/split_k.parquet`, `predictions/<member>_{validation,test}.parquet`, `models/quantile_split_k_member_r.{joblib,json}`, `models/quantile_ensemble.joblib`, `models/manifest.json`.

**Features:** configured in `configs/data.yaml`. `engineered_feature_names` (spherical `ncoord_x/y/z` from lat/lon, cyclical day-of-year) are computed in `features.py` and must not be read from the parquet. Data prep in `prepare_data` also applies salinity bounds, TA QC flag filtering (`target_filtering.py`) and GLODAP cruise adjustments (`dataio.add_talk_adjustment`, CSV in `data/`).

**Inference inputs** (`src/highres_ta/inference.py`): gridded predictors come from the public OceanSODA-ETHZv2 GeoZarr pyramid on `data.up.ethz.ch` (level `0` = 0.25°; opened over HTTP), bathymetry from a local netCDF (both in `paths.yaml`). `open_inference_data` builds training-named features from `inference.feature_sources` (fillna chain in listed order), and this is where product/unit mismatches live: SST is converted K→°C from the `units` attr, elevation (`positive: up`) → positive depth, `chl_cci_gapfilled` stands in for `chl_globcolour`, `mld_soda` for `mld_dens_soda`.

Notebooks `01`–`05` are numbered analysis steps (EDA, training variants, validation, output checks). `scripts/get_timeseries_stations.py` fetches station time series (BATS, HOT, ESTOC, …) for evaluation.

## Notes

- Notebook-only packages are in the `dev` dependency group (installed by default with `uv sync`).
- `glodap/bottle_data.py` contains legacy functions importing a `utils` module that no longer exists; they are not used by the current pipeline.
- `data/`, `models/`, `*.nc`, `*.pkl`, `*.db` are gitignored; `ta.resolve_path` resolves relative paths against the repo root (found via `pyproject.toml`).
