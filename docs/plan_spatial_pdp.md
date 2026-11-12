# Plan: spatial partial dependence (per-feature sensitivity maps)

Revised after review (Fable).

## Goal

For each predictor, map **where** surface TA responds to it: on the gridded inference
fields, perturb one feature in every ocean cell (all other features fixed) and map the
change in the ensemble's Q50 prediction. A local, ICE-style partial dependence evaluated
at every grid cell. Then show the figure on the methods website.

## Definition

For cell `x`, feature `j`, member `m`, step `δ_j`:

    x± = x_j ± δ_j            (log features: x_j · 10^(±δ_j); clipped at 0 for nonnegative)
    S_jm(x) = [f_m(x+) − f_m(x−)] · (2δ_j / (x+ − x−)) / 2      (µmol kg⁻¹ per +δ_j)

i.e. the secant slope over the **realised** step, scaled to a step of `δ_j`
(log features: realised step measured in log10 space). NaN where `x+ == x−` or `x_j` is NaN.

- `f_m` = member Q50 (`predict_members(...)[:, :, quantile_column(model.quantiles, 0.5)]`).
  The full model (linear salinity baseline + CatBoost residual) is perturbed.
- `δ_j = delta_scale × robust_scale_j`, `robust_scale_j = IQR_j / 1.349` of feature `j`
  over all `prepare_data` rows (train + test blocks; log10 of values > 0 for log
  features). Default `delta_scale: 0.25`. `δ_j` stored in feature units (log10 units for
  log features).
- **Members that dropped feature `j`** (`j in estimator.catboost_kwargs.get("ignored_features", [])`)
  are masked (NaN) for that feature, not counted as 0.
- Summary across the remaining members (nan-aware): `sensitivity` = median,
  `sensitivity_spread` = std (ddof=0), `sign_agreement` = fraction of members with the
  sign of the median (NaN where median == 0), `n_members(feature)` = members used.
  Note in attrs: summary uses the median across members regardless of `model.aggregation`.
- Across dates: nan-mean of each per-date field.
- `out_of_range` = fraction of dates where `x+` **or** `x−` is outside the training
  p01–p99 of feature `j` (marginal only; does not detect jointly implausible combinations).
- Excluded by default: `ncoord_x/y/z` (one spherical coordinate cannot move alone).

## Config: `configs/partial_dependence.yaml` (top-level `partial_dependence:`)

```yaml
partial_dependence:
  dates: ["2019-01-17", "2019-04-15", "2019-07-12", "2019-10-08"]  # one per season
  delta_scale: 0.25        # x robust scale (IQR/1.349) of the training feature
  stride: 4                # every 4th lat/lon cell -> 1°
  log_features: [chl_globcolour]
  nonnegative_features: [chl_globcolour, nitrate, phosphate, silicate, bottomdepth]
  exclude_features: [ncoord_x, ncoord_y, ncoord_z]
  max_cpus: 126
  feature_units:
    salinity: ""
    temperature: "°C"
    ssh_adt: "m"
    chl_globcolour: "log10 mg m⁻³"
    nitrate: "µmol kg⁻¹"
    phosphate: "µmol kg⁻¹"
    silicate: "µmol kg⁻¹"
    bottomdepth: "m"
    mld_dens_soda: "m"
```

## A. `src/highres_ta/partial_dependence.py` + `tests/test_partial_dependence.py` (Sonnet)

```python
def training_feature_stats(X: pd.DataFrame, features, log_features) -> pd.DataFrame
    # index=feature; columns: scale (IQR/1.349), p01, p99 — log10 space for log_features

def perturb_column(values: np.ndarray, delta: float, sign: int, *, log: bool,
                   nonnegative: bool) -> np.ndarray

def member_sensitivity(model, X: pd.DataFrame, feature: str, delta: float, *,
                       log: bool, nonnegative: bool) -> np.ndarray
    # (members, rows); ONE predict_members call on concat([X+, X−]); realised-step
    # rescaling; NaN for members whose ignored_features contain `feature`, and for rows
    # where X[feature] is NaN or x+ == x−.

def summarise_members(S: np.ndarray) -> dict[str, np.ndarray]
    # nan-median, nan-std, sign_agreement, n_members per row

def sensitivity_for_date(model, features: xr.Dataset, date, stats, *, delta_scale,
                         stride, log_features, nonnegative_features,
                         exclude_features) -> xr.Dataset
    # features.isel(lat=slice(stride//2, None, stride), lon=same) -> .load() ->
    # select_inference_date(...) -> droplevel("time") -> one row per (lat, lon)
    # features = [f for f in model.get_feature_names() if f not in exclude_features]
    # Output dims (feature, lat, lon) on the full strided grid (land = NaN):
    #   sensitivity, sensitivity_spread, sign_agreement, out_of_range (0/1)
    # coords: feature, delta(feature), n_members(feature)
    # attrs: delta_scale, units "umol kg-1", log_features, member_summary "median",
    #        model_aggregation

def spatial_sensitivity(model, features, dates, stats, **kw) -> xr.Dataset
    # inside joblib.parallel_config(backend="threading"); loop dates (log each),
    # nan-mean over dates; out_of_range -> fraction; attrs["dates"] = list of str
```

Export `spatial_sensitivity`, `training_feature_stats` from `highres_ta/__init__.py`.

Tests (unittest, no real data). Fake model: `quantiles=[0.1,0.5,0.9]`,
`get_feature_names()`, `estimators=[SimpleNamespace(catboost_kwargs={...})]`,
`predict_members(X)` linear in features. Fake Dataset: dims `time`, `dayofyear`, `lat`,
`lon` with `salinity` + other features, some land (NaN salinity). Cases: linear
`f = a x_j` → `S = a δ_j`; opposite-sign members → sign agreement/spread; member with
feature in `ignored_features` excluded (n_members); log perturbation analytic; clipping
rescaling exact for linear f; NaN feature cells → NaN; excluded features absent; strided
grid shape; two-date mean; out_of_range flag.

## B. `scripts/spatial_pdp.py` + `configs/partial_dependence.yaml` + CLAUDE.md line (Haiku)

Typer CLI: `--config`, `--override` (repeatable; copy from `scripts/inference.py`),
`--run-dir` (default `paths.output_dir`), `--output-dir` (default
`<run>/partial_dependence/`), `--date` (repeatable), `--stride`, `--delta-scale`,
`--max-cpus`, `--show`.

1. `config = ta.load_config(config_path, overrides)`. If `<run>/configs.json` exists and
   `--run-dir` was given, load the run's config from it for the data section (as
   `validation_runner.py` does); assert `training.data_fingerprint(data) ==
   manifest["data_fingerprint"]` if available, else warn.
2. Model `<run>/models/quantile_ensemble.joblib`, `model.optimise_cpus(max_cpus)`.
3. `data = ta.prepare_data(config)`; `stats = training_feature_stats(data[list(model.get_feature_names())], ...)`.
4. `features = ta.open_inference_data(config, model.get_feature_names())`.
5. `ds = ta.spatial_sensitivity(...)` with HTTP retry per date (3 attempts, like
   `scripts/inference.py`); write `sensitivity.nc`, `summary.csv` (per feature: n_members,
   delta, area-weighted (cos lat) mean |S|, median S, fraction of cells sign_agreement ≥ 0.8).
6. Figures: `sensitivity_maps.{png,pdf}` — one panel per feature ordered by mean |S|,
   `RdBu_r`, symmetric per-panel limits at p98 |S|, coastlines from
   `highres_ta/assets/ne_110m_coastline.json`, title `"<feature>  +δ = <delta> <unit>"`,
   stipple where sign_agreement < 0.8, hatch where out_of_range > 0.5;
   `sensitivity_spread.{png,pdf}` with `viridis`. Save PNG with `transparent=True` too
   (`sensitivity_maps_transparent.png`) with mid-grey text/coastlines for the website.
7. CLAUDE.md: one line under Commands.

## Run

```bash
uv run python scripts/spatial_pdp.py --override configs/variants/stacked_salinity.yaml --date 2019-07-12 --stride 16   # smoke
uv run python scripts/spatial_pdp.py --override configs/variants/stacked_salinity.yaml --stride 2                     # final
```

Smoke-check step stability with `--delta-scale 0.1/0.25/0.5` (pattern and sign stable).

## Website (after the figure exists)

- `web/app.js`: `case 'figure'` in the block switch: `<figure class="figure-block">` with
  `<img loading="lazy">` (alt via attribute), `<figcaption>` rendered with `this.md`,
  optional `full` link. Style in `style.css` as a bordered light card (like `.atlas`) so
  it reads in dark mode.
- `web/maps/sensitivity.webp` from the transparent PNG (Pillow, ~1600 px wide).
- `web/content.json`: section `drivers` (with `kicker`, `title`, `lead`) between
  `ensemble` and `uncertainty`; add `{id: "drivers", label: "Drivers"}` to `site.nav`;
  renumber kickers after it. Blocks: `text` (what is perturbed, how to read it),
  `figure`, `callout` (info: local one-at-a-time, correlated predictors, marginal
  out-of-range only, trees flat outside training range). Numbers from `summary.csv`.
- `web/README.md`: add `figure` to the block-type list.

## Out of scope

Interaction maps, SHAP, Q10/Q90 sensitivity, time series of sensitivity.
