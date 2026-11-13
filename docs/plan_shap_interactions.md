# Plan: SHAP contributions and interaction plots

Explain the ensemble's Q50 with additive per-predictor contributions (SHAP), pairwise
interaction strength, and 2-D partial dependence for the strongest pairs.

## Facts the design relies on (checked on highresTA-stacked-salinity-20261109)

- `CatBoostResidualRegressor` = linear model on `linear_features` (salinity only, polynomial
  degree 1) + CatBoost `MultiQuantile` residual. Prediction = `linear_model_.predict(X)[:, None] + boosting_model_.predict(X)`.
- `boosting_model_.get_feature_importance(Pool(X), type="ShapValues")` → `(n, 3, n_features + 1)`
  (last column = expected value), exactly additive to `boosting_model_.predict(X)` (error 1e-12).
  2000 rows ≈ 0.4 s per member.
- `type="ShapInteractionValues"` → `(n, 3, n_features + 1, n_features + 1)`; 300 rows ≈ 6 s per member (8 threads).
- `X` must have columns in `estimator.feature_names_in_` order. Members can have `ignored_features`; their SHAP is 0 for those features, which is correct for the ensemble mean.
- The ensemble aggregation is `mean` (42 members), so ensemble SHAP = mean of member SHAP (exact). If `aggregation == "median"`, still use the member mean and record it in attrs.
- Quantile column: `quantile_column(model.quantiles, 0.5)` (Q50 index 1).

## Definitions

Member contribution for feature `j` at row `x` (Q50 output `k`):
- boosted: CatBoost SHAP `φ_j^boost(x)`;
- linear: for each linear feature (salinity), `φ_sal^lin(x) = lin(x) − mean_bg lin`, where `bg` is the
  background frame passed in (the explained rows themselves by default); `mean_bg lin` is added to the base.
- `φ_j = φ_j^boost + φ_j^lin`; `base = E^boost + mean_bg lin`; `Σ_j φ_j + base = member Q50` exactly.

Ensemble: mean over members of `φ_j` and `base`; also `shap_spread` = std over members (ddof=0).

Groups: `location = ncoord_x + ncoord_y + ncoord_z` (additivity makes summing valid; for interactions,
sum the block). Group map from config.

Interaction values: CatBoost SHAP interaction (boosted part only; the linear salinity term is a pure
main effect, add `φ_sal^lin` to the salinity diagonal). Ensemble mean over members.
Strength matrix `I[a, b] = mean_rows |Φ_ab + Φ_ba|` for a≠b (CatBoost splits the pair symmetrically),
main effect `M[a] = mean_rows |Φ_aa|`.

2-D partial dependence for pair (a, b): grid `n_grid × n_grid` over the background p02–p98 of each
feature (log10 spacing for `log_features`); for each grid point set both columns on every background
row, ensemble Q50 mean over rows → `PD(a, b)`. Also 1-D `PD(a)`, `PD(b)` on the same grids, and the
pure interaction surface `PD(a,b) − PD(a) − PD(b) + mean(PD(a,b))` and Friedman's H² for the pair.
One `predict_members` call per pair on the stacked frame (chunk if > 2e6 rows).

## A. `src/highres_ta/explain.py` + `tests/test_explain.py` (Sonnet)

```python
def member_shap(estimator, X: pd.DataFrame, quantile_index: int,
                background: pd.DataFrame | None = None) -> tuple[np.ndarray, float]
    # (n, n_model_features) contributions in model.get_feature_names() order, and base (float)

def ensemble_shap(model, X, quantile=0.5, background=None, groups: dict | None = None,
                  n_jobs: int = 1) -> xr.Dataset
    # dims (row, feature); vars shap (mean), shap_spread (std over members), value (feature value;
    # NaN for groups), coords base (scalar mean base), prediction (row) = model member-mean Q50.
    # groups applied after averaging (sum of member features). Raises if |Σφ + base − prediction|
    # > 1e-6 * scale. Members run in a joblib threading pool (CatBoost releases the GIL).

def ensemble_shap_interactions(model, X, quantile=0.5, groups=None, n_jobs=1) -> xr.Dataset
    # vars: interaction (row, feature, feature2) ensemble-mean values (grouped),
    #       strength (feature, feature2) = mean |Φ_ab + Φ_ba| off-diagonal, NaN on diagonal,
    #       main (feature) = mean |Φ_aa|

def top_pairs(strength: xr.DataArray, k: int) -> list[tuple[str, str]]   # by strength, a<b, no duplicates

def partial_dependence_2d(model, background: pd.DataFrame, a: str, b: str, *, n_grid=20,
                          quantile=0.5, log_features=()) -> xr.Dataset
    # coords a_value(n_grid), b_value(n_grid); vars pd (a_value, b_value), pd_a (a_value),
    # pd_b (b_value), interaction (a_value, b_value); attrs a, b, h_squared

def shap_maps(model, features: xr.Dataset, dates, *, stride, groups=None, quantile=0.5, n_jobs=1) -> xr.Dataset
    # per date: strided grid → rows (reuse the selection logic of partial_dependence.sensitivity_for_date:
    # factor it into a private helper `_grid_frame(features, date, stride) -> (X, lat, lon)` in
    # partial_dependence.py and import it); ensemble_shap with background = that date's rows;
    # vars shap (feature, lat, lon), mean over dates; attrs dates, quantile, base (mean)
```

Export `ensemble_shap`, `ensemble_shap_interactions`, `partial_dependence_2d`, `shap_maps` from `highres_ta/__init__.py`.

Tests (unittest; fake estimators): build a real tiny `CatBoostResidualRegressor` with
`loss_function="MultiQuantile:alpha=0.1,0.5,0.9"`, few iterations, synthetic data where
`y = 50*s + 3*t*n + noise`, two members in a real `QuantileRegressionEnsemble`. Check additivity,
linear salinity contribution, ignored feature → zero SHAP, grouping sums, interaction strength
highest for (t, n), 2-D PD shape/grids and H² > 0 for (t, n) and ≈0 for an additive pair,
shap_maps grid shape and land NaN (fake Dataset with time/dayofyear dims, as in test_partial_dependence).

## B. `scripts/explain_model.py` + `configs/explain.yaml` + CLAUDE.md line (Haiku)

```yaml
explain:
  n_rows: 4000             # bottles explained (random sample of prepare_data rows, seed 0)
  n_rows_interactions: 1000
  n_background_pd: 400
  n_grid: 20
  top_pairs: 4
  map_dates: ["2019-01-17", "2019-04-15", "2019-07-12", "2019-10-08"]
  map_stride: 4
  groups: {location: [ncoord_x, ncoord_y, ncoord_z]}
  log_features: [chl_globcolour]
  max_cpus: 126
  labels: {salinity: Salinity, temperature: Temperature, ssh_adt: Sea surface height,
           chl_globcolour: Chlorophyll, nitrate: Nitrate, phosphate: Phosphate,
           silicate: Silicate, bottomdepth: Bottom depth, location: Location}
```

CLI like `scripts/spatial_pdp.py` (`--config`, `--override`, `--run-dir`, `--output-dir`
default `<run>/explain/`, `--stage bottles|interactions|pdp|maps|all` default all, `--max-cpus`).
Use the run's `configs.json` + fingerprint check exactly as `spatial_pdp.py` does (copy the code).
Outputs in `<run>/explain/`:
- `shap_bottles.nc` (+ lat/lon/time of the sampled rows as coords), `shap_summary.{png,pdf}`:
  beeswarm per feature ordered by mean |SHAP|, points coloured by feature value percentile
  (RdYlBu_r), jittered vertically, plus mean |SHAP| text; x label "Contribution to Q50 (µmol kg⁻¹)".
- `shap_dependence.{png,pdf}`: one panel per feature, SHAP vs feature value, coloured by the
  feature's strongest interaction partner (from interactions stage if available, else none).
- `interactions.nc`, `interaction_strength.csv`, `interaction_heatmap.{png,pdf}`: matrix with main
  effects on the diagonal (separate colour scale or annotated), off-diagonal pair strengths, annotated values.
- `pdp2d.nc`, `pdp2d_<a>__<b>.{png,pdf}` per top pair: left = PD surface (contourf, viridis),
  right = pure interaction surface (RdBu_r, symmetric), background points as a light scatter, H² in title.
- `shap_maps.nc`, `shap_maps.{png,pdf}`: one panel per feature (incl. location), RdBu_r symmetric
  p98 limits, coastlines as in `spatial_pdp.py`.
- `summary.csv`: feature, mean |SHAP| on bottles, mean |SHAP| on maps.

## Run

```bash
uv run python scripts/explain_model.py --override configs/variants/stacked_salinity.yaml
```

Website integration (later, separate step): SHAP maps via the existing `sensitivity-maps` style.
