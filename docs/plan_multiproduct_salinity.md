# Plan: multi-product salinity variant (gridded maps)

Goal: test whether giving each gridded salinity product its own feature column beats the single stacked
salinity, for the gridded TA maps. Point and station use with measured salinity stays with the default in-situ model.

Design: `stacked_salinity` + the individual products as extra CatBoost features.
- `salinity` = stacked anchor (sss_cci → salt_soda → sss_multiobs, falling back to in situ), as in `stacked_salinity`.
  It is used by the linear baseline and is protected.
- `sss_cci`, `salt_soda` = individual products. NaN where missing (CatBoost handles NaN), and droppable during tuning.
- `sss_multiobs` is **not** a feature, only part of the anchor stack: the store's `sss_multi_nrt` only has data from about 2020–2021
  (checked mid-June 1995/2005/2012/2019: empty; 2021/2024: filled), whereas `sss_multiobs` is filled in 90 % of training rows.
  As a feature that would be a train/inference mismatch. Revisit if the reprocessed product is added to the store.
- Product mapping, confirmed: `salt_soda` ≡ zarr `sss_soda`, `sss_multiobs` ≡ zarr `sss_multi_nrt`.

## Why this needs no code changes (checked)

- `prepare_data` loads `feature_names` + `salinity_source` columns (`training.py:122-126`). `stack_salinity`
  (`training.py:86-108`) only overwrites `salinity` and adds `salinity_insitu`. It leaves the product columns in place,
  so they reach the output when they are listed in `feature_names` (`:127`, `:160`).
- NaN in the product columns is allowed. Only `non_null_columns` (talk, salinity, temperature, nitrate) are enforced.
- Rows, blocks and the test block are identical to the in-situ and `stacked_salinity` models: same rows, and
  stratification on `salinity_insitu` (`train_test_split.py:197-198`).
- Inference: `feature_sources` is a generic name → [zarr vars] mapping (`inference.py:79-87`), and dicts deep-merge in overlays,
  so the variant only adds three keys.

## Variant `configs/variants/multiproduct_salinity.yaml`

```yaml
# Stacked salinity anchor + each gridded salinity product as its own (often missing) feature.
#   scripts/train_splits.sh --override configs/variants/multiproduct_salinity.yaml
paths:
  output_dir: models/highresTA-multiproduct-salinity-20261110/
data:
  salinity_source: [sss_cci, salt_soda, sss_multiobs]
  feature_names:   # lists replace, so give the full list
  - salinity
  - temperature
  - ssh_adt
  - chl_globcolour
  - nitrate
  - phosphate
  - silicate
  - bottomdepth
  - ncoord_x
  - ncoord_y
  - ncoord_z
  - sss_cci
  - salt_soda
inference:
  model_path: null
  output_zarr: null
  feature_sources:
    sss_cci: [sss_cci]
    salt_soda: [sss_soda]
```

Effect on tuning: there are 11 droppable features instead of 9, so `dropped_feature_options` grows from 46 to 67 (`max_dropped_features: 2`).
That is acceptable. Rashomon members that drop the products show directly whether they matter.

## Status (2026-11-10)

Variant created. Dry-run: 30,193 rows, 13 features, split sizes identical to `stacked_salinity`. The smoke test on split 1 (with `sss_multiobs`
still as a feature, in a scratch output dir) trained fine. A member predicted a full grid for 2015-06-15. Full training not started yet.

Also noted: `sss_cci` and `sss_soda` are empty in the store in 2024, so the anchor in recent years comes only from `sss_multi_nrt`
(this also affects `stacked_salinity`). Product columns contain values down to about 12 (not bounds-checked).

## Steps

1. Write the variant above.
2. `uv run python scripts/train_quantile_ensemble.py --dry-run --override configs/variants/multiproduct_salinity.yaml`
   - check: row count and test-block cruises equal to `stacked_salinity`; product columns present; NaN fractions roughly
     sss_cci 33 %, salt_soda 99 % non-null (after data prep).
3. Smoke test: `--stage train --split 1 --n-trials 8 --maximum-iterations 20 --parallel-trials 1 --threads-per-trial 2` with the override.
4. Check that inference produces a single date (`scripts/inference.py --override … --start-date … --end-date …`), confirming the product columns
   are loaded and predict raises no "Missing feature columns" error.
5. Full training: `scripts/train_splits.sh --override configs/variants/multiproduct_salinity.yaml`.
6. Evaluation, compared against `stacked_salinity` on the same test block (block 0):
   - overall RMSE/bias and q10–q90 coverage;
   - split by sss_cci available or not (≈ after/before 2010) and by number of products available;
   - stations: `evaluate_stations.py --salinity gridded` only. Its `insitu` mode swaps only `salinity` and would mix in-situ salinity with gridded product columns.
   - `validate_model.py` per-product swaps: they replace only `salinity` and leave the product columns untouched, so interpret them with care,
     or switch them off in the variant (`validation.salinity_products: [salinity]`, `validation.salinity_stack: null`; check the key semantics first).

## Known caveats (not fixed in v1)

- The individual product columns are not bounds-checked as features (only the stacked anchor is, in training). Inference has no
  bounds check at all, which is an existing difference that also affects `stacked_salinity`. It only matters for values outside 20–40.
- Difference features (e.g. `sss_cci − salt_soda`) would need code in both training and inference. Consider them for v2 if the products
  turn out to help.
- Not covered: whether the extra features change the uncertainty split between data splits and HPO. That shows up in validation.
