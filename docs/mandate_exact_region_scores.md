# Mandate: exact region scores for the Evaluation page

## Why

`web/evaluation.html` (logic in `web/evaluation.js`) shows MAE / RMSE / bias of the gridded-salinity model
(model `E`, run `models/highresTA-stacked-salinity-20261109`) per region, with pills for Deep ocean, Coastal ocean,
High lat, Mid and low lat, Mediterranean and Excl. Mediterranean. The page reads pooled scores from
`web/comparison-data.json`. Those exist only for depth x latitude cells and for the Mediterranean as a whole, so
"Excl. Mediterranean" combined with a depth or latitude pill is **estimated** (`keyScores` in `evaluation.js`: the
Mediterranean's overall scores, scaled by its share of a 6,000-point map sample, are subtracted) and Mediterranean
cannot be combined with other pills. Replace the estimate with exact scores computed from per-sample predictions.

## You need

The model run folder `models/highresTA-stacked-salinity-20261109/` (gitignored, **not** on the machine where the
page was built). Work on the machine that has it. Do not retrain anything.

## Steps

1. If `<run>/validation/predictions.parquet` is missing, create it:
   `uv run python scripts/validate_model.py --override configs/variants/stacked_salinity.yaml`
   (it refuses to run if the prepared data fingerprint differs from the training run; do not bypass that).
2. Run the export (already written and unit-tested on synthetic data, `tests/test_export_region_scores.py`):
   `uv run python scripts/export_region_scores.py --run models/highresTA-stacked-salinity-20261109`
   It writes `web/evaluation-regions.json`: `scores[key] = {n, mae, rmse, bias, medbias, cov80, width}` for every
   combination of depth (any/deep/coastal) x latitude (any/highlat/lowlat) x Mediterranean (any/med/nomed), keyed
   `depth,lat,med` with omitted parts dropped and `all` for no restriction (`deep,lowlat,nomed`, `coastal,med`, `highlat`).
3. **Reconcile before wiring it in** (report any mismatch, do not hide it):
   - `all` n should be about 30,123 (test 4,314 + out-of-fold 25,809 in `valscores.E`, product `salinity`).
   - `deep` (n 24,500, MAE 15.1), `coastal` (5,623, 42.0), `highlat` (6,611, 37.5) should match the page's current
     values. The existing region masks in `src/highres_ta/validation.py::_region_masks` use `|lat| >= 60` for high
     latitude; the page text says "> 60°". Pick one, make the code and page text agree.
   - The Mediterranean mask is a lat/lon box (`MED_BOX` in the script, Black Sea excluded). The page's basin table
     (`comparison-data.json`, `heldout.all.basin.E.Mediterranean`) has n = 1,206 and MAE 52.5, RMSE 99.9, bias -32.1.
     Find how that basin was defined (not in this repository; the builder of `comparison-data.json` lives elsewhere, ask
     the user if it cannot be found) and make `MED_BOX` / `pill_masks` reproduce n = 1,206 and those scores. CARIMED has 1,205 rows.
4. Update the page to use the export:
   - `web/evaluation.js`: load `evaluation-regions.json` for the Validation regions grouping, drop `keyScores`' estimate,
     the "(est.)" suffix and the `MED_PTS` machinery, and let the Mediterranean pill combine with the other pills
     (stop disabling the others; Med and Excl. Med stay mutually exclusive). Selection key must match the export's key
     (parts in the order depth, latitude, med/nomed). Keep the four default table rows and the extra row for any other
     selection. Other groupings (basin, salinity, 5-year) are unchanged.
   - `web/content.json` (`evaluation.group_note.valregion`, `evaluation.footnote`): remove the estimate wording, and say
     all combinations are exact.
   - `web/README.md`: add `evaluation-regions.json` and how to regenerate it.
5. Check in the browser (`preview_start` with `highresta-web`, open `/evaluation.html`): toggle every pill combination,
   confirm the table row and map highlight follow, no console errors, row counts of disjoint pill sets add up
   (deep + coastal = all rows with known depth, med + nomed = all).

## Constraints

- Keep it simple and readable; no new dependencies. Use `.venv/bin/python` if `uv` cannot write its cache (sandbox).
- Do not commit; leave that to the user. Do not hand-edit `comparison-data.json`.
- Report: the reconciliation results from step 3 (numbers, not "matches"), what changed, and anything still estimated.
