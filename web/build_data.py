"""Export a completed quantile ensemble run to web/data.json.

Run from the repository root:  uv run python web/build_data.py
Only reads existing outputs (no training). Re-run after retraining to refresh the page.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bottle_baseline import bottle_baselines

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import highres_ta as ta  # noqa: E402
from highres_ta.training import stack_salinity  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, help="Run directory; defaults to paths.output_dir.")
parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data.json", help="Destination JSON file.")
args = parser.parse_args()
config = ta.load_config()
RUN = ta.resolve_path(args.run or config["paths"]["output_dir"])
if not (RUN / "ensemble_test_metrics.json").is_file():
    parser.error(f"{RUN} has no assembled test metrics. Finish training and assembly first; data.json was not changed.")
snapshot = RUN / "configs.json"
legacy = RUN.name == "quantile_ensemble_split_hpo_1993-2022"
if snapshot.is_file():
    config = json.loads(snapshot.read_text())
elif legacy:
    # Explicit compatibility with the already-published historical run.
    config["data"]["feature_names"] = [n for n in config["data"]["feature_names"] if n != "chl_globcolour"]
    config["data"]["non_null_columns"] = ["talk", "salinity", "temperature", "nitrate", "ssh_adt"]
else:
    parser.error(f"Missing {snapshot}; export requires the configuration used for training.")
if config["model"]["quantiles"] != [0.1, 0.5, 0.9] or config["ensemble"]["aggregation"] != "mean":
    parser.error("The web explorer requires Q10/Q50/Q90 and mean aggregation.")
QS = [0.1, 0.5, 0.9]
N_SAMPLE = 700
rng = np.random.default_rng(0)

dc = config["data"]
non_null = dc["non_null_columns"]

# ---- preprocessing funnel -------------------------------------------------
raw = ta.load_data(str(ta.resolve_path(config["paths"]["training_data"])))
if legacy and "year" in raw:
    raw = raw[raw.year.between(1993, 2022)]
raw_feat = [n for n in dc["feature_names"] if n not in dc["engineered_feature_names"]]
salinity_sources = dc.get("salinity_source") or []
cols = list(dict.fromkeys(dc["coordinate_columns"] + raw_feat + ["talk"] + dc["quality_columns"] + list(salinity_sources)))
df = raw[cols].set_index(dc["coordinate_columns"], drop=False)
funnel = [("Collocated bottle samples", len(df), df.expocode.nunique())]
df = ta.drop_extreme_salinities(df, min=dc["minimum_salinity"], max=dc["maximum_salinity"])
funnel.append((f"Salinity within {dc['minimum_salinity']}-{dc['maximum_salinity']}", len(df), df.expocode.nunique()))
df = ta.add_talk_adjustment(df, fname=str(ta.resolve_path(config["paths"]["talk_adjustment_csv"])))
df = ta.drop_bad_quality_talk(df)
funnel.append(("Good TA flag and |adjustment| <= 6 umol/kg", len(df), df.expocode.nunique()))
df = df.dropna(subset=non_null)
funnel.append(("Required values present: " + ", ".join(non_null), len(df), df.expocode.nunique()))
df = df.drop_duplicates(subset=dc["coordinate_columns"], keep="first")
funnel.append(("Duplicate coordinates removed", len(df), df.expocode.nunique()))
meta = df.reset_index(drop=True)[["expocode", "time", "lat", "lon", "depth", "bottomdepth", "talk"]]  # bottomdepth: positive metres
print("final rows", len(df))

# ---- blocks / splits ----------------------------------------------------------
splits_df = pd.read_parquet(RUN / "splits" / "split_1.parquet")
assert len(splits_df) == len(meta), (len(splits_df), len(meta))
assert np.array_equal(splits_df.sort_values("row_id").expocode, meta.expocode), "Training cruise order does not match the saved splits."
blocks_info = []
test_ids = set(splits_df.query("split=='test'").row_id)
block_of = {}
cv = config["cross_validation"]
# recover block membership: block 0 = test; block k = validation set of split k
for k in range(1, cv["n_parts"]):
    s = pd.read_parquet(RUN / "splits" / f"split_{k}.parquet")
    for r in s.query("split=='validation'").row_id:
        block_of[r] = k
for r in test_ids:
    block_of[r] = 0
meta["block"] = meta.index.map(block_of)
assert meta.block.notna().all()
for b in range(cv["n_parts"]):
    m = meta[meta.block == b]
    blocks_info.append({"block": b, "observations": int(len(m)), "cruises": int(m.expocode.nunique())})

# ---- members ------------------------------------------------------------------
metrics = json.loads((RUN / "ensemble_test_metrics.json").read_text())
names = metrics["members"]
members = []
preds = []
for name in names:
    j = json.loads((RUN / "models" / f"quantile_{name}.json").read_text())
    sp, rep = int(name.split("_")[1]), int(name.split("_")[3])
    t = j["scores"]["test"]["all"]
    v = j["scores"]["validation"]["all"]
    members.append({
        "name": name, "split": sp, "replicate": rep, "seed": 100 * sp + rep,
        "params": {k: round(float(x), 4) if isinstance(x, (int, float)) else x
                   for k, x in j.get("params", j.get("best_params", {})).items()},
        "trial_number": j.get("trial_number"), "loss_rank": j.get("loss_rank"),
        "dropped_features": j.get("dropped_features", []),
        "in_rashomon_set": j.get("in_rashomon_set"),
        "iterations": j["refit_iterations"], "inner_cv_loss": round(j.get("inner_cv_loss", j["best_inner_cv_loss"]), 3),
        "n_train": j["observation_counts"]["train"], "n_validation": j["observation_counts"]["validation"],
        "cruises_train": j["cruise_counts"]["train"],
        "test": {"rmse": round(t["median_rmse"], 2), "mae": round(t["median_mae"], 2),
                 "crps": round(t["quantile_crps"], 3), "coverage80": round(t["interval_80pct_coverage"], 3)},
        "validation": {"rmse": round(v["median_rmse"], 2), "mae": round(v["median_mae"], 2)},
    })
    p = pd.read_parquet(RUN / "predictions" / f"{name}_test.parquet").sort_values("row_id")
    preds.append(p[[f"q_{q}" for q in QS]].to_numpy())
P = np.stack(preds)  # (members, obs, quantiles)
test = pd.read_parquet(RUN / "predictions" / "ensemble_test.parquet").sort_values("row_id")
rows = test.row_id.to_numpy()
y = test.observed.to_numpy()
assert np.allclose(meta.iloc[rows].talk.to_numpy(), y), "Training targets do not match the saved predictions."
ens = P.mean(0)  # mean aggregation like the saved ensemble
assert np.allclose(ens, test[[f"q_{q}" for q in QS]].to_numpy(), atol=1e-3)
sp_idx = np.array([m["split"] for m in members])

# ---- uncertainty diagnostics on the full test set -------------------------------------
med = P[:, :, 1]
ens_med = med.mean(0)
err = y - ens_med
member_std = med.std(0)                                   # spread of member medians
iq = (ens[:, 2] - ens[:, 0]) / 2.563                      # scaled Q90-Q10
total = np.sqrt(member_std**2 + iq**2)
def binned(u, nb=8):
    edges = np.quantile(u, np.linspace(0, 1, nb + 1)); out = []
    for i in range(nb):
        m = (u >= edges[i]) & (u <= edges[i + 1]) if i == nb - 1 else (u >= edges[i]) & (u < edges[i + 1])
        out.append({"u": round(float(np.sqrt(np.mean(u[m] ** 2))), 2), "rmse": round(float(np.sqrt(np.mean(err[m] ** 2))), 2), "n": int(m.sum())})
    return out
spread = {"member_std": binned(member_std), "scaled_iqr": binned(iq), "combined": binned(total),
          "corr_abs_err": {"member_std": round(float(np.corrcoef(member_std, np.abs(err))[0, 1]), 3),
                           "scaled_iqr": round(float(np.corrcoef(iq, np.abs(err))[0, 1]), 3)}}

# error / coverage as ensemble grows (random subsets, 30 draws)
grow = []
for n in sorted({n for n in [1, 2, 3, 5, 7, 10, 14, 21, 28, 42, len(members)] if n <= len(members)}):
    r, a, c, w = [], [], [], []
    for _ in range(30):
        sel = rng.choice(len(members), n, replace=False)
        e = P[sel].mean(0)
        r.append(np.sqrt(np.mean((y - e[:, 1]) ** 2)))
        a.append(np.mean(np.abs(y - e[:, 1])))
        c.append(np.mean((y >= e[:, 0]) & (y <= e[:, 2])))
        w.append(np.mean(e[:, 2] - e[:, 0]))
    grow.append({"n": n, "rmse": round(float(np.mean(r)), 2), "mae": round(float(np.mean(a)), 2), "coverage80": round(float(np.mean(c)), 3), "width80": round(float(np.mean(w)), 2)})

# scores on the full test set; open ocean = bottom depth > 500 m and |lat| < 60
def scores(mask):
    e = ens[mask, 1] - y[mask]
    return {"n": int(mask.sum()), "mae": round(float(np.abs(e).mean()), 2), "rmse": round(float(np.sqrt((e**2).mean())), 2),
            "bias": round(float(e.mean()), 2), "r2": round(float(1 - (e**2).sum() / ((y[mask] - y[mask].mean()) ** 2).sum()), 3),
            "coverage80": round(float(np.mean((y[mask] >= ens[mask, 0]) & (y[mask] <= ens[mask, 2]))), 3)}
test_meta = meta.iloc[rows]
open_ocean = ((test_meta.bottomdepth > 500) & (test_meta.lat.abs() < 60)).to_numpy()
regional_scores = {"all": scores(np.ones(len(y), dtype=bool)), "open_ocean": scores(open_ocean)}

calib = {"quantile": QS, "empirical": [round(float(np.mean(y <= ens[:, i])), 3) for i in range(3)]}

# ---- sample of test observations with all member predictions --------------------------
m_test = meta.iloc[rows].reset_index(drop=True)
pick = np.sort(rng.choice(len(rows), min(N_SAMPLE, len(rows)), replace=False))
if salinity_sources:
    df = stack_salinity(df, salinity_sources, dc["minimum_salinity"], dc["maximum_salinity"])
sample_rows = rows[pick]
baseline = None
if config["model"]["linear_features"] == ["salinity"] and config["model"]["polynomial_degree"] == 1:
    baseline = bottle_baselines(df, meta.block.to_numpy(), sp_idx, sample_rows)
sample = []
for position, i in enumerate(pick):
    sample.append({
        "id": int(rows[i]), "cruise": m_test.expocode[i],
        "date": str(m_test.time[i])[:10], "lat": round(float(m_test.lat[i]), 2), "lon": round(float(m_test.lon[i]), 2),
        "depth": round(float(m_test.depth[i]), 0), "observed": round(float(y[i]), 1),
        "pred": np.round(P[:, i, :], 1).tolist(),  # [member][quantile]
    })
    if baseline is not None:
        sample[-1].update(salinity=round(float(df.salinity.iloc[rows[i]]), 4), baseline=round(float(baseline[position]), 2))

# distinct sample locations with their block, for the train/test map (block 0 = test)
loc = meta.assign(lat=meta.lat.round(1), lon=(((meta.lon + 180) % 360) - 180).round(1))[["lat", "lon", "block"]].drop_duplicates()
split_map = {"lat": loc.lat.tolist(), "lon": loc.lon.tolist(), "block": loc.block.astype(int).tolist()}

out = {
    "run": {"name": RUN.name, "n_members": len(members), "n_splits": int(sp_idx.max()), "n_replicates": max(m["replicate"] for m in members) + 1,
            "method": "independent_hpo" if legacy else "rashomon",
            "n_total": int(len(meta)), "n_cruises": int(meta.expocode.nunique()), "test_observations": int(len(rows)),
            "years": [int(pd.to_datetime(meta.time).dt.year.min()), int(pd.to_datetime(meta.time).dt.year.max())], "sample_size": len(pick)},
    "funnel": [{"step": s, "rows": int(n), "cruises": int(c)} for s, n, c in funnel],
    "blocks": blocks_info,
    "members": members,
    "ensemble_scores": metrics["scores"],
    "regional_scores": regional_scores,
    "variance_means": metrics["variance_means"],
    "spread_vs_error": spread, "growth": grow, "calibration": calib,
    "test_sample": sample, "member_order": names,
    "split_map": split_map,
}
args.output.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", args.output, round(args.output.stat().st_size / 1e6, 2), "MB")
print(json.dumps(out["funnel"], indent=1)); print(blocks_info); print(spread["corr_abs_err"]); print(grow)
