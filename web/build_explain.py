"""Export a run's SHAP contributions and interactions to web/explain-data.json for the method page.

Reads <run>/explain/ (written by scripts/explain_model.py) and writes one compact JSON file:

  bottles       a random subset of the explained bottles: per feature (ordered by mean |SHAP| over all
                explained bottles) SHAP rounded to 0.1, the feature-value percentile (0-99, null for
                groups and missing values) and the raw value (4 significant digits, null if missing)
  maps          shared ocean mask (bits, row-major from the north-west corner) and per feature the SHAP
                over ocean cells only: int16 x 10 (µmol kg⁻¹), -32768 = no value, delta-coded along the
                cells, split into a low-byte and a high-byte plane, zlib-compressed, base64-encoded
                (the page inflates it with DecompressionStream)
  interactions  feature order, strength matrix (mean |Φab + Φba|, null on the diagonal) and main effects
  pdp2d         one entry per feature pair: grids, joint partial dependence, pure-interaction surface, H²

    uv run python web/build_explain.py --run models/highresTA-stacked-salinity-20261109
"""

import argparse
import base64
import json
import zlib
from pathlib import Path

import numpy as np
import xarray as xr

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
SCALE = 10
NO_S = -32768
N_BOTTLES = 1500


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def pack_map(values: np.ndarray) -> str:
    """int16 values -> delta-coded, byte-shuffled, zlib, base64."""
    delta = np.diff(values.astype(np.int32), prepend=0).astype("<i2")
    raw = delta.view(np.uint8).reshape(-1, 2)
    shuffled = np.concatenate([raw[:, 0], raw[:, 1]])
    return base64.b64encode(zlib.compress(shuffled.tobytes(), 9)).decode("ascii")


def sig(x: float, digits: int = 4):
    return None if not np.isfinite(x) else float(f"{x:.{digits}g}")


def round_list(a: np.ndarray, d: int) -> list:
    return np.round(np.asarray(a, dtype=float), d).tolist()


def export_bottles(ds: xr.Dataset, rng: np.random.Generator) -> dict:
    shap = ds.shap.transpose("row", "feature").values
    value = ds.value.transpose("row", "feature").values
    mean_abs = np.abs(shap).mean(axis=0)
    order = np.argsort(-mean_abs)
    n = shap.shape[0]
    rows = np.sort(rng.choice(n, size=min(N_BOTTLES, n), replace=False))
    features = []
    for j in order:
        v = value[:, j]
        finite = np.isfinite(v)
        pct = np.full(n, np.nan)
        if finite.any():
            ranks = np.argsort(np.argsort(v[finite]))
            pct[finite] = np.minimum(99, np.floor(ranks / finite.sum() * 100))
        features.append({
            "id": str(ds.feature.values[j]),
            "mean_abs": round(float(mean_abs[j]), 2),
            "n_missing": int((~finite).sum()) if finite.any() else None,
            "shap": [round(float(x), 1) for x in shap[rows, j]],
            "pct": [None if np.isnan(x) else int(x) for x in pct[rows]],
            "value": [sig(x) for x in v[rows]],
        })
    times = ds.time.values
    return {
        "n_total": int(n),
        "n_shown": int(len(rows)),
        "base": round(float(ds.base), 1),
        "years": [int(str(times.min())[:4]), int(str(times.max())[:4])],
        "features": features,
    }


def export_maps(ds: xr.Dataset) -> dict:
    ds = ds.sortby("lat", ascending=False).sortby("lon")
    lat, lon = ds.lat.values, ds.lon.values
    shap = ds.shap.transpose("feature", "lat", "lon").values
    ocean = np.isfinite(shap).any(axis=0)
    flat = ocean.ravel()
    if np.nanmax(np.abs(shap)) * SCALE >= 32767:
        raise ValueError("SHAP exceeds the int16 range; lower SCALE.")
    lat_w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    features = []
    for i, name in enumerate(ds.feature.values):
        s = shap[i].ravel()[flat]
        enc = np.where(np.isfinite(s), np.round(s * SCALE), NO_S).astype("<i2")
        w = np.where(np.isfinite(shap[i]), lat_w, 0)
        features.append({
            "id": str(name),
            "mean_abs": float(np.sum(np.abs(np.nan_to_num(shap[i])) * w) / w.sum()),
            "limit": float(np.percentile(np.abs(s[np.isfinite(s)]), 98)),
            "s": pack_map(enc),
        })
    features.sort(key=lambda f: -f["mean_abs"])
    return {
        "dates": json.loads(ds.attrs.get("dates", "[]")),
        "base": round(float(ds.attrs.get("base", np.nan)), 1),
        "grid": {
            "nlat": len(lat), "nlon": len(lon),
            "lat0": float(lat[0]), "dlat": float(lat[1] - lat[0]),
            "lon0": float(lon[0]), "dlon": float(lon[1] - lon[0]),
        },
        "scale": SCALE,
        "n_cells": int(flat.sum()),
        "mask": b64(np.packbits(flat.astype(np.uint8), bitorder="little")),
        "features": features,
    }


def export_interactions(ds: xr.Dataset, order: list[str]) -> dict:
    strength = ds.strength.sel(feature=order, feature2=order).values
    return {
        "features": order,
        "strength": [[None if np.isnan(x) else round(float(x), 2) for x in row] for row in strength],
        "main": round_list(ds.main.sel(feature=order).values, 2),
        "n_rows": int(ds.sizes["row"]),
    }


def export_pdp(path: Path) -> list:
    pairs = json.loads(xr.open_dataset(path).attrs["pairs"])
    out = []
    for a, b in pairs:
        g = xr.open_dataset(path, group=f"{a}__{b}").load()
        out.append({
            "a": a, "b": b,
            "h2": round(float(g.attrs["h_squared"]), 6),
            "a_value": [sig(x, 5) for x in g.a_value.values],
            "b_value": [sig(x, 5) for x in g.b_value.values],
            "pd": round_list(g.pd.transpose("a_value", "b_value").values, 1),
            "interaction": round_list(g.interaction.transpose("a_value", "b_value").values, 2),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder with explain/.")
    parser.add_argument("--out", type=Path, default=WEB / "explain-data.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    src = run / "explain"

    bottles_ds = xr.open_dataset(src / "shap_bottles.nc").load()
    bottles = export_bottles(bottles_ds, np.random.default_rng(args.seed))
    order = [f["id"] for f in bottles["features"]]
    out = {
        "run": run.name,
        "quantile": float(bottles_ds.attrs.get("quantile", 0.5)),
        "n_members": int(bottles_ds.attrs.get("n_members", 0)),
        "bottles": bottles,
        "maps": export_maps(xr.open_dataset(src / "shap_maps.nc").load()),
        "interactions": export_interactions(xr.open_dataset(src / "interactions.nc").load(), order),
        "pdp2d": export_pdp(src / "pdp2d.nc"),
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    parts = {k: len(json.dumps(v, separators=(",", ":"))) for k, v in out.items() if isinstance(v, (dict, list))}
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)", {k: f"{v / 1e3:.0f} kB" for k, v in parts.items()})


if __name__ == "__main__":
    main()
