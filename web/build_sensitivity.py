"""Export a run's spatial sensitivity maps to web/sensitivity-data.json for the method page.

Reads <run>/partial_dependence/sensitivity.nc (written by scripts/spatial_pdp.py) and packs it:
one shared ocean mask (bits, row-major from the north-west corner), then per feature over the
ocean cells only
  s: int16 little-endian, sensitivity x 100 (µmol kg⁻¹ per +δ); -32768 = no value
  q: uint8, round(sign_agreement x 50) x 5 + round(out_of_range x 4); 255 = no value
all base64-encoded.

    uv run python web/build_sensitivity.py --run models/highresTA-stacked-salinity-20261109
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import xarray as xr

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
SCALE = 100
NO_S, NO_Q = -32768, 255


def b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Run folder with partial_dependence/.")
    parser.add_argument("--out", type=Path, default=WEB / "sensitivity-data.json")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run

    ds = xr.open_dataset(run / "partial_dependence" / "sensitivity.nc").load()
    ds = ds.sortby("lat", ascending=False).sortby("lon")
    lat, lon = ds.lat.values, ds.lon.values
    sens = ds.sensitivity.transpose("feature", "lat", "lon").values
    agree = ds.sign_agreement.transpose("feature", "lat", "lon").values
    oor = ds.out_of_range.transpose("feature", "lat", "lon").values

    ocean = np.isfinite(sens).any(axis=0)
    flat = ocean.ravel()
    if np.nanmax(np.abs(sens)) * SCALE >= 32767:
        raise ValueError("Sensitivity exceeds the int16 range; lower SCALE.")

    lat_w = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(lon)[None, :]
    features = []
    for i, name in enumerate(ds.feature.values):
        s = sens[i].ravel()[flat]
        a = agree[i].ravel()[flat]
        o = oor[i].ravel()[flat]
        s_enc = np.where(np.isfinite(s), np.round(s * SCALE), NO_S).astype("<i2")
        has = np.isfinite(s)
        q = np.round(np.nan_to_num(a, nan=0) * 50) * 5 + np.round(np.nan_to_num(o, nan=0) * 4)
        q_enc = np.where(has & np.isfinite(a), q, NO_Q).astype(np.uint8)
        valid = np.isfinite(sens[i])
        w = np.where(valid, lat_w, 0)
        finite = np.abs(s[np.isfinite(s)])
        features.append({
            "id": str(name),
            "delta": float(ds.delta.values[i]),
            "n_members": int(ds.n_members.values[i]),
            "mean_abs": float(np.sum(np.abs(np.nan_to_num(sens[i])) * w) / w.sum()),
            "median": float(np.nanmedian(s)),
            "limit": float(np.percentile(finite, 98)) if finite.size else 1.0,
            "s": b64(s_enc),
            "q": b64(q_enc),
        })
    features.sort(key=lambda f: -f["mean_abs"])

    out = {
        "run": run.name,
        "dates": json.loads(ds.attrs.get("dates", "[]")),
        "delta_scale": float(ds.attrs.get("delta_scale", np.nan)),
        "n_members_total": int(ds.n_members.max()),
        "grid": {
            "nlat": len(lat), "nlon": len(lon),
            "lat0": float(lat[0]), "dlat": float(lat[1] - lat[0]),
            "lon0": float(lon[0]), "dlon": float(lon[1] - lon[0]),
        },
        "scale": SCALE,
        "mask": b64(np.packbits(flat.astype(np.uint8), bitorder="little")),
        "features": features,
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {flat.sum():,} ocean cells)")


if __name__ == "__main__":
    main()
