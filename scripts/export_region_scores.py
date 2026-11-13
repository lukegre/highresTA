"""Export exact held-out scores for every combination of the Evaluation page's region pills.

Reads ``<run>/validation/predictions.parquet`` (written by ``scripts/validate_model.py``), pools the
test block and the out-of-fold validation rows (each observation appears once) and scores every
combination of depth (any / deep / coastal), latitude (any / high / mid and low) and Mediterranean
(any / only / excluded). Writes ``web/evaluation-regions.json`` keyed like the page's selections:
parts in the order depth, latitude, Mediterranean joined by ",", ``all`` when nothing restricts, e.g.
``deep,lowlat,nomed``, ``coastal,med``, ``highlat``.

    uv run python scripts/export_region_scores.py --run models/highresTA-stacked-salinity-20261109
"""

import json
from itertools import product
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from highres_ta.training import resolve_path

app = typer.Typer()
WEB = Path(__file__).resolve().parent.parent / "web"
# Mediterranean Sea, excluding the Black Sea. The west edge (-5.5°E, i.e. east of the Strait of Gibraltar) reproduces the
# basin table of comparison-data.json exactly: n = 1,206, MAE 52.52 on the "salinity" product.
MED_BOX = (30.0, 46.0, -5.5, 36.0)  # lat_min, lat_max, lon_min, lon_max


def pill_masks(pred: pd.DataFrame, coastal_depth: float, med_box: tuple[float, float, float, float]) -> dict:
    """Boolean masks per pill value. Same rules as ``validation._region_masks``, which defines the
    existing region scores (missing depth or latitude is never classified)."""
    depth_known, lat_known = pred.bottomdepth.notna(), pred.lat.notna()
    coastal = pred.bottomdepth.abs() <= coastal_depth
    high = pred.lat.abs() >= 60
    lat_min, lat_max, lon_min, lon_max = med_box
    lon = ((pred.lon + 180) % 360) - 180
    med = pred.lat.between(lat_min, lat_max) & lon.between(lon_min, lon_max)
    # The Black Sea (east of 27.5°E, north of 40.5°N) is not part of the Mediterranean basin.
    med &= ~((lon > 27.5) & (pred.lat > 40.5))
    everything = pd.Series(True, index=pred.index)
    return {
        "depth": {None: everything, "deep": depth_known & ~coastal, "coastal": coastal},
        "lat": {None: everything, "highlat": high, "lowlat": lat_known & ~high},
        "med": {None: everything, "med": med, "nomed": ~med},
    }


def score(part: pd.DataFrame) -> dict | None:
    """MAE, RMSE, bias, median bias and q10-q90 coverage / width; residual is predicted minus observed."""
    part = part.dropna(subset=["observed", "predicted"])
    if part.empty:
        return None
    residual = part.predicted - part.observed
    out = {"n": int(len(part)), "mae": float(residual.abs().mean()), "rmse": float(np.sqrt((residual**2).mean())),
           "bias": float(residual.mean()), "medbias": float(residual.median())}
    if {"q_0.1", "q_0.9"} <= set(part.columns):
        out["cov80"] = float(((part.observed >= part["q_0.1"]) & (part.observed <= part["q_0.9"])).mean())
        out["width"] = float((part["q_0.9"] - part["q_0.1"]).mean())
    return out


def selection_scores(pred: pd.DataFrame, coastal_depth: float = 500, med_box: tuple = MED_BOX) -> dict:
    """Scores for all 3 x 3 x 3 pill combinations that contain at least one row, keyed by selection."""
    masks = pill_masks(pred, coastal_depth, med_box)
    out = {}
    for depth, lat, med in product(masks["depth"], masks["lat"], masks["med"]):
        key = ",".join(p for p in (depth, lat, med) if p) or "all"
        mask = masks["depth"][depth] & masks["lat"][lat] & masks["med"][med]
        result = score(pred.loc[mask])
        if result:
            out[key] = result
    return out


@app.command(help=__doc__)
def main(
    run: Annotated[Path, typer.Option(help="Model run folder (the one with validation/predictions.parquet).")],
    salinity_product: Annotated[str, typer.Option(help="Product in predictions.parquet; 'stacked' is what the page's region scores (valscores) use; 'salinity' is the in situ salinity run (basin table).")] = "stacked",
    coastal_depth: Annotated[float, typer.Option(help="Bottom depth (m) separating coastal from deep ocean.")] = 500,
    med_box: Annotated[str, typer.Option(help="Mediterranean box 'lat_min,lat_max,lon_min,lon_max'.")] = ",".join(map(str, MED_BOX)),
    out: Annotated[Path, typer.Option(help="Output JSON.")] = WEB / "evaluation-regions.json",
) -> None:
    run = resolve_path(run)
    pred = pd.read_parquet(run / "validation" / "predictions.parquet")
    pred = pred.loc[(pred.salinity_product == salinity_product) & pred.dataset.isin(["test", "validation"])]
    if pred.empty:
        raise ValueError(f"No rows for salinity_product={salinity_product!r}; found {sorted(pd.read_parquet(run / 'validation' / 'predictions.parquet').salinity_product.unique())}.")
    if pred.row_id.duplicated().any():
        raise ValueError("A row appears in both the test block and a validation split; pooling would double count it.")
    box = tuple(float(x) for x in med_box.split(","))
    scores = selection_scores(pred, coastal_depth, box)
    payload = {"run": run.name, "salinity_product": salinity_product, "coastal_depth": coastal_depth, "med_box": box,
               "rows": "test block + out-of-fold validation, pooled", "residual": "predicted minus observed (µmol kg⁻¹)",
               "scores": scores}
    out.write_text(json.dumps(payload, indent=1) + "\n")
    typer.echo(f"Wrote {out}: {len(scores)} selections; all n={scores['all']['n']}, Mediterranean n={scores.get('med', {}).get('n')}")


if __name__ == "__main__":
    app()
