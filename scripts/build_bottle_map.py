#!/usr/bin/env python3
"""Draw GLODAP and CARIMED bottle locations as a bare plate carrée image for the website."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

import highres_ta as ta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from medsea_data import drop_glodap_cruises, load_carimed  # noqa: E402

OUT = ROOT / "web" / "maps" / "bottles.webp"
LAND_OUT = ROOT / "web" / "maps" / "land.png"
SIZE = (12.6, 6.02)  # inches at 100 dpi -> 1260 x 602 px
OCEAN, LAND, COAST = "#dcebe9", "#3f5457", "#3f5457"  # LAND matches --map-land in web/site.css
GLODAP_COLOUR, CARIMED_COLOUR = "#12635f", "#c8553d"


def wrap_lon(lon):
    return (np.asarray(lon) + 180) % 360 - 180


def save_land_mask(elevation):
    """Transparent plate carrée PNG (2:1) with only the land filled, to sit under SVG points."""
    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.contourf(elevation.lon, elevation.lat, (elevation >= 0).astype(float), levels=[0.5, 1.5], colors=LAND)
    fig.savefig(LAND_OUT, dpi=100, transparent=True)
    plt.close(fig)


def main():
    config = ta.load_config()
    paths = config["paths"]
    glodap = pd.read_parquet(ta.resolve_path(paths["glodap_training_data"]), columns=["expocode", "lat", "lon"])
    carimed = drop_glodap_cruises(load_carimed(paths["carimed_source"], ta.resolve_path(paths["raw_dir"])), glodap)

    # ETOPO elevation (positive up) is 0 on land and negative in the ocean
    elevation = xr.open_dataarray(ROOT / "data/bathymetry_etopo2022_25km.nc")
    coast = json.loads((ROOT / "src/highres_ta/assets/ne_110m_coastline.json").read_text())
    x = np.array([v if v is not None else np.nan for v in coast["lon"]])
    y = np.array([v if v is not None else np.nan for v in coast["lat"]])
    x[np.r_[False, np.abs(np.diff(x)) > 180]] = np.nan

    fig = plt.figure(figsize=SIZE, dpi=100, facecolor=OCEAN)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.contourf(elevation.lon, elevation.lat, (elevation >= 0).astype(float), levels=[0.5, 1.5], colors=LAND)
    ax.plot(x, y, color=COAST, lw=0.4)
    for df, colour in [(glodap, GLODAP_COLOUR), (carimed, CARIMED_COLOUR)]:
        ax.scatter(wrap_lon(df["lon"]), df["lat"], s=1.2, color=colour, linewidths=0, zorder=3)
    tmp = OUT.with_suffix(".png")
    fig.savefig(tmp, dpi=100, facecolor=OCEAN)
    plt.close(fig)
    with Image.open(tmp) as image:
        image.save(OUT, "WEBP", quality=88, method=6)
    tmp.unlink()

    save_land_mask(elevation)
    print(f"Wrote {LAND_OUT}")

    for name, df in [("GLODAP", glodap), ("CARIMED", carimed)]:
        n_loc = len(df[["lat", "lon"]].drop_duplicates())
        print(f"{name}: {len(df):,} records, {n_loc:,} distinct locations")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
