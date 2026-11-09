"""Surface time-series stations (BATS, HOT, SPOTS, Irminger, ESTOC) for independent TA evaluation."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import xarray as xr
from loguru import logger

from .quantiles import quantile_column, score_quantile_predictions

NUTRIENTS = ("nitrate", "phosphate", "silicate")
INSITU = ("salinity", "temperature", *NUTRIENTS)
BASE_COLUMNS = ["station", "time", "lat", "lon", "depth", "salinity_insitu", "temperature_insitu", "talk"]


def _fetch(url: str, cache_dir: str | Path | None, fname: str | None = None) -> str:
    return pooch.retrieve(url, known_hash=None, fname=fname or url.split("/")[-1], path=cache_dir)


def _standardise(df: pd.DataFrame, station: str | None) -> pd.DataFrame:
    """Common cleaning; ``station=None`` keeps an existing ``station`` column (SPOTS)."""
    df = df.copy()
    if station is not None:
        df["station"] = station
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    for column in ["lat", "lon", "depth", "talk", *INSITU]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["lon"] = (df["lon"] + 180) % 360 - 180
    df = df.dropna(subset=["time", "talk", "salinity", "temperature"])
    df = df.rename(columns={name: f"{name}_insitu" for name in INSITU})
    nutrients = [f"{name}_insitu" for name in NUTRIENTS if f"{name}_insitu" in df]
    df = df[BASE_COLUMNS + nutrients].reset_index(drop=True)
    logger.info("{}: {:,} samples", ", ".join(sorted(df["station"].unique())), len(df))
    return df


def load_bats(url: str, max_depth: float = 20.0, cache_dir: str | Path | None = None) -> pd.DataFrame:
    cols = {
        "ISO_DateTime_UTC": "time", "Latitude": "lat", "Longitude": "lon", "Depth": "depth",
        "Salinity": "salinity", "Temperature": "temperature", "Alkalinity": "talk",
        "PO4": "phosphate", "NO3_plus_NO2": "nitrate_nitrite", "NO2": "nitrite", "Silicate": "silicate",
    }
    df = pd.read_csv(_fetch(url, cache_dir))[list(cols)].rename(columns=cols)
    df["nitrate"] = df["nitrate_nitrite"] - df["nitrite"]
    df = df[df["depth"] <= max_depth].drop(columns=["nitrate_nitrite", "nitrite"])
    return _standardise(df, "BATS")


def load_hot(url: str, max_depth: float = 20.0, cache_dir: str | Path | None = None) -> pd.DataFrame:
    cols = {
        "Sampling_ISO_DateTime_UTC": "time", "Latitude": "lat", "Longitude": "lon", "CTDPRS": "depth",
        "CTDTMP": "temperature", "ALKALIN": "talk", "CTDSAL": "salinity_ctd", "SALNITY": "salinity_bottle",
    }
    df = pd.read_csv(_fetch(url, cache_dir))[list(cols)].rename(columns=cols)
    df["salinity"] = df["salinity_bottle"].combine_first(df["salinity_ctd"])
    # Station ALOHA only (22°45'N, 158°00'W); excludes other HOT stations
    df = df[df["lat"].between(22.25, 23.25) & df["lon"].between(-158.5, -157.5)]
    df = df[df["depth"] <= max_depth].drop(columns=["salinity_bottle", "salinity_ctd"])
    return _standardise(df, "HOT")


def load_spot(url: str, max_depth: float = 20.0, cache_dir: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(_fetch(url, cache_dir), na_values=[-999, -999.0])
    df["time"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")  # DATE is YYYYMMDD
    cols = {
        "TimeSeriesSite": "station", "time": "time", "LATITUDE": "lat", "LONGITUDE": "lon",
        "CTDPRS": "depth", "CTDTMP": "temperature", "ALKALI": "talk", "ALKALI_FLAG_W": "talk_flag",
        "SALNTY": "salinity_bottle", "SALNTY_FLAG_W": "salinity_bottle_flag",
        "CTDSAL": "salinity_ctd", "CTDSAL_FLAG_W": "salinity_ctd_flag",
        "NITRAT": "nitrate", "SILCAT": "silicate", "PHSPHT": "phosphate",
    }
    df = df[list(cols)].rename(columns=cols)
    # bottle salinity (flag 2) preferred, CTD salinity (flag 2) as fallback
    bottle = df["salinity_bottle"].where(df["salinity_bottle_flag"] == 2)
    ctd = df["salinity_ctd"].where(df["salinity_ctd_flag"] == 2)
    df["salinity"] = bottle.combine_first(ctd)
    df = df[(df["talk_flag"] == 2) & (df["depth"] <= max_depth)]
    return _standardise(df, None)


def load_irminger(url: str, max_depth: float = 30.0, cache_dir: str | Path | None = None) -> pd.DataFrame:
    cols = {
        "CTD_Bottle_Closure_Time": "time", "CTD_Latitude": "lat", "CTD_Longitude": "lon",
        "CTD_Depth": "depth", "CTD_Temperature_1": "temperature", "CTD_Salinity_1": "salinity_ctd",
        "Discrete_Salinity": "salinity_bottle", "Discrete_Alkalinity": "talk",
    }
    df = pd.read_csv(_fetch(url, cache_dir))[list(cols)].rename(columns=cols)
    df["salinity"] = df["salinity_bottle"].combine_first(df["salinity_ctd"])
    df = df[df["depth"] <= max_depth].drop(columns=["salinity_bottle", "salinity_ctd"])
    return _standardise(df, "Irminger")


def load_estoc(url: str, max_depth: float = 20.0, cache_dir: str | Path | None = None) -> pd.DataFrame:
    path = _fetch(url, cache_dir, fname="estoc_timeseries.txt")
    # PANGAEA text files wrap metadata in /* ... */; skip through the closing marker
    with open(path) as file:
        for n_skip, line in enumerate(file):
            if line.startswith("*/"):
                break
    cols = {
        "Date/Time": "time", "Latitude": "lat", "Longitude": "lon", "Depth water [m]": "depth",
        "Temp [°C]": "temperature", "Sal": "salinity", "AT [µmol/kg]": "talk",
    }
    df = pd.read_csv(path, sep="\t", skiprows=n_skip + 1)[list(cols)].rename(columns=cols)
    df = df[pd.to_numeric(df["depth"], errors="coerce") <= max_depth]
    return _standardise(df, "ESTOC")


STATION_LOADERS: dict[str, Callable[..., pd.DataFrame]] = {
    "bats": load_bats,
    "hot": load_hot,
    "spot": load_spot,
    "irminger": load_irminger,
    "estoc": load_estoc,
}


def average_casts(df: pd.DataFrame) -> pd.DataFrame:
    """Average the surface bottles of each cast (same station and time)."""
    return df.groupby(["station", "time"], as_index=False).mean(numeric_only=True)[list(df.columns)]


def collocate_stations(stations: pd.DataFrame, features: xr.Dataset) -> pd.DataFrame:
    """Append the nearest gridded features to each station row (pointwise, no interpolation)."""
    stations = stations.reset_index(drop=True)

    def obs(values):
        return xr.DataArray(np.asarray(values), dims="obs")

    grid_time = features.time.sel(time=obs(stations["time"]), method="nearest")
    indexers = {"time": grid_time, "lat": obs(stations["lat"]), "lon": obs(stations["lon"])}
    if "dayofyear" in features.dims:  # climatologies, as in inference.select_inference_date
        indexers["dayofyear"] = grid_time.dt.dayofyear
    selected = features.sel(indexers, method="nearest").load()
    values = selected.reset_coords(drop=True).to_dataframe().reset_index(drop=True)
    grid = pd.DataFrame({
        "grid_time": selected["time"].values, "grid_lat": selected["lat"].values,
        "grid_lon": selected["lon"].values,
    })
    return pd.concat([stations, values, grid], axis=1)


def predict_stations(model, df: pd.DataFrame, salinity: str = "gridded",
                     feature_names=None) -> pd.DataFrame:
    """Predict quantiles at collocated stations, using gridded or in-situ salinity/temperature."""
    if salinity not in ("gridded", "insitu"):
        raise ValueError("salinity must be 'gridded' or 'insitu'.")
    feature_names = list(feature_names or model.get_feature_names())
    df = df.copy()
    if salinity == "insitu":
        df["salinity"] = df["salinity_insitu"]
        df["temperature"] = df["temperature_insitu"]
    required = ["talk", *model.estimators[0].linear_features]
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df):
        prediction = model.predict_quantiles(df[feature_names])
    else:
        prediction = np.empty((0, len(model.quantiles)))
    out = df[["station", "time", "lat", "lon"]].copy()
    out["salinity_source"] = salinity
    out["observed"] = df["talk"].to_numpy()
    for index, alpha in enumerate(model.quantiles):
        out[f"q_{alpha:g}"] = prediction[:, index]
    out["predicted"] = prediction[:, quantile_column(model.quantiles, 0.5)]
    out["residual"] = out["predicted"] - out["observed"]
    return out


def _quantile_columns(predictions: pd.DataFrame) -> tuple[list[str], list[float]]:
    columns = [c for c in predictions.columns if c.startswith("q_")]
    return columns, [float(c[2:]) for c in columns]


def score_stations(predictions: pd.DataFrame, quantiles,
                   intervals: Mapping[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """One row of quantile scores per station and salinity source."""
    columns = [f"q_{alpha:g}" for alpha in quantiles]
    rows = []
    for (station, source), group in predictions.groupby(["station", "salinity_source"]):
        scores = score_quantile_predictions(group["observed"], group[columns], quantiles, intervals)
        rows.append({"station": station, "salinity_source": source, **scores})
    return pd.DataFrame(rows)


def plot_station(predictions: pd.DataFrame, title: str | None = None):
    """Observed vs predicted scatter (left) and time series with q10-q90 band (right)."""
    import matplotlib.pyplot as plt

    predictions = predictions.sort_values("time")
    columns, quantiles = _quantile_columns(predictions)
    scores = score_quantile_predictions(predictions["observed"], predictions[columns], quantiles)
    text = "\n".join(f"{name:<16}{scores[name]:8.2f}" for name in
                     ("median_rmse", "median_bias_mean", "median_r2", "count"))
    label = "TA (µmol kg$^{-1}$)"

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), width_ratios=[1, 2], constrained_layout=True)
    ax = axs[0]
    y, yhat = predictions["observed"], predictions["predicted"]
    ax.scatter(y, yhat, alpha=0.7)
    limits = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    ax.plot(limits, limits, "k--", label="1:1 line")
    ax.set(xlabel=f"Observed {label}", ylabel=f"Predicted {label}")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.02, text, transform=ax.transAxes, fontsize=6, va="bottom", ha="right",
            fontfamily="monospace", bbox={"edgecolor": "none", "facecolor": "white", "alpha": 0.8})

    ax = axs[1]
    if {"q_0.1", "q_0.9"} <= set(predictions.columns):
        ax.fill_between(predictions["time"], predictions["q_0.1"], predictions["q_0.9"],
                        alpha=0.3, label="q10-q90")
    ax.plot(predictions["time"], yhat, label="Predicted (q50)")
    ax.plot(predictions["time"], y, "o", ms=3, alpha=0.7, label="Observed")
    ax.set_ylabel(label)
    ax.legend()
    if title:
        fig.suptitle(title)
    return fig, axs
