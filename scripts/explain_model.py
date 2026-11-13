"""Explain the ensemble's median (Q50) TA with SHAP, pairwise interactions and 2-D partial dependence.

Additive per-predictor contributions (SHAP) for a random sample of bottles, pairwise interaction
strengths for a subsample, 2-D partial dependence for the strongest pairs, and SHAP maps on the
gridded inference domain. Definitions: docs/plan_shap_interactions.md. Settings: configs/explain.yaml.

Stages (each writes its own outputs and reads what it needs from earlier stages):
  bottles        shap_bottles.nc, shap_summary.{png,pdf}, summary.csv, shap_dependence.{png,pdf}
  interactions   interactions.nc, interaction_strength.csv, interaction_heatmap.{png,pdf},
                 shap_dependence.{png,pdf} (re-drawn with partner colours)
  pdp            pdp2d.nc (one group per pair), pdp2d_<a>__<b>.{png,pdf}
  maps           shap_maps.nc, shap_maps.{png,pdf}, summary.csv (adds the maps column)

Outputs go to <run>/explain/ by default.

    uv run python scripts/explain_model.py
    uv run python scripts/explain_model.py --override configs/variants/stacked_salinity.yaml --stage bottles
"""

import asyncio
import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xarray as xr
from loguru import logger
from matplotlib.colors import Normalize, to_rgba
from matplotlib.cm import ScalarMappable

import highres_ta as ta
from highres_ta.explain import (
    ensemble_shap,
    ensemble_shap_interactions,
    partial_dependence_2d,
    shap_maps,
    top_pairs,
)
from highres_ta.training import CONFIG_DIR, data_fingerprint, resolve_path

N_ATTEMPTS = 3  # per date, for transient HTTP errors from the remote predictor store
QUANTILE = 0.5
ROW_COORDS = ("expocode", "time", "lat", "lon", "depth")
SHAP_LABEL = "Contribution to Q50 (µmol kg⁻¹)"
GREY = "#9a9a9a"

app = typer.Typer()


class Stage(str, Enum):
    bottles = "bottles"
    interactions = "interactions"
    pdp = "pdp"
    maps = "maps"
    all = "all"


@dataclass(frozen=True)
class Settings:
    output_dir: Path
    labels: dict[str, str]
    units: dict[str, str]
    log_features: list[str]
    groups: dict[str, list[str]]


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _name(feature: str, s: Settings) -> str:
    """Label with unit, e.g. 'Salinity' or 'Chlorophyll (log10 mg m⁻³)'."""
    label = s.labels.get(feature, feature)
    unit = s.units.get(feature, "")
    return f"{label} ({unit})" if unit else label


def _axis_values(feature: str, values: np.ndarray, s: Settings) -> np.ndarray:
    """Plot log features on log10 axes, matching the units in configs (e.g. 'log10 mg m⁻³')."""
    values = np.asarray(values, dtype=float)
    if feature in s.log_features:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(values)
    return values


def _netcdf_ready(ds: xr.Dataset) -> xr.Dataset:
    """netCDF attributes must be scalars; serialise lists, dicts and None as JSON."""
    def clean(attrs: dict) -> dict:
        return {k: v if isinstance(v, (str, int, float)) else json.dumps(v, default=str)
                for k, v in attrs.items()}
    out = ds.copy()
    out.attrs = clean(ds.attrs)
    for var in out.variables.values():
        var.attrs = clean(var.attrs)
    return out


def _attach_rows(ds: xr.Dataset, X: pd.DataFrame) -> xr.Dataset:
    """Replace the row index (a MultiIndex of coordinates) by plain integer rows plus coordinate variables."""
    ds = ds.drop_vars([name for name in ("row",) if name in ds.variables])
    ds = ds.assign_coords(row=np.arange(ds.sizes["row"]))
    coords = {
        name: ("row", X.index.get_level_values(name).to_numpy())
        for name in ROW_COORDS if name in X.index.names
    }
    return ds.assign_coords(coords)


def _robust_limit(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return fallback
    limit = float(np.percentile(finite, 98))
    return limit if limit > 0 else fallback


def _panel_grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.2 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def _map_axes(ax: Any, coast: dict, color: str) -> None:
    """Bundled public-domain Natural Earth coastlines keep the figure offline."""
    ax.plot(np.asarray(coast["lon"], dtype=float), np.asarray(coast["lat"], dtype=float),
            color=color, linewidth=.4, zorder=3)
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _subset(frame: pd.DataFrame, order: np.ndarray, n: int) -> pd.DataFrame:
    """First n rows of a fixed random permutation, so that every stage uses nested samples."""
    return frame.iloc[order[: min(int(n), len(frame))]]


def load_frame(training_config: dict, features: list[str], run_dir: Path) -> pd.DataFrame:
    data = ta.prepare_data(training_config)
    fingerprint = data_fingerprint(data)
    manifest_path = run_dir / "models" / "manifest.json"
    expected = json.loads(manifest_path.read_text()).get("data_fingerprint") if manifest_path.exists() else None
    if expected is None:
        logger.warning("No data fingerprint in {}; the prepared data cannot be verified against the run.", manifest_path)
    elif fingerprint != expected:
        logger.warning("Prepared data fingerprint differs from the saved run; check the configuration before trusting the output.")
    return data[features]


# ---------------------------------------------------------------------------------------------
# bottles: SHAP beeswarm, dependence, summary
# ---------------------------------------------------------------------------------------------


def _beeswarm_offsets(s: np.ndarray, rng: np.random.Generator, half_width: float = 0.4, bins: int = 40) -> np.ndarray:
    """Vertical offsets: uniform jitter scaled by the local density of the SHAP values."""
    counts, edges = np.histogram(s, bins=bins)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    density = counts[idx] / max(counts.max(), 1)
    return rng.uniform(-1.0, 1.0, s.size) * half_width * density


def _percentile(v: np.ndarray) -> np.ndarray:
    out = np.full(v.shape, np.nan)
    ok = np.isfinite(v)
    if ok.sum() > 1:
        out[ok] = np.argsort(np.argsort(v[ok])) / (ok.sum() - 1)
    return out


def plot_beeswarm(bottles: xr.Dataset, s: Settings) -> plt.Figure:
    shap, value = bottles["shap"], bottles["value"]
    mean_abs = np.abs(shap).mean("row").to_pandas()
    order = mean_abs.sort_values(ascending=False).index.tolist()
    n = len(order)
    cmap = plt.get_cmap("RdYlBu_r")
    rng = np.random.default_rng(0)
    base = float(bottles["base"])

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * n + 2.0))
    for pos, feature in enumerate(order):
        y0 = n - 1 - pos
        sv = shap.sel(feature=feature).values.astype(float)
        vv = value.sel(feature=feature).values.astype(float)
        pct = _percentile(vv)
        rgba = np.tile(np.array(to_rgba(GREY)), (sv.size, 1))
        has = np.isfinite(pct)
        rgba[has] = cmap(pct[has])
        ax.scatter(sv, y0 + _beeswarm_offsets(sv, rng), s=3, c=rgba, alpha=0.5,
                   rasterized=True, linewidths=0, zorder=2)
        ax.text(1.02, y0, f"{mean_abs[feature]:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8)
    ax.text(1.02, n - 0.3, "mean |SHAP|", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8, fontweight="bold")
    ax.axvline(0, color="#444444", linewidth=.6, zorder=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels([s.labels.get(f, f) for f in reversed(order)])
    ax.set_ylim(-0.6, n + 0.2)
    ax.set_xlabel(SHAP_LABEL, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"SHAP contributions to ensemble median TA (n = {bottles.sizes['row']:,} bottles)", fontsize=10)

    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.2, aspect=50,
                        label="Feature value (percentile)")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7)
    fig.text(.5, -.01, va="top", ha="center", fontsize=8, s=(
        "One point per bottle; colour is the percentile of the feature value (grey: no value, e.g. groups). "
        "Points spread vertically where they are dense. Contributions plus the base value "
        f"({base:.1f} µmol kg⁻¹) give the ensemble median prediction."))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _strongest_partner(feature: str, strength: pd.DataFrame | None, value: xr.DataArray) -> str | None:
    if strength is None or feature not in strength.index:
        return None
    row = strength.loc[feature].drop(feature, errors="ignore").dropna()
    row = row[[f for f in row.index if np.isfinite(value.sel(feature=f).values).any()]]
    return None if row.empty else str(row.idxmax())


def plot_dependence(bottles: xr.Dataset, strength: pd.DataFrame | None, s: Settings) -> plt.Figure:
    shap, value = bottles["shap"], bottles["value"]
    features = [f for f in bottles.feature.values if np.isfinite(value.sel(feature=f).values).any()]
    fig, axes = _panel_grid(len(features))
    for ax, feature in zip(axes, features):
        x = _axis_values(feature, value.sel(feature=feature).values, s)
        y = shap.sel(feature=feature).values.astype(float)
        partner = _strongest_partner(feature, strength, value)
        if partner is None:
            ax.scatter(x, y, s=3, alpha=0.5, rasterized=True, linewidths=0, color="#4a6fa5")
        else:
            c = _axis_values(partner, value.sel(feature=partner).values, s)
            mesh = ax.scatter(x, y, c=c, cmap="viridis", s=3, alpha=0.5, rasterized=True, linewidths=0)
            fig.colorbar(mesh, ax=ax, shrink=.85, label=_name(partner, s)).ax.tick_params(labelsize=7)
        ax.axhline(0, color="#444444", linewidth=.6)
        ax.set_xlabel(_name(feature, s), fontsize=9)
        ax.set_ylabel(SHAP_LABEL, fontsize=8)
        ax.tick_params(labelsize=7)
        title = s.labels.get(feature, feature)
        ax.set_title(f"{title}  (coloured by {s.labels.get(partner, partner)})" if partner else title, fontsize=10)
    fig.text(.5, -.01, va="top", ha="center", fontsize=8, s=(
        "SHAP contribution to ensemble median TA against the feature value, one point per bottle. "
        "Colour: strongest interaction partner (mean |Φ_ab + Φ_ba|) where interactions are available."))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def write_summary(output_dir: Path, s: Settings) -> None:
    """mean |SHAP| per feature on the bottles and on the maps, whichever stages have run."""
    columns: dict[str, pd.Series] = {}
    bottles_path = output_dir / "shap_bottles.nc"
    maps_path = output_dir / "shap_maps.nc"
    if bottles_path.exists():
        with xr.open_dataset(bottles_path) as ds:
            columns["mean_abs_shap_bottles"] = np.abs(ds["shap"]).mean("row").load().to_pandas()
    if maps_path.exists():
        with xr.open_dataset(maps_path) as ds:
            weights = np.cos(np.deg2rad(ds["lat"]))
            columns["mean_abs_shap_maps"] = np.abs(ds["shap"]).weighted(weights).mean(
                ["lat", "lon"], skipna=True).load().to_pandas()
    if not columns:
        logger.warning("No shap_bottles.nc or shap_maps.nc in {}; summary not written.", output_dir)
        return
    summary = pd.DataFrame(columns)
    summary.index.name = "feature"
    sort_by = "mean_abs_shap_bottles" if "mean_abs_shap_bottles" in summary else "mean_abs_shap_maps"
    summary = summary.sort_values(sort_by, ascending=False)
    summary.to_csv(output_dir / "summary.csv")
    print(summary.round(4).to_string())


def write_dependence(s: Settings) -> None:
    """Re-drawn whenever bottles or interactions change; partners come from interactions.nc if present."""
    bottles_path = s.output_dir / "shap_bottles.nc"
    if not bottles_path.exists():
        logger.warning("No shap_bottles.nc in {}; dependence plot skipped.", s.output_dir)
        return
    with xr.open_dataset(bottles_path) as ds:
        bottles = ds.load()
    strength = None
    inter_path = s.output_dir / "interactions.nc"
    if inter_path.exists():
        with xr.open_dataset(inter_path) as ds:
            strength = ds["strength"].load().to_pandas()
    _save(plot_dependence(bottles, strength, s), s.output_dir, "shap_dependence")


def run_bottles(model: ta.QuantileRegressionEnsemble, frame: pd.DataFrame, order: np.ndarray,
                cfg: dict, s: Settings, n_members: int) -> None:
    X = _subset(frame, order, cfg["n_rows"])
    logger.info("SHAP for {:,} bottles, {} members", len(X), n_members)
    ds = ensemble_shap(model, X, quantile=QUANTILE, groups=s.groups, n_jobs=n_members)
    ds = _attach_rows(ds, X)
    _netcdf_ready(ds).to_netcdf(s.output_dir / "shap_bottles.nc")
    _save(plot_beeswarm(ds, s), s.output_dir, "shap_summary")
    write_summary(s.output_dir, s)
    write_dependence(s)


# ---------------------------------------------------------------------------------------------
# interactions
# ---------------------------------------------------------------------------------------------


def plot_interaction_heatmap(ds: xr.Dataset, s: Settings) -> plt.Figure:
    main = ds["main"].to_pandas()
    strength = ds["strength"].to_pandas()
    order = main.sort_values(ascending=False).index.tolist()
    n = len(order)
    S = strength.loc[order, order].to_numpy(dtype=float)
    M = np.full((n, n), np.nan)
    np.fill_diagonal(M, main.loc[order].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(0.8 * n + 3.5, 0.75 * n + 2.2))
    pairs = ax.imshow(np.ma.masked_invalid(S), cmap="YlOrRd", vmin=0)
    mains = ax.imshow(np.ma.masked_invalid(M), cmap="Greys", vmin=0)
    for i in range(n):
        for j in range(n):
            v = M[i, j] if i == j else S[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    names = [s.labels.get(f, f) for f in order]
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(pairs, ax=ax, shrink=.8, label="Pair strength mean |Φ_ab + Φ_ba| (µmol kg⁻¹)")
    fig.colorbar(mains, ax=ax, shrink=.8, label="Main effect mean |Φ_aa| (µmol kg⁻¹)")
    ax.set_title("Interaction strength (off-diagonal) and main effects (diagonal)", fontsize=10)
    fig.text(.5, -.01, va="top", ha="center", fontsize=8, s=(
        "Boosted-model SHAP interaction values averaged over members and bottles; the salinity main "
        "effect includes the linear term. Features ordered by main effect."))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def run_interactions(model: ta.QuantileRegressionEnsemble, frame: pd.DataFrame, order: np.ndarray,
                     cfg: dict, s: Settings, n_members: int) -> None:
    X = _subset(frame, order, cfg["n_rows_interactions"])
    logger.info("SHAP interactions for {:,} bottles, {} members", len(X), n_members)
    ds = ensemble_shap_interactions(model, X, quantile=QUANTILE, groups=s.groups, n_jobs=n_members)
    ds = _attach_rows(ds, X)
    _netcdf_ready(ds).to_netcdf(s.output_dir / "interactions.nc")

    main = ds["main"].to_pandas()
    mat = ds["strength"].to_pandas()
    arr = mat.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, main.reindex(mat.index).to_numpy(dtype=float))
    pd.DataFrame(arr, index=mat.index, columns=mat.columns).round(6).to_csv(
        s.output_dir / "interaction_strength.csv")

    _save(plot_interaction_heatmap(ds, s), s.output_dir, "interaction_heatmap")
    write_dependence(s)


# ---------------------------------------------------------------------------------------------
# 2-D partial dependence
# ---------------------------------------------------------------------------------------------


def plot_pair(pdx: xr.Dataset, background: pd.DataFrame, a: str, b: str, s: Settings) -> plt.Figure:
    xa = _axis_values(a, pdx["a_value"].values, s)
    xb = _axis_values(b, pdx["b_value"].values, s)
    A, B = np.meshgrid(xa, xb, indexing="ij")
    surface = pdx["pd"].transpose("a_value", "b_value").values
    interaction = pdx["interaction"].transpose("a_value", "b_value").values
    h2 = float(pdx.attrs["h_squared"])
    bg_a = _axis_values(a, background[a].to_numpy(), s)
    bg_b = _axis_values(b, background[b].to_numpy(), s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    mesh = ax1.contourf(A, B, surface, levels=15, cmap="viridis")
    ax1.scatter(bg_a, bg_b, s=2, color="white", alpha=0.35, linewidths=0, rasterized=True)
    fig.colorbar(mesh, ax=ax1, shrink=.85, label="Ensemble Q50 TA (µmol kg⁻¹)")
    ax1.set_title("Partial dependence PD(a, b)", fontsize=10)

    limit = _robust_limit(interaction, fallback=1.0)
    mesh2 = ax2.contourf(A, B, interaction, levels=np.linspace(-limit, limit, 17),
                         cmap="RdBu_r", extend="both")
    ax2.scatter(bg_a, bg_b, s=2, color="#333333", alpha=0.25, linewidths=0, rasterized=True)
    fig.colorbar(mesh2, ax=ax2, shrink=.85, label="Pure interaction (µmol kg⁻¹)")
    ax2.set_title("Pure interaction PD(a,b) − PD(a) − PD(b) + mean", fontsize=10)

    for ax in (ax1, ax2):
        ax.set_xlabel(_name(a, s), fontsize=9)
        ax.set_ylabel(_name(b, s), fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{s.labels.get(a, a)} × {s.labels.get(b, b)}   (H² = {h2:.3f})", fontsize=11)
    fig.text(.5, -.01, va="top", ha="center", fontsize=8, s=(
        "Left: mean ensemble Q50 when both features are set to each grid value on every background bottle. "
        "Right: the part of the joint effect not explained by the two one-way effects. White/grey points: background bottles."))
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return fig


def run_pdp(model: ta.QuantileRegressionEnsemble, frame: pd.DataFrame, order: np.ndarray,
            cfg: dict, s: Settings) -> None:
    inter_path = s.output_dir / "interactions.nc"
    if not inter_path.exists():
        raise FileNotFoundError(f"{inter_path} is missing; run --stage interactions first.")
    with xr.open_dataset(inter_path) as ds:
        strength = ds["strength"].load()
    # groups (e.g. location) are sums of several inputs and cannot be swept on one axis
    model_features = [f for f in strength["feature"].values if f in model.get_feature_names()]
    strength = strength.sel(feature=model_features, feature2=model_features)
    pairs = top_pairs(strength, int(cfg["top_pairs"]))
    logger.info("Top pairs: {}", pairs)
    background = _subset(frame, order, cfg["n_background_pd"])

    path = s.output_dir / "pdp2d.nc"
    xr.Dataset(attrs={"pairs": json.dumps(pairs), "quantile": QUANTILE}).to_netcdf(path, mode="w")
    for a, b in pairs:
        logger.info("Partial dependence {} x {} on {} background rows", a, b, len(background))
        pdx = partial_dependence_2d(model, background, a, b, n_grid=int(cfg["n_grid"]),
                                    quantile=QUANTILE, log_features=s.log_features)
        _netcdf_ready(pdx).to_netcdf(path, mode="a", group=f"{a}__{b}")
        _save(plot_pair(pdx, background, a, b, s), s.output_dir, f"pdp2d_{a}__{b}")


# ---------------------------------------------------------------------------------------------
# SHAP maps
# ---------------------------------------------------------------------------------------------


def shap_maps_with_retries(model: ta.QuantileRegressionEnsemble, features: xr.Dataset, date: str,
                           **kwargs: Any) -> xr.Dataset:
    """One date per call so that a transient HTTP failure only repeats that date."""
    for attempt in range(1, N_ATTEMPTS + 1):
        try:
            return shap_maps(model, features, [date], **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == N_ATTEMPTS:
                raise
            logger.warning("{}: attempt {}/{} failed ({!r}); retrying", date, attempt, N_ATTEMPTS, e)
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def combine_dates(parts: list[xr.Dataset], dates: list[str]) -> xr.Dataset:
    """Nan-mean over dates of the SHAP fields; base is the mean of the per-date bases."""
    if len(parts) == 1:
        combined = parts[0][["shap"]].copy()
    else:
        stacked = xr.concat([part[["shap"]] for part in parts], dim="date", coords="minimal", compat="override")
        combined = stacked.mean("date", skipna=True, keep_attrs=True)
    base = float(np.mean([float(part.attrs["base"]) for part in parts]))
    combined.attrs = {**parts[0].attrs, "dates": list(dates), "base": base}
    return combined


def plot_maps(ds: xr.Dataset, order: list[str], s: Settings, dates: list[str]) -> plt.Figure:
    coast = json.loads(files("highres_ta").joinpath("assets/ne_110m_coastline.json").read_text())
    lon, lat = ds.lon.values, ds.lat.values
    fig, axes = _panel_grid(len(order))
    for ax, feature in zip(axes, order):
        values = ds["shap"].sel(feature=feature).transpose("lat", "lon").values.astype(float)
        limit = _robust_limit(values)
        mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(values), shading="auto", rasterized=True,
                             cmap="RdBu_r", vmin=-limit, vmax=limit)
        _map_axes(ax, coast, "#222222")
        ax.set_title(s.labels.get(feature, feature), fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=.85, label=SHAP_LABEL).ax.tick_params(labelsize=7)
    fig.supxlabel("Longitude (°E)", fontsize=9)
    fig.supylabel("Latitude (°N)", fontsize=9)
    fig.text(.5, -.01, va="top", ha="center", fontsize=8, s=(
        f"Per-cell contribution to the ensemble median TA, averaged over {len(dates)} date(s) "
        f"({', '.join(dates)}). Location is the sum of the three Cartesian coordinate contributions."))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def run_maps(model: ta.QuantileRegressionEnsemble, config: dict, features: list[str], cfg: dict,
             s: Settings, n_members: int) -> None:
    dates = [str(d) for d in cfg["map_dates"]]
    inference = ta.open_inference_data(config, features)
    logger.info("SHAP maps on stride {} for {} date(s), {} members", cfg["map_stride"], len(dates), n_members)
    parts = [
        shap_maps_with_retries(model, inference, date, stride=int(cfg["map_stride"]), groups=s.groups,
                               quantile=QUANTILE, n_jobs=n_members)
        for date in dates
    ]
    ds = combine_dates(parts, dates)
    _netcdf_ready(ds).to_netcdf(s.output_dir / "shap_maps.nc")

    mean_abs = np.abs(ds["shap"]).weighted(np.cos(np.deg2rad(ds["lat"]))).mean(["lat", "lon"], skipna=True)
    order = mean_abs.to_pandas().sort_values(ascending=False).index.tolist()
    _save(plot_maps(ds, order, s, dates), s.output_dir, "shap_maps")
    write_summary(s.output_dir, s)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


@app.command(help=__doc__)
def main(
    config_path: Annotated[
        Path, typer.Option("--config", help="Configuration directory or a single YAML file.", exists=True)
    ] = CONFIG_DIR,
    overrides: Annotated[
        list[Path],
        typer.Option(
            "--override",
            help="YAML file deep-merged over the configuration; repeatable, applied in order.",
            exists=True, dir_okay=False, readable=True,
        ),
    ] = [],
    run_dir: Annotated[
        Path | None, typer.Option(help="Run folder; defaults to paths.output_dir.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Output folder; defaults to <run>/explain.")
    ] = None,
    stage: Annotated[
        Stage, typer.Option(help="Stage to run; 'all' runs bottles, interactions, pdp, maps in order.")
    ] = Stage.all,
    max_cpus: Annotated[
        int | None, typer.Option(min=1, help="Override explain.max_cpus.")
    ] = None,
) -> None:
    config = ta.load_config(config_path, overrides)
    cfg = config["explain"]
    run_dir = resolve_path(run_dir) if run_dir is not None else resolve_path(config["paths"]["output_dir"])
    output_dir = resolve_path(output_dir) if output_dir is not None else run_dir / "explain"
    max_cpus = max_cpus or cfg["max_cpus"]
    settings = Settings(
        output_dir=output_dir,
        labels={str(k): str(v) for k, v in cfg["labels"].items()},
        units={str(k): str(v) for k, v in config.get("partial_dependence", {}).get("feature_units", {}).items()},
        log_features=[str(f) for f in cfg["log_features"]],
        groups={str(k): [str(f) for f in v] for k, v in cfg["groups"].items()},
    )

    # The run's saved configuration defines data preparation (as in spatial_pdp); the current
    # configuration is used for the inference predictors.
    snapshot = run_dir / "configs.json"
    if snapshot.exists():
        training_config = json.loads(snapshot.read_text())
    else:
        logger.warning("No saved configs.json in {}; using the current configuration for data.", run_dir)
        training_config = config

    model_path = run_dir / "models" / "quantile_ensemble.joblib"
    logger.info("model: {} -> output: {}", model_path, output_dir)
    model = ta.QuantileRegressionEnsemble.load(model_path)
    model.optimise_cpus(max_cpus)
    features = list(model.get_feature_names())
    n_members = len(model.estimators)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = ["bottles", "interactions", "pdp", "maps"] if stage is Stage.all else [stage.value]
    frame = None
    order = None
    if any(s in stages for s in ("bottles", "interactions", "pdp")):
        frame = load_frame(training_config, features, run_dir)
        order = np.random.default_rng(0).permutation(len(frame))

    if "bottles" in stages:
        run_bottles(model, frame, order, cfg, settings, n_members)
    if "interactions" in stages:
        run_interactions(model, frame, order, cfg, settings, n_members)
    if "pdp" in stages:
        run_pdp(model, frame, order, cfg, settings)
    if "maps" in stages:
        run_maps(model, config, features, cfg, settings, n_members)

    logger.success("Explanation stages {} written to {}", stages, output_dir)
    plt.close("all")


if __name__ == "__main__":
    app()
