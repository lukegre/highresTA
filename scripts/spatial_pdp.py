"""Map where surface TA responds to each predictor: local one-at-a-time partial dependence.

On a strided lat/lon grid, each predictor is shifted by +/-delta in every ocean cell while the
other predictors stay fixed. The change in the ensemble Q50 is reported as a secant slope per
+delta, computed per member, summarised across members (median, spread, sign agreement) and
averaged over dates. Definitions: docs/plan_spatial_pdp.md. Settings: configs/partial_dependence.yaml.

Outputs (in <run>/partial_dependence/ by default):
  sensitivity.nc                        gridded fields (feature, lat, lon)
  summary.csv                           per feature: delta, n_members, area-weighted mean |S|,
                                        median S, fraction of cells with sign_agreement >= 0.8
  sensitivity_maps.{png,pdf}            median slope maps, one panel per feature
  sensitivity_maps_transparent.png      same, transparent background with grey text (website)
  sensitivity_spread.{png,pdf}          across-member standard deviation maps

    uv run python scripts/spatial_pdp.py
    uv run python scripts/spatial_pdp.py --override configs/variants/stacked_salinity.yaml --date 2019-07-12 --stride 16
"""

import asyncio
import json
import math
import time
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

import highres_ta as ta
from highres_ta.partial_dependence import spatial_sensitivity, training_feature_stats
from highres_ta.training import CONFIG_DIR, data_fingerprint, resolve_path

N_ATTEMPTS = 3  # per date, for transient HTTP errors from the remote predictor store
FIELDS = ("sensitivity", "sensitivity_spread", "sign_agreement", "out_of_range")
STIPPLE_BELOW = 0.8  # stipple where fewer than this fraction of members agree on the sign
HATCH_ABOVE = 0.5  # hatch where more than this fraction of dates are out of range
GREY = "#7d828a"  # mid-grey for the transparent website figure

app = typer.Typer()


def sensitivity_with_retries(
    model: ta.QuantileRegressionEnsemble,
    features: xr.Dataset,
    date: str,
    stats: pd.DataFrame,
    **kwargs: Any,
) -> xr.Dataset:
    """One date per call so that a transient HTTP failure only repeats that date."""
    for attempt in range(1, N_ATTEMPTS + 1):
        try:
            return spatial_sensitivity(model, features, [date], stats, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == N_ATTEMPTS:
                raise
            logger.warning("{}: attempt {}/{} failed ({!r}); retrying", date, attempt, N_ATTEMPTS, e)
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def combine_dates(parts: list[xr.Dataset], dates: list[str]) -> xr.Dataset:
    """Nan-mean of each per-date field, matching spatial_sensitivity's multi-date rule."""
    if len(parts) == 1:
        combined = parts[0][list(FIELDS)].copy()
    else:
        stacked = xr.concat(
            [part[list(FIELDS)] for part in parts], dim="date", coords="minimal", compat="override"
        )
        combined = stacked.mean("date", skipna=True, keep_attrs=True)
    combined.attrs = {**parts[0].attrs, "dates": list(dates)}
    return combined


def summarise(ds: xr.Dataset, units: dict[str, str]) -> pd.DataFrame:
    """One row per feature; area weights are cos(latitude)."""
    ds = ds.transpose("feature", "lat", "lon")
    weights = np.broadcast_to(np.cos(np.deg2rad(ds.lat.values))[:, None], (ds.lat.size, ds.lon.size))
    rows = []
    for i, feature in enumerate(ds.feature.values):
        s = ds.sensitivity.values[i]
        sign = ds.sign_agreement.values[i]
        valid = np.isfinite(s)
        w = np.where(valid, weights, 0.0)
        total_w = w.sum()
        mean_abs = float(np.sum(np.abs(np.where(valid, s, 0.0)) * w) / total_w) if total_w > 0 else np.nan
        has_sign = np.isfinite(sign)
        rows.append({
            "feature": str(feature),
            "unit": units.get(str(feature), ""),
            "delta": float(ds.delta.values[i]),
            "n_members": int(ds.n_members.values[i]),
            "mean_abs_sensitivity": mean_abs,
            "median_sensitivity": float(np.nanmedian(s)) if valid.any() else np.nan,
            "sign_agree_fraction": float((sign[has_sign] >= STIPPLE_BELOW).mean()) if has_sign.any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("mean_abs_sensitivity", ascending=False, ignore_index=True)


def _map_axes(ax: Any, coast: dict, color: str) -> None:
    """Bundled public-domain Natural Earth coastlines keep the figure offline."""
    ax.plot(np.asarray(coast["lon"], dtype=float), np.asarray(coast["lat"], dtype=float),
            color=color, linewidth=.4, zorder=3)
    ax.set(xlim=(-180, 180), ylim=(-90, 90))
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)


def _panel_grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 2.9 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def _robust_limit(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return fallback
    limit = float(np.percentile(finite, 98))
    return limit if limit > 0 else fallback


def plot_maps(
    ds: xr.Dataset, order: list[str], units: dict[str, str], *, field: str, theme: str
) -> plt.Figure:
    """Panels for one field. theme "paper" is opaque; "web" uses grey text and coastlines."""
    coast = json.loads(files("highres_ta").joinpath("assets/ne_110m_coastline.json").read_text())
    fg = GREY if theme == "web" else "#222222"
    rc = {"text.color": fg, "axes.labelcolor": fg, "axes.edgecolor": fg, "xtick.color": fg,
          "ytick.color": fg, "axes.titlecolor": fg, "hatch.color": fg}
    lon, lat = ds.lon.values, ds.lat.values
    with plt.rc_context(rc):
        fig, axes = _panel_grid(len(order))
        for ax, feature in zip(axes, order):
            sel = ds.sel(feature=feature).transpose("lat", "lon")
            values = sel[field].values
            unit = units.get(feature, "")
            if field == "sensitivity":
                limit = _robust_limit(values)
                kwargs = {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}
                label = "µmol kg⁻¹ per +δ"
                title = f"{feature}  +δ = {float(sel.delta):.3g} {unit}".rstrip()
            else:
                kwargs = {"cmap": "viridis", "vmin": 0.0, "vmax": _robust_limit(values, 0.0) or 1.0}
                label = "across-member std (µmol kg⁻¹ per +δ)"
                title = f"{feature}  spread"
            mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(values), shading="auto",
                                 rasterized=True, **kwargs)
            if field == "sensitivity":
                agree = sel.sign_agreement.values
                weak = (np.isfinite(agree) & (agree < STIPPLE_BELOW)).astype(float)
                if weak.any():
                    ax.contourf(lon, lat, weak, levels=[.5, 1.5], colors="none", hatches=[".."])
                oor = np.nan_to_num(sel.out_of_range.values.astype(float))
                if (oor > HATCH_ABOVE).any():
                    ax.contourf(lon, lat, oor, levels=[HATCH_ABOVE, 1.5], colors="none", hatches=["////"])
            _map_axes(ax, coast, fg)
            ax.set_title(title, fontsize=10)
            fig.colorbar(mesh, ax=ax, shrink=.85, label=label).ax.tick_params(labelsize=7)
        fig.supxlabel("Longitude (°E)", fontsize=9)
        fig.supylabel("Latitude (°N)", fontsize=9)
        if field == "sensitivity":
            fig.text(.5, -.01, va="top", s=
                     "Median over members of the secant slope of ensemble Q50 per +δ, averaged over dates. "
                     "Stippling: sign agreement < 0.8. Hatching: out of training range on > 50% of dates.",
                     ha="center", fontsize=8, color=fg)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


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
        Path | None, typer.Option(help="Output folder; defaults to <run>/partial_dependence.")
    ] = None,
    dates: Annotated[
        list[str] | None,
        typer.Option("--date", help="Inference date (repeatable). Defaults to partial_dependence.dates."),
    ] = None,
    stride: Annotated[
        int | None, typer.Option(min=1, help="Every n-th lat/lon cell. Defaults to partial_dependence.stride.")
    ] = None,
    delta_scale: Annotated[
        float | None,
        typer.Option(min=0.0, help="Step as a multiple of the robust feature scale. Defaults to partial_dependence.delta_scale."),
    ] = None,
    max_cpus: Annotated[
        int | None, typer.Option(min=1, help="Override partial_dependence.max_cpus.")
    ] = None,
    show: Annotated[bool, typer.Option(help="Also open the figures interactively.")] = False,
) -> None:
    config = ta.load_config(config_path, overrides)
    cfg = config["partial_dependence"]
    run_dir = resolve_path(run_dir) if run_dir is not None else resolve_path(config["paths"]["output_dir"])
    output_dir = resolve_path(output_dir) if output_dir is not None else run_dir / "partial_dependence"
    dates = list(dates) if dates else [str(d) for d in cfg["dates"]]
    stride = stride or cfg["stride"]
    delta_scale = float(delta_scale if delta_scale is not None else cfg["delta_scale"])
    max_cpus = max_cpus or cfg["max_cpus"]
    log_features = list(cfg["log_features"])
    kwargs = {
        "delta_scale": delta_scale,
        "stride": stride,
        "log_features": log_features,
        "nonnegative_features": list(cfg["nonnegative_features"]),
        "exclude_features": list(cfg["exclude_features"]),
    }

    # The run's saved configuration defines data preparation (as in validation_runner); the
    # current configuration is used for the inference predictors.
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

    data = ta.prepare_data(training_config)
    fingerprint = data_fingerprint(data)
    manifest_path = run_dir / "models" / "manifest.json"
    expected = json.loads(manifest_path.read_text()).get("data_fingerprint") if manifest_path.exists() else None
    if expected is None:
        logger.warning("No data fingerprint in {}; the prepared data cannot be verified against the run.", manifest_path)
    elif fingerprint != expected:
        logger.warning("Prepared data fingerprint differs from the saved run; check the configuration before trusting the output.")
    stats = training_feature_stats(data[features], features, log_features)

    inference_features = ta.open_inference_data(config, features)
    logger.info("Sensitivity on stride {} for {} date(s), delta_scale {}", stride, len(dates), delta_scale)
    parts = [sensitivity_with_retries(model, inference_features, date, stats, **kwargs) for date in dates]
    ds = combine_dates(parts, dates)
    units = {str(k): str(v) for k, v in cfg["feature_units"].items()}

    summary = summarise(ds, units)
    order = summary.feature.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    _netcdf_ready(ds).to_netcdf(output_dir / "sensitivity.nc")
    summary.to_csv(output_dir / "summary.csv", index=False)

    figures = {
        "sensitivity_maps": plot_maps(ds, order, units, field="sensitivity", theme="paper"),
        "sensitivity_spread": plot_maps(ds, order, units, field="sensitivity_spread", theme="paper"),
    }
    for name, fig in figures.items():
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    web = plot_maps(ds, order, units, field="sensitivity", theme="web")
    web.savefig(output_dir / "sensitivity_maps_transparent.png", dpi=200, bbox_inches="tight", transparent=True)

    logger.success("Saved partial dependence for {} features to {}", len(order), output_dir)
    print(summary.round(4).to_string(index=False))
    if show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    app()
