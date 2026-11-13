from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from .inference import select_inference_date
from .quantile_ensemble import QuantileRegressionEnsemble
from .quantiles import quantile_column

# IQR of a normal distribution is 1.349 standard deviations
IQR_TO_SIGMA = 1.349


def _log10_positive(values: np.ndarray) -> np.ndarray:
    """log10 of the positive values; everything else becomes NaN."""
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(values > 0, np.log10(np.where(values > 0, values, 1.0)), np.nan)


def training_feature_stats(
    X: pd.DataFrame, features: Sequence[str], log_features: Sequence[str]
) -> pd.DataFrame:
    """Robust scale (IQR / 1.349) and p01/p99 per feature; log10 space for ``log_features``."""
    rows = {}
    for name in features:
        values = X[name].to_numpy(dtype=float)
        if name in log_features:
            values = _log10_positive(values)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"Feature {name!r} has no finite training values.")
        q01, q25, q75, q99 = np.quantile(values, [0.01, 0.25, 0.75, 0.99])
        rows[name] = {"scale": (q75 - q25) / IQR_TO_SIGMA, "p01": q01, "p99": q99}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["scale", "p01", "p99"])


def perturb_column(
    values: np.ndarray, delta: float, sign: int, *, log: bool, nonnegative: bool
) -> np.ndarray:
    """Shift by ``sign * delta`` (log features: multiply by 10 ** (sign * delta)); clip at 0 if nonnegative."""
    values = np.asarray(values, dtype=float)
    shifted = values * 10.0 ** (sign * delta) if log else values + sign * delta
    return np.maximum(shifted, 0.0) if nonnegative else shifted


def _ignored_feature_names(model: QuantileRegressionEnsemble, estimator: Any) -> set[str]:
    """Features CatBoost ignores in ``estimator``; the linear baseline still uses its own features."""
    names = list(model.get_feature_names())
    ignored = getattr(estimator, "catboost_kwargs", {}).get("ignored_features", [])
    resolved = {
        names[i] if isinstance(i, (int, np.integer)) else str(i) for i in ignored
    }
    return resolved - set(getattr(estimator, "linear_features", None) or ())


def member_sensitivity(
    model: QuantileRegressionEnsemble,
    X: pd.DataFrame,
    feature: str,
    delta: float,
    *,
    log: bool,
    nonnegative: bool,
) -> np.ndarray:
    """Secant slope of every member's Q50 per ``+delta`` of ``feature``, shaped (members, rows).

    The slope is taken over the realised step (log10 space for log features), so
    clipped perturbations are still scaled to a step of ``delta``. NaN where the
    feature is NaN or the step collapses, and for members that ignore the feature.
    """
    values = X[feature].to_numpy(dtype=float)
    x_plus = perturb_column(values, delta, +1, log=log, nonnegative=nonnegative)
    x_minus = perturb_column(values, delta, -1, log=log, nonnegative=nonnegative)

    if log:
        realised = _log10_positive(x_plus) - _log10_positive(x_minus)
    else:
        realised = x_plus - x_minus
    realised = np.where(np.isfinite(realised) & (realised != 0), realised, np.nan)

    X_plus = X.assign(**{feature: x_plus})
    X_minus = X.assign(**{feature: x_minus})
    column = quantile_column(model.quantiles, 0.5)
    predictions = model.predict_members(pd.concat([X_plus, X_minus], ignore_index=True))[
        :, :, column
    ].astype(float)
    n = len(X)
    sensitivity = (predictions[:, :n] - predictions[:, n:]) * (delta / realised)

    for member, estimator in enumerate(model.estimators):
        if feature in _ignored_feature_names(model, estimator):
            sensitivity[member] = np.nan
    return sensitivity


def summarise_members(S: np.ndarray) -> dict[str, np.ndarray]:
    """NaN-aware median, std, sign agreement and member count across axis 0 of (members, rows)."""
    S = np.asarray(S, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(S, axis=0)
        spread = np.nanstd(S, axis=0)
    n_members = np.isfinite(S).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = (np.sign(S) == np.sign(median)).sum(axis=0) / n_members
    agree = np.where((median == 0) | (n_members == 0), np.nan, agree)
    return {
        "sensitivity": median,
        "sensitivity_spread": spread,
        "sign_agreement": agree,
        "n_members": n_members,
    }


def _grid_frame(
    features: xr.Dataset, date, stride: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Ocean cells of the strided grid at ``date``: (frame indexed by (lat, lon), lat, lon)."""
    offset = stride // 2
    grid = features.isel(lat=slice(offset, None, stride), lon=slice(offset, None, stride))
    selected = grid.sel(time=[date], method="nearest")
    if "dayofyear" in grid.dims:
        doy = int(selected.time.dt.dayofyear.item())
        selected = selected.sel(dayofyear=[doy], method="nearest")
    # load only the selected time step, not the full stack
    df = select_inference_date(selected.load(), date).droplevel("time")
    return df, grid.lat.values, grid.lon.values


def sensitivity_for_date(
    model: QuantileRegressionEnsemble,
    features: xr.Dataset,
    date,
    stats: pd.DataFrame,
    *,
    delta_scale: float,
    stride: int,
    log_features: Sequence[str],
    nonnegative_features: Sequence[str],
    exclude_features: Sequence[str],
) -> xr.Dataset:
    """Per-feature sensitivity maps (feature, lat, lon) for one date on a strided grid."""
    df, lat, lon = _grid_frame(features, date, stride)
    X = df.loc[:, list(model.get_feature_names())]

    names = [name for name in model.get_feature_names() if name not in exclude_features]
    logger.info(f"Sensitivity for {date}: {len(X):,} cells, {len(names)} features")

    deltas, n_members = [], []
    columns: dict[str, dict[str, np.ndarray]] = {key: {} for key in (
        "sensitivity", "sensitivity_spread", "sign_agreement", "out_of_range")}
    for name in names:
        log, nonnegative = name in log_features, name in nonnegative_features
        delta = float(delta_scale * stats.loc[name, "scale"])
        S = member_sensitivity(model, X, name, delta, log=log, nonnegative=nonnegative)
        summary = summarise_members(S)
        for key in ("sensitivity", "sensitivity_spread", "sign_agreement"):
            columns[key][name] = summary[key]

        values = X[name].to_numpy(dtype=float)
        outside = np.zeros(len(X), dtype=bool)
        for sign in (+1, -1):
            shifted = perturb_column(values, delta, sign, log=log, nonnegative=nonnegative)
            if log:
                shifted = _log10_positive(shifted)
            outside |= (shifted < stats.loc[name, "p01"]) | (shifted > stats.loc[name, "p99"])
        columns["out_of_range"][name] = np.where(np.isfinite(values), outside, np.nan)

        deltas.append(delta)
        n_members.append(sum(
            name not in _ignored_feature_names(model, estimator) for estimator in model.estimators
        ))

    data_vars = {}
    for key, per_feature in columns.items():
        frame = pd.DataFrame(per_feature, index=X.index).rename_axis(columns="feature")
        da = frame.stack("feature", future_stack=True).to_xarray()
        da = da.reindex(lat=lat, lon=lon, feature=names).transpose("feature", "lat", "lon")
        data_vars[key] = da

    ds = xr.Dataset(data_vars).assign_coords(
        delta=("feature", deltas), n_members=("feature", n_members)
    )
    ds.attrs = {
        "delta_scale": delta_scale,
        "units": "umol kg-1",
        "log_features": list(log_features),
        "member_summary": "median",
        "model_aggregation": model.aggregation,
    }
    return ds


def spatial_sensitivity(
    model: QuantileRegressionEnsemble,
    features: xr.Dataset,
    dates: Sequence,
    stats: pd.DataFrame,
    **kw: Any,
) -> xr.Dataset:
    """Sensitivity maps averaged over ``dates``; ``out_of_range`` becomes the fraction of dates."""
    per_date = []
    with joblib.parallel_config(backend="threading"):
        for date in dates:
            logger.info(f"Spatial sensitivity: {date}")
            per_date.append(sensitivity_for_date(model, features, date, stats, **kw))

    ds = xr.concat(
        per_date, dim="date", coords="minimal", compat="override", combine_attrs="override"
    )
    ds = ds.mean("date", skipna=True, keep_attrs=True)
    ds.attrs = {**per_date[0].attrs, "dates": [str(date) for date in dates]}
    return ds
