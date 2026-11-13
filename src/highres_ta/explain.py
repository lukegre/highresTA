"""SHAP contributions, SHAP interactions and 2-D partial dependence of the Q50 ensemble.

A member's output is ``linear(x) + boosting(x)``. CatBoost SHAP explains the boosting part;
the linear baseline is a main effect, ``lin(x) - mean_bg lin``, added to its own feature(s).
Contributions plus ``base`` then add up to the member's quantile exactly. The ensemble
explanation is the member mean (exact for ``aggregation="mean"``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from catboost import Pool
from loguru import logger

from .partial_dependence import _grid_frame
from .quantile_ensemble import QuantileRegressionEnsemble
from .quantiles import quantile_column

ADDITIVITY_RTOL = 1e-6
MAX_PREDICT_ROWS = 2_000_000


# ---------------------------------------------------------------------- #
# Members                                                                  #
# ---------------------------------------------------------------------- #


def _member_features(estimator: Any) -> list[str]:
    return [str(name) for name in estimator.feature_names_in_]


def _threads_per_member(n_jobs: int) -> int:
    """CatBoost threads per concurrently running member."""
    return max(1, (os.cpu_count() or 1) // max(1, joblib.effective_n_jobs(n_jobs)))


def _linear_contributions(
    estimator: Any, X: pd.DataFrame, background: pd.DataFrame
) -> tuple[dict[str, np.ndarray], float]:
    """Main effects of the linear features, ``phi_j = lin_j(x) - mean_bg``, and the mean base."""
    names = _member_features(estimator)
    linear = [str(name) for name in (estimator.linear_features or ())]
    unknown = [name for name in linear if name not in names]
    if unknown:
        raise ValueError(f"Linear features {unknown} are not model features.")

    lin_model = estimator.linear_model_
    X_df, B_df = X[names], background[names]
    lin_x = np.asarray(lin_model.predict(X_df), dtype=float)
    base = float(np.mean(lin_model.predict(B_df)))

    if len(linear) <= 1:
        return {name: lin_x - base for name in linear}, base
    if estimator.polynomial_degree != 1:
        raise NotImplementedError(
            "Several linear features are only supported for polynomial_degree == 1."
        )

    # linear in x: vary one feature at a time from the background mean
    reference = B_df[linear].mean()
    ref_row = B_df.iloc[[0]].copy()
    ref_row.loc[:, linear] = reference.to_numpy()
    lin_ref = float(lin_model.predict(ref_row)[0])
    contributions = {}
    for name in linear:
        varied = X_df.copy()
        for other in linear:
            if other != name:
                varied[other] = reference[other]
        contributions[name] = np.asarray(lin_model.predict(varied), dtype=float) - lin_ref
    return contributions, lin_ref


def _prepare_member(
    estimator: Any, X: pd.DataFrame, background: pd.DataFrame | None
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    names = _member_features(estimator)
    missing = [name for name in names if name not in X.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {', '.join(missing)}")
    if background is None:
        background = X
    missing = [name for name in names if name not in background.columns]
    if missing:
        raise ValueError(f"Background is missing feature columns: {', '.join(missing)}")
    return names, X, background


def _add_linear(
    phi: np.ndarray, names: list[str], contributions: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Add linear main effects to columns of ``phi`` (n, p) or the diagonal of (n, p, p)."""
    for name, values in contributions.items():
        j = names.index(name)
        if phi.ndim == 2:
            phi[:, j] += values
        else:
            phi[:, j, j] += values
    return phi


def member_shap(
    estimator: Any,
    X: pd.DataFrame,
    quantile_index: int,
    background: pd.DataFrame | None = None,
    *,
    thread_count: int = -1,
) -> tuple[np.ndarray, float]:
    """Additive contributions (n, n_features) of one member's quantile output, and the base.

    Columns follow ``estimator.feature_names_in_`` (the ensemble's ``get_feature_names()``
    order). ``contributions.sum(1) + base`` equals the member's prediction exactly. The
    background (default ``X``) only sets the reference of the linear baseline.
    """
    names, X, background = _prepare_member(estimator, X, background)
    pool = Pool(X[names])
    values = estimator.boosting_model_.get_feature_importance(
        pool, type="ShapValues", thread_count=thread_count
    )
    values = np.asarray(values, dtype=float)[:, quantile_index, :]
    phi, base = values[:, :-1].copy(), float(values[0, -1])

    contributions, base_linear = _linear_contributions(estimator, X, background)
    return _add_linear(phi, names, contributions), base + base_linear


def member_shap_interactions(
    estimator: Any,
    X: pd.DataFrame,
    quantile_index: int,
    *,
    thread_count: int = -1,
) -> np.ndarray:
    """SHAP interaction values (n, n_features, n_features) of one member's quantile output.

    The linear main effect (reference: ``X`` itself) is added to its feature's diagonal, so
    summing over the last axis gives :func:`member_shap` contributions.
    """
    names, X, background = _prepare_member(estimator, X, None)
    pool = Pool(X[names])
    values = estimator.boosting_model_.get_feature_importance(
        pool, type="ShapInteractionValues", thread_count=thread_count
    )
    p = len(names)
    phi = np.asarray(values, dtype=float)[:, quantile_index, :p, :p].copy()
    contributions, _ = _linear_contributions(estimator, X, background)
    return _add_linear(phi, names, contributions)


def _member_prediction(
    estimator: Any, X: pd.DataFrame, quantile_index: int, thread_count: int
) -> np.ndarray:
    """Float64 Q-column prediction of a member (linear + boosted)."""
    X_df = X[_member_features(estimator)]
    lin = np.asarray(estimator.linear_model_.predict(X_df), dtype=float)
    boosted = estimator.boosting_model_.predict(X_df, thread_count=thread_count)
    return lin + np.asarray(boosted, dtype=float)[:, quantile_index]


# ---------------------------------------------------------------------- #
# Groups                                                                   #
# ---------------------------------------------------------------------- #


def _group_matrix(
    names: Sequence[str], groups: Mapping[str, Sequence[str]] | None
) -> tuple[list[str], np.ndarray, list[str]]:
    """New feature names, the (p, p') summing matrix and the names that are groups.

    Ungrouped features keep their order, groups follow in the order given.
    """
    names = list(names)
    groups = {key: list(members) for key, members in (groups or {}).items()}
    used: list[str] = []
    for key, members in groups.items():
        unknown = [name for name in members if name not in names]
        if unknown:
            raise ValueError(f"Group {key!r} has unknown features {unknown}.")
        if key in names:
            raise ValueError(f"Group name {key!r} clashes with a feature name.")
        used += members
    if len(set(used)) != len(used):
        raise ValueError("A feature may belong to only one group.")

    singles = [name for name in names if name not in used]
    new_names = singles + list(groups)
    matrix = np.zeros((len(names), len(new_names)))
    for name in singles:
        matrix[names.index(name), new_names.index(name)] = 1.0
    for key, members in groups.items():
        for name in members:
            matrix[names.index(name), new_names.index(key)] = 1.0
    return new_names, matrix, list(groups)


def _reorder(phi: np.ndarray, member_names: list[str], names: list[str]) -> np.ndarray:
    """Reorder the feature axes (all axes after the first) from member to ensemble order."""
    if member_names == names:
        return phi
    if set(member_names) != set(names):
        raise ValueError("Member features differ from the ensemble feature names.")
    order = [member_names.index(name) for name in names]
    for axis in range(1, phi.ndim):
        phi = np.take(phi, order, axis=axis)
    return phi


def _run_members(model: QuantileRegressionEnsemble, func, n_jobs: int) -> list:
    """Run ``func(estimator, thread_count)`` per member in a threading pool."""
    threads = _threads_per_member(n_jobs)
    return joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(func)(estimator, threads) for estimator in model.estimators
    )


# ---------------------------------------------------------------------- #
# Ensemble SHAP                                                            #
# ---------------------------------------------------------------------- #


def ensemble_shap(
    model: QuantileRegressionEnsemble,
    X: pd.DataFrame,
    quantile: float = 0.5,
    background: pd.DataFrame | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    n_jobs: int = 1,
) -> xr.Dataset:
    """Ensemble-mean SHAP contributions of one quantile, dims (row, feature).

    Variables: ``shap`` (member mean), ``shap_spread`` (std over members, ddof=0) and
    ``value`` (feature value; NaN for groups). Coordinates: ``base`` (scalar mean base) and
    ``prediction`` (row; member-mean quantile). ``groups`` maps a name to features whose
    contributions are summed per member. Raises if contributions + base differ from the
    prediction by more than 1e-6 of its scale.
    """
    names = list(model.get_feature_names())
    k = quantile_column(model.quantiles, quantile)
    if model.aggregation != "mean":
        logger.warning(
            f"Ensemble aggregation is {model.aggregation!r}; SHAP uses the member mean."
        )
    new_names, matrix, group_names = _group_matrix(names, groups)

    def explain(estimator: Any, threads: int):
        phi, base = member_shap(estimator, X, k, background, thread_count=threads)
        phi = _reorder(phi, _member_features(estimator), names)
        return phi, base, _member_prediction(estimator, X, k, threads)

    results = _run_members(model, explain, n_jobs)
    phi = np.stack([r[0] for r in results]) @ matrix  # (members, rows, features')
    base = float(np.mean([r[1] for r in results]))
    prediction = np.mean([r[2] for r in results], axis=0)

    mean = phi.mean(axis=0)
    error = np.abs(mean.sum(axis=1) + base - prediction)
    scale = max(1.0, float(np.nanmax(np.abs(prediction))))
    if error.max() > ADDITIVITY_RTOL * scale:
        raise ValueError(
            f"SHAP additivity violated: max |sum(phi) + base - prediction| = {error.max():.3g}."
        )

    values = np.full((len(X), len(new_names)), np.nan)
    values[:, : len(new_names) - len(group_names)] = X[
        [name for name in new_names if name not in group_names]
    ].to_numpy(dtype=float)

    ds = xr.Dataset(
        {
            "shap": (("row", "feature"), mean),
            "shap_spread": (("row", "feature"), phi.std(axis=0)),
            "value": (("row", "feature"), values),
        },
        coords={
            "row": np.arange(len(X)),
            "feature": new_names,
            "base": base,
            "prediction": ("row", prediction),
        },
    )
    ds.attrs = {
        "quantile": quantile,
        "units": "umol kg-1",
        "n_members": len(model.estimators),
        "model_aggregation": model.aggregation,
        "groups": {key: list(members) for key, members in (groups or {}).items()},
    }
    return ds


def ensemble_shap_interactions(
    model: QuantileRegressionEnsemble,
    X: pd.DataFrame,
    quantile: float = 0.5,
    groups: Mapping[str, Sequence[str]] | None = None,
    n_jobs: int = 1,
) -> xr.Dataset:
    """Ensemble-mean SHAP interaction values of one quantile.

    Variables: ``interaction`` (row, feature, feature2), ``strength`` (feature, feature2) =
    mean |Phi_ab + Phi_ba| with NaN on the diagonal, and ``main`` (feature) = mean |Phi_aa|.
    Groups sum the whole block of their features, so a group's diagonal includes the
    interactions inside the group.
    """
    names = list(model.get_feature_names())
    k = quantile_column(model.quantiles, quantile)
    new_names, matrix, _ = _group_matrix(names, groups)

    def explain(estimator: Any, threads: int):
        phi = member_shap_interactions(estimator, X, k, thread_count=threads)
        return _reorder(phi, _member_features(estimator), names)

    phi = np.mean(_run_members(model, explain, n_jobs), axis=0)  # (rows, p, p)
    phi = np.einsum("nab,ac,bd->ncd", phi, matrix, matrix)

    strength = np.abs(phi + phi.transpose(0, 2, 1)).mean(axis=0)
    np.fill_diagonal(strength, np.nan)
    main = np.abs(np.einsum("naa->na", phi)).mean(axis=0)

    ds = xr.Dataset(
        {
            "interaction": (("row", "feature", "feature2"), phi),
            "strength": (("feature", "feature2"), strength),
            "main": (("feature",), main),
        },
        coords={"row": np.arange(len(X)), "feature": new_names, "feature2": new_names},
    )
    ds.attrs = {
        "quantile": quantile,
        "units": "umol kg-1",
        "n_members": len(model.estimators),
        "model_aggregation": model.aggregation,
        "groups": {key: list(members) for key, members in (groups or {}).items()},
    }
    return ds


def top_pairs(strength: xr.DataArray, k: int) -> list[tuple[str, str]]:
    """The ``k`` strongest feature pairs (a before b in feature order), strongest first."""
    names = [str(name) for name in strength["feature"].values]
    values = np.asarray(strength.transpose("feature", "feature2").values, dtype=float)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = np.nanmax([values[i, j], values[j, i]]) if (
                np.isfinite(values[i, j]) or np.isfinite(values[j, i])
            ) else np.nan
            if np.isfinite(value):
                pairs.append((value, i, j))
    pairs.sort(key=lambda item: -item[0])
    return [(names[i], names[j]) for _, i, j in pairs[:k]]


# ---------------------------------------------------------------------- #
# 2-D partial dependence                                                   #
# ---------------------------------------------------------------------- #


def _feature_grid(values: np.ndarray, n_grid: int, log: bool) -> np.ndarray:
    """p02-p98 grid; log10-spaced for ``log`` features."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)] if log else values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Background has no usable values for a feature.")
    lo, hi = np.percentile(values, [2, 98])
    return np.geomspace(lo, hi, n_grid) if log else np.linspace(lo, hi, n_grid)


def _mean_prediction(
    model: QuantileRegressionEnsemble,
    background: pd.DataFrame,
    overrides: Mapping[str, tuple[np.ndarray, np.ndarray]],
    column: int,
    max_rows: int | None = None,
) -> np.ndarray:
    """Member-mean prediction averaged over background rows, for each of P points.

    ``overrides[name] = (values (P,), mask (P,))`` sets column ``name`` to ``values`` at
    points where ``mask`` is True (the background column otherwise).
    """
    n = len(background)
    n_points = len(next(iter(overrides.values()))[0])
    chunk = max(1, (max_rows or MAX_PREDICT_ROWS) // n)
    out = np.empty(n_points)
    for start in range(0, n_points, chunk):
        stop = min(start + chunk, n_points)
        frame = background.iloc[np.tile(np.arange(n), stop - start)].reset_index(drop=True)
        for name, (values, mask) in overrides.items():
            frame[name] = np.where(
                np.repeat(mask[start:stop], n),
                np.repeat(values[start:stop], n),
                frame[name].to_numpy(dtype=float),
            )
        with joblib.parallel_config(backend="threading"):
            member = model.predict_members(frame)[:, :, column].astype(float)
        out[start:stop] = member.mean(axis=0).reshape(stop - start, n).mean(axis=1)
    return out


def partial_dependence_2d(
    model: QuantileRegressionEnsemble,
    background: pd.DataFrame,
    a: str,
    b: str,
    *,
    n_grid: int = 20,
    quantile: float = 0.5,
    log_features: Sequence[str] = (),
) -> xr.Dataset:
    """2-D partial dependence of the member-mean quantile for the pair (a, b).

    Grids span the background p02-p98 of each feature (log10-spaced for ``log_features``).
    Variables: ``pd`` (a_value, b_value), ``pd_a`` (a_value), ``pd_b`` (b_value) and the
    pure ``interaction`` surface, i.e. the mean-centred ``pd`` minus the mean-centred
    ``pd_a`` and ``pd_b`` (zero for an additive pair). Attrs: ``a``, ``b`` and Friedman's
    ``h_squared`` = sum(interaction^2) / sum(centred pd^2) over the grid.
    """
    if a == b:
        raise ValueError("a and b must be different features.")
    names = list(model.get_feature_names())
    for name in (a, b):
        if name not in names:
            raise ValueError(f"{name!r} is not a model feature.")
    base = background[names].reset_index(drop=True).copy()
    column = quantile_column(model.quantiles, quantile)

    ga = _feature_grid(base[a].to_numpy(), n_grid, a in log_features)
    gb = _feature_grid(base[b].to_numpy(), n_grid, b in log_features)

    # points: n*n pair grid, then n a-only and n b-only points (other column untouched)
    n = n_grid
    ones, zeros = np.ones, np.zeros
    a_values = np.concatenate([np.repeat(ga, n), ga, zeros(n)])
    b_values = np.concatenate([np.tile(gb, n), zeros(n), gb])
    a_mask = np.concatenate([ones(n * n + n, bool), zeros(n, bool)])
    b_mask = np.concatenate([ones(n * n, bool), zeros(n, bool), ones(n, bool)])
    logger.info(f"2-D PD {a} x {b}: {len(a_values) * len(base):,} predictions")
    pd_all = _mean_prediction(
        model, base, {a: (a_values, a_mask), b: (b_values, b_mask)}, column
    )

    pd_ab = pd_all[: n * n].reshape(n, n)
    pd_a, pd_b = pd_all[n * n : n * n + n], pd_all[n * n + n :]
    centred = pd_ab - pd_ab.mean()
    interaction = centred - (pd_a - pd_a.mean())[:, None] - (pd_b - pd_b.mean())[None, :]
    denominator = float((centred**2).sum())
    h_squared = float((interaction**2).sum() / denominator) if denominator > 0 else float("nan")

    ds = xr.Dataset(
        {
            "pd": (("a_value", "b_value"), pd_ab),
            "pd_a": (("a_value",), pd_a),
            "pd_b": (("b_value",), pd_b),
            "interaction": (("a_value", "b_value"), interaction),
        },
        coords={"a_value": ga, "b_value": gb},
    )
    ds.attrs = {
        "a": a,
        "b": b,
        "h_squared": h_squared,
        "quantile": quantile,
        "units": "umol kg-1",
        "n_background": len(base),
        "log_features": [name for name in (a, b) if name in log_features],
    }
    return ds


# ---------------------------------------------------------------------- #
# Maps                                                                     #
# ---------------------------------------------------------------------- #


def shap_maps(
    model: QuantileRegressionEnsemble,
    features: xr.Dataset,
    dates: Sequence,
    *,
    stride: int,
    groups: Mapping[str, Sequence[str]] | None = None,
    quantile: float = 0.5,
    n_jobs: int = 1,
) -> xr.Dataset:
    """SHAP contribution maps ``shap`` (feature, lat, lon) averaged over ``dates``.

    Each date is explained on its own strided grid with that date's cells as background.
    Attrs: ``dates``, ``quantile`` and ``base`` (mean over dates).
    """
    names = list(model.get_feature_names())
    per_date, bases = [], []
    for date in dates:
        df, lat, lon = _grid_frame(features, date, stride)
        X = df.loc[:, names]
        logger.info(f"SHAP map for {date}: {len(X):,} cells")
        ds = ensemble_shap(
            model, X, quantile, background=X, groups=groups, n_jobs=n_jobs
        )
        bases.append(float(ds["base"]))
        frame = pd.DataFrame(ds["shap"].values, index=X.index, columns=ds["feature"].values)
        da = frame.rename_axis(columns="feature").stack("feature", future_stack=True).to_xarray()
        per_date.append(
            da.reindex(lat=lat, lon=lon, feature=ds["feature"].values).transpose(
                "feature", "lat", "lon"
            )
        )

    shap = xr.concat(per_date, dim="date").mean("date", skipna=True)
    out = xr.Dataset({"shap": shap})
    out.attrs = {
        "dates": [str(date) for date in dates],
        "quantile": quantile,
        "base": float(np.mean(bases)),
        "units": "umol kg-1",
        "model_aggregation": model.aggregation,
        "groups": {key: list(members) for key, members in (groups or {}).items()},
    }
    return out


__all__ = [
    "ensemble_shap",
    "ensemble_shap_interactions",
    "member_shap",
    "member_shap_interactions",
    "partial_dependence_2d",
    "shap_maps",
    "top_pairs",
]
