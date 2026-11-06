"""Leakage-safe evaluation of saved quantile ensemble splits."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import huber
from sklearn.metrics import r2_score

from .quantiles import quantile_column


def load_splits(split_files: Sequence[str | Path], data: pd.DataFrame, model,
                fingerprint: str | None = None) -> dict[int, pd.DataFrame]:
    """Verify positional split files against the original, unmodified prepared data.

    If a training fingerprint is available, pass it here to verify the entire data
    frame, including its order and feature values, before using positional row IDs.
    """
    if fingerprint is not None:
        from .training import data_fingerprint
        if data_fingerprint(data) != fingerprint:
            raise ValueError("Prepared data fingerprint does not match the training run.")
    member_splits = getattr(model, "member_splits", None)
    if member_splits is None or len(member_splits) != len(model.estimators):
        raise ValueError("Ensemble member_splits are required for leakage-safe validation.")
    result = {}
    for path in map(Path, split_files):
        match = re.fullmatch(r"split_(\d+)\.parquet", path.name)
        if not match:
            raise ValueError(f"Cannot parse split ID from {path}.")
        split_id = int(match.group(1))
        if split_id in result:
            raise ValueError(f"Duplicate split {split_id}.")
        frame = pd.read_parquet(path)
        if not {"row_id", "split", "expocode"}.issubset(frame.columns):
            raise ValueError(f"Missing split columns in {path}.")
        ids = frame.row_id.to_numpy()
        if (len(ids) != len(data) or not np.issubdtype(ids.dtype, np.integer)
                or not np.array_equal(np.sort(ids), np.arange(len(data)))):
            raise ValueError(f"Invalid positional row IDs in {path}.")
        if set(frame.split.unique()) != {"train", "validation", "test"}:
            raise ValueError(f"Invalid partitions in {path}.")
        if not np.array_equal(data.index.get_level_values("expocode").to_numpy()[ids],
                              frame.expocode.to_numpy()):
            raise ValueError(f"Expocodes in {path} do not match prepared data rows.")
        if (frame.groupby("expocode")["split"].nunique() > 1).any():
            raise ValueError(f"Cruise leakage across partitions in {path}.")
        result[split_id] = frame.sort_values("row_id", ignore_index=True)
    if not result or set(result) != set(member_splits):
        raise ValueError("Split files must match the ensemble member splits exactly.")
    tests = [set(f.loc[f.split == "test", "row_id"]) for f in result.values()]
    if any(test != tests[0] for test in tests[1:]):
        raise ValueError("Saved splits do not share the same test rows.")
    validation = [set(f.loc[f.split == "validation", "row_id"]) for f in result.values()]
    if sum(map(len, validation)) != len(set().union(*validation)):
        raise ValueError("Validation sets overlap between splits.")
    if set().union(*validation) != set(range(len(data))) - tests[0]:
        raise ValueError("Validation sets do not cover all non-test rows.")
    return dict(sorted(result.items()))


def swap_salinity(data: pd.DataFrame, raw: pd.DataFrame, product: str,
                  coordinate_columns: Sequence[str], minimum: float, maximum: float,
                  salinity_column: str = "salinity") -> pd.DataFrame:
    """Replace salinity using raw coordinate keys, retaining original positions."""
    if product == salinity_column:
        return data.copy()
    indexed = raw.set_index(list(coordinate_columns))
    indexed = indexed.loc[~indexed.index.duplicated(keep="first")]
    values = indexed[product].reindex(data.index)
    result = data.copy()
    result[salinity_column] = values.where(values.between(minimum, maximum)).to_numpy()
    return result


def predict_splits(data: pd.DataFrame, splits: Mapping[int, pd.DataFrame], model,
                   feature_names: Sequence[str], target_name: str, *,
                   non_null_columns: Sequence[str] = (), aggregation: str = "median") -> pd.DataFrame:
    """Predict the common test set and each validation set with eligible members only."""
    if aggregation not in ("median", "mean"):
        raise ValueError("aggregation must be 'median' or 'mean'.")
    aggregate = np.median if aggregation == "median" else np.mean
    q50 = quantile_column(model.quantiles, 0.5)
    member_splits = np.asarray(model.member_splits)
    coordinate_names = list(data.index.names)
    columns = list(dict.fromkeys(coordinate_names + ["bottomdepth"]))
    frames = []
    work = [("test", None, next(iter(splits.values())))] + [
        ("validation", split_id, split) for split_id, split in splits.items()]
    for dataset, split_id, split in work:
        ids = split.loc[split.split == dataset, "row_id"].to_numpy(dtype=int)
        subset = data.iloc[ids]
        required = list(dict.fromkeys(list(non_null_columns) + ["salinity", target_name]))
        keep = subset[required].notna().all(axis=1).to_numpy()
        ids, subset = ids[keep], subset.iloc[keep]
        member_ids = np.arange(len(model.estimators)) if split_id is None else np.flatnonzero(member_splits == split_id)
        if not len(member_ids):
            raise ValueError(f"No ensemble members for split {split_id}.")
        coords = subset.index.to_frame(index=False)
        for name in columns:
            if name not in coords and name in subset:
                coords[name] = subset[name].to_numpy()
        coords["row_id"] = ids
        coords["split_id"] = split_id
        coords["dataset"] = dataset
        coords["observed"] = subset[target_name].to_numpy()
        if len(subset):
            predictions = np.stack([model.estimators[i].predict_quantiles(subset[list(feature_names)]) for i in member_ids])
            expected = (len(member_ids), len(subset), len(model.quantiles))
            if predictions.shape != expected:
                raise ValueError(f"Expected member predictions {expected}, got {predictions.shape}.")
            if not np.isfinite(predictions).all():
                raise ValueError("Ensemble produced non-finite predictions.")
            prediction = aggregate(predictions, axis=0)
        else:
            prediction = np.empty((0, len(model.quantiles)))
        for index, alpha in enumerate(model.quantiles):
            coords[f"q_{alpha:g}"] = prediction[:, index]
        coords["predicted"] = prediction[:, q50]
        coords["residual"] = coords.predicted - coords.observed
        frames.append(coords)
    return pd.concat(frames, ignore_index=True)


def regional_scores(predictions: pd.DataFrame, depth_threshold: float = 500) -> pd.DataFrame:
    """Notebook regional scores; empty regions have n=0 and NaN metrics."""
    coastal = predictions.bottomdepth.abs() <= depth_threshold
    high = predictions.lat.abs() >= 60
    # Missing coordinates are not silently classified as oceanic or low latitude.
    depth_known = predictions.bottomdepth.notna()
    lat_known = predictions.lat.notna()
    masks = {"all": np.ones(len(predictions), dtype=bool), "coastal": coastal,
             "oceanic": depth_known & ~coastal,
             "oceanic & < 60deg": depth_known & ~coastal & lat_known & ~high,
             "coastal & < 60deg": coastal & lat_known & ~high,
             "coastal & > 60deg": coastal & high,
             "oceanic & > 60deg": depth_known & ~coastal & high, "> 60deg": high}
    rows = []
    for region, mask in masks.items():
        part = predictions.loc[mask, ["observed", "predicted"]].dropna()
        y = part.observed.to_numpy()
        residual = part.predicted.to_numpy() - y
        n = len(y)
        rows.append({"region": region, "n": n,
                     "root_mean_squared_error": np.sqrt(np.mean(residual ** 2)) if n else np.nan,
                     "mean_absolute_error": np.mean(np.abs(residual)) if n else np.nan,
                     "huber_loss": np.mean(huber(1.35, residual)) if n else np.nan,
                     "median_absolute_error": np.median(np.abs(residual)) if n else np.nan,
                     "mean_bias": np.mean(residual) if n else np.nan,
                     "median_bias": np.median(residual) if n else np.nan,
                     "r2_score": r2_score(y, part.predicted) if n >= 2 else np.nan})
    return pd.DataFrame(rows)


def evaluate_model(model, data: pd.DataFrame, config: dict, split_files: Sequence[str | Path],
                   salinity_products: Mapping[str, pd.DataFrame], *, aggregation: str = "median",
                   depth_threshold: float = 500, fingerprint: str | None = None):
    """Evaluate each position-preserving salinity product against verified splits."""
    feature_names = config["data"]["feature_names"]
    if hasattr(model, "get_feature_names") and tuple(model.get_feature_names()) != tuple(feature_names):
        raise ValueError("Model feature names do not match the saved configuration.")
    splits = load_splits(split_files, data, model, fingerprint=fingerprint)
    predictions, scores = [], []
    for name, product_data in salinity_products.items():
        if not product_data.index.equals(data.index):
            raise ValueError(f"Salinity product {name!r} changed the observation order.")
        pred = predict_splits(product_data, splits, model, config["data"]["feature_names"],
                              config["data"]["target_name"],
                              non_null_columns=config["data"]["non_null_columns"], aggregation=aggregation)
        row_ids = pred["row_id"].to_numpy(dtype=int)
        pred["reference_salinity"] = data["salinity"].to_numpy()[row_ids]
        pred["salinity"] = product_data["salinity"].to_numpy()[row_ids]
        pred["salinity_delta"] = pred["salinity"] - pred["reference_salinity"]
        pred["salinity_product"] = name
        predictions.append(pred)
        for dataset in ("test", "validation"):
            score = regional_scores(pred.loc[pred.dataset == dataset], depth_threshold)
            score["dataset"] = dataset
            score["salinity_product"] = name
            scores.append(score)
    if not predictions:
        raise ValueError("At least one salinity product is required.")
    return pd.concat(predictions, ignore_index=True), pd.concat(scores, ignore_index=True)
