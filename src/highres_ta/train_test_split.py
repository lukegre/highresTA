from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold

FoldIndices = tuple[np.ndarray, np.ndarray]
SPLIT_NAMES = ("train", "validation", "test")


def make_salinity_bins(
    salinity: pd.Series, bins: list[float] | float = 0.2, **cut_kwargs
) -> pd.Series:
    """
    Create salinity bins from a pandas Series of salinity values.

    Parameters:
    - salinity: pd.Series containing salinity values.
    - bins: different behaviors depending on the type:
        - list of floats: use these values as bin edges.
        - int: create this many equal-width bins between the min and max salinity.
        - float: create bins with this percentile width (e.g., 0.2 for 20% width).

    Returns:
    - pd.Series with the same index as the input, containing the bin labels.
    """

    if isinstance(bins, list):
        # Use the provided list of bin edges
        bin_edges = bins
    elif isinstance(bins, (float, int)) and (bins > 1) and not (bins % 1):
        # must be an integer greater than 1, create equal-width bins
        bins = int(bins)
        bin_edges = np.linspace(salinity.min(), salinity.max(), bins + 1)
    elif isinstance(bins, float) and (0 < bins < 1) and (float(1 / bins)).is_integer():
        # Create bins based on percentiles
        quantiles = np.arange(0, 1 + bins / 2, bins)
        if quantiles[0] != 0.0:
            raise ValueError(
                "For percentile bins, the first quantile must be 0.0. Adjust the bins value accordingly."
            )
        if quantiles[-1] != 1.0:
            raise ValueError(
                "For percentile bins, the last quantile must be 1.0. Adjust the bins value accordingly."
            )
        bin_edges = salinity.quantile(quantiles).values
    else:
        raise ValueError("bins must be a list of floats, an int[1:inf], or a float[0:1]")

    logger.debug(f"Using the following bin edges for salinity: {bin_edges}")

    # Create the bins and return the binned series
    cut_default_kwargs = {"include_lowest": True, "right": True, "labels": False}
    cut_passed_kwargs = cut_default_kwargs | cut_kwargs
    binned_salinity = pd.cut(salinity, bins=bin_edges, **cut_passed_kwargs)  # type: ignore

    return binned_salinity


def add_salinity_bins(
    df: pd.DataFrame,
    salinity_col_name: str = "salinity",
    bins: list[float] | float = 0.2,
    **cut_kwargs,
) -> pd.DataFrame:
    """
    Add a new column to the DataFrame with salinity bins.

    Parameters:
    - df: pd.DataFrame containing the data.
    - salinity_col_name: Name of the column in df that contains salinity values.
    - bins: different behaviors depending on the type:
        - list of floats: use these values as bin edges.
        - int: create this many equal-width bins between the min and max salinity.
        - float: create bins with this percentile width (e.g., 0.2 for 20% width).

    Returns:
    - pd.DataFrame with an additional column 'salinity_bin' containing the bin labels.
    """

    df = df.copy()
    df["salinity_bin"] = make_salinity_bins(df[salinity_col_name], bins=bins, **cut_kwargs)

    return df


def stratified_group_folds(
    data: pd.DataFrame,
    *,
    stratify_by: str = "salinity_bin",
    group_by: str = "expocode",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> list[FoldIndices]:
    """Return train/validation positional indices for stratified group CV."""

    logger.debug(f"Making train-test splits stratified by {stratify_by} and grouped by {group_by}")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )

    return list(
        splitter.split(
            X=data,
            y=data[stratify_by],
            groups=data[group_by],
        )
    )


def make_train_test_folds(
    df: pd.DataFrame,
    *,
    expocode_col_name: str = "expocode",
    salinity_col_name: str = "salinity",
    salinity_bins: list[float] | float = 0.2,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
) -> list[FoldIndices]:
    """Return train/test positional indices for stratified group CV."""

    df = df.copy()
    df["salinity_bin"] = make_salinity_bins(df[salinity_col_name], bins=salinity_bins)

    return stratified_group_folds(
        data=df.reset_index(allow_duplicates=True, drop=False),
        stratify_by="salinity_bin",
        group_by=expocode_col_name,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def make_train_test_split(
    df: pd.DataFrame,
    *,
    expocode_col_name: str = "expocode",
    salinity_col_name: str = "salinity",
    salinity_bins: list[float] | float = 0.2,
    n_splits: int = 5,
    with_validation: bool = False,
    shuffle: bool = True,
    random_state: int | None = 42,
    fold_index: int = 0,
) -> list[np.ndarray]:
    """Return train/test positional indices for stratified group CV."""

    folds = make_train_test_folds(
        df,
        expocode_col_name=expocode_col_name,
        salinity_col_name=salinity_col_name,
        salinity_bins=salinity_bins,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    i = fold_index % n_splits
    if with_validation:
        train, test = folds[i]
        valid = folds[(i + 1) % n_splits][-1]
        train = np.setdiff1d(train, valid)
        folds = [train, test, valid]
    else:
        train, test = folds[i]
        folds = [train, test]

    unique = set()
    for f in folds:
        unique.update(f)
    if len(unique) != len(df):
        raise ValueError("Train/test/validation splits do not cover all data points.")

    # Return only the first fold as train/test split
    return folds


def outer_blocks(data: pd.DataFrame, config: dict[str, Any]) -> list[np.ndarray]:
    """Return the shared partition. Block 0 always tests, so no member trains on it."""
    cv_config = config["cross_validation"]
    n_parts = int(cv_config["n_parts"])
    if n_parts < 3:
        raise ValueError("cross_validation.n_parts must be at least 3.")
    return [
        test_index
        for _, test_index in make_train_test_folds(
            data,
            # stratify on in situ salinity so blocks match when salinity is a gridded product
            salinity_col_name="salinity_insitu" if "salinity_insitu" in data else "salinity",
            n_splits=n_parts,
            shuffle=True,
            random_state=int(cv_config["split_random_state"]),
        )
    ]


def split_indices(blocks: Sequence[np.ndarray], split: int) -> dict[str, Any]:
    """Return train/validation/test positional indices when block ``split`` validates."""
    if not 1 <= split < len(blocks):
        raise ValueError(f"Split {split} is outside 1..{len(blocks) - 1}.")
    train_blocks = [np.sort(block) for index, block in enumerate(blocks) if index not in (0, split)]
    return {
        "train": np.sort(np.concatenate(train_blocks)),
        "validation": np.sort(blocks[split]),
        "test": np.sort(blocks[0]),
        "train_blocks": train_blocks,
    }


def tuning_folds(split: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-block-out folds as positions within ``split["train"]``."""
    train = split["train"]
    blocks = split["train_blocks"]
    folds = []
    for held_out, block in enumerate(blocks):
        fit = np.concatenate([other for index, other in enumerate(blocks) if index != held_out])
        folds.append((np.searchsorted(train, np.sort(fit)), np.searchsorted(train, block)))
    return folds


def validate_split(data: pd.DataFrame, split: dict[str, Any], index: int) -> None:
    covered = np.concatenate([split[name] for name in SPLIT_NAMES])
    if len(covered) != len(data) or len(np.unique(covered)) != len(data):
        raise ValueError(f"Split {index} does not cover every row exactly once.")
    groups = data.index.get_level_values("expocode").to_numpy()
    for left, right in combinations(SPLIT_NAMES, 2):
        overlap = set(groups[split[left]]).intersection(groups[split[right]])
        if overlap:
            raise ValueError(f"Split {index} leaks cruises between {left} and {right}: {overlap}")
    logger.info(
        "Split {}: {:,} train / {:,} validation / {:,} test observations",
        index,
        len(split["train"]),
        len(split["validation"]),
        len(split["test"]),
    )


def write_split(output_dir: Path, data: pd.DataFrame, split: dict[str, Any], index: int):
    frame = pd.DataFrame({
        "row_id": np.concatenate([split[name] for name in SPLIT_NAMES]),
        "split": np.repeat(SPLIT_NAMES, [len(split[name]) for name in SPLIT_NAMES]),
    }).sort_values("row_id", ignore_index=True)
    frame["expocode"] = data.index.get_level_values("expocode").to_numpy()[frame["row_id"]]
    path = output_dir / "splits" / f"split_{index}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
