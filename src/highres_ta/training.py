"""Quantile-ensemble training: data preparation, HPO, member fitting and assembly."""

import hashlib
import itertools
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dotenv
import numpy as np
import optuna
import pandas as pd
import yaml
from loguru import logger

from .dataio import add_talk_adjustment, load_data
from .estimators import CatBoostResidualRegressor
from .features import add_cyclical_dayofyear, add_spherical_coords
from .quantile_ensemble import QuantileRegressionEnsemble, decompose_variance
from .quantiles import score_quantile_predictions
from .target_filtering import drop_bad_quality_talk, drop_extreme_salinities
from .train_test_split import SPLIT_NAMES, split_indices, tuning_folds

ROOT = Path(dotenv.find_dotenv("pyproject.toml")).parent
CONFIG_DIR = ROOT / "configs"


@dataclass(frozen=True)
class Member:
    """One ensemble member: ``replicate`` is its index among the Rashomon members of a split."""

    split: int
    replicate: int

    @property
    def seed(self) -> int:
        return 100 * self.split + self.replicate

    @property
    def name(self) -> str:
        return f"split_{self.split}_member_{self.replicate}"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``: nested mappings recursively, any other value replaced."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path = CONFIG_DIR, overrides: Sequence[Path] = ()) -> dict[str, Any]:
    """Load one YAML file, or merge every ``*.yaml`` in a directory (top-level keys must be unique).

    Each file in ``overrides`` is then deep-merged on top, in order.
    """
    path = Path(path)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    config: dict[str, Any] = {}
    for file in files:
        content = yaml.safe_load(file.read_text())
        if not isinstance(content, dict):
            raise TypeError(f"Configuration in {file} must be a mapping.")
        if duplicates := config.keys() & content.keys():
            raise ValueError(f"Duplicate top-level keys {sorted(duplicates)} in {file}.")
        config.update(content)
    for file in overrides:
        content = yaml.safe_load(Path(file).read_text())
        if not isinstance(content, dict):
            raise TypeError(f"Configuration in {file} must be a mapping.")
        config = deep_merge(config, content)
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stack_salinity(
    data: pd.DataFrame, sources: Sequence[str], minimum: float, maximum: float,
    salinity_col_name: str = "salinity",
) -> pd.DataFrame:
    """Set salinity to the fillna stack of ``sources`` (in order), keeping in situ salinity
    where the stack is missing or outside the bounds. In situ values go to ``salinity_insitu``."""
    insitu = data[salinity_col_name]
    stack = pd.Series(np.nan, index=data.index)
    origin = pd.Series("in situ", index=data.index)
    for name in sources:
        take = stack.isna() & data[name].between(minimum, maximum)
        stack[take] = data.loc[take, name]
        origin[take] = name
    data = data.copy()
    data["salinity_insitu"] = insitu
    data[salinity_col_name] = stack.fillna(insitu)
    counts = origin.value_counts()
    logger.info(
        "Stacked salinity: {} (fallback to in situ: {:,} of {:,} rows)",
        ", ".join(f"{name} {counts.get(name, 0):,}" for name in sources),
        counts.get("in situ", 0), len(data),
    )
    return data


def prepare_data(
    config: dict[str, Any], show_all_columns=False, salinity_col_name: str = "salinity"
) -> pd.DataFrame:
    data_config = config["data"]
    coordinate_columns = data_config["coordinate_columns"]
    feature_names = data_config["feature_names"]
    engineered_feature_names = data_config["engineered_feature_names"]
    raw_feature_names = [name for name in feature_names if name not in engineered_feature_names]
    target_name = data_config["target_name"]
    quality_columns = data_config["quality_columns"]
    salinity_source = data_config.get("salinity_source") or []
    required_columns = list(
        dict.fromkeys(
            coordinate_columns + raw_feature_names + [target_name] + quality_columns + list(salinity_source)
        )
    )
    output_columns = feature_names + [target_name] + (["salinity_insitu"] if salinity_source else [])

    if show_all_columns:
        logger.info(
            "\n" + str(load_data(str(resolve_path(config["paths"]["training_data"]))).columns)
        )

    data = (
        load_data(str(resolve_path(config["paths"]["training_data"])))[required_columns]
        .set_index(coordinate_columns, drop=False)
        .pipe(
            drop_extreme_salinities,
            salinity_col_name=salinity_col_name,
            min=data_config["minimum_salinity"],
            max=data_config["maximum_salinity"],
        )
        .pipe(
            add_talk_adjustment,
            fname=str(resolve_path(config["paths"]["talk_adjustment_csv"])),
        )
        .pipe(drop_bad_quality_talk)
        .dropna(subset=data_config["non_null_columns"])
        .drop_duplicates(subset=coordinate_columns, keep="first")
    )
    if salinity_source:
        data = stack_salinity(
            data, salinity_source, data_config["minimum_salinity"],
            data_config["maximum_salinity"], salinity_col_name,
        )
    data = (
        data.select_dtypes(include=[np.number])
        .pipe(add_cyclical_dayofyear)
        .pipe(add_spherical_coords)
        .loc[:, output_columns]
    )
    logger.info("Prepared {:,} observations with {} features", len(data), len(feature_names))
    return data


def data_fingerprint(data: pd.DataFrame) -> str:
    row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def quantiles(config: dict[str, Any]) -> np.ndarray:
    return np.asarray(config["model"]["quantiles"], dtype=float)


def interval_config(config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        str(label): (float(bounds[0]), float(bounds[1]))
        for label, bounds in config["evaluation"]["intervals"].items()
    }


def fixed_catboost_params(config: dict[str, Any], seed: int) -> dict[str, Any]:
    model_config = config["model"]
    alpha_string = ",".join(f"{alpha:g}" for alpha in quantiles(config))
    return {
        "loss_function": f"MultiQuantile:alpha={alpha_string}",
        "iterations": model_config["maximum_iterations"],
        "random_seed": seed,
        "allow_writing_files": False,
        "thread_count": config["optimization"]["threads_per_trial"],
        "verbose": False,
        "early_stopping_rounds": model_config["early_stopping_rounds"],
    }


# (low, high, log-scaled); integer bounds give integer parameters. Shared by the sampler
# and the Rashomon diversity distance so they cannot drift apart.
PARAMETER_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "learning_rate": (0.03, 0.3, True),
    "l2_leaf_reg": (1.0, 100.0, True),
    "depth": (4, 8, False),
    "random_strength": (1.0, 10.0, False),
}


def dropped_feature_options(config: dict[str, Any]) -> list[str]:
    """Categorical choices for the features a trial ignores: "none", singles, pairs, ..."""
    optimization = config["optimization"]
    protected = set(optimization["protected_features"])
    droppable = [name for name in config["data"]["feature_names"] if name not in protected]
    options = ["none"]
    for size in range(1, int(optimization["max_dropped_features"]) + 1):
        options += ["+".join(combination) for combination in itertools.combinations(droppable, size)]
    return options


def decode_dropped_features(value: str) -> list[str]:
    return [] if value == "none" else value.split("+")


def suggest_parameters(trial: optuna.Trial, config: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, (low, high, log) in PARAMETER_BOUNDS.items():
        if isinstance(low, int):
            params[name] = trial.suggest_int(name, low, high, log=log)
        else:
            params[name] = trial.suggest_float(name, low, high, log=log)
    params["dropped_features"] = trial.suggest_categorical(
        "dropped_features", dropped_feature_options(config)
    )
    return params


def catboost_params(params: dict[str, Any]) -> dict[str, Any]:
    """Turn sampled/trial params into CatBoost kwargs (``dropped_features`` -> ``ignored_features``)."""
    params = dict(params)
    dropped = decode_dropped_features(params.pop("dropped_features"))
    if dropped:
        params["ignored_features"] = dropped
    return params


def make_model(
    config: dict[str, Any], seed: int, catboost_params: dict[str, Any]
) -> CatBoostResidualRegressor:
    return CatBoostResidualRegressor(
        linear_features=config["model"]["linear_features"],
        feature_names=config["data"]["feature_names"],
        polynomial_degree=config["model"]["polynomial_degree"],
        **catboost_params,
    )


def validation_loss(model: CatBoostResidualRegressor) -> float:
    scores = model.boosting_model_.get_best_score()["validation"]
    loss_name = model.loss_function
    if loss_name in scores:
        return float(scores[loss_name])
    if len(scores) == 1:
        return float(next(iter(scores.values())))
    raise KeyError(f"Could not identify {loss_name!r} in validation scores: {scores}")


def objective_factory(
    data: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    seed: int,
):
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]
    alpha = quantiles(config)
    intervals = interval_config(config)
    fixed_params = fixed_catboost_params(config, seed)

    def objective(trial: optuna.Trial) -> float:
        params = fixed_params | catboost_params(suggest_parameters(trial, config))
        fold_losses: list[float] = []
        fold_scores: list[dict[str, float]] = []
        fold_best_iterations: list[int] = []

        for fold_index, (train_index, validation_index) in enumerate(folds):
            train = data.iloc[train_index]
            validation = data.iloc[validation_index]
            model = make_model(config, seed, params)
            model.fit(
                train.loc[:, feature_names],
                train.loc[:, target_name],
                eval_set=(
                    validation.loc[:, feature_names],
                    validation.loc[:, target_name],
                ),
            )

            prediction = model.predict_quantiles(validation.loc[:, feature_names])
            fold_losses.append(validation_loss(model))
            fold_scores.append(
                score_quantile_predictions(
                    validation.loc[:, target_name], prediction, alpha, intervals
                )
            )
            fold_best_iterations.append(model.boosting_model_.get_best_iteration() + 1)
            trial.report(float(np.median(fold_losses)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_multi_quantile_loss", fold_losses)
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_best_iterations", fold_best_iterations)
        trial.set_user_attr("refit_iterations", int(np.median(fold_best_iterations)))
        return float(np.median(fold_losses))

    return objective


def optimize(
    data: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    split: int,
    storage_path: Path,
) -> optuna.Study:
    """Run or resume the split's single study until ``optimization.n_trials`` trials complete."""
    optimization = config["optimization"]
    seed = 100 * split
    study_name = f"catboost_quantile_split_{split}"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        # Optuna reseeds the sampler per worker when n_jobs > 1, so the seed only makes
        # the study reproducible with parallel_trials == 1.
        sampler=optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=optimization["n_startup_trials"], constant_liar=True
        ),
        pruner=optuna.pruners.PercentilePruner(
            75.0, n_startup_trials=optimization["n_startup_trials"], n_warmup_steps=0
        ),
        # The heartbeat makes Optuna fail trials left RUNNING by a killed process on the next start.
        storage=optuna.storages.RDBStorage(
            f"sqlite:///{storage_path}", heartbeat_interval=60, grace_period=180
        ),
        study_name=study_name,
        load_if_exists=True,
    )
    fold_digest = hashlib.sha256()
    for train_index, validation_index in folds:
        fold_digest.update(np.asarray(train_index, dtype=np.int64).tobytes())
        fold_digest.update(np.asarray(validation_index, dtype=np.int64).tobytes())
    # n_trials, parallel_trials and threads_per_trial are deliberately excluded: they may change on resume.
    signature_payload = {
        "data_fingerprint": data_fingerprint(data),
        "fold_digest": fold_digest.hexdigest(),
        "features": config["data"]["feature_names"],
        "target": config["data"]["target_name"],
        "model": config["model"],
        "split": split,
        "seed": seed,
        "protected_features": optimization["protected_features"],
        "max_dropped_features": optimization["max_dropped_features"],
        "n_startup_trials": optimization["n_startup_trials"],
        "dropped_feature_options": dropped_feature_options(config),
        "parameter_bounds": {name: list(bounds) for name, bounds in PARAMETER_BOUNDS.items()},
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    existing_signature = study.user_attrs.get("study_signature")
    if existing_signature is not None and existing_signature != signature:
        raise ValueError(
            f"Study {study_name!r} was created for different data, folds, or model "
            "settings. Use a new output directory or remove the mismatched study."
        )
    study.set_user_attr("study_signature", signature)
    study.set_metric_names(["multi_quantile_loss"])
    target = int(optimization["n_trials"])
    completed = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    logger.info("Study {} has {}/{} completed trials", study_name, completed, target)
    if completed < target:
        parallel_trials = optimization["parallel_trials"]
        study.optimize(
            objective_factory(data, folds, config, seed),
            n_jobs=parallel_trials,
            # Pruned trials do not count, so stop on completed trials rather than n_trials.
            callbacks=[
                optuna.study.MaxTrialsCallback(target, states=(optuna.trial.TrialState.COMPLETE,))
            ],
        )
    return study


def completed_trials(trials: Sequence[optuna.trial.FrozenTrial]) -> list[optuna.trial.FrozenTrial]:
    """Completed trials, best first (ties by trial number)."""
    complete = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    return sorted(complete, key=lambda t: (t.value, t.number))


def rashomon_set(
    trials: Sequence[optuna.trial.FrozenTrial], config: dict[str, Any]
) -> list[optuna.trial.FrozenTrial]:
    """Completed trials within ``rashomon_epsilon`` (relative) of the best loss, best first."""
    ranked = completed_trials(trials)
    if not ranked:
        return []
    limit = (1.0 + float(config["ensemble"]["rashomon_epsilon"])) * ranked[0].value
    return [t for t in ranked if t.value <= limit]


def trial_distance(a: optuna.trial.FrozenTrial, b: optuna.trial.FrozenTrial) -> float:
    """Half Jaccard distance of dropped-feature sets, half mean normalised hyperparameter distance."""
    dropped_a = set(decode_dropped_features(a.params["dropped_features"]))
    dropped_b = set(decode_dropped_features(b.params["dropped_features"]))
    union = dropped_a | dropped_b
    jaccard = 1.0 - len(dropped_a & dropped_b) / len(union) if union else 0.0
    distances = []
    for name, (low, high, log) in PARAMETER_BOUNDS.items():
        x_a, x_b = a.params[name], b.params[name]
        if log:
            x_a, x_b, low, high = np.log10([x_a, x_b, low, high])
        distances.append(abs(x_a - x_b) / (high - low))
    return 0.5 * jaccard + 0.5 * float(np.mean(distances))


def select_rashomon_members(
    trials: Sequence[optuna.trial.FrozenTrial], config: dict[str, Any]
) -> list[tuple[optuna.trial.FrozenTrial, bool]]:
    """Pick ``n_members_per_split`` diverse near-optimal trials as ``(trial, in_rashomon_set)``.

    Greedy max-min on :func:`trial_distance`, starting from the best trial. If the Rashomon
    set is too small it is taken whole and topped up with the next-best trials outside it.
    """
    n_members = int(config["ensemble"]["n_members_per_split"])
    if float(config["ensemble"]["rashomon_epsilon"]) < 0:
        raise ValueError("ensemble.rashomon_epsilon must not be negative.")
    ranked = completed_trials(trials)
    if len(ranked) < n_members:
        raise ValueError(f"Need {n_members} completed trials to select members, got {len(ranked)}.")
    rashomon = rashomon_set(ranked, config)

    selected = [rashomon[0]]
    remaining = rashomon[1:]
    while len(selected) < n_members and remaining:
        # Max-min distance; ties go to the lower loss, then the lower trial number.
        best = max(
            remaining,
            key=lambda t: (min(trial_distance(t, s) for s in selected), -t.value, -t.number),
        )
        selected.append(best)
        remaining.remove(best)
    result = [(t, True) for t in selected]

    if len(result) < n_members:
        logger.warning(
            "Rashomon set has only {} trials (< {} members); filling with the next-best trials",
            len(rashomon),
            n_members,
        )
        in_set = {t.number for t in rashomon}
        fill = [t for t in ranked if t.number not in in_set][: n_members - len(result)]
        result += [(t, False) for t in fill]
    return result


def rashomon_table(
    trials: Sequence[optuna.trial.FrozenTrial],
    selection: Sequence[tuple[optuna.trial.FrozenTrial, bool]],
    config: dict[str, Any],
) -> pd.DataFrame:
    """One row per Rashomon-set trial plus any fill-ins, with selection status."""
    ranked = completed_trials(trials)
    ranks = {t.number: rank for rank, t in enumerate(ranked, start=1)}
    member_index = {t.number: index for index, (t, _) in enumerate(selection)}
    in_set = {t.number for t in rashomon_set(ranked, config)}
    rows = []
    for t in ranked:
        if t.number in in_set or t.number in member_index:
            rows.append({
                "trial": t.number,
                "value": t.value,
                "rank": ranks[t.number],
                **t.params,
                "in_rashomon_set": t.number in in_set,
                "selected": t.number in member_index,
                "member": member_index.get(t.number),
            })
    table = pd.DataFrame(rows)
    table["member"] = table["member"].astype("Int64")
    return table


def selected_params(trial: optuna.trial.FrozenTrial, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """CatBoost kwargs for refitting a selected trial on the full training blocks."""
    optimization = config["optimization"]
    params = fixed_catboost_params(config, seed) | catboost_params(trial.params)
    params["iterations"] = trial.user_attrs["refit_iterations"]
    params.pop("early_stopping_rounds", None)
    # The split's own core budget; splits run as separate concurrent processes.
    params["thread_count"] = optimization["parallel_trials"] * optimization["threads_per_trial"]
    return params


def score_subsets(
    data: pd.DataFrame, prediction: np.ndarray, config: dict[str, Any]
) -> dict[str, dict[str, float]]:
    target_name = config["data"]["target_name"]
    alpha = quantiles(config)
    intervals = interval_config(config)
    scores = {"all": score_quantile_predictions(data[target_name], prediction, alpha, intervals)}
    minimum_bottomdepth = config["evaluation"].get("minimum_bottomdepth")
    if minimum_bottomdepth is not None:
        mask = data["bottomdepth"].to_numpy() > minimum_bottomdepth
        if np.any(mask):
            scores["deep"] = score_quantile_predictions(
                data.loc[mask, target_name], prediction[mask], alpha, intervals
            )
    return scores


def prediction_frame(
    data: pd.DataFrame,
    row_ids: np.ndarray,
    prediction: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    target_name = config["data"]["target_name"]
    frame = pd.DataFrame({
        "row_id": row_ids,
        "observed": data[target_name].to_numpy(),
        "bottomdepth": data["bottomdepth"].to_numpy(),
    })
    for index, alpha in enumerate(quantiles(config)):
        frame[f"q_{alpha:g}"] = prediction[:, index]
    return frame


def prediction_columns(config: dict[str, Any]) -> list[str]:
    return [f"q_{alpha:g}" for alpha in quantiles(config)]


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def members(config: dict[str, Any], split: int | None = None) -> list[Member]:
    n_splits = int(config["cross_validation"]["n_parts"]) - 1
    n_members = int(config["ensemble"]["n_members_per_split"])
    if split is not None and not 1 <= split <= n_splits:
        raise ValueError(f"Split {split} is outside 1..{n_splits}.")
    return [
        Member(k, r)
        for k in range(1, n_splits + 1)
        for r in range(n_members)
        if split in (None, k)
    ]


def train_split(
    split: int,
    data: pd.DataFrame,
    config: dict[str, Any],
    blocks: Sequence[np.ndarray],
    fingerprint: str,
) -> list[Path]:
    """Run/resume the split's study, select Rashomon members and refit/save each of them."""
    output_dir = resolve_path(config["paths"]["output_dir"])
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]

    indices = split_indices(blocks, split)
    train = data.iloc[indices["train"]]

    study = optimize(
        train,
        tuning_folds(indices),
        config,
        split,
        output_dir / "studies" / f"split_{split}.sqlite3",
    )
    selection = select_rashomon_members(study.trials, config)
    table = rashomon_table(study.trials, selection, config)
    table.to_csv(output_dir / "studies" / f"split_{split}_rashomon.csv", index=False)
    best_loss = completed_trials(study.trials)[0].value
    set_size = len(rashomon_set(study.trials, config))
    ranks = dict(zip(table["trial"], table["rank"]))

    paths = []
    for member, (trial, in_set) in zip(members(config, split), selection):
        params = selected_params(trial, config, member.seed)
        model = make_model(config, member.seed, params)
        model.fit(train[feature_names], train[target_name])

        scores = {}
        for name in ("validation", "test"):
            subset = data.iloc[indices[name]]
            prediction = model.predict_quantiles(subset[feature_names])
            scores[name] = score_subsets(subset, prediction, config)
            path = output_dir / "predictions" / f"{member.name}_{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            prediction_frame(subset, indices[name], prediction, config).to_parquet(path, index=False)

        model.training_metadata_ = {
            "split": member.split,
            "replicate": member.replicate,
            "seed": member.seed,
            "data_fingerprint": fingerprint,
            "observation_counts": {name: len(indices[name]) for name in SPLIT_NAMES},
            "cruise_counts": {
                name: len(set(data.index.get_level_values("expocode").to_numpy()[indices[name]]))
                for name in SPLIT_NAMES
            },
            "trial_number": trial.number,
            "inner_cv_loss": trial.value,
            "loss_rank": int(ranks[trial.number]),
            "best_inner_cv_loss": best_loss,
            "rashomon_epsilon": config["ensemble"]["rashomon_epsilon"],
            "rashomon_set_size": set_size,
            "in_rashomon_set": in_set,
            "dropped_features": decode_dropped_features(trial.params["dropped_features"]),
            "params": trial.params,
            "refit_iterations": trial.user_attrs["refit_iterations"],
            "scores": scores,
        }

        model_path = output_dir / "models" / f"quantile_{member.name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        write_json(output_dir / "models" / f"quantile_{member.name}.json", model.training_metadata_)
        logger.success("Saved member {} to {}", member.name, model_path)
        paths.append(model_path)
    return paths


def summarize_ensemble_test(
    data: pd.DataFrame,
    config: dict[str, Any],
    members: Sequence[Member],
    fingerprint: str,
) -> None:
    output_dir = resolve_path(config["paths"]["output_dir"])
    paths = [output_dir / "predictions" / f"{member.name}_test.parquet" for member in members]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot summarize the held-out test set; missing member predictions: "
            + ", ".join(str(path) for path in missing)
        )

    frames = [pd.read_parquet(path).sort_values("row_id", ignore_index=True) for path in paths]
    row_ids = frames[0]["row_id"].to_numpy()
    if any(not np.array_equal(frame["row_id"].to_numpy(), row_ids) for frame in frames):
        raise ValueError("Ensemble members were evaluated on different test rows.")

    columns = prediction_columns(config)
    splits = sorted({member.split for member in members})
    # The replicate axis is the Rashomon member index within a split.
    replicates = sorted({member.replicate for member in members})
    if len(members) != len(splits) * len(replicates):
        raise ValueError("The ensemble must be a balanced splits x members design.")
    member_predictions = np.stack([frame[columns].to_numpy() for frame in frames])
    predictions = member_predictions.reshape(
        len(splits), len(replicates), *member_predictions.shape[1:]
    )
    decomposition = decompose_variance(predictions)
    prediction = decomposition["mean"]
    test = data.iloc[row_ids]

    frame = prediction_frame(test, row_ids, prediction, config)
    variance_means = {}
    for index, alpha in enumerate(quantiles(config)):
        frame[f"split_var_q_{alpha:g}"] = decomposition["split_var"][:, index]
        frame[f"hpo_var_q_{alpha:g}"] = decomposition["hpo_var"][:, index]
        variance_means[f"q_{alpha:g}"] = {
            "split_var": float(decomposition["split_var"][:, index].mean()),
            "hpo_var": float(decomposition["hpo_var"][:, index].mean()),
        }
    frame.to_parquet(output_dir / "predictions" / "ensemble_test.parquet", index=False)

    split_scores = {
        str(split): score_subsets(test, predictions[index].mean(axis=0), config)
        for index, split in enumerate(splits)
    }
    member_scores = {
        member.name: score_subsets(test, member_predictions[index], config)
        for index, member in enumerate(members)
    }
    write_json(
        output_dir / "ensemble_test_metrics.json",
        {
            "members": [member.name for member in members],
            "seeds": [member.seed for member in members],
            "data_fingerprint": fingerprint,
            "aggregation": "mean",
            "test_observations": len(row_ids),
            "scores": score_subsets(test, prediction, config),
            "variance_means": variance_means,
            "split_mean_scores": split_scores,
            "member_scores": member_scores,
        },
    )
    logger.success("Wrote held-out ensemble test metrics over {} members", len(members))


def assemble_ensemble(config: dict[str, Any], members: Sequence[Member], fingerprint: str) -> Path:
    output_dir = resolve_path(config["paths"]["output_dir"])
    model_paths = [output_dir / "models" / f"quantile_{member.name}.joblib" for member in members]
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot assemble ensemble; missing members: " + ", ".join(str(path) for path in missing)
        )
    estimators = [CatBoostResidualRegressor.load(path) for path in model_paths]
    ensemble = QuantileRegressionEnsemble(
        estimators=estimators,
        seeds=[member.seed for member in members],
        quantiles=quantiles(config),
        aggregation=config["ensemble"]["aggregation"],
        member_splits=[member.split for member in members],
        member_replicates=[member.replicate for member in members],
    )
    ensemble_path = output_dir / "models" / "quantile_ensemble.joblib"
    ensemble.save(ensemble_path)
    write_json(
        output_dir / "models" / "manifest.json",
        {
            "quantiles": quantiles(config),
            "aggregation": config["ensemble"]["aggregation"],
            "data_fingerprint": fingerprint,
            "feature_names": config["data"]["feature_names"],
            "target_name": config["data"]["target_name"],
            "members": [
                {
                    "split": member.split,
                    "replicate": member.replicate,
                    "seed": member.seed,
                    "file": path.name,
                }
                for member, path in zip(members, model_paths)
            ],
            "ensemble": ensemble_path.name,
        },
    )
    logger.success("Saved {}-member ensemble to {}", len(members), ensemble_path)
    return ensemble_path
