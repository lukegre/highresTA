"""Quantile-ensemble training: data preparation, HPO, member fitting and assembly."""

import hashlib
import json
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


@dataclass(frozen=True)
class Member:
    split: int
    replicate: int

    @property
    def seed(self) -> int:
        return 100 * self.split + self.replicate

    @property
    def name(self) -> str:
        return f"split_{self.split}_hpo_{self.replicate}"



def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Configuration in {path} must be a mapping.")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
    required_columns = list(
        dict.fromkeys(coordinate_columns + raw_feature_names + [target_name] + quality_columns)
    )

    if show_all_columns:
        logger.info(
            "\n" + str(load_data(str(resolve_path(data_config["parquet_glob"]))).columns)
        )

    data = (
        load_data(str(resolve_path(data_config["parquet_glob"])))[required_columns]
        .set_index(coordinate_columns, drop=False)
        .pipe(
            drop_extreme_salinities,
            salinity_col_name=salinity_col_name,
            min=data_config["minimum_salinity"],
            max=data_config["maximum_salinity"],
        )
        .pipe(
            add_talk_adjustment,
            fname=str(resolve_path(data_config["talk_adjustment_csv"])),
        )
        .pipe(drop_bad_quality_talk)
        .dropna(subset=data_config["non_null_columns"])
        .drop_duplicates(subset=coordinate_columns, keep="first")
        .select_dtypes(include=[np.number])
        .pipe(add_cyclical_dayofyear)
        .pipe(add_spherical_coords)
        .loc[:, feature_names + [target_name]]
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
        "thread_count": model_config["trial_thread_count"],
        "verbose": False,
        "early_stopping_rounds": model_config["early_stopping_rounds"],
    }


def suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "depth": trial.suggest_int("depth", 4, 8),
        "random_strength": trial.suggest_float("random_strength", 1.0, 10.0),
    }


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
        params = fixed_params | suggest_parameters(trial)
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
    member: Member,
    storage_path: Path,
    study_name: str,
    target_completed_trials: int,
    optuna_jobs: int,
) -> optuna.Study:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=member.seed, n_startup_trials=20),
        storage=f"sqlite:///{storage_path}",
        study_name=study_name,
        load_if_exists=True,
    )
    fold_digest = hashlib.sha256()
    for train_index, validation_index in folds:
        fold_digest.update(np.asarray(train_index, dtype=np.int64).tobytes())
        fold_digest.update(np.asarray(validation_index, dtype=np.int64).tobytes())
    signature_payload = {
        "data_fingerprint": data_fingerprint(data),
        "fold_digest": fold_digest.hexdigest(),
        "features": config["data"]["feature_names"],
        "target": config["data"]["target_name"],
        "model": config["model"],
        "split": member.split,
        "replicate": member.replicate,
        "seed": member.seed,
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
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, target_completed_trials - completed)
    logger.info(
        "Study {} has {}/{} completed trials; running {}",
        study_name,
        completed,
        target_completed_trials,
        remaining,
    )
    if remaining:
        study.optimize(
            objective_factory(data, folds, config, member.seed),
            n_trials=remaining,
            n_jobs=optuna_jobs,
            show_progress_bar=optuna_jobs == 1,
        )
    return study


def selected_params(
    study: optuna.Study, config: dict[str, Any], seed: int, final_fit: bool
) -> dict[str, Any]:
    params = fixed_catboost_params(config, seed) | study.best_trial.params
    params["iterations"] = study.best_trial.user_attrs["refit_iterations"]
    params.pop("early_stopping_rounds", None)
    if final_fit:
        params["thread_count"] = config["model"]["final_thread_count"]
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
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    temporary_path.replace(path)


def members(
    config: dict[str, Any], split: int | None = None, replicate: int | None = None
) -> list[Member]:
    n_splits = int(config["cross_validation"]["n_parts"]) - 1
    n_replicates = int(config["ensemble"]["n_hpo_replicates"])
    if split is not None and not 1 <= split <= n_splits:
        raise ValueError(f"Split {split} is outside 1..{n_splits}.")
    if replicate is not None and not 0 <= replicate < n_replicates:
        raise ValueError(f"HPO replicate {replicate} is outside 0..{n_replicates - 1}.")
    return [
        Member(k, r)
        for k in range(1, n_splits + 1)
        for r in range(n_replicates)
        if split in (None, k) and replicate in (None, r)
    ]



def train_member(
    member: Member,
    data: pd.DataFrame,
    config: dict[str, Any],
    blocks: Sequence[np.ndarray],
    target_trials: int,
    optuna_jobs: int,
    fingerprint: str,
) -> Path:
    output_dir = resolve_path(config["output_dir"])
    feature_names = config["data"]["feature_names"]
    target_name = config["data"]["target_name"]

    split = split_indices(blocks, member.split)
    train = data.iloc[split["train"]]

    study = optimize(
        train,
        tuning_folds(split),
        config,
        member,
        output_dir / "studies" / f"{member.name}.sqlite3",
        f"catboost_quantile_{member.name}",
        target_trials,
        optuna_jobs,
    )
    model = make_model(
        config, member.seed, selected_params(study, config, member.seed, final_fit=True)
    )
    model.fit(train[feature_names], train[target_name])

    scores = {}
    for name in ("validation", "test"):
        subset = data.iloc[split[name]]
        prediction = model.predict_quantiles(subset[feature_names])
        scores[name] = score_subsets(subset, prediction, config)
        path = output_dir / "predictions" / f"{member.name}_{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction_frame(subset, split[name], prediction, config).to_parquet(path, index=False)

    model.training_metadata_ = {
        "split": member.split,
        "replicate": member.replicate,
        "seed": member.seed,
        "data_fingerprint": fingerprint,
        "observation_counts": {name: len(split[name]) for name in SPLIT_NAMES},
        "cruise_counts": {
            name: len(set(data.index.get_level_values("expocode").to_numpy()[split[name]]))
            for name in SPLIT_NAMES
        },
        "best_trial": study.best_trial.number,
        "best_inner_cv_loss": study.best_value,
        "best_params": study.best_trial.params,
        "refit_iterations": study.best_trial.user_attrs["refit_iterations"],
        "scores": scores,
    }

    model_path = output_dir / "models" / f"quantile_{member.name}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    write_json(output_dir / "models" / f"quantile_{member.name}.json", model.training_metadata_)
    logger.success("Saved member {} to {}", member.name, model_path)
    return model_path


def summarize_ensemble_test(
    data: pd.DataFrame,
    config: dict[str, Any],
    members: Sequence[Member],
    fingerprint: str,
) -> None:
    output_dir = resolve_path(config["output_dir"])
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
    replicates = sorted({member.replicate for member in members})
    if len(members) != len(splits) * len(replicates):
        raise ValueError("The ensemble must be a balanced splits x replicates design.")
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
    output_dir = resolve_path(config["output_dir"])
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
