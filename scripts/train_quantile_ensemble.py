"""Tune and fit a quantile ensemble over data splits and HPO replicates.

The prepared data is partitioned once into ``n_parts`` stratified, cruise-grouped
blocks using a fixed ``split_random_state``. Block 0 is the test set for every
ensemble member and is never trained on, so the assembled ensemble keeps an
honest held-out score.

The ensemble is a nested two-factor design. Split ``k`` (1..n_parts-1) reserves
block ``k`` for validation and trains on the remaining blocks. Within each split,
``n_hpo_replicates`` independent Optuna studies each tune and fit one member
(seed ``100 * k + r``). Hyperparameters are tuned by leave-one-block-out
cross-validation inside the training blocks only, with folds fixed per split, so
replicates differ only through the optimisation path and model randomness. The
test-set variance decomposes into a data-split term and an HPO term.

Examples
--------
Smoke-test one member::

    uv run python scripts/train_quantile_ensemble.py --stage train --split 1 \
        --hpo-replicate 0 --n-trials 2 --optuna-jobs 1

Check the splits without training anything::

    uv run python scripts/train_quantile_ensemble.py --dry-run

Run or resume every member and assemble the ensemble::

    uv run python scripts/train_quantile_ensemble.py --stage all
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from highres_ta.training import (
    ROOT,
    assemble_ensemble,
    data_fingerprint,
    load_config,
    members,
    prepare_data,
    resolve_path,
    summarize_ensemble_test,
    train_member,
)
from highres_ta.train_test_split import outer_blocks, split_indices, validate_split, write_split

DEFAULT_CONFIG = ROOT / "scripts" / "quantile_ensemble_config.yaml"
app = typer.Typer()


class Stage(str, Enum):
    TRAIN = "train"
    ASSEMBLE = "assemble"
    ALL = "all"


@app.command(help=__doc__)
def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Path to the ensemble configuration file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = DEFAULT_CONFIG,
    stage: Annotated[
        Stage,
        typer.Option(help="Pipeline stage to run. Existing Optuna studies are resumed."),
    ] = Stage.ALL,
    split: Annotated[
        int | None,
        typer.Option(help="Run only this split (validation block 1..n_parts-1)."),
    ] = None,
    hpo_replicate: Annotated[
        int | None,
        typer.Option(help="Run only this HPO replicate within each selected split."),
    ] = None,
    n_trials: Annotated[
        int | None,
        typer.Option(help="Override the target number of completed trials per study."),
    ] = None,
    optuna_jobs: Annotated[
        int | None,
        typer.Option(help="Override the number of concurrent Optuna trials."),
    ] = None,
    maximum_iterations: Annotated[
        int | None,
        typer.Option(help="Override CatBoost's maximum iterations (useful for smoke tests)."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override the artifact directory."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Load data and validate the requested splits without training."),
    ] = False,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config = load_config(config_path)
    if maximum_iterations is not None:
        config["model"]["maximum_iterations"] = maximum_iterations
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    selected = members(config, split, hpo_replicate)
    all_members = members(config)
    target_trials = n_trials or int(config["optimization"]["n_trials"])
    resolved_optuna_jobs = optuna_jobs or int(config["optimization"]["n_jobs"])
    if target_trials < 1:
        raise ValueError("The target number of completed trials must be positive.")
    if resolved_optuna_jobs == 0:
        raise ValueError("Optuna jobs must be nonzero.")
    if config["model"]["maximum_iterations"] < 1:
        raise ValueError("CatBoost maximum iterations must be positive.")
    data = prepare_data(config)
    fingerprint = data_fingerprint(data)
    logger.info("Data fingerprint: {}", fingerprint)
    blocks = outer_blocks(data, config)
    selected_splits = sorted({member.split for member in selected})
    if dry_run:
        for index in selected_splits:
            validate_split(data, split_indices(blocks, index), index)
        logger.success("Dry run completed; no studies or models were written")
        return

    if stage in (Stage.TRAIN, Stage.ALL):
        output_dir = resolve_path(config["output_dir"])
        for index in selected_splits:
            split_dict = split_indices(blocks, index)
            validate_split(data, split_dict, index)
            write_split(output_dir, data, split_dict, index)
        for member in selected:
            train_member(
                member,
                data,
                config,
                blocks,
                target_trials,
                resolved_optuna_jobs,
                fingerprint,
            )

    filtered = split is not None or hpo_replicate is not None
    if stage is Stage.ASSEMBLE or (stage is Stage.ALL and not filtered):
        summarize_ensemble_test(data, config, all_members, fingerprint)
        assemble_ensemble(config, all_members, fingerprint)


if __name__ == "__main__":
    app()

