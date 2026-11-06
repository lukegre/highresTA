"""Tune and fit a quantile ensemble over data splits and Rashomon members.

The prepared data is partitioned once into ``n_parts`` stratified, cruise-grouped
blocks using a fixed ``split_random_state``. Block 0 is the test set for every
ensemble member and is never trained on, so the assembled ensemble keeps an
honest held-out score.

Split ``k`` (1..n_parts-1) reserves block ``k`` for validation and trains on the
remaining blocks. Each split runs one Optuna study (sampler seed ``100 * k``),
tuned by leave-one-block-out cross-validation inside the training blocks only.
Besides CatBoost hyperparameters, each trial may ignore 0..``max_dropped_features``
non-protected features. Unpromising trials are pruned. From the finished study,
``n_members_per_split`` members are chosen from the Rashomon set (trials within
``rashomon_epsilon`` of the best inner-CV loss), greedily maximising diversity in
dropped features and hyperparameters, and each is refitted (seed ``100 * k + r``).
The test-set variance decomposes into a data-split term and a within-split
(Rashomon member) term.

Examples
--------
Smoke-test one split::

    uv run python scripts/train_quantile_ensemble.py --stage train --split 1 \
        --n-trials 8 --maximum-iterations 20 --parallel-trials 1 --threads-per-trial 2

Check the splits without training anything::

    uv run python scripts/train_quantile_ensemble.py --dry-run

Run or resume every split and assemble the ensemble::

    uv run python scripts/train_quantile_ensemble.py --stage all
"""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from highres_ta.training import (
    CONFIG_DIR,
    assemble_ensemble,
    data_fingerprint,
    load_config,
    members,
    prepare_data,
    resolve_path,
    summarize_ensemble_test,
    train_split,
)
from highres_ta.train_test_split import outer_blocks, split_indices, validate_split, write_split

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
            help="Configuration directory (all *.yaml merged) or a single YAML file.",
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
        ),
    ] = CONFIG_DIR,
    stage: Annotated[
        Stage,
        typer.Option(help="Pipeline stage to run. Existing Optuna studies are resumed."),
    ] = Stage.ALL,
    split: Annotated[
        int | None,
        typer.Option(help="Run only this split (validation block 1..n_parts-1)."),
    ] = None,
    n_trials: Annotated[
        int | None,
        typer.Option(help="Override the target number of completed trials per split's study."),
    ] = None,
    parallel_trials: Annotated[
        int | None,
        typer.Option(help="Override the number of concurrent Optuna trials per study."),
    ] = None,
    threads_per_trial: Annotated[
        int | None,
        typer.Option(help="Override the number of CatBoost threads per trial."),
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
        config["paths"]["output_dir"] = str(output_dir)
    if parallel_trials is not None:
        config["optimization"]["parallel_trials"] = parallel_trials
    if threads_per_trial is not None:
        config["optimization"]["threads_per_trial"] = threads_per_trial
    if n_trials is not None:
        config["optimization"]["n_trials"] = n_trials
    all_members = members(config)
    selected_splits = sorted({member.split for member in members(config, split)})
    if config["optimization"]["n_trials"] < 1:
        raise ValueError("The target number of completed trials must be positive.")
    if config["optimization"]["n_trials"] < config["ensemble"]["n_members_per_split"]:
        raise ValueError(
            "optimization.n_trials must be at least ensemble.n_members_per_split "
            f"({config['ensemble']['n_members_per_split']}) to select that many members."
        )
    if config["ensemble"]["rashomon_epsilon"] < 0:
        raise ValueError("ensemble.rashomon_epsilon must not be negative.")
    for key in ("parallel_trials", "threads_per_trial"):
        if config["optimization"][key] < 1:
            raise ValueError(f"optimization.{key} must be at least 1.")
    if config["model"]["maximum_iterations"] < 1:
        raise ValueError("CatBoost maximum iterations must be positive.")
    data = prepare_data(config)
    fingerprint = data_fingerprint(data)
    logger.info("Data fingerprint: {}", fingerprint)
    blocks = outer_blocks(data, config)
    if dry_run:
        for index in selected_splits:
            validate_split(data, split_indices(blocks, index), index)
        logger.success("Dry run completed; no studies or models were written")
        return

    if stage in (Stage.TRAIN, Stage.ALL):
        output_dir = resolve_path(config["paths"]["output_dir"])
        for index in selected_splits:
            split_dict = split_indices(blocks, index)
            validate_split(data, split_dict, index)
            write_split(output_dir, data, split_dict, index)
        for index in selected_splits:
            train_split(index, data, config, blocks, fingerprint)

    filtered = split is not None
    if stage is Stage.ASSEMBLE or (stage is Stage.ALL and not filtered):
        summarize_ensemble_test(data, config, all_members, fingerprint)
        assemble_ensemble(config, all_members, fingerprint)


if __name__ == "__main__":
    app()

