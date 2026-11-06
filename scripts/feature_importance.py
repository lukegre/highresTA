"""Plot CatBoost residual feature importance across saved ensemble members.

Uses PredictionValuesChange, normalised by CatBoost to sum to 100 per member.
Zero importances are masked before computing each feature's mean and standard deviation.
These importances describe the boosted residual
jointly across quantiles; they exclude the separate linear salinity baseline.

    uv run python scripts/feature_importance.py
    uv run python scripts/feature_importance.py --run-dir models/<run> --show
"""

from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger

from highres_ta.estimators import CatBoostResidualRegressor
from highres_ta.training import CONFIG_DIR, load_config, resolve_path

app = typer.Typer()


def member_importances(model_dir: Path, threads: int = 2) -> pd.DataFrame:
    """Return one row per saved member and one column per candidate feature."""
    paths = sorted(model_dir.glob("quantile_split_*_member_*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No ensemble member models found in {model_dir}.")

    rows = []
    for path in paths:
        model = CatBoostResidualRegressor.load(path)
        names = list(model.feature_names_in_)
        values = np.asarray(
            model.boosting_model_.get_feature_importance(
                type="PredictionValuesChange", thread_count=threads
            ),
            dtype=float,
        )
        if values.shape != (len(names),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid feature importances in {path}.")
        rows.append(pd.Series(values, index=names, name=path.stem.removeprefix("quantile_")))
        logger.info("Read {}", path.name)

    # Align by name; features absent from a member also contribute zero.
    return pd.DataFrame(rows).fillna(0.0).rename_axis("member")


def plot_importances(summary: pd.DataFrame, n_members: int):
    """Draw mean importance with one population standard deviation as error bars."""
    ordered = summary.dropna(subset=["mean"]).sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(ordered) + 1.8)))
    ax.barh(
        ordered.index,
        ordered["mean"],
        xerr=ordered["std"],
        color="#4575b4",
        ecolor="#333333",
        capsize=3,
    )
    ax.set_xlabel("PredictionValuesChange importance (%) · mean ± 1 std")
    ax.set_title(f"CatBoost residual feature importance across {n_members} members")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, 0.015,
        "Zero importances excluded per feature. Linear salinity baseline excluded. Std uses ddof=0.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


@app.command(help=__doc__)
def main(
    config_path: Annotated[Path, typer.Option("--config", exists=True)] = CONFIG_DIR,
    run_dir: Annotated[
        Path | None, typer.Option(help="Run folder; defaults to paths.output_dir.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Output folder; defaults to <run>/feature_importance.")
    ] = None,
    threads: Annotated[int, typer.Option(min=1, help="CatBoost threads per member.")] = 2,
    show: Annotated[bool, typer.Option(help="Also open the plot interactively.")] = False,
) -> None:
    if run_dir is None:
        run_dir = resolve_path(load_config(config_path)["paths"]["output_dir"])
    else:
        run_dir = resolve_path(run_dir)
    output_dir = (
        resolve_path(output_dir) if output_dir is not None else run_dir / "feature_importance"
    )

    importance = member_importances(run_dir / "models", threads)
    importance = importance.mask(importance.eq(0))
    summary = pd.DataFrame({
        "mean": importance.mean(axis=0),
        "std": importance.std(axis=0, ddof=0),
        "n_members": importance.count(axis=0),
    }).sort_values("mean", ascending=False).rename_axis("feature")

    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "per_member.csv")
    summary.to_csv(output_dir / "summary.csv")
    fig = plot_importances(summary, len(importance))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"feature_importance.{suffix}", dpi=200, bbox_inches="tight")
    logger.success("Saved importances for {} members to {}", len(importance), output_dir)
    print(summary.round(3).to_string())
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    app()
