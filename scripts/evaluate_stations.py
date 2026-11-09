"""Collocate time-series stations with the gridded predictors and evaluate a saved ensemble on them."""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import typer  # noqa: E402
from loguru import logger  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import highres_ta as ta  # noqa: E402
from highres_ta.training import CONFIG_DIR, interval_config  # noqa: E402

app = typer.Typer()


class Stage(str, Enum):
    collocate = "collocate"
    evaluate = "evaluate"
    all = "all"


class Salinity(str, Enum):
    gridded = "gridded"
    insitu = "insitu"
    both = "both"


def collocate_all(config: dict, stations: list[str]) -> bool:
    """Write one collocated parquet per station; return True if all succeeded."""
    paths, settings = config["paths"], config["validation"]
    station_dir = ta.resolve_path(paths["station_dir"])
    max_depth = settings["station_max_depth"]
    features = ta.open_inference_data(config)
    ok = True
    for name in stations:
        try:
            df = ta.STATION_LOADERS[name](
                paths["station_sources"][name],
                max_depth=max_depth.get(name, max_depth["default"]),
                cache_dir=station_dir / "raw",
            )
            df = ta.collocate_stations(ta.average_casts(df), features)
            station_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(station_dir / f"{name}_collocated.parquet")
        except Exception:
            logger.exception("Collocation failed for {}", name)
            ok = False
    return ok


def evaluate_all(config: dict, stations: list[str], model_path: Path, output_dir: Path,
                 modes: list[str], max_cpus: int) -> None:
    station_dir = ta.resolve_path(config["paths"]["station_dir"])
    data = pd.concat([pd.read_parquet(station_dir / f"{name}_collocated.parquet") for name in stations],
                     ignore_index=True)
    model = ta.QuantileRegressionEnsemble.load(model_path)
    model.optimise_cpus(max_cpus)
    predictions = pd.concat([ta.predict_stations(model, data, mode) for mode in modes], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet")
    scores = ta.score_stations(predictions, model.quantiles, interval_config(config))
    scores.to_csv(output_dir / "scores.csv", index=False)
    with PdfPages(output_dir / "stations.pdf") as pdf:
        for (station, mode), group in predictions.groupby(["station", "salinity_source"]):
            fig, _ = ta.plot_station(group, title=f"{station} ({mode} salinity)")
            pdf.savefig(fig)
            plt.close(fig)
    logger.info("Wrote predictions, scores and PDF to {}", output_dir)


@app.command(help=__doc__)
def main(
    config_path: Annotated[Path, typer.Option("--config", exists=True)] = CONFIG_DIR,
    overrides: Annotated[
        list[Path],
        typer.Option(
            "--override",
            help="YAML file deep-merged over the configuration; repeatable, applied in order.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = [],
    stage: Annotated[Stage, typer.Option(help="collocate, evaluate or both.")] = Stage.all,
    stations: Annotated[list[str] | None, typer.Option("--stations", help="Repeatable; default all configured.")] = None,
    model_path: Annotated[Path | None, typer.Option(help="Defaults to <paths.output_dir>/models/quantile_ensemble.joblib.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Defaults to <model run>/stations.")] = None,
    salinity: Annotated[Salinity, typer.Option(help="Salinity/temperature source for prediction.")] = Salinity.both,
    max_cpus: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    config = ta.load_config(config_path, overrides)
    stations = stations or list(config["paths"]["station_sources"])
    unknown = set(stations) - set(ta.STATION_LOADERS)
    if unknown:
        raise typer.BadParameter(f"Unknown stations: {sorted(unknown)}")
    if stage in (Stage.collocate, Stage.all) and not collocate_all(config, stations):
        sys.exit(1)
    if stage in (Stage.evaluate, Stage.all):
        model_path = ta.resolve_path(model_path or Path(config["paths"]["output_dir"]) / "models" / "quantile_ensemble.joblib")
        destination = ta.resolve_path(output_dir) if output_dir else model_path.parent.parent / "stations"
        modes = ["gridded", "insitu"] if salinity is Salinity.both else [salinity.value]
        evaluate_all(config, stations, model_path, destination, modes,
                     max_cpus or config["validation"]["max_cpus"])


if __name__ == "__main__":
    app()
