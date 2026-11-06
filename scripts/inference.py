import asyncio
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import aiohttp
import joblib
import typer
import xarray as xr
from loguru import logger

import highres_ta as ta
from highres_ta.training import CONFIG_DIR

N_ATTEMPTS = 3  # per date, for transient HTTP errors from the remote predictor store

typer_app = typer.Typer()


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Configuration directory (all *.yaml merged) or a single YAML file.",
            exists=True,
        ),
    ] = CONFIG_DIR,
    model_path: Annotated[str | None, typer.Option(help="Override inference.model_path.")] = None,
    zarr_fname_out: Annotated[str | None, typer.Option("--zarr-out", help="Override inference.output_zarr.")] = None,
    n_workers: Annotated[int | None, typer.Option(help="Override inference.n_workers.")] = None,
    start_date: Annotated[str | None, typer.Option(help="Override inference.start_date.")] = None,
    end_date: Annotated[str | None, typer.Option(help="Override inference.end_date.")] = None,
    max_cpus: Annotated[int | None, typer.Option(help="Override inference.max_cpus.")] = None,
    aggregation: Annotated[str | None, typer.Option(help="Override inference.aggregation (mean|median).")] = None,
) -> None:
    config = ta.load_config(config_path)
    cfg = config["inference"]
    model_path = model_path or cfg["model_path"]
    zarr_fname_out = zarr_fname_out or cfg["output_zarr"]
    n_workers = n_workers or cfg["n_workers"]
    start_date = start_date or cfg["start_date"]
    end_date = end_date or cfg["end_date"]
    max_cpus = max_cpus or cfg["max_cpus"]
    aggregation = aggregation or cfg["aggregation"]

    model = load_model(str(ta.resolve_path(model_path)), max_cpus=max_cpus, aggregation=aggregation)
    ds = get_inference_data(model, config)
    dates = ds.time.sel(time=slice(start_date, end_date)).values
    if len(dates) == 0:
        raise typer.BadParameter(f"No time steps in the predictor store between {start_date} and {end_date}.")

    zarr_fname_out = str(ta.resolve_path(zarr_fname_out))
    with joblib.parallel_config(backend="threading"):
        ta.create_output_template(model, ds, dates, zarr_fname_out, chunks=cfg["chunks"])
    run_inference_pipeline(model, ds, zarr_fname_out, dates, n_workers=n_workers, chunks=cfg["chunks"])


def load_model(
    model_path: str, max_cpus: int, aggregation: str | None = None
) -> ta.QuantileRegressionEnsemble:
    model = ta.QuantileRegressionEnsemble.load(model_path)
    model.optimise_cpus(max_cpus=max_cpus)
    if aggregation is not None:
        model.aggregation = aggregation
    return model


def get_inference_data(model: ta.QuantileRegressionEnsemble, config: dict):
    single_estimator = model.estimators[0]
    if single_estimator.feature_names is None:
        raise ValueError("Model does not have feature names. Cannot load inference data.")
    return ta.open_inference_data(config, single_estimator.feature_names)


def process_date(
    model: ta.QuantileRegressionEnsemble,
    features_ds: xr.Dataset,
    zarr_fname_out: str,
    date: str,
    chunks: dict,
    delay: int = 0,
) -> str:

    if delay > 0:
        logger.info(f"Sleeping for {delay} seconds before processing date {date}")
        time.sleep(delay)

    # Predictors are read over HTTP; retry transient network errors before giving up on a date.
    for attempt in range(1, N_ATTEMPTS + 1):
        try:
            with joblib.parallel_config(backend="threading"):
                (
                    ta
                    .predict(model, features_ds, date=date)
                    .pipe(ta.calc_variance_decomposition)
                    .chunk(chunks)
                    .to_zarr(
                        zarr_fname_out,
                        region="auto",
                        zarr_format=2,
                        align_chunks=True,
                    )
                )
            return date
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == N_ATTEMPTS:
                raise
            logger.warning(f"{date}: attempt {attempt}/{N_ATTEMPTS} failed ({e!r}); retrying")
            time.sleep(10 * attempt)


def calc_delay(i, n_workers):
    if i > n_workers:
        return 0
    else:
        return i * int(10 / n_workers)


def run_inference_pipeline(
    model: ta.QuantileRegressionEnsemble,
    ds: xr.Dataset,
    zarr_fname_out: str,
    dates: list,
    n_workers: int,
    chunks: dict,
):

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                process_date,
                model,
                ds,
                zarr_fname_out,
                date,
                chunks=chunks,
                delay=calc_delay(i, n_workers),
            ): date
            for i, date in enumerate(dates)
        }
        n_failed = 0
        for fut in as_completed(futures):
            try:
                logger.success(f"done: {fut.result()}")
            except Exception as e:
                n_failed += 1
                logger.error(f"FAILED: {futures[fut]}: {e!r}")

    logger.info(f"{len(futures) - n_failed}/{len(futures)} dates succeeded")
    if n_failed:
        logger.error(f"{n_failed} dates failed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer_app.command()(main)
    typer_app()
