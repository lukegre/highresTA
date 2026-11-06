import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import typer
import xarray as xr
from loguru import logger

import highres_ta as ta

N_CONCURRENT_MODELS = 3
MAX_CPUS_PER_MODEL = 42
CHUNKS = {"time": 1, "lat": 720, "lon": 1440, "variance_type": 4}
START_DATE = "1982-01-01"
END_DATE = "2025-12-31"

typer_app = typer.Typer()


def main(
    model_path: str,
    zarr_fname_out: str,
    n_workers: int = N_CONCURRENT_MODELS,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    max_cpus: int = MAX_CPUS_PER_MODEL,
    aggregation: str | None = None,
) -> None:

    model = load_model(model_path, max_cpus=max_cpus, aggregation=aggregation)
    ds = get_inference_data(model)
    dates = ds.time.sel(time=slice(start_date, end_date)).values

    run_inference_pipeline(model, ds, zarr_fname_out, dates, n_workers=n_workers)


def load_model(
    model_path: str, max_cpus: int = MAX_CPUS_PER_MODEL, aggregation: str | None = None
) -> ta.QuantileRegressionEnsemble:
    model = ta.QuantileRegressionEnsemble.load(model_path)
    model.optimise_cpus(max_cpus=max_cpus)
    if aggregation is not None:
        model.aggregation = aggregation
    return model


def get_inference_data(model: ta.QuantileRegressionEnsemble):
    single_estimator = model.estimators[0]
    if single_estimator.feature_names is not None:
        x_names = tuple(single_estimator.feature_names)
    else:
        raise ValueError("Model does not have feature names. Cannot load inference data.")
    return ta.open_inference_data(x_names)


def process_date(
    model: ta.QuantileRegressionEnsemble,
    features_ds: xr.Dataset,
    zarr_fname_out: str,
    date: str,
    chunks: dict = CHUNKS,
    delay: int = 0,
) -> str:

    if delay > 0:
        logger.info(f"Sleeping for {delay} seconds before processing date {date}")
        time.sleep(delay)

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
    n_workers: int = N_CONCURRENT_MODELS,
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
                delay=calc_delay(i, n_workers),
            ): date
            for i, date in enumerate(dates)
        }
        for fut in as_completed(futures):
            try:
                logger.success(f"done: {fut.result()}")
            except Exception as e:
                logger.error(f"FAILED: {futures[fut]}: {e!r}")


if __name__ == "__main__":
    typer_app.command()(main)
    typer_app()
