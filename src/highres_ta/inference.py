import string
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from highres_ta.quantile_ensemble import QuantileRegressionEnsemble

from .features import latlon_to_spherical_coords
from .training import load_config, resolve_path


def _location(value: str) -> str:
    """Repo-relative paths become absolute; URLs pass through untouched."""
    return str(value) if "://" in str(value) else str(resolve_path(value))


def open_processed_zarr(fname: str) -> xr.Dataset:
    """Open the gridded predictors; for a multiscale pyramid, use its full-resolution level."""
    storage_options = {"client_kwargs": {"trust_env": True}} if fname.startswith("http") else None
    ds = xr.open_zarr(fname, storage_options=storage_options)
    if "multiscales" in ds.attrs:
        level = ds.attrs["multiscales"]["layout"][0]["asset"]
        ds = xr.open_zarr(fname, group=level, storage_options=storage_options)
    return ds


def open_bottomdepth(fname: str) -> xr.DataArray:
    """Open bathymetry as positive depth (m), the convention used in the training data."""
    da = xr.open_dataset(fname)
    da = da.rename({
        k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in da.dims
    })
    da = da[list(da.data_vars)[0]]
    if da.attrs.get("positive") == "up":
        da = -da
    return da


def to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Training temperatures are in °C; convert fields whose units say Kelvin."""
    if da.attrs.get("units", "").lower() in ("k", "kelvin"):
        return da - 273.15
    return da


@lru_cache(1)
def _open_gridded_data(fname_processed: str, fname_bathymetry: str) -> xr.Dataset:
    data_8D_025deg = open_processed_zarr(fname_processed)

    bottomdepth = open_bottomdepth(fname_bathymetry).interp_like(data_8D_025deg)
    coords = bottomdepth.to_series().index.to_frame()
    ncoords = latlon_to_spherical_coords(coords.lat, coords.lon).to_xarray()

    data_8D_025deg["bottomdepth"] = bottomdepth
    for key in ncoords.data_vars:
        data_8D_025deg[key] = ncoords[key]

    return data_8D_025deg.astype(np.float32)


def open_gridded_data(config: dict[str, Any] | None = None) -> xr.Dataset:
    """Gridded predictors plus bathymetry and spherical coordinates, with paths from ``config``."""
    paths = (config or load_config())["paths"]
    return _open_gridded_data(
        _location(paths["gridded_predictors"]), _location(paths["bathymetry"])
    )


def open_inference_data(
    config: dict[str, Any] | None = None, x_names: list[str] | tuple[str, ...] | None = None
) -> xr.Dataset:
    """Features named as in training. ``x_names`` defaults to ``config["data"]["feature_names"]``."""
    config = config or load_config()
    ds = open_gridded_data(config)
    sources = config["inference"]["feature_sources"]

    features = xr.Dataset()
    for name in x_names or config["data"]["feature_names"]:
        if name in sources:
            # first source wins, later ones fill its gaps
            first, *rest = (to_celsius(ds[v]) for v in sources[name])
            for da in rest:
                first = first.fillna(da)
            features[name] = first
        else:
            features[name] = ds[name]  # bottomdepth, ncoord_*
    return features


def create_labels_for_members(model: QuantileRegressionEnsemble) -> pd.MultiIndex:
    """
    Creates labels for each member of the ensemble based on their split and replicate.
    """
    if model.member_splits is None or model.member_replicates is None:
        raise ValueError("Model must have member_splits and member_replicates set.")
    replicates = [string.ascii_lowercase[r] for r in model.member_replicates]
    return pd.MultiIndex.from_arrays(
        [np.asarray(model.member_splits, dtype=int), replicates], names=["split", "replicate"]
    )


def select_inference_date(
    ds: xr.Dataset, date: str | list[str | pd.Timestamp] | pd.Timestamp
) -> pd.DataFrame:
    """
    Selects the nearest date from the dataset.
    """
    time = ds.time.sel(time=date, method="nearest")
    dayofyear = time.dt.dayofyear

    if not isinstance(date, list):
        date = [date]

    ds = ds.sel(time=date, dayofyear=dayofyear, method="nearest").drop_vars("dayofyear")
    df = ds.to_dataframe().dropna(subset="salinity")
    return df


def predict(model: QuantileRegressionEnsemble, features: xr.Dataset, date: str) -> xr.DataArray:
    """
    Predicts the target variable using the provided model and features.
    """
    logger.info(f"Selecting features for {date}")
    features_df = select_inference_date(features, date)

    logger.info(f"Predicting ensemble members for {date}")
    yhat = model.predict_members(features_df)

    da = (
        xr
        .DataArray(
            yhat,
            dims=["member", "coords", "quantile"],
            coords={
                "member": create_labels_for_members(model),
                "coords": features_df.index,
                "quantile": model.quantiles,
            },
        )
        .unstack(["member", "coords"])
        .chunk({"lat": 180, "lon": 180, "quantile": 1})
    )

    da.replicate.attrs = {
        "description": "Replicate of the model for each split, each with its own hyperparameters."
    }
    da.split.attrs = {
        "description": "Split of the model for each fold of cross-validation (shared across replicates)."
    }
    da.lat.attrs = {"units": "degrees_north", "long_name": "Latitude"}
    da.lon.attrs = {"units": "degrees_east", "long_name": "Longitude"}
    da["quantile"].attrs = {"units": "dimensionless", "long_name": "Quantile"}

    da["replicate"] = da.replicate.astype("str")
    da["split"] = da.split.astype("int32")
    # logger.debug(f"Output dims: {dict(da.sizes)}".replace("'", ""))

    return da


def calc_variance_decomposition(da: xr.DataArray) -> xr.Dataset:
    """Summarize the ensemble prediction and its sources of variance.

    The median prediction is summarized across all splits and replicates. The
    variance output contains the total ensemble variance, separate split and
    replicate components, and a variance estimate derived from the 10th-to-90th
    quantile range.
    """
    logger.info("Calculating variance decomposition")
    ensemble_dims = ["split", "replicate"]
    median_predictions = da.sel(quantile=0.5).reset_coords(drop=True)

    talk_median = median_predictions.median(dim=ensemble_dims)
    total_variance = median_predictions.var(dim=ensemble_dims)
    # replicates are nested within splits (replicate "a" of split 1 is unrelated to
    # replicate "a" of split 2), so split_var + var_replicate == total_var
    split_variance = median_predictions.mean(dim="replicate").var(dim="split")
    replicate_variance = median_predictions.var(dim="replicate").mean(dim="split")

    quantile_predictions = da.sel(quantile=[0.1, 0.9])
    quantile_range = (
        quantile_predictions
        .diff("quantile")
        .isel(quantile=0, drop=True)
        .median(dim=ensemble_dims)
        .reset_coords(drop=True)
    )
    # for a Gaussian, q90 - q10 = 2 * 1.2816 * sigma
    quantile_range_scale = 2 * 1.2816
    quantile_standard_deviation = quantile_range / quantile_range_scale
    quantile_range_variance = quantile_standard_deviation**2

    variance_components = xr.Dataset({
        "total_var": total_variance,
        "var_split": split_variance,
        "var_replicate": replicate_variance,
        "q90_minus_q10": quantile_range_variance,
    }).to_array(dim="variance_type")

    standard_dev = (
        variance_components
        .sel(variance_type=["total_var", "q90_minus_q10"])
        .sum(dim="variance_type")
        .pipe(lambda x: np.sqrt(x))
    )

    return xr.Dataset({
        "talk_median": talk_median,
        "talk_var": variance_components,
        "talk_total_stdev": standard_dev,
    })


def create_output_template(
    model: QuantileRegressionEnsemble,
    features: xr.Dataset,
    dates: list,
    zarr_fname_out: str,
    chunks: dict,
) -> None:
    """Create the output zarr (metadata only, NaN-filled) so dates can be written with ``region="auto"``.

    The layout comes from one real prediction, expanded to the full lat/lon grid of ``features`` and
    to the time steps that ``dates`` select. An existing store is kept (to resume or add dates) after
    checking that it covers every requested time step on the same grid.
    """
    times = (
        pd
        .DatetimeIndex(features.time.sel(time=list(dates), method="nearest").values)
        .unique()
        .sort_values()
    )

    if Path(zarr_fname_out).exists():
        existing = xr.open_zarr(zarr_fname_out)
        missing = times.difference(pd.DatetimeIndex(existing.time.values))
        if len(missing) > 0:
            raise ValueError(
                f"{zarr_fname_out} lacks {len(missing)} requested time steps (first: {missing[0]}); "
                "use a new output path or delete the store."
            )
        for dim in ("lat", "lon"):
            if not np.array_equal(existing[dim].values, features[dim].values):
                raise ValueError(
                    f"{zarr_fname_out} has a different {dim} grid than the inference data."
                )
        logger.info(f"Using existing output store {zarr_fname_out}")
        return

    first = predict(model, features, dates[0]).pipe(calc_variance_decomposition)
    template = xr.full_like(
        first.reindex(lat=features.lat, lon=features.lon, time=times).chunk(), np.nan
    ).chunk(chunks)
    logger.info(f"Creating output store {zarr_fname_out} with dims {dict(template.sizes)}")
    template.to_zarr(zarr_fname_out, compute=False, zarr_format=2)
