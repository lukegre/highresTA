import os
import string
from functools import lru_cache

import dotenv
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from highres_ta.quantile_ensemble import QuantileRegressionEnsemble

from .features import latlon_to_spherical_coords

dotenv.load_dotenv()

FNAME_PROCESSED = os.environ.get(
    "HIGHRES_TA_FNAME_PROCESSED",
    "/net/sea/work/gregorl/projects/oceansoda_v2/data/processed/OceanSODA_ETHZv2-gridded_data-8D_025deg.zarr",
)
FNAME_BATHYMETRY = os.environ.get(
    "HIGHRES_TA_FNAME_BATHYMETRY",
    "/net/sea/work/datasets/grd/ocean/2d/obs/bathymetry/ETOPO1/ETOPO15_mean_bath.nc",
)
FNAME_NUTRIENTS = os.environ.get(
    "HIGHRES_TA_FNAME_NUTRIENTS",
    "/net/sea/work/gregorl/projects/OceanSODA-ETHZv2/data-in/carbsys/nutrients_woa18_t46y720x1440.zarr",
)


@lru_cache(1)
def open_gridded_data(fname_processed=FNAME_PROCESSED, fname_bathymetry=FNAME_BATHYMETRY):
    data_8D_025deg = xr.open_zarr(fname_processed)

    bottomdepth = (
        xr
        .open_dataarray(fname_bathymetry)
        .rename({"latitude": "lat", "longitude": "lon"})
        .interp_like(data_8D_025deg)
    )
    coords = bottomdepth.to_series().index.to_frame()
    ncoords = latlon_to_spherical_coords(coords.lat, coords.lon).to_xarray()

    data_8D_025deg["bottomdepth"] = bottomdepth
    for key in ncoords.data_vars:
        data_8D_025deg[key] = ncoords[key]

    return data_8D_025deg.astype(np.float32)


@lru_cache(1)
def open_inference_data(x_names: tuple[str, ...]):
    ds = open_gridded_data()
    x_names_list = list(x_names)

    features = xr.Dataset()
    features["salinity"] = ds.sss_cci.fillna(ds.sss_soda).fillna(ds.sss_multi_nrt)
    features["temperature"] = ds.sst_cci.fillna(ds.sst_c3s_icdr)
    features["ssh_adt"] = ds.adt_duacs
    features["bottomdepth"] = ds.bottomdepth
    features["silicate"] = ds.silicate_woa
    features["nitrate"] = ds.nitrate_woa
    features["phosphate"] = ds.phosphate_woa
    features["ncoord_x"] = ds.ncoord_x
    features["ncoord_y"] = ds.ncoord_y
    features["ncoord_z"] = ds.ncoord_z

    features = features[x_names_list]

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
    })

    return xr.Dataset({
        "talk_median": talk_median,
        "talk_var": variance_components.to_array(dim="variance_type"),
    })
