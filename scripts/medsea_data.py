"""Add CARIMED Mediterranean bottle data (Jiang et al. 2022, NCEI 0309255) to the training data.

Stages:
  collocate  download CARIMED, keep surface TA samples in GLODAP column names, match them to the
             gridded predictors (nearest neighbour) and write ``paths.carimed_collocated``.
  combine    append those rows to ``paths.glodap_training_data`` and write ``paths.training_data``.

The matchups come from the OceanSODA-ETHZv2 store (``paths.gridded_predictors``), whereas the
GLODAP rows were matched to the original product files, so ``chl_globcolour`` (CCI gap-filled here)
and the SODA fields are near but not identical products. In situ features come from the bottles.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pooch
import typer
import xarray as xr
from loguru import logger

import highres_ta as ta
from highres_ta.training import CONFIG_DIR

app = typer.Typer()

MAX_PRESSURE = 30.0  # dbar; the GLODAP training rows go down to ~30 m
FIRST_YEAR = 1982  # start of the satellite era

# CARIMED variable -> GLODAP training column (2QC = second-QC, already adjusted values)
CARIMED_COLUMNS = {
    "EXPOCODE": "expocode",
    "time": "time",
    "latitude": "lat",
    "longitude": "lon",
    "DEPTH": "bottomdepth",
    "MAXSAMPDEPTH": "maxsampdepth",
    "pressure": "depth",  # dbar ~ m at <= 30 dbar
    "CTDTMP": "temperature",
    "CTDSAL_2QC": "salinity",
    "OXYGEN_2QC": "oxygen",
    "AOU": "aou",
    "NITRAT_2QC": "nitrate",
    "NITRIT": "nitrite",
    "SILCAT_2QC": "silicate",
    "PHSPHT_2QC": "phosphate",
    "TCARBN_2QC": "tco2",
    "PH_TOT_IS_2QC": "phtsinsitutp",
    "ALKALI_2QC": "talk",
    "ALKALI_FLAG_W": "talkf",
}

# gridded store variable -> GLODAP column holding the equivalent matchup (raw units, as in GLODAP)
GRIDDED_COLUMNS = {
    "sst_cci": "sst_cci",
    "sss_cci": "sss_cci",
    "sss_soda": "salt_soda",
    "sss_multi_nrt": "sss_multiobs",
    "mld_soda": "mld_dens_soda",
    "adt_duacs": "ssh_adt",
    "chl_cci_gapfilled": "chl_globcolour",
    "bottomdepth": "bottomdepth_grid",
}


class Stage(str, Enum):
    collocate = "collocate"
    combine = "combine"
    all = "all"


def load_carimed(url: str, cache_dir: Path) -> pd.DataFrame:
    """Surface (<= MAX_PRESSURE dbar) TA samples with flag 2 from FIRST_YEAR, as GLODAP columns."""
    fname = pooch.retrieve(url, known_hash=None, fname=url.split("/")[-1], path=cache_dir)
    df = (
        xr.open_dataset(fname)
        .reset_coords()[list(CARIMED_COLUMNS)]
        .to_dataframe()
        .reset_index(drop=True)
        .rename(columns=CARIMED_COLUMNS)
    )
    keep = (
        (df["depth"] <= MAX_PRESSURE)
        & df["talk"].notna()
        & (df["talkf"] == 2)
        & (df["time"].dt.year >= FIRST_YEAR)
    )
    df = df[keep].assign(
        lon=lambda x: (x.lon + 180) % 360 - 180,
        talkqc=np.nan,
        year=lambda x: x.time.dt.year.astype("int32"),
    )
    logger.info("CARIMED: {:,} surface TA samples from {} cruises", len(df), df["expocode"].nunique())
    return df.reset_index(drop=True)


def drop_glodap_cruises(df: pd.DataFrame, glodap: pd.DataFrame) -> pd.DataFrame:
    """Drop cruises that GLODAP already contains (GLODAP wins)."""
    overlap = sorted(set(df["expocode"]) & set(glodap["expocode"]))
    logger.info("Dropping {} cruises already in GLODAP: {}", len(overlap), ", ".join(overlap))
    return df[~df["expocode"].isin(overlap)].reset_index(drop=True)


def collocate(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Nearest gridded value at each sample, under the GLODAP matchup column names."""
    ds = ta.open_gridded_data(config)
    missing = [name for name in GRIDDED_COLUMNS if name not in ds]
    if missing:
        logger.warning("Not in the gridded store, left empty: {}", ", ".join(missing))
    available = {k: v for k, v in GRIDDED_COLUMNS.items() if k in ds}
    df = ta.collocate_stations(df, ds[list(available)].rename(available))
    df["bottomdepth"] = df["bottomdepth"].fillna(df.pop("bottomdepth_grid"))
    for column in set(GRIDDED_COLUMNS.values()) - set(available.values()) - {"bottomdepth_grid"}:
        df[column] = np.nan
    return df


@app.command(help=__doc__)
def main(
    config_path: Annotated[Path, typer.Option("--config", exists=True)] = CONFIG_DIR,
    stage: Annotated[Stage, typer.Option()] = Stage.all,
) -> None:
    config = ta.load_config(config_path)
    paths = config["paths"]
    glodap_path = ta.resolve_path(paths["glodap_training_data"])
    carimed_path = ta.resolve_path(paths["carimed_collocated"])

    if stage in (Stage.collocate, Stage.all):
        glodap = pd.read_parquet(glodap_path, columns=["expocode"])
        df = load_carimed(paths["carimed_source"], ta.resolve_path(paths["raw_dir"]))
        df = collocate(drop_glodap_cruises(df, glodap), config)
        carimed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(carimed_path)
        logger.info("Wrote {:,} collocated CARIMED rows to {}", len(df), carimed_path)

    if stage in (Stage.combine, Stage.all):
        glodap = pd.read_parquet(glodap_path).assign(source="GLODAPv2023")
        carimed = pd.read_parquet(carimed_path).assign(source="CARIMED")
        combined = pd.concat([glodap, carimed], ignore_index=True)
        out = ta.resolve_path(paths["training_data"])
        combined.to_parquet(out)
        logger.info("Wrote {:,} rows ({:,} CARIMED) to {}", len(combined), len(carimed), out)


if __name__ == "__main__":
    app()
