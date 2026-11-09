import unittest

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")

from highres_ta.stations import (  # noqa: E402
    _standardise,
    average_casts,
    collocate_stations,
    plot_station,
    predict_stations,
    score_stations,
)


class Member:
    linear_features = ["salinity"]

    def predict_quantiles(self, x):
        base = x["salinity"].to_numpy() * 10
        return np.column_stack([base - 1, base, base + 1])


class Ensemble:
    quantiles = np.array([0.1, 0.5, 0.9])
    estimators = [Member()]

    def get_feature_names(self):
        return ("salinity", "temperature")

    def predict_quantiles(self, x):
        return self.estimators[0].predict_quantiles(x)


class StandardiseTests(unittest.TestCase):
    def test_columns_and_drops(self):
        raw = pd.DataFrame({
            "time": ["2000-01-01T12:00:00Z", "2000-01-02", "2000-01-03", None],
            "lat": [30, 30, 30, 30], "lon": [190, 190, 190, 190], "depth": [1, 1, 1, 1],
            "salinity": [35, 35, 35, 35], "temperature": [20, 20, np.nan, 20],
            "talk": [2300, np.nan, 2300, 2300], "nitrate": [1, 1, 1, 1],
            "flag": [2, 2, 2, 2],
        })
        out = _standardise(raw, "BATS")
        self.assertEqual(len(out), 1)
        self.assertEqual(out.columns.tolist(), [
            "station", "time", "lat", "lon", "depth", "salinity_insitu",
            "temperature_insitu", "talk", "nitrate_insitu"])
        self.assertIsNone(out["time"].dt.tz)
        self.assertEqual(out["station"].iloc[0], "BATS")
        self.assertEqual(out["lon"].iloc[0], -170)

    def test_average_casts(self):
        df = pd.DataFrame({"station": "A", "time": pd.to_datetime(["2000-01-01"] * 2 + ["2000-01-02"]),
                           "talk": [2300.0, 2310.0, 2200.0], "depth": [1.0, 5.0, 2.0]})
        out = average_casts(df)
        self.assertEqual(out["talk"].tolist(), [2305.0, 2200.0])
        self.assertEqual(out.columns.tolist(), list(df.columns))


class CollocateTests(unittest.TestCase):
    def setUp(self):
        time = pd.date_range("2000-01-01", periods=4, freq="8D")  # doy 1, 9, 17, 25
        lat, lon = [10.0, 0.0, -10.0], [-20.0, 0.0, 20.0]
        sst = np.arange(4)[:, None, None] * 100 + np.arange(3)[None, :, None] * 10 + np.arange(3)[None, None, :]
        self.ds = xr.Dataset(
            {"sst": (("time", "lat", "lon"), sst),
             "clim": (("dayofyear", "lat", "lon"), np.arange(4)[:, None, None] * 1000.0 + sst[0])},
            coords={"time": time, "lat": lat, "lon": lon, "dayofyear": [5.0, 13.0, 21.0, 29.0]},
        )

    def test_nearest_values(self):
        stations = pd.DataFrame({
            "time": pd.to_datetime(["2000-01-01", "2000-01-18", "2000-01-10"]),
            "lat": [10.0, -9.0, 4.0], "lon": [-20.0, 19.0, 11.0], "talk": [1.0, 2.0, 3.0]})
        out = collocate_stations(stations, self.ds)
        # point 3: lat 4 -> index 1 (lat 0), lon 11 -> index 2 (lon 20), time -> 2000-01-09
        self.assertEqual(out["sst"].tolist(), [0, 222, 112])
        self.assertEqual(out["grid_time"].tolist(), list(pd.to_datetime(["2000-01-01", "2000-01-17", "2000-01-09"])))
        self.assertEqual(out["grid_lat"].tolist(), [10.0, -10.0, 0.0])
        self.assertEqual(out["grid_lon"].tolist(), [-20.0, 20.0, 20.0])
        # clim = 1000 * doy index + sst at time 0: doy 1 -> 0, 17 -> 2, 9 -> 1
        self.assertEqual(out["clim"].tolist(), [0, 2022, 1012])
        self.assertNotIn("dayofyear", out.columns)
        self.assertEqual(out["talk"].tolist(), [1.0, 2.0, 3.0])

    def test_without_dayofyear(self):
        stations = pd.DataFrame({"time": pd.to_datetime(["2000-01-09"]), "lat": [0.0], "lon": [0.0]})
        out = collocate_stations(stations, self.ds[["sst"]])
        self.assertEqual(out["sst"].tolist(), [111])


class PredictScoreTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "station": ["A"] * 4,
            "time": pd.date_range("2000-01-01", periods=4, freq="8D"),
            "lat": 1.0, "lon": 2.0,
            "salinity": [30.0, 31.0, 32.0, 33.0], "temperature": 15.0,
            "salinity_insitu": [35.0, 36.0, 37.0, 38.0], "temperature_insitu": 20.0,
            "talk": [300.0, np.nan, 320.0, 330.0],
        })

    def test_sources_and_columns(self):
        gridded = predict_stations(Ensemble(), self.df, "gridded")
        insitu = predict_stations(Ensemble(), self.df, "insitu")
        self.assertEqual(len(gridded), 3)
        self.assertEqual(gridded["predicted"].tolist(), [300, 320, 330])
        self.assertEqual(insitu["predicted"].tolist(), [350, 370, 380])
        self.assertEqual(set(insitu["salinity_source"]), {"insitu"})
        self.assertEqual(gridded.columns.tolist(), [
            "station", "time", "lat", "lon", "salinity_source", "observed",
            "q_0.1", "q_0.5", "q_0.9", "predicted", "residual"])
        np.testing.assert_allclose(gridded["residual"], [0, 0, 0])
        self.assertEqual(self.df["salinity"].tolist(), [30.0, 31.0, 32.0, 33.0])
        with self.assertRaises(ValueError):
            predict_stations(Ensemble(), self.df, "other")

    def test_score_and_plot(self):
        preds = pd.concat([predict_stations(Ensemble(), self.df, m) for m in ("gridded", "insitu")])
        scores = score_stations(preds, Ensemble.quantiles, {"80pct": (0.1, 0.9)})
        self.assertEqual(len(scores), 2)
        self.assertEqual(set(scores["salinity_source"]), {"gridded", "insitu"})
        self.assertIn("interval_80pct_coverage", scores.columns)
        fig, axs = plot_station(preds[preds.salinity_source == "gridded"], title="A")
        self.assertEqual(len(axs), 2)


if __name__ == "__main__":
    unittest.main()
