import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr

from highres_ta.partial_dependence import (
    member_sensitivity,
    perturb_column,
    sensitivity_for_date,
    spatial_sensitivity,
    summarise_members,
    training_feature_stats,
)

FEATURES = ("salinity", "temperature", "chl", "nitrate", "ncoord_x")
KWARGS = dict(
    delta_scale=0.5,
    stride=2,
    log_features=["chl"],
    nonnegative_features=["nitrate"],
    exclude_features=["ncoord_x"],
)


class FakeModel:
    """Members predict a_m . x (q10/q90 offset by -1/+1); a_m is one row of ``coefficients``."""

    quantiles = np.array([0.1, 0.5, 0.9])
    aggregation = "mean"

    def __init__(self, coefficients, ignored=None, linear_features=("salinity",)):
        self.coefficients = np.asarray(coefficients, dtype=float)
        ignored = ignored or [[] for _ in self.coefficients]
        self.estimators = [
            SimpleNamespace(
                catboost_kwargs={"ignored_features": dropped} if dropped else {},
                linear_features=list(linear_features),
            )
            for dropped in ignored
        ]
        self.calls = 0

    def get_feature_names(self):
        return FEATURES

    def predict_members(self, X):
        self.calls += 1
        values = np.nan_to_num(X[list(FEATURES)].to_numpy(dtype=float))  # trees handle NaN
        base = values @ self.coefficients.T
        return np.stack([base.T - 1, base.T, base.T + 1], axis=-1)


def coefficients(**by_feature):
    """One row per member; ``by_feature`` maps feature -> list of per-member coefficients."""
    n = len(next(iter(by_feature.values())))
    out = np.zeros((n, len(FEATURES)))
    for name, values in by_feature.items():
        out[:, FEATURES.index(name)] = values
    return out


def make_frame(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "salinity": rng.uniform(34, 36, n),
        "temperature": rng.uniform(5, 25, n),
        "chl": rng.uniform(0.05, 2, n),
        "nitrate": rng.uniform(1, 10, n),
        "ncoord_x": rng.uniform(-1, 1, n),
    })


def make_features(n=8):
    lat = np.linspace(-40, 40, n)
    lon = np.linspace(0, 70, n)
    time = pd.to_datetime(["2019-01-17", "2019-07-12"])
    dayofyear = np.array([17, 193])
    rng = np.random.default_rng(1)
    shape = (n, n)
    salinity = rng.uniform(34, 36, shape)
    salinity[0, :] = np.nan  # land
    return xr.Dataset(
        {
            "salinity": (("lat", "lon"), salinity),
            "temperature": (
                ("time", "lat", "lon"),
                np.stack([np.full(shape, 10.0), np.full(shape, 20.0)]),
            ),
            "chl": (("dayofyear", "lat", "lon"), rng.uniform(0.1, 1, (2, *shape))),
            "nitrate": (("lat", "lon"), rng.uniform(2, 8, shape)),
            "ncoord_x": (("lat", "lon"), rng.uniform(-1, 1, shape)),
        },
        coords={"lat": lat, "lon": lon, "time": time, "dayofyear": dayofyear},
    )


def make_stats(**scales):
    index = list(FEATURES)
    return pd.DataFrame(
        {
            "scale": [scales.get(name, 1.0) for name in index],
            "p01": [-1e9] * len(index),
            "p99": [1e9] * len(index),
        },
        index=index,
    )


class StatsTests(unittest.TestCase):
    def test_scale_and_percentiles(self):
        X = pd.DataFrame({"a": np.arange(101.0), "b": 10.0 ** np.arange(101) / 1e50})
        stats = training_feature_stats(X, ["a", "b"], ["b"])
        self.assertAlmostEqual(stats.loc["a", "scale"], 50 / 1.349)
        self.assertAlmostEqual(stats.loc["a", "p01"], 1.0)
        self.assertAlmostEqual(stats.loc["a", "p99"], 99.0)
        # log10(b) = i - 50, spaced by 1
        self.assertAlmostEqual(stats.loc["b", "scale"], 50 / 1.349)
        self.assertAlmostEqual(stats.loc["b", "p99"], 49.0)

    def test_log_ignores_nonpositive(self):
        X = pd.DataFrame({"c": [0.0, -1.0, np.nan, 1.0, 10.0, 100.0]})
        stats = training_feature_stats(X, ["c"], ["c"])
        self.assertTrue(np.isfinite(stats.loc["c", "scale"]))
        self.assertGreaterEqual(stats.loc["c", "p01"], 0.0)


class PerturbTests(unittest.TestCase):
    def test_additive_and_log(self):
        x = np.array([1.0, 2.0, np.nan])
        np.testing.assert_allclose(
            perturb_column(x, 0.5, +1, log=False, nonnegative=False), [1.5, 2.5, np.nan]
        )
        np.testing.assert_allclose(
            perturb_column(x, 1.0, -1, log=True, nonnegative=False), [0.1, 0.2, np.nan]
        )

    def test_nonnegative_clip(self):
        out = perturb_column(np.array([0.2, 3.0]), 0.5, -1, log=False, nonnegative=True)
        np.testing.assert_allclose(out, [0.0, 2.5])


class MemberSensitivityTests(unittest.TestCase):
    def test_linear_recovers_slope_times_delta(self):
        model = FakeModel(coefficients(temperature=[2.0, -3.0], nitrate=[0.5, 0.5]))
        X = make_frame()
        S = member_sensitivity(model, X, "temperature", 0.7, log=False, nonnegative=False)
        self.assertEqual(S.shape, (2, len(X)))
        np.testing.assert_allclose(S[0], 2.0 * 0.7)
        np.testing.assert_allclose(S[1], -3.0 * 0.7)

    def test_one_predict_call_per_feature(self):
        model = FakeModel(coefficients(temperature=[1.0, 1.0]))
        member_sensitivity(model, make_frame(), "temperature", 0.5, log=False, nonnegative=False)
        self.assertEqual(model.calls, 1)

    def test_log_feature_is_linear_in_log10(self):
        model = FakeModel(coefficients(chl=[4.0]))
        X = make_frame()
        # f = 4 chl; secant over [x 10^-d, x 10^d] per d of log10 units
        d = 0.3
        S = member_sensitivity(model, X, "chl", d, log=True, nonnegative=False)
        x = X["chl"].to_numpy()
        expected = 4.0 * x * (10**d - 10**-d) / 2
        np.testing.assert_allclose(S[0], expected, rtol=1e-5)

    def test_clipping_rescales_by_realised_step(self):
        model = FakeModel(coefficients(nitrate=[3.0]))
        X = make_frame()
        X.loc[0, "nitrate"] = 0.1  # x- clipped at 0, realised step 0.1 + delta
        S = member_sensitivity(model, X, "nitrate", 0.5, log=False, nonnegative=True)
        np.testing.assert_allclose(S[0], 3.0 * 0.5, rtol=1e-6)

    def test_nan_feature_gives_nan(self):
        model = FakeModel(coefficients(temperature=[1.0]))
        X = make_frame()
        X.loc[2, "temperature"] = np.nan
        S = member_sensitivity(model, X, "temperature", 0.5, log=False, nonnegative=False)
        self.assertTrue(np.isnan(S[0, 2]))
        self.assertEqual(np.isfinite(S[0]).sum(), len(X) - 1)

    def test_ignored_member_is_nan_not_zero(self):
        model = FakeModel(
            coefficients(temperature=[1.0, 1.0]), ignored=[["temperature"], []]
        )
        S = member_sensitivity(model, make_frame(), "temperature", 0.5, log=False, nonnegative=False)
        self.assertTrue(np.isnan(S[0]).all())
        np.testing.assert_allclose(S[1], 0.5)

    def test_ignored_index_resolved_and_linear_feature_kept(self):
        # index 1 is "temperature"; "salinity" is a linear feature the baseline still uses
        model = FakeModel(
            coefficients(temperature=[1.0, 1.0], salinity=[1.0, 1.0]),
            ignored=[[1, "salinity"], []],
        )
        X = make_frame()
        temp = member_sensitivity(model, X, "temperature", 0.5, log=False, nonnegative=False)
        sal = member_sensitivity(model, X, "salinity", 0.5, log=False, nonnegative=False)
        self.assertTrue(np.isnan(temp[0]).all())
        self.assertTrue(np.isfinite(sal).all())


class SummariseTests(unittest.TestCase):
    def test_sign_agreement_spread_and_counts(self):
        S = np.array([[1.0, 2.0, 0.0], [3.0, -2.0, 0.0], [5.0, np.nan, 0.0], [np.nan] * 3])
        out = summarise_members(S)
        np.testing.assert_allclose(out["sensitivity"], [3.0, 0.0, 0.0])
        np.testing.assert_allclose(out["sensitivity_spread"][0], np.std([1.0, 3.0, 5.0]))
        self.assertEqual(out["n_members"].tolist(), [3, 2, 3])
        self.assertAlmostEqual(out["sign_agreement"][0], 1.0)
        self.assertTrue(np.isnan(out["sign_agreement"][1]))  # median == 0

    def test_opposite_signs(self):
        out = summarise_members(np.array([[1.0], [1.0], [-1.0]]))
        self.assertAlmostEqual(out["sign_agreement"][0], 2 / 3)

    def test_all_nan_rows(self):
        out = summarise_members(np.full((2, 2), np.nan))
        self.assertTrue(np.isnan(out["sensitivity"]).all())
        self.assertTrue(np.isnan(out["sign_agreement"]).all())
        self.assertEqual(out["n_members"].tolist(), [0, 0])


class DateTests(unittest.TestCase):
    def setUp(self):
        self.features = make_features()
        self.model = FakeModel(
            coefficients(temperature=[1.0, 3.0], chl=[0.0, 0.0], nitrate=[2.0, -2.0]),
            ignored=[[], ["temperature"]],
        )
        self.stats = make_stats(temperature=2.0)

    def test_shape_coords_and_values(self):
        ds = sensitivity_for_date(
            self.model, self.features, "2019-01-17", self.stats, **KWARGS
        )
        self.assertEqual(
            set(ds.data_vars),
            {"sensitivity", "sensitivity_spread", "sign_agreement", "out_of_range"},
        )
        self.assertEqual(ds.sensitivity.dims, ("feature", "lat", "lon"))
        self.assertEqual(ds.sizes["lat"], 4)
        self.assertEqual(ds.sizes["lon"], 4)
        self.assertEqual(list(ds.feature.values), ["salinity", "temperature", "chl", "nitrate"])
        self.assertNotIn("ncoord_x", ds.feature.values)
        np.testing.assert_allclose(ds.delta.sel(feature="temperature"), 0.5 * 2.0)
        self.assertEqual(int(ds.n_members.sel(feature="temperature")), 1)
        self.assertEqual(int(ds.n_members.sel(feature="nitrate")), 2)
        self.assertEqual(ds.attrs["member_summary"], "median")
        self.assertEqual(ds.attrs["model_aggregation"], "mean")
        self.assertEqual(ds.attrs["units"], "umol kg-1")

        # only member 0 (slope 1) contributes for temperature
        temp = ds.sensitivity.sel(feature="temperature")
        np.testing.assert_allclose(temp.values[np.isfinite(temp.values)], 1.0 * 1.0)
        # nitrate: opposite members, median 0, agreement undefined, spread = 2 * delta
        nitrate_delta = float(ds.delta.sel(feature="nitrate"))
        np.testing.assert_allclose(
            ds.sensitivity.sel(feature="nitrate").values[np.isfinite(ds.sensitivity.values[3])], 0.0,
            atol=1e-5,
        )
        spread = ds.sensitivity_spread.sel(feature="nitrate").values
        np.testing.assert_allclose(spread[np.isfinite(spread)], 2.0 * nitrate_delta, rtol=1e-5)

    def test_land_is_nan_and_strided_grid(self):
        ds = sensitivity_for_date(
            self.model, self.features, "2019-01-17", self.stats, **KWARGS
        )
        expected_lat = self.features.lat.values[1::2]
        np.testing.assert_allclose(ds.lat.values, expected_lat)
        # stride offset 1 skips lat index 0 (all land), so check land via a NaN salinity cell
        features = self.features.copy(deep=True)
        features["salinity"][1, 1] = np.nan
        ds = sensitivity_for_date(self.model, features, "2019-01-17", self.stats, **KWARGS)
        self.assertTrue(np.isnan(ds.sensitivity.isel(lat=0, lon=0)).all())
        self.assertTrue(np.isfinite(ds.sensitivity.isel(lat=0, lon=1)).any())

    def test_nan_feature_cell_is_nan_for_that_feature_only(self):
        features = self.features.copy(deep=True)
        features["nitrate"][3, 3] = np.nan
        ds = sensitivity_for_date(self.model, features, "2019-01-17", self.stats, **KWARGS)
        cell = ds.isel(lat=1, lon=1)  # grid index 3 -> strided position 1
        self.assertTrue(np.isnan(cell.sensitivity.sel(feature="nitrate")))
        self.assertTrue(np.isfinite(cell.sensitivity.sel(feature="temperature")))
        self.assertTrue(np.isnan(cell.out_of_range.sel(feature="nitrate")))

    def test_out_of_range_flag(self):
        stats = self.stats.copy()
        stats.loc["temperature", ["p01", "p99"]] = [9.9, 12.0]  # date 1 temperature is 10
        ds = sensitivity_for_date(self.model, self.features, "2019-01-17", stats, **KWARGS)
        flag = ds.out_of_range.sel(feature="temperature").values
        self.assertTrue((flag[np.isfinite(flag)] == 1).all())  # 10 - 1 = 9 < 9.9
        other = ds.out_of_range.sel(feature="salinity").values
        self.assertTrue((other[np.isfinite(other)] == 0).all())

    def test_two_date_mean(self):
        stats = self.stats.copy()
        stats.loc["temperature", ["p01", "p99"]] = [0.0, 15.0]  # date 1 in range, date 2 not
        ds = spatial_sensitivity(
            self.model, self.features, ["2019-01-17", "2019-07-12"], stats, **KWARGS
        )
        self.assertNotIn("date", ds.dims)
        self.assertEqual(ds.attrs["dates"], ["2019-01-17", "2019-07-12"])
        flag = ds.out_of_range.sel(feature="temperature").values
        np.testing.assert_allclose(flag[np.isfinite(flag)], 0.5)
        self.assertEqual(ds.sensitivity.dims, ("feature", "lat", "lon"))
        self.assertEqual(int(ds.n_members.sel(feature="temperature")), 1)
        temp = ds.sensitivity.sel(feature="temperature").values
        np.testing.assert_allclose(temp[np.isfinite(temp)], 1.0)


if __name__ == "__main__":
    unittest.main()
