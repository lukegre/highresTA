import unittest

import numpy as np
import pandas as pd
import xarray as xr
from catboost import Pool

import highres_ta.explain as explain
from highres_ta.estimators import CatBoostResidualRegressor
from highres_ta.explain import (
    ensemble_shap,
    ensemble_shap_interactions,
    member_shap,
    partial_dependence_2d,
    shap_maps,
    top_pairs,
)
from highres_ta.partial_dependence import _grid_frame
from highres_ta.quantile_ensemble import QuantileRegressionEnsemble

FEATURES = ("salinity", "temperature", "nitrate", "chl", "ncoord_x", "ncoord_y")
LOSS = "MultiQuantile:alpha=0.1,0.5,0.9"
Q50 = 1


def make_data(n=400, seed=0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "salinity": rng.uniform(34, 36, n),
        "temperature": rng.uniform(-2, 2, n),
        "nitrate": rng.uniform(-2, 2, n),
        "chl": rng.uniform(0.05, 2, n),
        "ncoord_x": rng.uniform(-1, 1, n),
        "ncoord_y": rng.uniform(-1, 1, n),
    })
    y = (
        50 * X["salinity"]
        + 3 * X["temperature"] * X["nitrate"]
        + 2 * X["chl"]
        + rng.normal(0, 0.3, n)
    )
    return X, y.to_numpy()


def make_estimator(X, y, seed, **kwargs) -> CatBoostResidualRegressor:
    return CatBoostResidualRegressor(
        linear_features=("salinity",),
        feature_names=list(FEATURES),
        loss_function=LOSS,
        iterations=60,
        depth=3,
        learning_rate=0.2,
        thread_count=2,
        verbose=False,
        random_seed=seed,
        **kwargs,
    ).fit(X, y)


class ExplainCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, y = make_data()
        cls.members = [
            make_estimator(cls.X, y, 1),
            make_estimator(cls.X, y, 2, ignored_features=["ncoord_x"]),
        ]
        cls.model = QuantileRegressionEnsemble(
            cls.members, seeds=[1, 2], quantiles=[0.1, 0.5, 0.9]
        )
        cls.Xs = cls.X.iloc[:120].reset_index(drop=True)


class MemberShapTests(ExplainCase):
    def test_additive_to_member_prediction(self):
        for member in self.members:
            phi, base = member_shap(member, self.Xs, Q50, thread_count=2)
            self.assertEqual(phi.shape, (len(self.Xs), len(FEATURES)))
            np.testing.assert_allclose(
                phi.sum(axis=1) + base, member.predict_quantiles(self.Xs)[:, Q50], atol=1e-8
            )

    def test_linear_salinity_contribution(self):
        member = self.members[0]
        phi, base = member_shap(member, self.Xs, Q50)
        lin = member.linear_model_.predict(self.Xs)
        boost = member.boosting_model_.get_feature_importance(
            Pool(self.Xs[list(FEATURES)]), type="ShapValues"
        )[:, Q50, :]
        np.testing.assert_allclose(
            phi[:, 0] - boost[:, 0], lin - lin.mean(), atol=1e-8
        )
        np.testing.assert_allclose(base, boost[0, -1] + lin.mean(), atol=1e-8)
        # salinity has a slope of ~50 per unit
        slope = np.polyfit(self.Xs["salinity"], phi[:, 0], 1)[0]
        self.assertGreater(slope, 30)

    def test_background_shifts_base_only(self):
        member = self.members[0]
        background = self.X.iloc[200:300]
        phi, base = member_shap(member, self.Xs, Q50, background=background)
        phi0, base0 = member_shap(member, self.Xs, Q50)
        lin = member.linear_model_.predict
        shift = lin(self.Xs).mean() - lin(background).mean()
        np.testing.assert_allclose(phi[:, 0] - phi0[:, 0], shift, atol=1e-8)
        np.testing.assert_allclose(phi[:, 1:], phi0[:, 1:], atol=1e-10)
        np.testing.assert_allclose(base - base0, -shift, atol=1e-8)

    def test_ignored_feature_is_zero(self):
        phi, _ = member_shap(self.members[1], self.Xs, Q50)
        np.testing.assert_array_equal(phi[:, FEATURES.index("ncoord_x")], 0.0)
        self.assertGreater(np.abs(phi[:, 1]).max(), 0)  # temperature still used

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            member_shap(self.members[0], self.Xs.drop(columns="chl"), Q50)


class EnsembleShapTests(ExplainCase):
    def test_structure_and_additivity(self):
        ds = ensemble_shap(self.model, self.Xs, n_jobs=2)
        self.assertEqual(ds.shap.dims, ("row", "feature"))
        self.assertEqual(list(ds.feature.values), list(FEATURES))
        self.assertEqual(set(ds.data_vars), {"shap", "shap_spread", "value"})
        self.assertEqual(ds["base"].ndim, 0)
        np.testing.assert_allclose(
            ds.shap.sum("feature") + ds["base"], ds["prediction"], atol=1e-8
        )
        np.testing.assert_allclose(
            ds["prediction"], self.model.predict_quantiles(self.Xs)[:, Q50], atol=1e-3
        )
        np.testing.assert_allclose(ds.value.values, self.Xs[list(FEATURES)].to_numpy())

    def test_mean_and_spread_over_members(self):
        ds = ensemble_shap(self.model, self.Xs)
        per_member = np.stack([member_shap(m, self.Xs, Q50)[0] for m in self.members])
        np.testing.assert_allclose(ds.shap.values, per_member.mean(axis=0), atol=1e-10)
        np.testing.assert_allclose(ds.shap_spread.values, per_member.std(axis=0), atol=1e-10)
        # member 2 ignores ncoord_x: its zero halves the ensemble value of the other member
        np.testing.assert_allclose(
            ds.shap.sel(feature="ncoord_x").values, per_member[0, :, 4] / 2, atol=1e-10
        )

    def test_other_quantile(self):
        ds = ensemble_shap(self.model, self.Xs, quantile=0.9)
        np.testing.assert_allclose(
            ds.shap.sum("feature") + ds["base"],
            self.model.predict_quantiles(self.Xs)[:, 2],
            atol=1e-3,
        )

    def test_groups_sum_members(self):
        groups = {"location": ["ncoord_x", "ncoord_y"]}
        flat = ensemble_shap(self.model, self.Xs)
        grouped = ensemble_shap(self.model, self.Xs, groups=groups)
        self.assertEqual(
            list(grouped.feature.values), ["salinity", "temperature", "nitrate", "chl", "location"]
        )
        np.testing.assert_allclose(
            grouped.shap.sel(feature="location"),
            flat.shap.sel(feature=["ncoord_x", "ncoord_y"]).sum("feature"),
            atol=1e-10,
        )
        self.assertTrue(np.isnan(grouped.value.sel(feature="location")).all())
        self.assertTrue(np.isfinite(grouped.value.sel(feature="chl")).all())
        np.testing.assert_allclose(
            grouped.shap.sum("feature") + grouped["base"], grouped["prediction"], atol=1e-8
        )
        # spread of a group is the std of the member sums, not the sum of the stds
        per_member = np.stack([member_shap(m, self.Xs, Q50)[0] for m in self.members])
        expected = per_member[:, :, 4:6].sum(axis=2).std(axis=0)
        np.testing.assert_allclose(
            grouped.shap_spread.sel(feature="location"), expected, atol=1e-10
        )

    def test_bad_groups_raise(self):
        for groups in (
            {"g": ["nope"]},
            {"salinity": ["chl"]},
            {"g1": ["chl", "nitrate"], "g2": ["nitrate"]},
        ):
            with self.assertRaises(ValueError):
                ensemble_shap(self.model, self.Xs, groups=groups)


class InteractionTests(ExplainCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.ds = ensemble_shap_interactions(cls.model, cls.Xs, n_jobs=2)

    def test_structure(self):
        ds = self.ds
        p = len(FEATURES)
        self.assertEqual(ds.interaction.dims, ("row", "feature", "feature2"))
        self.assertEqual(ds.interaction.shape, (len(self.Xs), p, p))
        self.assertEqual(ds.strength.dims, ("feature", "feature2"))
        self.assertEqual(ds.main.dims, ("feature",))
        self.assertTrue(np.isnan(np.diag(ds.strength.values)).all())
        off = ds.strength.values[~np.eye(p, dtype=bool)]
        self.assertTrue(np.isfinite(off).all())
        np.testing.assert_allclose(ds.strength.values, ds.strength.values.T, equal_nan=True)

    def test_row_sums_equal_shap(self):
        shap = ensemble_shap(self.model, self.Xs)
        np.testing.assert_allclose(
            self.ds.interaction.sum("feature2").values, shap.shap.values, atol=1e-6
        )

    def test_strongest_pair_is_temperature_nitrate(self):
        self.assertEqual(top_pairs(self.ds.strength, 1), [("temperature", "nitrate")])

    def test_grouped_blocks(self):
        groups = {"location": ["ncoord_x", "ncoord_y"]}
        grouped = ensemble_shap_interactions(self.model, self.Xs, groups=groups)
        flat = self.ds.interaction
        np.testing.assert_allclose(
            grouped.interaction.sel(feature="location", feature2="location"),
            flat.sel(feature=["ncoord_x", "ncoord_y"], feature2=["ncoord_x", "ncoord_y"]).sum(
                ["feature", "feature2"]
            ),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            grouped.interaction.sel(feature="chl", feature2="location"),
            flat.sel(feature="chl", feature2=["ncoord_x", "ncoord_y"]).sum("feature2"),
            atol=1e-10,
        )
        shap = ensemble_shap(self.model, self.Xs, groups=groups)
        np.testing.assert_allclose(
            grouped.interaction.sum("feature2").values, shap.shap.values, atol=1e-6
        )


class TopPairsTests(unittest.TestCase):
    def test_order_unique_and_k(self):
        names = ["a", "b", "c"]
        values = np.array([[np.nan, 1.0, 5.0], [1.0, np.nan, 3.0], [5.0, 3.0, np.nan]])
        strength = xr.DataArray(
            values, dims=("feature", "feature2"), coords={"feature": names, "feature2": names}
        )
        self.assertEqual(top_pairs(strength, 2), [("a", "c"), ("b", "c")])
        self.assertEqual(len(top_pairs(strength, 10)), 3)
        self.assertEqual(top_pairs(strength, 0), [])


class PartialDependence2DTests(ExplainCase):
    def test_shape_grids_and_interaction(self):
        bg = self.X.iloc[:100]
        ds = partial_dependence_2d(self.model, bg, "temperature", "nitrate", n_grid=7)
        self.assertEqual(ds.pd.dims, ("a_value", "b_value"))
        self.assertEqual(ds.pd.shape, (7, 7))
        self.assertEqual(ds.pd_a.dims, ("a_value",))
        self.assertEqual(ds.pd_b.dims, ("b_value",))
        self.assertEqual(ds.interaction.shape, (7, 7))
        self.assertEqual((ds.attrs["a"], ds.attrs["b"]), ("temperature", "nitrate"))
        lo, hi = np.percentile(bg["temperature"], [2, 98])
        np.testing.assert_allclose(ds.a_value.values, np.linspace(lo, hi, 7))
        self.assertAlmostEqual(float(ds.interaction.mean()), 0.0, places=8)

        # product term: pd(a, b) has the saddle shape 3 a b
        corners = ds.pd.values[[0, -1]][:, [0, -1]]
        self.assertGreater(corners[0, 0], corners[0, 1])
        self.assertGreater(corners[1, 1], corners[1, 0])
        self.assertGreater(ds.attrs["h_squared"], 0.3)

    def test_additive_pair_has_small_h2(self):
        bg = self.X.iloc[:100]
        inter = partial_dependence_2d(self.model, bg, "temperature", "nitrate", n_grid=6)
        additive = partial_dependence_2d(self.model, bg, "salinity", "chl", n_grid=6)
        self.assertLess(additive.attrs["h_squared"], 0.1)
        self.assertGreater(inter.attrs["h_squared"], 3 * additive.attrs["h_squared"])
        # salinity main effect of ~50 per unit shows up in pd_a
        slope = np.polyfit(additive.a_value, additive.pd_a, 1)[0]
        self.assertGreater(slope, 30)

    def test_log_grid_and_chunking(self):
        bg = self.X.iloc[:50]
        ds = partial_dependence_2d(
            self.model, bg, "chl", "temperature", n_grid=5, log_features=["chl"]
        )
        ratios = ds.a_value.values[1:] / ds.a_value.values[:-1]
        np.testing.assert_allclose(ratios, ratios[0])
        self.assertEqual(ds.attrs["log_features"], ["chl"])

        original = explain.MAX_PREDICT_ROWS
        try:
            explain.MAX_PREDICT_ROWS = 120  # a few grid points per chunk
            chunked = partial_dependence_2d(
                self.model, bg, "chl", "temperature", n_grid=5, log_features=["chl"]
            )
        finally:
            explain.MAX_PREDICT_ROWS = original
        np.testing.assert_allclose(chunked.pd.values, ds.pd.values)
        np.testing.assert_allclose(chunked.pd_a.values, ds.pd_a.values)

    def test_invalid_features(self):
        with self.assertRaises(ValueError):
            partial_dependence_2d(self.model, self.X, "chl", "chl")
        with self.assertRaises(ValueError):
            partial_dependence_2d(self.model, self.X, "chl", "unknown")


def make_features(n=8) -> xr.Dataset:
    lat = np.linspace(-40, 40, n)
    lon = np.linspace(0, 70, n)
    time = pd.to_datetime(["2019-01-17", "2019-07-12"])
    rng = np.random.default_rng(3)
    shape = (n, n)
    salinity = rng.uniform(34, 36, shape)
    salinity[1, 1] = np.nan  # land cell that survives the stride-2 offset
    return xr.Dataset(
        {
            "salinity": (("lat", "lon"), salinity),
            "temperature": (
                ("time", "lat", "lon"),
                np.stack([rng.uniform(-2, 2, shape), rng.uniform(-2, 2, shape)]),
            ),
            "nitrate": (("lat", "lon"), rng.uniform(-2, 2, shape)),
            "chl": (("dayofyear", "lat", "lon"), rng.uniform(0.1, 1, (2, *shape))),
            "ncoord_x": (("lat", "lon"), rng.uniform(-1, 1, shape)),
            "ncoord_y": (("lat", "lon"), rng.uniform(-1, 1, shape)),
        },
        coords={"lat": lat, "lon": lon, "time": time, "dayofyear": np.array([17, 193])},
    )


class ShapMapTests(ExplainCase):
    def setUp(self):
        self.features = make_features()
        self.groups = {"location": ["ncoord_x", "ncoord_y"]}

    def test_single_date_shape_land_and_additivity(self):
        ds = shap_maps(
            self.model, self.features, ["2019-01-17"], stride=2, groups=self.groups
        )
        self.assertEqual(ds.shap.dims, ("feature", "lat", "lon"))
        self.assertEqual(ds.sizes["lat"], 4)
        self.assertEqual(ds.sizes["lon"], 4)
        self.assertEqual(
            list(ds.feature.values), ["salinity", "temperature", "nitrate", "chl", "location"]
        )
        np.testing.assert_allclose(ds.lat.values, self.features.lat.values[1::2])
        # (1, 1) of the full grid is strided position (0, 0)
        self.assertTrue(np.isnan(ds.shap.isel(lat=0, lon=0)).all())
        self.assertTrue(np.isfinite(ds.shap.isel(lat=0, lon=1)).all())

        df, _, _ = _grid_frame(self.features, "2019-01-17", 2)
        self.assertEqual(len(df), 15)
        prediction = self.model.predict_quantiles(df[list(FEATURES)])[:, Q50]
        total = ds.shap.sum("feature", skipna=False) + ds.attrs["base"]
        stacked = total.stack(cell=("lat", "lon")).to_series().dropna()
        np.testing.assert_allclose(
            stacked.reindex(df.index).to_numpy(), prediction, atol=1e-3
        )

    def test_two_date_mean_and_attrs(self):
        dates = ["2019-01-17", "2019-07-12"]
        both = shap_maps(self.model, self.features, dates, stride=2)
        first = shap_maps(self.model, self.features, dates[:1], stride=2)
        second = shap_maps(self.model, self.features, dates[1:], stride=2)
        self.assertNotIn("date", both.dims)
        self.assertEqual(both.attrs["dates"], dates)
        self.assertEqual(both.attrs["quantile"], 0.5)
        self.assertAlmostEqual(
            both.attrs["base"], (first.attrs["base"] + second.attrs["base"]) / 2
        )
        np.testing.assert_allclose(
            both.shap.values, (first.shap.values + second.shap.values) / 2, atol=1e-10
        )
        self.assertEqual(both.shap.dims, ("feature", "lat", "lon"))
        self.assertEqual(list(both.feature.values), list(FEATURES))


if __name__ == "__main__":
    unittest.main()
