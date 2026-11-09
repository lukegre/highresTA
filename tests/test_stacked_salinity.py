import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import highres_ta as ta
from highres_ta.train_test_split import outer_blocks
from highres_ta.training import prepare_data

CONFIG = {
    "paths": {"training_data": "unused.pq", "talk_adjustment_csv": "unused.csv"},
    "data": {
        "target_name": "talk",
        "coordinate_columns": ["expocode", "time", "lat", "lon", "depth"],
        "quality_columns": ["talkqc", "talkf"],
        "non_null_columns": ["talk", "salinity", "temperature"],
        "minimum_salinity": 20,
        "maximum_salinity": 40,
        "engineered_feature_names": [],
        "feature_names": ["salinity", "temperature"],
        "salinity_source": None,
    },
    "cross_validation": {"n_parts": 3, "split_random_state": 0},
}


def raw_frame() -> pd.DataFrame:
    n = 12
    return pd.DataFrame({
        "expocode": [f"c{i // 2}" for i in range(n)],
        "time": pd.date_range("2020-01-01", periods=n, freq="7D"),
        "lat": 0.0, "lon": 0.0, "depth": 5.0,
        "talk": 2300.0 + np.arange(n), "talkqc": 1, "talkf": 2,
        "temperature": 15.0,
        "salinity": np.linspace(30, 36, n),
        # a: first choice, NaN at rows 0-1 and 5; b: fills row 0 only and is out of bounds at 5
        "a": [np.nan, np.nan, 31, 32, 33, np.nan, 34, 35, 45, 36, 37, 38],
        "b": [39, np.nan, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0],
        "c": 35.5,
    }).assign(salinity=lambda d: d.salinity.where(d.index != 11))  # row 11 dropped (non_null)


def prepare(salinity_source):
    config = {**CONFIG, "data": {**CONFIG["data"], "salinity_source": salinity_source}}
    with mock.patch("highres_ta.training.load_data", return_value=raw_frame()), mock.patch(
        "highres_ta.training.add_talk_adjustment", lambda df, fname: df.assign(talk_adj=0.0)
    ):
        return prepare_data(config)


class DeepMergeOverrideTest(unittest.TestCase):
    def test_override_merges_nested_and_replaces_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.yaml"
            base.write_text("paths: {x: 1, y: 2}\ndata: {names: [a, b], keep: 1}\n")
            first = Path(tmp) / "o1.yaml"
            first.write_text("paths: {x: 9}\ndata: {names: [c]}\n")
            second = Path(tmp) / "o2.yaml"
            second.write_text("model: {z: 3}\n")
            config = ta.load_config(base, overrides=[first, second])
        self.assertEqual(
            config, {"paths": {"x": 9, "y": 2}, "data": {"names": ["c"], "keep": 1}, "model": {"z": 3}}
        )

    def test_variant_not_loaded_by_default_and_keeps_n_parts(self):
        default = ta.load_config()
        self.assertIsNone(default["data"]["salinity_source"])
        variant = ta.load_config(overrides=[Path(ta.training.ROOT) / "configs/variants/stacked_salinity.yaml"])
        self.assertEqual(variant["data"]["salinity_source"], ["sss_cci", "salt_soda", "sss_multiobs"])
        self.assertEqual(variant["cross_validation"], default["cross_validation"])
        self.assertEqual(variant["data"]["feature_names"], default["data"]["feature_names"])


class StackedPrepareDataTest(unittest.TestCase):
    def test_default_output_has_no_extra_column(self):
        self.assertEqual(list(prepare(None).columns), ["salinity", "temperature", "talk"])

    def test_same_rows_and_order_as_in_situ(self):
        insitu, stacked = prepare(None), prepare(["a", "b", "c"])
        self.assertTrue(insitu.index.equals(stacked.index))
        self.assertEqual(len(stacked), 11)
        self.assertEqual(list(stacked.columns), ["salinity", "temperature", "talk", "salinity_insitu"])
        np.testing.assert_array_equal(stacked["salinity_insitu"], insitu["salinity"])

    def test_fill_order_and_fallback(self):
        stacked = prepare(["a", "b", "c"])["salinity"].to_numpy()
        self.assertEqual(stacked[0], 39)  # a missing -> b
        self.assertEqual(stacked[1], 35.5)  # a, b missing -> c
        self.assertEqual(stacked[2], 31)  # a wins over b (b is 0, out of bounds anyway)
        self.assertEqual(stacked[5], 35.5)  # a missing, b=10 out of bounds -> c
        self.assertEqual(stacked[8], 35.5)  # a=45 out of bounds, b=0 -> c

    def test_out_of_bounds_everywhere_falls_back_to_in_situ(self):
        stacked = prepare(["a", "b"])
        insitu = prepare(None)
        self.assertEqual(stacked["salinity"].iloc[8], insitu["salinity"].iloc[8])
        self.assertEqual(stacked["salinity"].iloc[5], insitu["salinity"].iloc[5])
        self.assertTrue(stacked["salinity"].between(20, 40).all())


class OuterBlocksTest(unittest.TestCase):
    def test_blocks_identical_for_insitu_and_stacked(self):
        rng = np.random.default_rng(1)
        n = 120
        insitu = pd.DataFrame({
            "expocode": [f"c{i // 4}" for i in range(n)],
            "salinity": rng.uniform(30, 37, n),
        })
        stacked = insitu.assign(salinity_insitu=insitu["salinity"], salinity=rng.uniform(30, 37, n))
        config = {"cross_validation": {"n_parts": 3, "split_random_state": 7}}
        for a, b in zip(outer_blocks(insitu, config), outer_blocks(stacked, config)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
