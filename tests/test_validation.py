import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from highres_ta.validation import evaluate_model, load_splits, salinity_penalty, swap_salinity


class Member:
    def __init__(self, value):
        self.value = value

    def predict_quantiles(self, x):
        return np.full((len(x), 1), self.value)


class Ensemble:
    quantiles = np.array([0.5])
    member_splits = [1, 2]
    estimators = [Member(10), Member(100)]


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.raw = pd.DataFrame({"expocode": list("abcd"), "lat": [0, 70, 10, 20],
                                 "lon": [1, 2, 3, 4], "salinity": [30., 31., 32., 33.],
                                 "alt": [30., np.nan, 100., 34.]})
        self.data = self.raw.set_index(["expocode", "lat", "lon"])[["salinity"]].copy()
        self.data["bottomdepth"] = [100, 600, 700, 200]
        self.data["target"] = [0., 1., 2., 3.]
        self.data["optional"] = np.nan
        self.config = {"data": {"feature_names": ["salinity", "optional"], "target_name": "target",
                                "non_null_columns": ["salinity", "target"]}}
        self.paths = []
        for split_id, labels in [(1, ["test", "validation", "train", "validation"]),
                                 (2, ["test", "train", "validation", "train"])]:
            path = Path(self.directory.name) / f"split_{split_id}.parquet"
            pd.DataFrame({"row_id": [0, 1, 2, 3], "split": labels,
                          "expocode": list("abcd")}).to_parquet(path)
            self.paths.append(path)

    def test_test_uses_all_members_validation_only_heldout_members(self):
        pred, scores = evaluate_model(Ensemble(), self.data, self.config, self.paths,
                                      {"salinity": self.data})
        self.assertEqual(pred.loc[pred.dataset == "test", "predicted"].tolist(), [55])
        self.assertEqual(pred.loc[pred.dataset == "validation", "predicted"].tolist(), [10, 10, 100])
        self.assertEqual(pred.loc[pred.dataset == "validation", "row_id"].tolist(), [1, 3, 2])
        self.assertEqual(scores.query("dataset == 'test' and region == 'oceanic'").iloc[0]["n"], 0)

    def test_reordered_input_and_invalid_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expocodes"):
            load_splits(self.paths, self.data.iloc[::-1], Ensemble())
        frame = pd.read_parquet(self.paths[0])
        frame.loc[0, "row_id"] = 1
        frame.to_parquet(self.paths[0])
        with self.assertRaisesRegex(ValueError, "row IDs"):
            load_splits(self.paths, self.data, Ensemble())

    def test_alternate_preserves_positions_and_empty_regions(self):
        alt = swap_salinity(self.data, self.raw, "alt", ["expocode", "lat", "lon"], 20, 40)
        self.assertTrue(alt.index.equals(self.data.index))
        self.assertEqual(alt.salinity.isna().tolist(), [False, True, True, False])
        pred, scores = evaluate_model(Ensemble(), self.data, self.config, self.paths, {"alt": alt})
        self.assertEqual(pred.row_id.tolist(), [0, 3])
        self.assertEqual(pred.salinity_delta.tolist(), [0, 1])
        valid = scores.query("dataset == 'validation'")
        empty = valid.query("region == 'oceanic'")
        self.assertEqual(empty.iloc[0]["n"], 0)
        self.assertTrue(np.isnan(empty.iloc[0].mean_bias))

    def test_stacked_salinity_fills_in_order_with_bounds(self):
        raw = self.raw.assign(a=[np.nan, 31.5, np.nan, 99.], b=[30.5, 32.5, np.nan, 33.5],
                              c=[1., 2., 36., 3.])
        out = swap_salinity(self.data, raw, ["a", "b", "c"], ["expocode", "lat", "lon"], 20, 40)
        # row 3: a=99 is out of bounds, so it falls through to b (as in training.stack_salinity)
        self.assertEqual(out.salinity.tolist(), [30.5, 31.5, 36., 33.5])

    def test_salinity_penalty_is_paired(self):
        def frame(name, predicted, delta, rows=(0, 1, 2, 3)):
            return pd.DataFrame({"dataset": "test", "row_id": list(rows), "salinity_product": name,
                                 "observed": [0.] * len(rows), "predicted": predicted,
                                 "salinity_delta": delta, "lat": [0, 70, 10, 20][:len(rows)] if len(rows) == 4 else [0, 10, 20],
                                 "bottomdepth": [100, 600, 700, 200][:len(rows)] if len(rows) == 4 else [100, 700, 200]})
        ref = frame("salinity", [3., 3., 3., 3.], 0.)
        same = frame("same", [3., 3., 3., 3.], 0.05)
        # missing row 1 (and a different, huge prediction there in the reference must not matter)
        ref.loc[1, "predicted"] = 100.
        part = frame("part", [5., 5., 5.], 0.3, rows=(0, 2, 3))
        pen = salinity_penalty(pd.concat([ref, same, part], ignore_index=True))
        self.assertEqual(set(pen.group_type), {"region", "abs_salinity_delta"})
        row = pen.query("salinity_product == 'same' and group == 'all'").iloc[0]
        self.assertEqual(row.n, 4)
        self.assertAlmostEqual(row.rms_prediction_shift, 97 * 0 + np.sqrt(97 ** 2 / 4))  # row 1 shifted by -97
        row = pen.query("salinity_product == 'part' and group == 'all'").iloc[0]
        self.assertEqual(row.n, 3)
        self.assertAlmostEqual(row.rmse_reference, 3.)
        self.assertAlmostEqual(row.rmse_product, 5.)
        self.assertAlmostEqual(row.added_error, 4.)
        self.assertAlmostEqual(row.rms_prediction_shift, 2.)
        self.assertAlmostEqual(row.rms_salinity_delta, 0.3)
        self.assertEqual(pen.query("salinity_product == 'part' and group == '[0.25, 0.5)'").iloc[0].n, 3)

    def test_salinity_penalty_zero_shift_when_equal(self):
        base = pd.DataFrame({"dataset": "test", "row_id": [0, 1], "observed": [1., 2.],
                             "predicted": [2., 4.], "salinity_delta": 0.0, "lat": [0., 1.],
                             "bottomdepth": [100., 100.]})
        pen = salinity_penalty(pd.concat([base.assign(salinity_product="salinity"),
                                          base.assign(salinity_product="x")]))
        row = pen.query("dataset == 'test' and group == 'all'").iloc[0]
        self.assertEqual(row.rms_prediction_shift, 0)
        self.assertEqual(row.added_error, 0)

    def test_fingerprint_and_validation_coverage(self):
        from highres_ta.training import data_fingerprint
        load_splits(self.paths, self.data, Ensemble(), fingerprint=data_fingerprint(self.data))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_splits(self.paths, self.data, Ensemble(), fingerprint="wrong")
        frame = pd.read_parquet(self.paths[0])
        frame.loc[3, "split"] = "train"
        frame.to_parquet(self.paths[0])
        with self.assertRaisesRegex(ValueError, "cover all non-test"):
            load_splits(self.paths, self.data, Ensemble())

    def test_model_feature_mismatch_rejected(self):
        class WrongFeatures(Ensemble):
            def get_feature_names(self):
                return ("different",)
        with self.assertRaisesRegex(ValueError, "feature names"):
            evaluate_model(WrongFeatures(), self.data, self.config, self.paths,
                           {"salinity": self.data})

    def test_nonfinite_predictions_rejected(self):
        class InvalidEnsemble(Ensemble):
            estimators = [Member(np.nan), Member(100)]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluate_model(InvalidEnsemble(), self.data, self.config, self.paths,
                           {"salinity": self.data})


if __name__ == "__main__":
    unittest.main()
