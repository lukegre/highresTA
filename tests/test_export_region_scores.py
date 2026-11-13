import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_region_scores import selection_scores  # noqa: E402


class SelectionScoreTests(unittest.TestCase):
    def setUp(self):
        self.pred = pd.DataFrame({
            "lat": [0.0, 70.0, 40.0, 40.0, -65.0],
            "lon": [-30.0, 10.0, 15.0, 15.0, 0.0],
            "bottomdepth": [4000.0, 200.0, 1500.0, 100.0, 3000.0],
            "observed": [2300.0, 2300.0, 2600.0, 2600.0, 2300.0],
            "predicted": [2310.0, 2290.0, 2620.0, 2560.0, 2300.0],
        })

    def test_keys_and_partition(self):
        scores = selection_scores(self.pred)
        self.assertEqual(scores["all"]["n"], 5)
        self.assertEqual(scores["med"]["n"], 2)
        self.assertEqual(scores["nomed"]["n"], 3)
        self.assertEqual(scores["deep"]["n"] + scores["coastal"]["n"], 5)
        self.assertEqual(scores["highlat"]["n"] + scores["lowlat"]["n"], 5)
        self.assertEqual(scores["deep,lowlat,med"]["n"], 1)
        self.assertNotIn("coastal,highlat,med", scores)  # empty combinations are left out

    def test_exact_metrics(self):
        s = selection_scores(self.pred)["coastal,med"]
        self.assertEqual(s["n"], 1)
        self.assertAlmostEqual(s["mae"], 40.0)
        self.assertAlmostEqual(s["bias"], -40.0)
        self.assertAlmostEqual(selection_scores(self.pred)["med"]["rmse"], np.sqrt((400 + 1600) / 2))


if __name__ == "__main__":
    unittest.main()
