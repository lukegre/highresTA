import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from highres_ta.validation import regional_scores
from highres_ta.validation_report import write_validation_reports


class ValidationReportTests(unittest.TestCase):
    def test_reports_preserve_nullable_values_and_work_offline(self):
        predictions = pd.DataFrame({
            "dataset": ["test"], "salinity_product": ["salinity"],
            "observed": [2300.], "predicted": [2301.], "residual": [1.],
            "lon": [-30.], "lat": [60.], "bottomdepth": [700.],
            "split_id": [np.nan], "row_id": [4],
            "time": [pd.Timestamp("2020-01-02")],
        })
        scores = regional_scores(predictions)
        scores["dataset"], scores["salinity_product"] = "test", "salinity"
        with tempfile.TemporaryDirectory() as directory:
            pdf, html = write_validation_reports(predictions, scores, Path(directory), {})
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            document = html.read_text()
            self.assertNotIn('<script src=', document)
            start = document.index("const DATA=") + len("const DATA=")
            payload, _ = json.JSONDecoder().raw_decode(document[start:])
            self.assertIsNone(payload["predictions"][0]["split_id"])
            self.assertEqual(payload["predictions"][0]["row_id"], 4)
            self.assertTrue(payload["predictions"][0]["time"].startswith("2020-01-02"))

            # A product with no available observations must still report n=0.
            empty = predictions.iloc[:0]
            empty_scores = regional_scores(empty)
            empty_scores["dataset"], empty_scores["salinity_product"] = "test", "missing_product"
            pdf, html = write_validation_reports(empty, empty_scores, Path(directory) / "empty", {})
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))
            document = html.read_text()
            start = document.index("const DATA=") + len("const DATA=")
            payload, _ = json.JSONDecoder().raw_decode(document[start:])
            self.assertEqual(payload["predictions"], [])
            self.assertTrue(all(row["n"] == 0 for row in payload["scores"]))


if __name__ == "__main__":
    unittest.main()
