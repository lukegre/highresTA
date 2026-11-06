import tempfile
import unittest
from pathlib import Path

import highres_ta as ta


class LoadConfigTest(unittest.TestCase):
    def test_directory_merges_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.yaml").write_text("paths: {x: 1}\n")
            (Path(tmp) / "b.yaml").write_text("model: {y: 2}\n")
            self.assertEqual(ta.load_config(Path(tmp)), {"paths": {"x": 1}, "model": {"y": 2}})

    def test_duplicate_top_level_key_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.yaml").write_text("paths: {x: 1}\n")
            (Path(tmp) / "b.yaml").write_text("paths: {y: 2}\n")
            with self.assertRaises(ValueError):
                ta.load_config(Path(tmp))

    def test_default_configs_have_unique_keys(self):
        config = ta.load_config()
        for key in ("paths", "data", "model", "inference"):
            self.assertIn(key, config)


if __name__ == "__main__":
    unittest.main()
