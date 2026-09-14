"""Regression tests for release configuration loading and dataset labels."""

import unittest
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "config"


class ConfigurationTest(unittest.TestCase):
    def test_yaml_configs_are_mappings(self):
        paths = sorted(CONFIG_DIR.rglob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=str(path.relative_to(CONFIG_DIR))):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertIsInstance(config, dict)

    def test_pgf_labels_supported_by_train_and_test(self):
        for mode in ("train", "test"):
            with self.subTest(mode=mode):
                path = CONFIG_DIR / (mode + "_config.yaml")
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["mode"], mode)
                self.assertEqual(config["label_dict"]["PolyGlotFake_Fake"], 1)
                self.assertEqual(config["label_dict"]["PolyGlotFake_Real"], 0)


if __name__ == "__main__":
    unittest.main()
