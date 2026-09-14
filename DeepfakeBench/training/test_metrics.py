"""Regression tests for cross-platform video grouping."""

import unittest

from metrics.utils import get_test_metrics


class VideoMetricsTest(unittest.TestCase):
    def test_path_styles_preserve_video_groups(self):
        # Real and fake videos may share a basename; keep their full directories.
        paths = [
            "Celeb-DF-v2/real/frames/001/000.png",
            "Celeb-DF-v2/real/frames/001/001.png",
            "Celeb-DF-v2/fake/frames/001/000.png",
            "Celeb-DF-v2/fake/frames/001/001.png",
            "Celeb-DF-v2/fake/frames/002/000.png",
            "Celeb-DF-v2/fake/frames/002/001.png",
        ]
        scores = [0.1, 0.7, 0.5, 0.7, 0.8, 0.9]
        labels = [0, 0, 1, 1, 1, 1]
        windows_paths = [path.replace("/", "\\") for path in paths]
        mixed_paths = [
            windows_paths[i] if i % 2 else path
            for i, path in enumerate(paths)
        ]
        for style, inputs in (
            ("posix", paths), ("windows", windows_paths), ("mixed", mixed_paths)
        ):
            with self.subTest(style=style):
                result = get_test_metrics(scores, labels, inputs)
                self.assertAlmostEqual(result["auc"], 0.8125)
                self.assertAlmostEqual(result["video_auc"], 1.0)
                self.assertAlmostEqual(result["video_eer"], 0.0)
                self.assertAlmostEqual(result["video_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
