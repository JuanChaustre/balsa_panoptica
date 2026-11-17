from __future__ import annotations

import unittest

import numpy as np

from balsa_panoptica.training.evaluate_mask2former import (
    average_precision_101,
    calculate_pixel_metrics,
    class_map_from_panoptic,
)


class EvaluateMask2FormerTests(unittest.TestCase):
    def test_pixel_metrics_use_macro_class_scores_and_void_column(self) -> None:
        confusion = np.array([[4, 1, 0], [1, 1, 1]], dtype=np.int64)
        metrics = calculate_pixel_metrics(confusion, ["wood", "hand"])

        self.assertAlmostEqual(metrics["IoU_macro"], (4 / 6 + 1 / 4) / 2)
        self.assertAlmostEqual(metrics["precision_macro"], (4 / 5 + 1 / 2) / 2)
        self.assertAlmostEqual(metrics["F1_macro"], (8 / 10 + 2 / 5) / 2)

    def test_average_precision_penalizes_false_positive_ranked_first(self) -> None:
        score = average_precision_101([(0.9, False), (0.8, True)], 1)
        self.assertAlmostEqual(score, 0.5)
        self.assertIsNone(average_precision_101([], 0))
        self.assertEqual(average_precision_101([], 1), 0.0)

    def test_panoptic_map_keeps_unassigned_pixels_void(self) -> None:
        segments = np.array([[0, 0, 1], [1, -1, -1]], dtype=np.int64)
        class_map = class_map_from_panoptic(
            segments,
            [{"id": 0, "label_id": 2}, {"id": 1, "label_id": 0}],
            class_count=4,
        )
        np.testing.assert_array_equal(class_map, [[2, 2, 0], [0, -1, -1]])


if __name__ == "__main__":
    unittest.main()
