from __future__ import annotations

import unittest

import numpy as np

from balsa_panoptica.live_app import estimate_mask_dimensions, overlay_panoptic


class LiveAppTests(unittest.TestCase):
    def test_mask_dimensions_return_long_then_short_extents_in_pixels(self) -> None:
        mask = np.zeros((12, 16), dtype=bool)
        mask[3:6, 4:10] = True

        self.assertEqual(estimate_mask_dimensions(mask), (6.0, 3.0))

    def test_rotated_mask_dimensions_follow_its_principal_axis(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        for offset in range(10):
            mask[10 + offset, 10 + offset : 13 + offset] = True

        length, width = estimate_mask_dimensions(mask)

        self.assertGreater(length, width)
        self.assertGreater(length, 10)

    def test_mask_dimensions_reject_empty_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "vacía"):
            estimate_mask_dimensions(np.zeros((4, 5), dtype=bool))

    def test_overlay_colors_the_predicted_segment_and_keeps_void_pixels(self) -> None:
        frame = np.zeros((30, 40, 3), dtype=np.uint8)
        segmentation = np.full((30, 40), -1, dtype=np.int32)
        segmentation[:2, :2] = 2
        result = overlay_panoptic(
            frame,
            segmentation,
            [{"id": 2, "label_id": 0, "score": 0.9}],
            {0: "balsa_piece"},
            alpha=0.5,
        )

        self.assertEqual(result.shape, frame.shape)
        self.assertFalse(np.array_equal(result, frame))
        np.testing.assert_array_equal(result[29, 39], frame[29, 39])
        self.assertTrue(np.all(result[0, 0] > 0))

    def test_overlay_rejects_mismatched_map_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensiones"):
            overlay_panoptic(
                np.zeros((4, 5, 3), dtype=np.uint8),
                np.zeros((3, 5), dtype=np.int32),
                [],
                {},
            )


if __name__ == "__main__":
    unittest.main()
