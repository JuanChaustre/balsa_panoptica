from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor

MODULE_PATH = Path(__file__).resolve().parents[1] / "training" / "coco_dataset.py"
SPEC = importlib.util.spec_from_file_location("balsa_coco_dataset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DATASET_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATASET_MODULE)


def encode_rle(mask: np.ndarray) -> dict:
    pixels = mask.astype(np.uint8).reshape(-1, order="F")
    boundaries = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [pixels.size]))).tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


class CocoPanopticDatasetTests(unittest.TestCase):
    def test_decodes_uncompressed_coco_rle(self) -> None:
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:5] = True
        decoded = DATASET_MODULE.decode_uncompressed_rle(
            encode_rle(mask),
            (4, 5),
        )
        np.testing.assert_array_equal(decoded, mask)

    def test_merges_stuff_and_resolves_mask_overlap_by_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (8, 6), (120, 120, 120)).save(image_path)
            annotations_path = root / "annotations.json"
            split_path = root / "splits.json"

            block = np.zeros((6, 8), dtype=bool)
            block[1:4, 2:6] = True
            hand = np.zeros_like(block)
            hand[2:5, 4:7] = True
            background = np.ones_like(block)
            coco = {
                "categories": [
                    {"id": 1, "name": "balsa_piece", "supercategory": "thing"},
                    {"id": 2, "name": "hand_arm", "supercategory": "thing"},
                    {"id": 3, "name": "work_surface", "supercategory": "stuff"},
                    {"id": 4, "name": "background", "supercategory": "stuff"},
                ],
                "images": [
                    {
                        "id": 1,
                        "file_name": "image.jpg",
                        "width": 8,
                        "height": 6,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": encode_rle(block),
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 2,
                        "segmentation": encode_rle(hand),
                    },
                    {
                        "id": 3,
                        "image_id": 1,
                        "category_id": 3,
                        "segmentation": encode_rle(background),
                    },
                    {
                        "id": 4,
                        "image_id": 1,
                        "category_id": 3,
                        "segmentation": encode_rle(background),
                    },
                ],
            }
            annotations_path.write_text(json.dumps(coco), encoding="utf-8")
            split_path.write_text(
                json.dumps({"train": ["image.jpg"], "validation": [], "test": []}),
                encoding="utf-8",
            )

            dataset = DATASET_MODULE.CocoPanopticDataset(
                annotations_path,
                images_root=root,
                splits_path=split_path,
                split="train",
            )
            sample = dataset[0]
            panoptic = sample["panoptic_map"]
            semantic_map = sample["instance_id_to_semantic_id"]

            self.assertGreater(sample["overlap_pixels_resolved"], 0)
            self.assertEqual(
                panoptic[2, 4],
                1,
                "The hand takes precedence over the overlapping wood mask.",
            )
            self.assertEqual(
                semantic_map[1],
                dataset.category_id_to_label[2],
            )
            self.assertEqual(
                semantic_map[2],
                dataset.category_id_to_label[1],
            )
            self.assertEqual(
                sum(value == dataset.category_id_to_label[3] for value in semantic_map.values()),
                1,
            )
            self.assertTrue(np.all(panoptic[background] != DATASET_MODULE.IGNORE_INDEX))
            self.assertEqual(sample["labeled_pixels"], int(background.sum()))

            processor = Mask2FormerImageProcessor()
            encoded = DATASET_MODULE.encode_for_mask2former(sample, processor, 32)
            self.assertEqual(tuple(encoded["pixel_values"].shape), (3, 32, 32))
            self.assertEqual(tuple(encoded["mask_labels"].shape), (3, 32, 32))
            self.assertEqual(encoded["class_labels"].numel(), 3)

    def test_rle_rejects_mismatched_pixel_count(self) -> None:
        with self.assertRaises(ValueError):
            DATASET_MODULE.decode_uncompressed_rle(
                {"size": [2, 2], "counts": [3]},
                (2, 2),
            )


if __name__ == "__main__":
    unittest.main()
