from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("balsa_annotation_app", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


class AnnotationAppTests(unittest.TestCase):
    def test_rle_round_trip_preserves_column_major_mask(self) -> None:
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:5] = True
        rle = APP.encode_uncompressed_rle(mask)

        decoded = []
        value = 0
        for count in rle["counts"]:
            decoded.extend([value] * count)
            value = 1 - value

        np.testing.assert_array_equal(
            np.asarray(decoded, dtype=bool).reshape((4, 5), order="F"),
            mask,
        )

    def test_polygon_mask_and_bbox(self) -> None:
        image_data = np.zeros((6, 8, 4), dtype=np.uint8)
        image_data[2:5, 3:7, :3] = APP.DRAW_COLOR

        mask = APP.polygon_from_canvas(image_data, 8, 6)

        self.assertEqual(int(mask.sum()), 12)
        self.assertEqual(APP.bbox_for_mask(mask), [3, 2, 4, 3])

    def test_path_guard_rejects_paths_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(APP.safe_path(root, "image.jpg"), root / "image.jpg")
            with self.assertRaises(ValueError):
                APP.safe_path(root, "../outside.jpg")

    def test_reviewed_annotations_load_and_export_as_coco(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            image_path = workspace / "images" / "frame.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (5, 4), (100, 100, 100)).save(image_path)

            run_dir = root / "run_test"
            run_dir.mkdir()
            corrections_root = root / "corrections"
            correction_dir = corrections_root / run_dir.name
            image_record = {
                "file_name": "images/frame.jpg",
                "width": 5,
                "height": 4,
                "annotations": [],
            }
            mask = np.zeros((4, 5), dtype=bool)
            mask[1:3, 2:5] = True
            annotations = [
                {
                    "id": 1,
                    "category_id": 1,
                    "detected_text": "Anotación manual",
                    "detection_score": None,
                    "sam_iou_estimate": None,
                    "mask_file": "",
                    "mask": mask,
                }
            ]
            corrections = {"source_proposals": run_dir.name, "images": {}}
            categories = [
                {"id": 1, "name": "balsa_piece", "isthing": True},
                {"id": 2, "name": "background", "isthing": False},
            ]

            with patch.object(APP, "CORRECTION_ROOT", corrections_root):
                APP.save_image_annotations(
                    correction_dir,
                    corrections,
                    image_record,
                    annotations,
                    reviewed=True,
                )
                saved = corrections["images"][image_record["file_name"]]
                loaded = APP.load_annotations(run_dir, image_record, saved)
                coco = APP.make_coco_export(workspace, run_dir, corrections, categories)

            np.testing.assert_array_equal(loaded[0]["mask"], mask)
            self.assertEqual(len(coco["images"]), 1)
            self.assertEqual(len(coco["annotations"]), 1)
            self.assertEqual(coco["annotations"][0]["area"], int(mask.sum()))
            self.assertEqual(coco["annotations"][0]["bbox"], [2, 1, 3, 2])

    def test_unreviewed_images_are_not_exported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            (workspace / "images").mkdir(parents=True)
            Image.new("RGB", (2, 2)).save(workspace / "images" / "frame.jpg")
            run_dir = root / "run_test"
            run_dir.mkdir()

            with patch.object(APP, "CORRECTION_ROOT", root / "corrections"):
                coco = APP.make_coco_export(
                    workspace,
                    run_dir,
                    {
                        "images": {
                            "images/frame.jpg": {
                                "reviewed": False,
                                "width": 2,
                                "height": 2,
                                "annotations": [],
                            }
                        }
                    },
                    [],
                )

            self.assertEqual(coco["images"], [])
            self.assertEqual(coco["annotations"], [])


if __name__ == "__main__":
    unittest.main()
