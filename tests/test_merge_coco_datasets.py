from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from balsa_panoptica.training.merge_coco_datasets import merge_datasets, merge_splits


class MergeCocoDatasetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images_dir = self.root / "images"
        self.images_dir.mkdir()
        (self.images_dir / "old.jpg").write_bytes(b"old")
        (self.images_dir / "WhatsApp reviewed.jpeg").write_bytes(b"reviewed")
        (self.images_dir / "WhatsApp defective.jpeg").write_bytes(b"defective")
        self.categories = [
            {"id": 1, "name": "balsa_piece", "supercategory": "thing"},
            {"id": 2, "name": "hand_arm", "supercategory": "thing"},
            {"id": 3, "name": "work_surface", "supercategory": "stuff"},
            {"id": 4, "name": "background", "supercategory": "stuff"},
        ]
        self.original: dict[str, Any] = {
            "categories": self.categories,
            "images": [
                {"id": 1, "file_name": "images/old.jpg", "width": 10, "height": 8}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": {},
                }
            ],
        }
        self.additional: dict[str, Any] = {
            "categories": self.categories,
            "images": [
                {
                    "id": 1,
                    "file_name": "images/WhatsApp reviewed.jpeg",
                    "width": 10,
                    "height": 8,
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 2,
                    "segmentation": {},
                }
            ],
        }
        self.splits = {
            "train": ["images/old.jpg"],
            "validation": [],
            "test": [],
            "excluded_temporal_gap": [],
        }
        self.original_path = self._write("original.json", self.original)
        self.additional_path = self._write("additional.json", self.additional)
        self.splits_path = self._write("splits.json", self.splits)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _merge(self, *, delete: bool = False) -> tuple[dict[str, Any], dict[str, list[str]], list[Path]]:
        return merge_datasets(
            self.original_path,
            self.additional_path,
            self.splits_path,
            self.root / "merged.json",
            self.root / "merged_splits.json",
            self.root,
            self.images_dir,
            new_image_pattern="WhatsApp*",
            delete_unreviewed=delete,
        )

    def test_merge_remaps_conflicting_ids_and_assigns_new_images(self) -> None:
        merged, splits, unreviewed = self._merge()

        self.assertEqual(len(merged["images"]), 2)
        self.assertEqual([image["id"] for image in merged["images"]], [1, 2])
        self.assertEqual([annotation["id"] for annotation in merged["annotations"]], [1, 2])
        self.assertEqual(merged["annotations"][1]["image_id"], 2)
        self.assertEqual(
            splits["train"],
            ["images/old.jpg", "images/WhatsApp reviewed.jpeg"],
        )
        self.assertEqual([path.name for path in unreviewed], ["WhatsApp defective.jpeg"])
        self.assertTrue((self.images_dir / "WhatsApp defective.jpeg").is_file())

    def test_new_capture_groups_are_distributed_across_all_splits(self) -> None:
        additional_images = []
        captures = (
            ("11.41.49", 7),
            ("11.41.50", 7),
            ("11.41.51", 5),
            ("11.41.52", 4),
        )
        for timestamp, count in captures:
            base = f"images/WhatsApp Image 2026-11-17 at {timestamp} AM"
            additional_images.append({"file_name": f"{base}.jpeg"})
            additional_images.extend(
                {"file_name": f"{base} ({index}).jpeg"}
                for index in range(1, count)
            )
        additional = {**self.additional, "images": list(reversed(additional_images))}

        splits = merge_splits(self.splits, self.original, additional)

        new_counts = {
            split_name: sum(name.startswith("images/WhatsApp") for name in names)
            for split_name, names in splits.items()
        }
        self.assertEqual(
            {name: new_counts[name] for name in ("train", "validation", "test")},
            {"train": 14, "validation": 5, "test": 4},
        )
        for timestamp, _ in captures:
            capture_files = [
                image["file_name"]
                for image in additional_images
                if f"at {timestamp} AM" in image["file_name"]
            ]
            memberships = {
                split_name
                for split_name in ("train", "validation", "test")
                if any(name in splits[split_name] for name in capture_files)
            }
            self.assertEqual(len(memberships), 1)

    def test_delete_option_removes_only_unreviewed_new_image(self) -> None:
        _, _, removed = self._merge(delete=True)

        self.assertEqual([path.name for path in removed], ["WhatsApp defective.jpeg"])
        self.assertFalse((self.images_dir / "WhatsApp defective.jpeg").exists())
        self.assertTrue((self.images_dir / "old.jpg").is_file())
        self.assertTrue((self.images_dir / "WhatsApp reviewed.jpeg").is_file())


if __name__ == "__main__":
    unittest.main()
