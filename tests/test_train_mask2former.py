from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from transformers import Mask2FormerImageProcessor

from balsa_panoptica.training.train_mask2former import loss_for_sample, run_epoch


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.received_labels: list[torch.Tensor] | None = None

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        mask_labels: list[torch.Tensor],
        class_labels: list[torch.Tensor],
    ) -> SimpleNamespace:
        assert pixel_values.shape == (1, 3, 16, 16)
        self.received_labels = class_labels
        return SimpleNamespace(loss=self.weight.square())


class TinyDataset:
    epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class TrainMask2FormerTests(unittest.TestCase):
    def test_loss_adapter_encodes_masks_and_preserves_class_labels(self) -> None:
        sample: dict[str, Any] = {
            "file_name": "sample.jpg",
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
            "panoptic_map": np.ones((16, 16), dtype=np.uint16),
            "instance_id_to_semantic_id": {1: 2},
        }
        model = TinyModel()
        loss = loss_for_sample(
            model,
            Mask2FormerImageProcessor(),
            sample,
            16,
            torch.device("cpu"),
        )
        loss.backward()

        self.assertEqual(loss.item(), 1.0)
        self.assertEqual(model.weight.grad.item(), 2.0)
        assert model.received_labels is not None
        self.assertEqual(model.received_labels[0].tolist(), [2])

    def test_epoch_runs_training_step_and_reports_overlap_count(self) -> None:
        sample: dict[str, Any] = {
            "file_name": "sample.jpg",
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
            "panoptic_map": np.ones((16, 16), dtype=np.uint16),
            "instance_id_to_semantic_id": {1: 2},
            "overlap_pixels_resolved": 3,
        }
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss, count, overlaps = run_epoch(
            model,
            Mask2FormerImageProcessor(),
            [[sample]],
            TinyDataset(),
            16,
            torch.device("cpu"),
            optimizer=optimizer,
            max_batches=1,
        )

        self.assertEqual(loss, 1.0)
        self.assertEqual(count, 1)
        self.assertEqual(overlaps, 3)
        self.assertAlmostEqual(model.weight.item(), 0.9)

    def test_loss_adapter_rejects_empty_targets(self) -> None:
        sample = {
            "file_name": "empty.jpg",
            "image": np.zeros((16, 16, 3), dtype=np.uint8),
            "panoptic_map": np.full((16, 16), 255, dtype=np.uint16),
            "instance_id_to_semantic_id": {},
        }
        with self.assertRaisesRegex(ValueError, "empty.jpg"):
            loss_for_sample(
                TinyModel(),
                Mask2FormerImageProcessor(),
                sample,
                16,
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
