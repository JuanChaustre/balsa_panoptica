#!/usr/bin/env python3
"""Calcula IoU, precisión, F1 y mAP de Mask2Former en validation o test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from balsa_panoptica.training.coco_dataset import (
    IGNORE_INDEX,
    WORKSPACE_ROOT,
    CocoPanopticDataset,
    collate_samples,
    encode_for_mask2former,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ANNOTATIONS = PROJECT_ROOT / "data/annotations/balsa_coco_merged.json"
DEFAULT_EVAL_SPLITS = PROJECT_ROOT / "data/splits_merged.json"


def parse_args() -> argparse.Namespace:
    """Configura evaluación de validation o test sin opciones de entrenamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_EVAL_ANNOTATIONS)
    parser.add_argument("--images-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_EVAL_SPLITS)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image_size is not None and args.image_size < 128:
        parser.error("--image-size debe ser al menos 128.")
    return args


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    """Calcula intersección sobre unión entre dos máscaras binarias."""
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return float(intersection / union) if union else 0.0


def average_precision_101(
    predictions: list[tuple[float, bool]],
    ground_truth_count: int,
) -> float | None:
    """Calcula AP interpolada en 101 niveles de recall para una clase/umbral."""
    if ground_truth_count == 0:
        return None
    if not predictions:
        return 0.0

    ranked = sorted(predictions, key=lambda item: item[0], reverse=True)
    true_positives = np.cumsum(
        [int(item[1]) for item in ranked],
        dtype=np.float64,
    )
    false_positives = np.cumsum(
        [int(not item[1]) for item in ranked],
        dtype=np.float64,
    )
    recall = true_positives / ground_truth_count
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(
        np.mean(
            [
                precision_envelope[recall >= level].max()
                if np.any(recall >= level)
                else 0.0
                for level in np.linspace(0.0, 1.0, 101)
            ]
        )
    )


def calculate_instance_map(
    ground_truth: dict[int, list[tuple[int, np.ndarray]]],
    predictions: dict[int, list[tuple[int, float, np.ndarray]]],
    thing_class_ids: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Promedia AP de máscaras por clase thing e IoU entre 0.50 y 0.95."""
    thresholds = [round(value, 2) for value in np.arange(0.5, 0.951, 0.05)]
    per_class: dict[str, Any] = {}
    class_aps: list[float] = []

    for class_id in thing_class_ids:
        gt_by_image = {
            image_id: [
                mask for label, mask in segments if label == class_id
            ]
            for image_id, segments in ground_truth.items()
        }
        gt_count = sum(len(masks) for masks in gt_by_image.values())
        class_predictions = [
            (image_id, score, mask)
            for image_id, segments in predictions.items()
            for label, score, mask in segments
            if label == class_id
        ]
        ap_by_iou: dict[str, float | None] = {}
        for threshold in thresholds:
            matched = {image_id: set() for image_id in gt_by_image}
            ranked_results: list[tuple[float, bool]] = []
            for image_id, score, pred_mask in sorted(
                class_predictions,
                key=lambda item: item[1],
                reverse=True,
            ):
                gt_masks = gt_by_image.get(image_id, [])
                candidates = [
                    (index, mask_iou(pred_mask, gt_mask))
                    for index, gt_mask in enumerate(gt_masks)
                    if index not in matched.get(image_id, set())
                ]
                if candidates:
                    best_index, best_iou = max(candidates, key=lambda item: item[1])
                    is_true_positive = best_iou >= threshold
                    if is_true_positive:
                        matched.setdefault(image_id, set()).add(best_index)
                else:
                    is_true_positive = False
                ranked_results.append((score, is_true_positive))

            ap_by_iou[f"{threshold:.2f}"] = average_precision_101(
                ranked_results,
                gt_count,
            )

        valid_ap = [value for value in ap_by_iou.values() if value is not None]
        class_map = float(np.mean(valid_ap)) if valid_ap else None
        per_class[class_names[class_id]] = {
            "ground_truth_instances": gt_count,
            "predicted_instances": len(class_predictions),
            "AP_by_IoU": ap_by_iou,
            "mAP_50_95": class_map,
        }
        if class_map is not None:
            class_aps.append(class_map)

    return {
        "definition": (
            "Mask AP over thing classes only; IoU thresholds 0.50:0.05:0.95; "
            "101-point interpolated precision."
        ),
        "mAP_50_95": float(np.mean(class_aps)) if class_aps else None,
        "per_class": per_class,
    }


def calculate_pixel_metrics(
    confusion: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Calcula IoU, precisión y F1 macro desde la matriz de confusión por píxel."""
    class_count = len(class_names)
    if confusion.shape != (class_count, class_count + 1):
        raise ValueError("La matriz de confusión tiene una forma inesperada.")
    per_class: dict[str, Any] = {}
    iou_values: list[float] = []
    precision_values: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(class_names):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        union = tp + fp + fn
        iou = tp / union if union else None
        precision = tp / (tp + fp) if tp + fp else (0.0 if fn else None)
        f1_denominator = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        per_class[class_name] = {
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
            "IoU": iou,
            "precision": precision,
            "F1": f1,
        }
        if iou is not None:
            iou_values.append(float(iou))
        if precision is not None:
            precision_values.append(float(precision))
        if f1 is not None:
            f1_values.append(float(f1))

    return {
        "definition": (
            "Macro mean of per-class pixel metrics over the four classes; "
            "ground-truth IGNORE_INDEX pixels are excluded."
        ),
        "IoU_macro": float(np.mean(iou_values)) if iou_values else None,
        "precision_macro": (
            float(np.mean(precision_values)) if precision_values else None
        ),
        "F1_macro": float(np.mean(f1_values)) if f1_values else None,
        "per_class": per_class,
    }


def class_map_from_panoptic(
    segmentation: np.ndarray,
    segments_info: list[dict[str, Any]],
    class_count: int,
) -> np.ndarray:
    """Convierte IDs de segmento panóptico a IDs de clase; deja void como -1."""
    class_map = np.full(segmentation.shape, -1, dtype=np.int16)
    for segment in segments_info:
        class_id = int(segment["label_id"])
        if not 0 <= class_id < class_count:
            raise ValueError(f"Predicción con clase fuera de rango: {class_id}.")
        class_map[segmentation == int(segment["id"])] = class_id
    return class_map


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Evalúa un checkpoint y devuelve métricas sin escribir archivos."""
    run_dir = args.run_dir.resolve()
    model_dir = run_dir / "best_model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No existe el checkpoint entrenado: {model_dir}")
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    image_size = args.image_size or int(run_config["image_size"])

    dataset = CocoPanopticDataset(
        args.annotations,
        images_root=args.images_root,
        splits_path=args.splits,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = device.type == "cuda" and not args.no_amp

    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir).to(device)
    class_names = [str(category["name"]) for category in dataset.categories]
    expected_labels = {index: name for index, name in enumerate(class_names)}
    actual_labels = {
        int(index): str(name)
        for index, name in model.config.id2label.items()
    }
    if actual_labels != expected_labels:
        raise ValueError(
            f"Las etiquetas del checkpoint {actual_labels} no coinciden con el "
            f"orden COCO {expected_labels}."
        )
    thing_class_ids = [
        index
        for index, category in enumerate(dataset.categories)
        if category.get("supercategory") == "thing"
        or bool(category.get("isthing", 0))
    ]
    stuff_class_ids = set(range(len(class_names))) - set(thing_class_ids)
    model.eval()

    confusion = np.zeros((len(class_names), len(class_names) + 1), dtype=np.int64)
    ground_truth_instances: dict[int, list[tuple[int, np.ndarray]]] = {}
    predicted_instances: dict[int, list[tuple[int, float, np.ndarray]]] = {}
    total_overlap_pixels = 0
    processed_images = 0

    for batch in loader:
        for sample in batch:
            height, width = int(sample["height"]), int(sample["width"])
            encoded = encode_for_mask2former(sample, processor, image_size)
            pixel_values = encoded["pixel_values"].unsqueeze(0).to(device)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    outputs = model(pixel_values=pixel_values)
                target_sizes = [(height, width)]
                panoptic = processor.post_process_panoptic_segmentation(
                    outputs,
                    label_ids_to_fuse=stuff_class_ids,
                    target_sizes=target_sizes,
                )[0]
                instance = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.0,
                    target_sizes=target_sizes,
                    return_binary_maps=True,
                )[0]

            ground_truth_map = sample["panoptic_map"]
            # Los píxeles sin ground truth se excluyen tanto de pixel metrics como de AP.
            valid = ground_truth_map != IGNORE_INDEX
            target_class_map = np.full((height, width), -1, dtype=np.int16)
            ground_truth_instances[sample["image_id"]] = []
            for segment_id, class_id in sample["instance_id_to_semantic_id"].items():
                segment_mask = ground_truth_map == int(segment_id)
                target_class_map[segment_mask] = int(class_id)
                if int(class_id) in thing_class_ids:
                    ground_truth_instances[sample["image_id"]].append(
                        (int(class_id), segment_mask & valid)
                    )

            predicted_class_map = class_map_from_panoptic(
                panoptic["segmentation"].cpu().numpy(),
                panoptic["segments_info"],
                len(class_names),
            )
            target_flat = target_class_map[valid].astype(np.int64)
            predicted_flat = predicted_class_map[valid].astype(np.int64)
            predicted_flat[predicted_flat < 0] = len(class_names)
            if np.any(target_flat < 0) or np.any(predicted_flat > len(class_names)):
                raise ValueError(f"Etiqueta de píxel inválida en {sample['file_name']}.")
            confusion += np.bincount(
                target_flat * (len(class_names) + 1) + predicted_flat,
                minlength=len(class_names) * (len(class_names) + 1),
            ).reshape(len(class_names), len(class_names) + 1)

            predicted_instances[sample["image_id"]] = []
            binary_maps = instance["segmentation"]
            if instance["segments_info"]:
                if not isinstance(binary_maps, torch.Tensor) or binary_maps.ndim != 3:
                    raise RuntimeError("Mask2Former no devolvió los mapas binarios de instancia.")
                if binary_maps.shape[0] != len(instance["segments_info"]):
                    raise RuntimeError("La cantidad de máscaras y segmentos no coincide.")
                for segment, mask_tensor in zip(instance["segments_info"], binary_maps):
                    class_id = int(segment["label_id"])
                    if class_id not in thing_class_ids:
                        continue
                    mask = mask_tensor.cpu().numpy().astype(bool) & valid
                    if mask.any():
                        score = float(segment["score"])
                        if not math.isfinite(score):
                            raise ValueError("La inferencia produjo una confianza no finita.")
                        predicted_instances[sample["image_id"]].append(
                            (class_id, score, mask)
                        )

            total_overlap_pixels += int(sample["overlap_pixels_resolved"])
            processed_images += 1
            print(
                f"[{processed_images}/{len(dataset)}] {sample['file_name']}: "
                f"predicciones thing={len(predicted_instances[sample['image_id']])}",
                flush=True,
            )

    metrics = {
        "split": args.split,
        "image_count": processed_images,
        "checkpoint": str(model_dir),
        "image_size": image_size,
        "device": str(device),
        "amp": amp,
        "overlap_pixels_resolved": total_overlap_pixels,
        "class_names": class_names,
        "metric_thresholds": {"IoU": 0.7, "mAP": 0.6, "F1": 0.8},
        "pixel_metrics": calculate_pixel_metrics(confusion, class_names),
        "instance_mAP": calculate_instance_map(
            ground_truth_instances,
            predicted_instances,
            thing_class_ids,
            class_names,
        ),
    }
    return metrics


def main() -> None:
    """Guarda el JSON de métricas y resume los resultados en consola."""
    args = parse_args()
    metrics = evaluate(args)
    output = args.output or args.run_dir.resolve() / f"{args.split}_metrics.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pixel = metrics["pixel_metrics"]
    print(f"\nMétricas guardadas en: {output}")
    print(f"IoU macro: {pixel['IoU_macro']}")
    print(f"mAP thing 0.50:0.95: {metrics['instance_mAP']['mAP_50_95']}")
    print(f"F1 macro: {pixel['F1_macro']}")
    print(f"Precisión macro: {pixel['precision_macro']}")
    for class_name, values in pixel["per_class"].items():
        print(
            f"  {class_name}: IoU={values['IoU']} "
            f"F1={values['F1']} precisión={values['precision']}"
        )


if __name__ == "__main__":
    main()
