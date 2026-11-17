#!/usr/bin/env python3
"""Genera propuestas locales de máscaras con Grounding DINO y SAM."""

from __future__ import annotations

import argparse
import fnmatch
import gc
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    SamModel,
    SamProcessor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DETECTION_PROMPT = "a wooden block. a human hand or arm."


def parse_args() -> argparse.Namespace:
    """Define los filtros, umbrales y recursos de una ejecución de pre-etiquetado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=WORKSPACE_ROOT / "images",
        help="Carpeta de imágenes JPG/JPEG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "prelabels",
        help="Carpeta donde se guardará una nueva ejecución de propuestas.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Dispositivo de inferencia. 'auto' usa CUDA si está disponible.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Procesa solo las primeras N imágenes, útil para probar el modelo.",
    )
    parser.add_argument(
        "--filename-pattern",
        default="*",
        help="Patrón glob aplicado a los nombres de archivo para procesar un subconjunto.",
    )
    return parser.parse_args()


def category_for_label(label: str) -> tuple[int, str] | None:
    """Mapea el texto de Grounding DINO a una de las clases thing del proyecto."""
    normalized = label.lower()
    if "hand" in normalized or "arm" in normalized:
        return 2, "hand_arm"
    if "wood" in normalized or "block" in normalized or "balsa" in normalized:
        return 1, "balsa_piece"
    return None


def box_iou(first: list[float], second: list[float]) -> float:
    """Calcula IoU entre dos cajas XYXY para filtrar propuestas duplicadas."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def suppress_duplicate_candidates(
    candidates: list[dict], iou_threshold: float = 0.65
) -> list[dict]:
    """Conserva primero la propuesta de mayor score por clase y región."""
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(
            candidate["category"][0] == kept["category"][0]
            and box_iou(candidate["box"], kept["box"]) > iou_threshold
            for kept in selected
        ):
            continue
        selected.append(candidate)
    return selected


def load_detector(device: torch.device):
    """Carga Grounding DINO Tiny para localizar balsa y manos/brazos."""
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector_processor = GroundingDinoProcessor.from_pretrained(detector_id)
    detector = GroundingDinoForObjectDetection.from_pretrained(detector_id).to(device)
    detector.eval()
    return detector_processor, detector


def load_segmenter(device: torch.device):
    """Carga SAM ViT-B para convertir las cajas de detección en máscaras."""
    segmenter_id = "facebook/sam-vit-base"
    segmenter_processor = SamProcessor.from_pretrained(segmenter_id)
    segmenter = SamModel.from_pretrained(segmenter_id).to(device)
    segmenter.eval()
    return segmenter_processor, segmenter


def main() -> None:
    """Genera propuestas revisables y máscaras sin modificar las imágenes fuente."""
    args = parse_args()
    if not 0 <= args.box_threshold <= 1 or not 0 <= args.text_threshold <= 1:
        raise ValueError("Los umbrales deben estar entre 0 y 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit debe ser mayor que cero.")
    if not args.images.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {args.images}")

    image_paths = sorted(
        path.resolve()
        for path in args.images.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg"}
        and fnmatch.fnmatchcase(path.name, args.filename_pattern)
    )
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"No se encontraron imágenes JPG/JPEG en {args.images}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    run_dir = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    masks_dir = run_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=False)
    detector_processor, detector = load_detector(device)
    detected_images = []

    # Se detectan todas las cajas primero para liberar la memoria antes de cargar SAM.
    for image_index, image_path in enumerate(image_paths, start=1):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        inputs = detector_processor(
            images=image,
            text=DETECTION_PROMPT,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = detector(**inputs)
        detections = detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        candidates = []
        for box, score, label in zip(
            detections["boxes"], detections["scores"], detections["labels"], strict=True
        ):
            category = category_for_label(label)
            if category is None:
                continue
            candidates.append(
                {
                    "box": [float(coordinate) for coordinate in box.tolist()],
                    "score": float(score.item()),
                    "label": label,
                    "category": category,
                }
            )
        candidates = suppress_duplicate_candidates(candidates)
        detected_images.append((image_path, width, height, candidates))
        print(
            f"[detección {image_index}/{len(image_paths)}] {image_path.name}: "
            f"{len(candidates)} propuestas"
        )

    del detector, detector_processor, inputs, outputs, detections
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sam_processor, sam = load_segmenter(device)
    all_results = []
    for image_index, (image_path, width, height, candidates) in enumerate(
        detected_images, start=1
    ):
        image = Image.open(image_path).convert("RGB")
        annotations = []
        if candidates:
            sam_inputs = sam_processor(
                images=image,
                input_boxes=[[candidate["box"] for candidate in candidates]],
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                sam_outputs = sam(
                    **sam_inputs,
                    multimask_output=True,
                )
            masks = sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]
            iou_scores = sam_outputs.iou_scores[0].cpu()

            for candidate_index, candidate in enumerate(candidates):
                best_mask_index = int(iou_scores[candidate_index].argmax().item())
                mask = masks[candidate_index][best_mask_index].numpy().astype(np.uint8)
                mask_name = f"{image_path.stem}_{candidate_index:03d}.png"
                Image.fromarray(mask * 255).save(masks_dir / mask_name)
                annotations.append(
                    {
                        "category_id": candidate["category"][0],
                        "category": candidate["category"][1],
                        "detected_text": candidate["label"],
                        "detection_score": candidate["score"],
                        "sam_iou_estimate": float(iou_scores[candidate_index][best_mask_index]),
                        "bbox_xyxy": candidate["box"],
                        "mask_file": f"masks/{mask_name}",
                    }
                )

        all_results.append(
            {
                "file_name": image_path.relative_to(WORKSPACE_ROOT.resolve()).as_posix(),
                "width": width,
                "height": height,
                "annotations": annotations,
                "review_required": True,
            }
        )
        print(
            f"[segmentación {image_index}/{len(image_paths)}] {image_path.name}: "
            f"{len(annotations)} propuestas"
        )

    (run_dir / "proposals.json").write_text(
        json.dumps(
            {
                "model": {
                    "detector": "IDEA-Research/grounding-dino-tiny",
                    "segmenter": "facebook/sam-vit-base",
                },
                "prompt": DETECTION_PROMPT,
                "device": str(device),
                "images": all_results,
                "warning": "Propuestas automáticas: corregir antes de usarlas para entrenamiento o evaluación.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Propuestas y máscaras guardadas en: {run_dir}")


if __name__ == "__main__":
    main()
