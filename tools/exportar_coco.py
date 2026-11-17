#!/usr/bin/env python3
"""Exporta las propuestas revisables como anotaciones COCO de instancias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Define el run de propuestas de origen y el destino del COCO exportado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directorio de una ejecución con proposals.json y la subcarpeta masks/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta de salida; por defecto, run_dir/coco_instances.json.",
    )
    return parser.parse_args()


def encode_uncompressed_rle(mask: np.ndarray) -> dict:
    """Codifica COCO RLE en orden column-major (Fortran), como requiere COCO."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    boundaries = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [pixels.size]))).tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def main() -> None:
    """Convierte propuestas y máscaras de una ejecución a COCO de instancias."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    proposals_path = run_dir / "proposals.json"
    if not proposals_path.is_file():
        raise FileNotFoundError(f"No se encontró {proposals_path}")

    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    config = json.loads(
        (PROJECT_ROOT / "annotation_config.json").read_text(encoding="utf-8")
    )
    categories = [
        {
            "id": category["id"],
            "name": category["name"],
            "supercategory": "thing" if category["isthing"] else "stuff",
        }
        for category in config["categories"]
    ]
    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for image_id, image in enumerate(proposals["images"], start=1):
        coco_images.append(
            {
                "id": image_id,
                "file_name": Path(image["file_name"]).name,
                "width": image["width"],
                "height": image["height"],
            }
        )
        for annotation in image["annotations"]:
            mask_path = (run_dir / annotation["mask_file"]).resolve()
            if not mask_path.is_relative_to(run_dir):
                raise ValueError(f"La máscara escapa del directorio de ejecución: {mask_path}")
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L")) > 0
            if mask.shape != (image["height"], image["width"]):
                raise ValueError(
                    f"Dimensiones incorrectas en {mask_path}: {mask.shape}"
                )
            area = int(mask.sum())
            if area == 0:
                raise ValueError(f"La máscara está vacía: {mask_path}")

            x1, y1, x2, y2 = annotation["bbox_xyxy"]
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": annotation["category_id"],
                    "segmentation": encode_uncompressed_rle(mask),
                    "area": area,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "iscrowd": 0,
                    "score": annotation["detection_score"],
                }
            )
            annotation_id += 1

    coco_data = {
        "info": {
            "description": "Propuestas automáticas revisables; no son ground truth.",
            "models": proposals["model"],
            "prompt": proposals["prompt"],
        },
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    output_path = (args.output or run_dir / "coco_instances.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(coco_data, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exportadas {len(coco_annotations)} propuestas en "
        f"{len(coco_images)} imágenes: {output_path}"
    )


if __name__ == "__main__":
    main()
