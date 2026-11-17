#!/usr/bin/env python3
"""Crea un manifiesto de imágenes y particiones temporales para el proyecto."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def parse_args() -> argparse.Namespace:
    """Define las rutas y el margen temporal que usa el preprocesamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=WORKSPACE_ROOT / "images",
        help="Carpeta con imágenes JPG/JPEG.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Carpeta donde se escribirán el manifiesto y las particiones.",
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=8,
        help="Fotogramas que se excluyen alrededor de cada límite entre particiones.",
    )
    return parser.parse_args()


def main() -> None:
    """Verifica las fotos y escribe manifiesto y particiones cronológicas."""
    args = parse_args()
    if args.gap_frames < 0:
        raise ValueError("--gap-frames debe ser mayor o igual a cero.")
    if not args.images.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {args.images}")

    image_paths = sorted(
        path.resolve()
        for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not image_paths:
        raise ValueError(f"No se encontraron imágenes JPG/JPEG en {args.images}")

    count = len(image_paths)
    train_end = round(count * 0.70)
    validation_end = round(count * 0.85)
    split_names = []
    # El margen excluido evita que fotogramas casi idénticos caigan en splits distintos.
    for index in range(count):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"

        if args.gap_frames and any(
            boundary - args.gap_frames <= index < boundary + args.gap_frames
            for boundary in (train_end, validation_end)
        ):
            split = "excluded_temporal_gap"
        split_names.append(split)

    manifest_rows = []
    for index, (path, split) in enumerate(zip(image_paths, split_names, strict=True)):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        manifest_rows.append(
            {
                "image_id": index + 1,
                "file_name": path.relative_to(WORKSPACE_ROOT.resolve()).as_posix(),
                "width": width,
                "height": height,
                "split": split,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    splits = {
        split: [
            row["file_name"] for row in manifest_rows if row["split"] == split
        ]
        for split in ("train", "validation", "test", "excluded_temporal_gap")
    }
    (args.output / "splits.json").write_text(
        json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Imágenes verificadas: {count}")
    for split, paths in splits.items():
        print(f"{split}: {len(paths)}")
    print(f"Manifiesto: {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
