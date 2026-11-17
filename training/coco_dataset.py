#!/usr/bin/env python3
"""Carga anotaciones COCO revisadas como targets panópticos de Mask2Former."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import Mask2FormerImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "annotations" / "balsa_coco.json"
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "splits.json"
IGNORE_INDEX = 255

# Prioriza objetos visibles sobre superficies y fondo cuando las máscaras se solapan.
DEFAULT_MASK_PRIORITY = {
    "hand_arm": 0,
    "balsa_piece": 1,
    "work_surface": 2,
    "background": 3,
}


def read_json(path: Path) -> dict[str, Any]:
    """Carga y valida la estructura raíz de un JSON requerido."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo no contiene JSON válido: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"El JSON debe contener un objeto en la raíz: {path}")
    return payload


def resolve_path(root: Path, relative_path: str) -> Path:
    """Resuelve una ruta del COCO y evita que salga de la raíz de imágenes."""
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"La ruta de COCO escapa del directorio de imágenes: {relative_path}")
    return path


def decode_uncompressed_rle(
    segmentation: dict[str, Any],
    expected_size: tuple[int, int],
) -> np.ndarray:
    """Decodifica RLE COCO sin comprimir, en orden column-major, a máscara booleana."""
    if not isinstance(segmentation, dict):
        raise ValueError("La segmentación COCO debe ser un objeto RLE.")
    if not isinstance(segmentation.get("counts"), list):
        raise ValueError(
            "Este cargador requiere RLE COCO sin comprimir (counts como lista)."
        )
    size = segmentation.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("El RLE COCO debe declarar size=[height, width].")
    height, width = (int(value) for value in size)
    if (height, width) != expected_size:
        raise ValueError(
            f"El tamaño del RLE {(height, width)} no coincide con la imagen {expected_size}."
        )
    runs = segmentation["counts"]
    if any(not isinstance(run, int) or run < 0 for run in runs):
        raise ValueError("Las longitudes RLE deben ser enteros no negativos.")
    if sum(runs) != height * width:
        raise ValueError(
            f"El RLE cubre {sum(runs)} píxeles; se esperaban {height * width}."
        )
    values = np.repeat(
        np.arange(len(runs), dtype=np.uint8) % 2,
        np.asarray(runs, dtype=np.int64),
    )
    return values.reshape((height, width), order="F").astype(bool, copy=False)


def load_split_names(path: Path) -> dict[str, set[str]]:
    """Carga las particiones y rechaza imágenes asignadas a más de un split."""
    payload = read_json(path)
    split_data = payload.get("splits", payload)
    if not isinstance(split_data, dict):
        raise ValueError(f"El archivo de particiones no tiene el formato esperado: {path}")
    splits: dict[str, set[str]] = {}
    seen: set[str] = set()
    for split_name, file_names in split_data.items():
        if not isinstance(file_names, list):
            continue
        names = {str(file_name) for file_name in file_names}
        duplicates = seen & names
        if duplicates:
            raise ValueError(
                f"Imágenes presentes en más de una partición: {sorted(duplicates)[:3]}"
            )
        seen.update(names)
        splits[str(split_name)] = names
    return splits


class CocoPanopticDataset(Dataset[dict[str, Any]]):
    """Lee instancias COCO y construye targets panópticos sin solapamientos.

    Las anotaciones thing conservan sus IDs de instancia; las máscaras stuff
    se agrupan por clase. Los píxeles sin anotación quedan como ``IGNORE_INDEX``
    en el mapa, pero no se excluyen automáticamente de la loss binaria.
    """

    def __init__(
        self,
        annotations_path: Path = DEFAULT_ANNOTATIONS,
        *,
        images_root: Path = WORKSPACE_ROOT,
        splits_path: Path | None = DEFAULT_SPLITS,
        split: str = "train",
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        """Lee COCO, valida sus referencias y prepara registros del split elegido."""
        payload = read_json(annotations_path)
        categories = payload.get("categories")
        images = payload.get("images")
        annotations = payload.get("annotations")
        if not isinstance(categories, list) or not categories:
            raise ValueError("El COCO debe incluir al menos una categoría.")
        if not isinstance(images, list) or not images:
            raise ValueError("El COCO debe incluir al menos una imagen.")
        if not isinstance(annotations, list):
            raise ValueError("El COCO debe incluir la lista annotations.")

        self.annotations_path = annotations_path.resolve()
        self.images_root = images_root.resolve()
        self.categories = categories
        self.category_by_id = {int(item["id"]): item for item in categories}
        if len(self.category_by_id) != len(categories):
            raise ValueError("El COCO contiene IDs de categoría duplicados.")
        self.category_id_to_label = {
            int(category["id"]): index for index, category in enumerate(categories)
        }

        if split == "all":
            split_names = None
            self.split_counts: dict[str, int] = {}
        else:
            if splits_path is None:
                raise ValueError("Se necesita --splits para cargar train/validation/test.")
            split_data = load_split_names(splits_path)
            if split not in split_data:
                raise ValueError(f"No existe la partición {split!r} en {splits_path}.")
            split_names = split_data[split]
            self.split_counts = {
                name: sum(
                    str(image["file_name"]) in names
                    for image in images
                )
                for name, names in split_data.items()
            }

        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            image_id = int(annotation["image_id"])
            if image_id not in {int(image["id"]) for image in images}:
                raise ValueError(
                    f"La anotación {annotation.get('id')} apunta a una imagen inexistente."
                )
            category_id = int(annotation["category_id"])
            if category_id not in self.category_by_id:
                raise ValueError(
                    f"La anotación {annotation.get('id')} usa la categoría "
                    f"desconocida {category_id}."
                )
            annotations_by_image.setdefault(image_id, []).append(annotation)

        self.records = []
        for image in images:
            file_name = str(image["file_name"])
            if split_names is not None and file_name not in split_names:
                continue
            image_path = resolve_path(self.images_root, file_name)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"No existe la imagen COCO {file_name}; se buscó en {image_path}."
                )
            self.records.append(
                {
                    "image_id": int(image["id"]),
                    "file_name": file_name,
                    "image_path": image_path,
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": annotations_by_image.get(int(image["id"]), []),
                }
            )
        if not self.records:
            raise ValueError(
                f"No hay imágenes COCO anotadas en la partición {split!r}."
            )
        self.split = split
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        """Devuelve la cantidad de imágenes cargadas para este split."""
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Actualiza el índice que hace reproducibles los aumentos por época."""
        self.epoch = epoch

    def _panoptic_target(
        self,
        record: dict[str, Any],
    ) -> tuple[np.ndarray, dict[int, int], int, int]:
        """Une stuff y resuelve solapamientos según prioridad para crear segmentos."""
        height, width = record["height"], record["width"]
        expected_size = (height, width)
        stuff_masks: dict[int, np.ndarray] = {}
        thing_masks: list[tuple[int, int, np.ndarray]] = []
        for annotation_index, annotation in enumerate(record["annotations"]):
            category_id = int(annotation["category_id"])
            category = self.category_by_id[category_id]
            mask = decode_uncompressed_rle(
                annotation["segmentation"],
                expected_size,
            )
            if not mask.any():
                raise ValueError(
                    f"La anotación {annotation.get('id')} tiene una máscara vacía."
                )
            isthing = category.get("supercategory") == "thing" or bool(
                category.get("isthing", 0)
            )
            if isthing:
                thing_masks.append((category_id, annotation_index, mask))
            else:
                stuff_masks.setdefault(category_id, np.zeros(expected_size, dtype=bool))
                stuff_masks[category_id] |= mask

        candidates: list[tuple[int, int, int, np.ndarray]] = []
        for category_id, annotation_index, mask in thing_masks:
            category_name = str(self.category_by_id[category_id]["name"])
            priority = DEFAULT_MASK_PRIORITY.get(category_name, 10)
            candidates.append((priority, annotation_index, category_id, mask))
        for category_id, mask in stuff_masks.items():
            category_name = str(self.category_by_id[category_id]["name"])
            priority = DEFAULT_MASK_PRIORITY.get(category_name, 10)
            candidates.append((priority, len(record["annotations"]), category_id, mask))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))

        panoptic_map = np.full(expected_size, IGNORE_INDEX, dtype=np.uint16)
        semantic_labels: dict[int, int] = {}
        occupied = np.zeros(expected_size, dtype=bool)
        overlap_pixels = 0
        next_segment_id = 1
        stuff_segment_ids: dict[int, int] = {}
        for _, _, category_id, original_mask in candidates:
            mask = original_mask & ~occupied
            overlap_pixels += int(original_mask.sum() - mask.sum())
            if not mask.any():
                continue

            category = self.category_by_id[category_id]
            isthing = category.get("supercategory") == "thing" or bool(
                category.get("isthing", 0)
            )
            if not isthing and category_id in stuff_segment_ids:
                segment_id = stuff_segment_ids[category_id]
            else:
                segment_id = next_segment_id
                next_segment_id += 1
                semantic_labels[segment_id] = self.category_id_to_label[category_id]
                if not isthing:
                    stuff_segment_ids[category_id] = segment_id
            panoptic_map[mask] = segment_id
            occupied[mask] = True
        return panoptic_map, semantic_labels, overlap_pixels, int(occupied.sum())

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Carga una imagen, su mapa panóptico y aplica reflejos reproducibles."""
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError(
                f"Dimensiones de imagen incorrectas en {record['image_path']}: "
                f"{image.size} != {(record['width'], record['height'])}."
            )
        panoptic_map, semantic_labels, overlap_pixels, labeled_pixels = (
            self._panoptic_target(record)
        )
        if self.augment:
            rng = random.Random(self.seed + self.epoch * len(self.records) + index)
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
                panoptic_map = np.fliplr(panoptic_map).copy()
            if rng.random() < 0.5:
                image = ImageOps.flip(image)
                panoptic_map = np.flipud(panoptic_map).copy()
        return {
            "image_id": record["image_id"],
            "file_name": record["file_name"],
            "image": np.asarray(image, dtype=np.uint8),
            "panoptic_map": panoptic_map,
            "instance_id_to_semantic_id": semantic_labels,
            "overlap_pixels_resolved": overlap_pixels,
            "labeled_pixels": labeled_pixels,
            "height": record["height"],
            "width": record["width"],
        }


def collate_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mantiene cada muestra separada porque varía el número de segmentos."""
    return samples


def encode_for_mask2former(
    sample: dict[str, Any],
    processor: Mask2FormerImageProcessor,
    image_size: int,
) -> dict[str, Any]:
    """Redimensiona imagen/máscaras y crea tensores de entrada para Mask2Former."""
    if image_size < 1:
        raise ValueError("image_size debe ser mayor que cero.")
    resized = Image.fromarray(sample["image"]).resize(
        (image_size, image_size),
        resample=Image.Resampling.BILINEAR,
    )
    resized_map = Image.fromarray(
        sample["panoptic_map"].astype(np.int32),
        mode="I",
    ).resize(
        (image_size, image_size),
        resample=Image.Resampling.NEAREST,
    )
    target_map = np.asarray(resized_map, dtype=np.uint16)
    masks: list[np.ndarray] = []
    labels: list[int] = []
    for segment_id, class_id in sample["instance_id_to_semantic_id"].items():
        mask = target_map == int(segment_id)
        if mask.any():
            masks.append(mask)
            labels.append(int(class_id))
    if masks:
        mask_labels = torch.from_numpy(np.stack(masks).astype(np.float32))
        class_labels = torch.tensor(labels, dtype=torch.long)
    else:
        mask_labels = torch.zeros((0, image_size, image_size), dtype=torch.float32)
        class_labels = torch.zeros((0,), dtype=torch.long)
    encoded_image = processor(
        images=np.asarray(resized),
        do_resize=False,
        return_tensors="pt",
    )["pixel_values"][0]
    return {
        "pixel_values": encoded_image,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }


def parse_args() -> argparse.Namespace:
    """Define los archivos COCO, split y tamaño para validar el cargador."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--images-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "excluded_temporal_gap", "all"),
        default="train",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.image_size < 128 or args.max_batches < 0:
        parser.error("--batch-size debe ser positivo, --image-size >= 128 y --max-batches >= 0.")
    return args


def main() -> None:
    """Recorre el split y reporta muestras y solapamientos resueltos."""
    args = parse_args()
    dataset = CocoPanopticDataset(
        args.annotations,
        images_root=args.images_root,
        splits_path=args.splits if args.split != "all" else None,
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    processor = Mask2FormerImageProcessor()
    category_names = {
        index: category["name"]
        for index, category in enumerate(dataset.categories)
    }
    image_count = 0
    total_segments = 0
    total_overlaps = 0
    total_labeled_pixels = 0
    class_counts: Counter[str] = Counter()
    print(
        f"COCO: {dataset.annotations_path}\n"
        f"Partición: {args.split} | imágenes: {len(dataset)} | "
        f"tamaño de entrada: {args.image_size}×{args.image_size}"
    )
    if dataset.split_counts:
        print(
            "Imágenes revisadas por partición: "
            + ", ".join(
                f"{split}={count}"
                for split, count in dataset.split_counts.items()
                if count
            )
        )
        if args.split == "validation" and len(dataset) < 10:
            print(
                "ADVERTENCIA: la partición de validación tiene menos de 10 "
                "imágenes revisadas."
            )

    for batch_index, batch in enumerate(loader, start=1):
        for sample in batch:
            encoded = encode_for_mask2former(sample, processor, args.image_size)
            if encoded["pixel_values"].shape != (3, args.image_size, args.image_size):
                raise RuntimeError("Forma inesperada en pixel_values.")
            if encoded["mask_labels"].shape[1:] != (args.image_size, args.image_size):
                raise RuntimeError("Forma inesperada en mask_labels.")
            for class_id in encoded["class_labels"].tolist():
                class_counts[category_names[class_id]] += 1
            image_count += 1
            total_segments += int(encoded["class_labels"].numel())
            total_overlaps += int(sample["overlap_pixels_resolved"])
            total_labeled_pixels += int(sample["labeled_pixels"])
            if image_count <= 3:
                print(
                    f"  {sample['file_name']}: "
                    f"pixel_values={tuple(encoded['pixel_values'].shape)}, "
                    f"máscaras={tuple(encoded['mask_labels'].shape)}, "
                    f"clases={[category_names[i] for i in encoded['class_labels'].tolist()]}, "
                    f"solapamientos resueltos={sample['overlap_pixels_resolved']}"
                )
        if args.max_batches and batch_index >= args.max_batches:
            break
    print(
        f"LOADER OK: imágenes procesadas={image_count}, "
        f"segmentos de entrenamiento={total_segments}, "
        f"píxeles solapados asignados por prioridad={total_overlaps}, "
        f"píxeles etiquetados={total_labeled_pixels}"
    )
    print("Segmentos por clase:", dict(class_counts))


if __name__ == "__main__":
    main()
