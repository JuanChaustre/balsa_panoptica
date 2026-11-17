#!/usr/bin/env python3
"""Combina exportaciones COCO sin modificar las fuentes ni sus anotaciones."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_ORIGINAL = PROJECT_ROOT / "data/annotations/balsa_coco.json"
DEFAULT_ADDITIONAL = PROJECT_ROOT / "data/annotations/balsa_coco2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/annotations/balsa_coco_merged.json"
DEFAULT_SPLITS = PROJECT_ROOT / "data/splits.json"
DEFAULT_MERGED_SPLITS = PROJECT_ROOT / "data/splits_merged.json"


def read_json(path: Path) -> dict[str, Any]:
    """Carga un archivo COCO o splits y exige que la raíz sea un objeto JSON."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo no contiene JSON válido: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"El JSON debe contener un objeto raíz: {path}")
    return payload


def category_signature(categories: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """Representa categorías en orden para detectar incompatibilidades entre COCO."""
    return [
        (
            int(category["id"]),
            str(category["name"]),
            str(category.get("supercategory", "")),
        )
        for category in categories
    ]


def _validate_dataset(
    payload: dict[str, Any],
    source: Path,
    workspace_root: Path,
) -> None:
    """Valida IDs, referencias, rutas y dimensiones antes de combinar datasets."""
    categories = payload.get("categories")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"{source} no declara categorías COCO.")
    if not isinstance(images, list) or not images:
        raise ValueError(f"{source} no declara imágenes COCO.")
    if not isinstance(annotations, list):
        raise ValueError(f"{source} no declara anotaciones COCO.")

    image_ids: set[int] = set()
    file_names: set[str] = set()
    for image in images:
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        if image_id in image_ids:
            raise ValueError(f"ID de imagen duplicado en {source}: {image_id}.")
        if file_name in file_names:
            raise ValueError(f"Imagen duplicada en {source}: {file_name}.")
        image_ids.add(image_id)
        file_names.add(file_name)
        path = (workspace_root / file_name).resolve()
        if not path.is_relative_to(workspace_root.resolve()):
            raise ValueError(f"La ruta de imagen escapa del workspace: {file_name}")
        if not path.is_file():
            raise FileNotFoundError(f"No existe la imagen COCO: {path}")
        if int(image["width"]) < 1 or int(image["height"]) < 1:
            raise ValueError(f"Dimensiones inválidas en {source}: {file_name}")

    category_ids = {int(category["id"]) for category in categories}
    annotation_ids: set[int] = set()
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise ValueError(
                f"ID de anotación duplicado en {source}: {annotation_id}."
            )
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(
                f"La anotación {annotation_id} referencia una imagen inexistente."
            )
        if int(annotation["category_id"]) not in category_ids:
            raise ValueError(
                f"La anotación {annotation_id} referencia una categoría inexistente."
            )
        annotation_ids.add(annotation_id)


def merge_payloads(
    original: dict[str, Any],
    additional: dict[str, Any],
) -> dict[str, Any]:
    """Combina anotaciones y remapea IDs del lote adicional para evitar colisiones."""
    categories = original["categories"]
    if category_signature(categories) != category_signature(additional["categories"]):
        raise ValueError("Las exportaciones tienen categorías distintas o en otro orden.")

    original_names = {str(item["file_name"]) for item in original["images"]}
    additional_names = {str(item["file_name"]) for item in additional["images"]}
    duplicates = original_names & additional_names
    if duplicates:
        raise ValueError(
            "Ambos COCO incluyen las mismas imágenes: "
            + ", ".join(sorted(duplicates)[:5])
        )

    next_image_id = max(int(item["id"]) for item in original["images"]) + 1
    image_id_map: dict[int, int] = {}
    additional_images: list[dict[str, Any]] = []
    for offset, image in enumerate(additional["images"]):
        old_id = int(image["id"])
        new_id = next_image_id + offset
        image_id_map[old_id] = new_id
        additional_images.append({**image, "id": new_id})

    next_annotation_id = max(
        (int(item["id"]) for item in original["annotations"]),
        default=0,
    ) + 1
    additional_annotations: list[dict[str, Any]] = []
    for offset, annotation in enumerate(additional["annotations"]):
        image_id = int(annotation["image_id"])
        if image_id not in image_id_map:
            raise ValueError(f"ID de imagen adicional inesperado: {image_id}.")
        additional_annotations.append(
            {
                **annotation,
                "id": next_annotation_id + offset,
                "image_id": image_id_map[image_id],
            }
        )

    merged = dict(original)
    merged["images"] = [*original["images"], *additional_images]
    merged["annotations"] = [
        *original["annotations"],
        *additional_annotations,
    ]
    return merged


def capture_group_key(file_name: str) -> str:
    """Agrupa nombres con sufijos de copia, como '(1)', bajo la misma captura."""
    image_name = Path(file_name).name
    return re.sub(r" \(\d+\)(?=\.[^.]+$)", "", image_name).casefold()


def merge_splits(
    base_splits: dict[str, Any],
    original: dict[str, Any],
    additional: dict[str, Any],
) -> dict[str, list[str]]:
    """Asigna grupos temporales nuevos a train/validation/test sin mover los antiguos."""
    splits = {
        str(name): [str(file_name) for file_name in file_names]
        for name, file_names in base_splits.items()
        if isinstance(file_names, list)
    }
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(splits):
        raise ValueError("Los splits base deben incluir train, validation y test.")
    existing_memberships: dict[str, str] = {}
    for split_name, file_names in splits.items():
        for file_name in file_names:
            if file_name in existing_memberships:
                raise ValueError(
                    f"{file_name} está repetida en las particiones "
                    f"{existing_memberships[file_name]} y {split_name}."
                )
            existing_memberships[file_name] = split_name

    original_names = {str(item["file_name"]) for item in original["images"]}
    if not original_names.issubset(existing_memberships):
        missing = sorted(original_names - existing_memberships.keys())
        raise ValueError(
            "Faltan imágenes COCO original en splits.json: "
            + ", ".join(missing[:5])
        )

    new_names = [str(item["file_name"]) for item in additional["images"]]
    overlap = set(new_names) & existing_memberships.keys()
    if overlap:
        raise ValueError(
            "El split original ya incluye imágenes de balsa_coco2: "
            + ", ".join(sorted(overlap)[:5])
        )
    ordered_new_names = sorted(
        new_names,
        key=lambda name: (
            capture_group_key(name),
            Path(name).name.casefold(),
        ),
    )
    capture_groups: list[list[str]] = []
    for file_name in ordered_new_names:
        if not capture_groups or capture_group_key(file_name) != capture_group_key(
            capture_groups[-1][-1]
        ):
            capture_groups.append([])
        capture_groups[-1].append(file_name)

    total_new = len(ordered_new_names)
    cumulative = 0
    for group in capture_groups:
        midpoint = (cumulative + len(group) / 2) / total_new
        if midpoint < 0.6:
            split_name = "train"
        elif midpoint < 0.8:
            split_name = "validation"
        else:
            split_name = "test"
        splits[split_name].extend(group)
        cumulative += len(group)

    merged_names = original_names | set(new_names)
    split_names = {
        name: [file_name for file_name in values if file_name in merged_names]
        for name, values in splits.items()
    }
    if set().union(*(set(values) for values in split_names.values())) != merged_names:
        raise ValueError("No se pudieron asignar todas las imágenes fusionadas.")
    return split_names


def unreviewed_new_images(
    images_root: Path,
    pattern: str,
    additional: dict[str, Any],
) -> list[Path]:
    """Encuentra fotos del lote que no aparecen en el COCO adicional revisado."""
    matched = sorted(
        path
        for path in images_root.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not matched:
        raise ValueError(
            f"El patrón {pattern!r} no encontró fotos nuevas en {images_root}."
        )
    reviewed = {str(image["file_name"]) for image in additional["images"]}
    reviewed_paths: set[str] = set()
    for file_name in reviewed:
        image_path = (images_root.parent / file_name).resolve()
        if not image_path.is_relative_to(images_root.resolve()):
            raise ValueError(f"La imagen adicional no está dentro de {images_root}: {file_name}")
        if not fnmatch.fnmatchcase(image_path.name, pattern):
            raise ValueError(
                f"La imagen adicional no coincide con el patrón del lote: {file_name}"
            )
        if image_path.parent != images_root.resolve():
            raise ValueError(f"La imagen adicional no está directamente en {images_root}.")
        reviewed_paths.add(str(image_path))
    return [path.resolve() for path in matched if str(path.resolve()) not in reviewed_paths]


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    """Escribe JSON mediante un archivo temporal y reemplazo atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"El destino ya existe: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def merge_datasets(
    original_path: Path,
    additional_path: Path,
    splits_path: Path,
    output_path: Path,
    merged_splits_path: Path,
    workspace_root: Path,
    new_images_dir: Path,
    *,
    new_image_pattern: str,
    delete_unreviewed: bool = False,
    overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, list[str]], list[Path]]:
    """Valida, fusiona y escribe los COCO/splits; borrar no revisadas es opcional."""
    original = read_json(original_path)
    additional = read_json(additional_path)
    base_splits = read_json(splits_path)
    workspace_root = workspace_root.resolve()
    new_images_dir = new_images_dir.resolve()
    _validate_dataset(original, original_path, workspace_root)
    _validate_dataset(additional, additional_path, workspace_root)
    missing_images = unreviewed_new_images(
        new_images_dir,
        new_image_pattern,
        additional,
    )
    if missing_images and not delete_unreviewed:
        print("Fotos nuevas sin revisión (excluidas del COCO fusionado):")
        for path in missing_images:
            print(f"  {path}")

    merged = merge_payloads(original, additional)
    merged_splits = merge_splits(base_splits, original, additional)
    _validate_dataset(merged, output_path, workspace_root)

    if not overwrite and (output_path.exists() or merged_splits_path.exists()):
        raise FileExistsError(
            "El COCO o las particiones fusionadas ya existen; "
            "usa otro destino o --overwrite."
        )
    write_json(output_path, merged, overwrite=overwrite)
    write_json(merged_splits_path, merged_splits, overwrite=overwrite)

    if delete_unreviewed:
        for path in missing_images:
            if (
                path.parent != new_images_dir
                or path.is_symlink()
                or not fnmatch.fnmatchcase(path.name, new_image_pattern)
            ):
                raise ValueError(f"Se rechazó borrar una ruta fuera del lote: {path}")
        for path in missing_images:
            path.unlink()

    return merged, merged_splits, missing_images


def parse_args() -> argparse.Namespace:
    """Define los archivos de origen, salidas y guardas para borrar fotos nuevas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--additional", type=Path, default=DEFAULT_ADDITIONAL)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merged-splits", type=Path, default=DEFAULT_MERGED_SPLITS)
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--new-images-dir", type=Path, default=WORKSPACE_ROOT / "images")
    parser.add_argument(
        "--new-image-pattern",
        default="WhatsApp Image 2026-11-17*",
        help="Patrón directo dentro de images/ que identifica solo el lote adicional.",
    )
    parser.add_argument(
        "--delete-unreviewed-new-images",
        action="store_true",
        help="Elimina únicamente fotos del lote adicional que no están en su COCO.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Ejecuta la fusión y resume conteos y fotos no revisadas."""
    args = parse_args()
    merged, splits, removed = merge_datasets(
        args.original,
        args.additional,
        args.splits,
        args.output,
        args.merged_splits,
        args.workspace_root,
        args.new_images_dir,
        new_image_pattern=args.new_image_pattern,
        delete_unreviewed=args.delete_unreviewed_new_images,
        overwrite=args.overwrite,
    )
    split_counts = {
        name: sum(image["file_name"] in values for image in merged["images"])
        for name, values in splits.items()
    }
    print(
        f"COCO fusionado: {len(merged['images'])} imágenes, "
        f"{len(merged['annotations'])} anotaciones"
    )
    print(f"Particiones: {split_counts}")
    print(f"COCO: {args.output.resolve()}")
    print(f"Particiones: {args.merged_splits.resolve()}")
    if args.delete_unreviewed_new_images:
        print(f"Fotos nuevas defectuosas eliminadas: {len(removed)}")
        for path in removed:
            print(f"  {path}")


if __name__ == "__main__":
    main()
