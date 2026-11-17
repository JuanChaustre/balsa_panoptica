#!/usr/bin/env python3
"""Interfaz local para revisar y corregir las propuestas panópticas de balsa."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image
import streamlit_drawable_canvas
from streamlit_drawable_canvas import st_canvas


PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
PRELABEL_ROOT = PROJECT_ROOT / "data" / "prelabels"
CORRECTION_ROOT = PROJECT_ROOT / "data" / "corrections"
CONFIG_PATH = PROJECT_ROOT / "annotation_config.json"
DRAW_COLOR = (255, 0, 255)
CLASS_COLORS = {
    1: (245, 70, 55),
    2: (30, 165, 255),
    3: (40, 210, 100),
    4: (220, 100, 235),
}
MASK_ALPHA = 105

if not hasattr(streamlit_drawable_canvas.st_image, "image_to_url"):
    from streamlit.elements.lib import image_utils
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def _canvas_image_to_url(
        image: Image.Image,
        width: int,
        clamp: bool,
        channels: str,
        output_format: str,
        image_id: str,
    ) -> str:
        return image_utils.image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    streamlit_drawable_canvas.st_image.image_to_url = _canvas_image_to_url


def read_json(path: Path) -> dict[str, Any]:
    """Carga un archivo JSON requerido para la app."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def safe_path(root: Path, relative_path: str) -> Path:
    """Resuelve una ruta relativa y rechaza cualquier salida de la carpeta raíz."""
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"La ruta sale de la carpeta permitida: {relative_path}")
    return resolved


def annotation_color(category_id: int) -> tuple[int, int, int]:
    try:
        return CLASS_COLORS[category_id]
    except KeyError as error:
        raise ValueError(f"No hay color para la clase {category_id}.") from error


def make_overlay(image: Image.Image, annotations: list[dict[str, Any]]) -> Image.Image:
    """Superpone las máscaras coloreadas conservando visible la foto original."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for annotation in annotations:
        mask = annotation["mask"]
        if mask.shape != (image.height, image.width):
            raise ValueError(f"Dimensiones de máscara inesperadas: {mask.shape}")
        color = annotation_color(int(annotation["category_id"]))
        layer = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        layer[mask] = (*color, MASK_ALPHA)
        overlay = Image.alpha_composite(overlay, Image.fromarray(layer))
    return Image.alpha_composite(base, overlay).convert("RGB")


def load_annotations(
    run_dir: Path,
    image_record: dict[str, Any],
    saved_image: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Carga máscaras corregidas si existen; de lo contrario, usa las propuestas."""
    source_annotations = (
        saved_image["annotations"]
        if saved_image is not None
        else image_record.get("annotations", [])
    )
    annotations = []
    for index, source in enumerate(source_annotations, start=1):
        mask_root = CORRECTION_ROOT / run_dir.name if saved_image is not None else run_dir
        mask_path = safe_path(mask_root, source["mask_file"])
        if not mask_path.is_file():
            raise FileNotFoundError(f"No existe la máscara: {mask_path}")
        with Image.open(mask_path) as source_mask:
            mask_image = source_mask.convert("L")
            if mask_image.size != (image_record["width"], image_record["height"]):
                raise ValueError(f"Dimensiones incorrectas en la máscara {mask_path}")
            mask = np.asarray(mask_image) > 0
        annotations.append(
            {
                "id": int(source.get("id", index)),
                "category_id": int(source["category_id"]),
                "detected_text": source.get("detected_text", "Anotación manual"),
                "detection_score": source.get("detection_score"),
                "sam_iou_estimate": source.get("sam_iou_estimate"),
                "mask_file": source["mask_file"],
                "mask": mask,
            }
        )
    return annotations


def bbox_for_mask(mask: np.ndarray) -> list[int]:
    """Devuelve el bounding box COCO [x, y, ancho, alto] de una máscara."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def save_image_annotations(
    correction_dir: Path,
    corrections: dict[str, Any],
    image_record: dict[str, Any],
    annotations: list[dict[str, Any]],
    reviewed: bool,
) -> None:
    """Persiste las máscaras y metadatos de una imagen sin alterar las propuestas."""
    mask_dir = correction_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_record["file_name"]).stem
    saved_annotations = []
    for annotation in annotations:
        mask_file = f"masks/{image_stem}_{annotation['id']:04d}.png"
        mask_path = safe_path(correction_dir, mask_file)
        Image.fromarray(annotation["mask"].astype(np.uint8) * 255).save(mask_path)
        saved_annotations.append(
            {
                "id": int(annotation["id"]),
                "category_id": int(annotation["category_id"]),
                "detected_text": annotation["detected_text"],
                "detection_score": annotation["detection_score"],
                "sam_iou_estimate": annotation["sam_iou_estimate"],
                "bbox_xywh": bbox_for_mask(annotation["mask"]),
                "area": int(annotation["mask"].sum()),
                "mask_file": mask_file,
            }
        )
    corrections["images"][image_record["file_name"]] = {
        "width": image_record["width"],
        "height": image_record["height"],
        "reviewed": reviewed,
        "annotations": saved_annotations,
    }
    correction_dir.mkdir(parents=True, exist_ok=True)
    destination = correction_dir / "corrections.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(corrections, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def polygon_from_canvas(
    image_data: np.ndarray | None, width: int, height: int
) -> np.ndarray:
    """Extrae del lienzo los píxeles pintados con el color reservado al polígono."""
    if image_data is None or image_data.shape[:2] != (height, width):
        raise ValueError("No hay un polígono válido en el lienzo.")
    pixels = image_data[..., :3]
    red, green, blue = DRAW_COLOR
    selected = (
        (pixels[..., 0] >= red - 35)
        & (pixels[..., 1] <= green + 45)
        & (pixels[..., 2] >= blue - 35)
    )
    if not np.any(selected):
        raise ValueError("Dibuja un polígono magenta sobre la imagen antes de aplicarlo.")
    return selected


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """Codifica una máscara como RLE COCO recorriendo primero las columnas."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    boundaries = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    counts = np.diff(np.concatenate(([0], boundaries, [pixels.size]))).tolist()
    if pixels[0]:
        counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def make_coco_export(
    workspace_root: Path,
    run_dir: Path,
    corrections: dict[str, Any],
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construye un COCO exportable que incluye solo imágenes marcadas revisadas."""
    images = []
    annotations = []
    annotation_id = 1
    for file_name, saved_image in corrections["images"].items():
        if not saved_image["reviewed"]:
            continue
        image_id = len(images) + 1
        image_path = safe_path(workspace_root, file_name)
        if not image_path.is_file():
            raise FileNotFoundError(f"No existe la imagen: {image_path}")
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": saved_image["width"],
                "height": saved_image["height"],
            }
        )
        correction_dir = CORRECTION_ROOT / run_dir.name
        for item in saved_image["annotations"]:
            mask_path = safe_path(correction_dir, item["mask_file"])
            with Image.open(mask_path) as source_mask:
                mask = np.asarray(source_mask.convert("L")) > 0
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": item["category_id"],
                    "segmentation": encode_uncompressed_rle(mask),
                    "area": int(mask.sum()),
                    "bbox": item["bbox_xywh"],
                    "iscrowd": 0,
                    "score": item["detection_score"],
                }
            )
            annotation_id += 1
    return {
        "info": {
            "description": (
                "Anotaciones corregidas manualmente; se incluyen solo imágenes "
                "marcadas como revisadas."
            ),
            "source_proposals": run_dir.name,
        },
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": category["id"],
                "name": category["name"],
                "supercategory": "thing" if category["isthing"] else "stuff",
            }
            for category in categories
        ],
    }


def category_name(category_id: int, categories_by_id: dict[int, dict[str, Any]]) -> str:
    return categories_by_id[category_id]["display_name"]


def main() -> None:
    """Ejecuta la interfaz Streamlit de revisión y corrección de anotaciones."""
    st.title("🪵 Etiquetado panóptico de madera balsa")
    st.caption(
        "Revisa las propuestas, corrige máscaras con polígonos y guarda "
        "anotaciones localmente. Las propuestas originales no se modifican."
    )

    runs = sorted(
        (path.parent for path in PRELABEL_ROOT.glob("*/proposals.json")),
        key=lambda path: path.name,
        reverse=True,
    )
    if not runs:
        st.error(f"No hay ejecuciones de pre-etiquetado en {PRELABEL_ROOT}.")
        return
    run_names = [path.name for path in runs]
    selected_run_name = st.sidebar.selectbox(
        "Ejecución de propuestas",
        run_names,
        format_func=lambda value: value.replace("run_", ""),
    )
    run_dir = PRELABEL_ROOT / selected_run_name
    proposals = read_json(run_dir / "proposals.json")
    category_config = read_json(CONFIG_PATH)["categories"]
    categories_by_id = {int(item["id"]): item for item in category_config}
    corrections_dir = CORRECTION_ROOT / run_dir.name
    corrections_path = corrections_dir / "corrections.json"
    corrections = (
        read_json(corrections_path)
        if corrections_path.is_file()
        else {"source_proposals": run_dir.name, "images": {}}
    )
    if corrections.get("source_proposals") != run_dir.name:
        st.error("Las correcciones no corresponden a esta ejecución de propuestas.")
        return

    image_records = proposals["images"]
    saved_images = corrections["images"]
    reviewed_count = sum(bool(item.get("reviewed")) for item in saved_images.values())
    st.sidebar.metric("Imágenes revisadas", f"{reviewed_count}/{len(image_records)}")
    st.sidebar.progress(reviewed_count / max(1, len(image_records)))
    image_filter = st.sidebar.selectbox(
        "Mostrar",
        ("Todas", "Pendientes de revisión", "Sin propuesta de balsa", "Con varias instancias"),
    )
    filtered_records = []
    for record in image_records:
        saved = saved_images.get(record["file_name"])
        annotations = saved["annotations"] if saved else record["annotations"]
        has_balsa = any(int(item["category_id"]) == 1 for item in annotations)
        if image_filter == "Pendientes de revisión" and saved and saved.get("reviewed"):
            continue
        if image_filter == "Sin propuesta de balsa" and has_balsa:
            continue
        if image_filter == "Con varias instancias" and len(annotations) < 2:
            continue
        filtered_records.append(record)
    if not filtered_records:
        st.info("No hay imágenes para mostrar con este filtro.")
        return

    selector_key = f"image_selector_{run_dir.name}_{image_filter}"
    filenames = [item["file_name"] for item in filtered_records]
    if st.session_state.get(selector_key) not in filenames:
        st.session_state[selector_key] = filenames[0]
    current_index = filenames.index(st.session_state[selector_key])
    previous, image_select, following = st.columns([1, 8, 1])
    if previous.button("← Anterior", use_container_width=True, disabled=current_index == 0):
        st.session_state[selector_key] = filenames[current_index - 1]
    if following.button(
        "Siguiente →",
        use_container_width=True,
        disabled=current_index == len(filenames) - 1,
    ):
        st.session_state[selector_key] = filenames[current_index + 1]
    selected_file = image_select.selectbox(
        "Imagen del dataset",
        filenames,
        key=selector_key,
        format_func=lambda value: Path(value).name,
        label_visibility="collapsed",
    )
    image_record = next(item for item in filtered_records if item["file_name"] == selected_file)
    saved_image = saved_images.get(selected_file)
    current_key = f"{run_dir.name}:{selected_file}"

    if st.session_state.get("loaded_image_key") != current_key:
        image_path = safe_path(WORKSPACE_ROOT, selected_file)
        if not image_path.is_file():
            st.error(f"No se encontró la imagen original: {image_path}")
            return
        with Image.open(image_path) as source:
            st.session_state["current_image"] = source.convert("RGB")
        st.session_state["current_annotations"] = load_annotations(
            run_dir,
            image_record,
            saved_image,
        )
        st.session_state["loaded_image_key"] = current_key
        st.session_state["canvas_revision"] = 0

    image = st.session_state["current_image"]
    annotations = st.session_state["current_annotations"]
    width, height = image.size
    progress = f"{filenames.index(selected_file) + 1}/{len(filenames)}"
    reviewed = bool(saved_image and saved_image.get("reviewed"))
    review_state_key = f"reviewed_{current_key}"
    st.subheader(f"{Path(selected_file).name} · {progress}")
    st.caption(
        f"{width} × {height} px · "
        f"{'Revisada' if reviewed else 'Pendiente de revisión'} · "
        f"{len(annotations)} regiones etiquetadas"
    )

    annotation_labels = {
        int(item["id"]): (
            f"#{int(item['id']):03d} · "
            f"{category_name(int(item['category_id']), categories_by_id)} · "
            f"{item['detected_text']}"
        )
        for item in annotations
    }
    with st.sidebar:
        st.divider()
        st.subheader("Instancias y regiones")
        if annotations:
            selected_annotation_id = st.selectbox(
                "Selecciona una máscara",
                list(annotation_labels),
                format_func=lambda value: annotation_labels[value],
                key=f"annotation_select_{current_key}",
            )
        else:
            selected_annotation_id = None
            st.info("No hay máscaras todavía. Dibuja un polígono para añadir una.")

        target_category_id = st.selectbox(
            "Clase para la corrección",
            list(categories_by_id),
            format_func=lambda value: (
                f"{categories_by_id[value]['display_name']} "
                f"({'thing' if categories_by_id[value]['isthing'] else 'stuff'})"
            ),
            key=f"category_select_{current_key}",
        )
        editing_action = st.radio(
            "Herramienta del lienzo",
            ("Visualizar", "Agregar polígono", "Borrar polígono de máscara"),
            key=f"editing_action_{current_key}",
        )
        if st.button(
            "Cambiar clase de la máscara seleccionada",
            disabled=selected_annotation_id is None,
            use_container_width=True,
        ):
            target = next(item for item in annotations if item["id"] == selected_annotation_id)
            target["category_id"] = target_category_id
            save_image_annotations(
                corrections_dir,
                corrections,
                image_record,
                annotations,
                bool(st.session_state.get(review_state_key, reviewed)),
            )
            st.success("Clase actualizada y guardada.")
            st.rerun()

        if st.button(
            "Eliminar máscara seleccionada",
            disabled=selected_annotation_id is None,
            use_container_width=True,
        ):
            st.session_state["current_annotations"] = [
                item for item in annotations if item["id"] != selected_annotation_id
            ]
            save_image_annotations(
                corrections_dir,
                corrections,
                image_record,
                st.session_state["current_annotations"],
                bool(st.session_state.get(review_state_key, reviewed)),
            )
            st.success("Máscara eliminada y cambios guardados.")
            st.rerun()

        st.divider()
        mark_reviewed = st.checkbox(
            "Marcar esta imagen como revisada",
            value=reviewed,
            key=review_state_key,
        )
        if st.button("Guardar revisión", type="primary", use_container_width=True):
            save_image_annotations(
                corrections_dir,
                corrections,
                image_record,
                annotations,
                mark_reviewed,
            )
            st.success("Revisión guardada localmente.")
            st.rerun()

    draw_mode = "polygon" if editing_action != "Visualizar" else "transform"
    canvas_key = (
        f"canvas_{run_dir.name}_{Path(selected_file).stem}_"
        f"{st.session_state['canvas_revision']}"
    )
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 255, 1.0)",
        stroke_width=2,
        stroke_color="#ff00ff",
        background_image=make_overlay(image, annotations),
        update_streamlit=True,
        height=height,
        width=width,
        drawing_mode=draw_mode,
        display_toolbar=editing_action != "Visualizar",
        key=canvas_key,
    )
    if editing_action != "Visualizar":
        st.caption(
            "Cierra cada polígono con clic derecho. El magenta marca el área a "
            "agregar o borrar; usa Aplicar para confirmar."
        )
        if st.button("Aplicar polígono", type="primary"):
            try:
                polygon_mask = polygon_from_canvas(canvas_result.image_data, width, height)
                if editing_action == "Agregar polígono":
                    if categories_by_id[target_category_id]["isthing"]:
                        annotation_id = max(
                            (item["id"] for item in annotations),
                            default=0,
                        ) + 1
                        annotations.append(
                            {
                                "id": annotation_id,
                                "category_id": target_category_id,
                                "detected_text": "Anotación manual",
                                "detection_score": None,
                                "sam_iou_estimate": None,
                                "mask_file": "",
                                "mask": polygon_mask,
                            }
                        )
                    else:
                        existing = next(
                            (
                                item
                                for item in annotations
                                if item["category_id"] == target_category_id
                                and not categories_by_id[item["category_id"]]["isthing"]
                            ),
                            None,
                        )
                        if existing is None:
                            annotation_id = max(
                                (item["id"] for item in annotations),
                                default=0,
                            ) + 1
                            annotations.append(
                                {
                                    "id": annotation_id,
                                    "category_id": target_category_id,
                                    "detected_text": "Anotación manual",
                                    "detection_score": None,
                                    "sam_iou_estimate": None,
                                    "mask_file": "",
                                    "mask": polygon_mask,
                                }
                            )
                        else:
                            existing["mask"] |= polygon_mask
                else:
                    if selected_annotation_id is None:
                        raise ValueError("Selecciona primero la máscara que quieres corregir.")
                    target = next(
                        item for item in annotations if item["id"] == selected_annotation_id
                    )
                    target["mask"] &= ~polygon_mask
                    if not np.any(target["mask"]):
                        annotations.remove(target)

                st.session_state["current_annotations"] = annotations
                save_image_annotations(
                    corrections_dir,
                    corrections,
                    image_record,
                    annotations,
                    bool(st.session_state.get(review_state_key, reviewed)),
                )
                st.session_state["canvas_revision"] += 1
                st.success("Máscara corregida y guardada.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))

    st.subheader("Exportar imágenes revisadas")
    st.caption(
        "El archivo COCO incluye únicamente imágenes guardadas como revisadas; "
        "contiene thing y stuff en formato de máscaras RLE."
    )
    if st.button("Preparar descarga COCO JSON"):
        coco = make_coco_export(WORKSPACE_ROOT, run_dir, corrections, category_config)
        if not coco["images"]:
            st.info("Marca al menos una imagen como revisada para exportar.")
        else:
            coco_bytes = json.dumps(coco, ensure_ascii=False).encode("utf-8")
            st.download_button(
                "Descargar anotaciones revisadas (COCO JSON)",
                data=coco_bytes,
                file_name=f"{run_dir.name}_anotaciones_revisadas.json",
                mime="application/json",
            )
            st.caption(
                f"{len(coco['images'])} imágenes revisadas · "
                f"{len(coco['annotations'])} máscaras exportadas"
            )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Etiquetado panóptico de balsa",
        page_icon="🪵",
        layout="wide",
    )
    main()
