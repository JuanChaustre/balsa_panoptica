#!/usr/bin/env python3
"""Evalúa Mask2Former y genera un reporte de comparación con Cartografía."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from balsa_panoptica.training.coco_dataset import (
    IGNORE_INDEX,
    WORKSPACE_ROOT,
    CocoPanopticDataset,
    encode_for_mask2former,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_ANNOTATIONS = PROJECT_ROOT / "data/annotations/balsa_coco_merged.json"
DEFAULT_MERGED_SPLITS = PROJECT_ROOT / "data/splits_merged.json"
DEFAULT_CARTOGRAPHY_RUN = (
    WORKSPACE_ROOT
    / "Taller 3Cartografia/cartografia_panoptica/training/runs/"
    "mask2former_stuff_reentrenado"
)
CLASS_COLORS = {
    "balsa_piece": (255, 75, 65),
    "hand_arm": (40, 165, 255),
    "work_surface": (50, 205, 90),
    "background": (210, 105, 235),
}
DISPLAY_NAMES = {
    "balsa_piece": "Pieza de balsa",
    "hand_arm": "Mano/brazo",
    "work_surface": "Superficie",
    "background": "Entorno",
}
METRIC_KEYS = {
    "IoU": ("pixel_metrics", "IoU_macro"),
    "mAP": ("instance_mAP", "mAP_50_95"),
    "F1": ("pixel_metrics", "F1_macro"),
    "Precisión": ("pixel_metrics", "precision_macro"),
}
PROJECT_THRESHOLDS = {"IoU": 0.7, "mAP": 0.6, "F1": 0.8}


def parse_args() -> argparse.Namespace:
    """Configura el checkpoint, el dataset y la cantidad de ejemplos visuales."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_MERGED_ANNOTATIONS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_MERGED_SPLITS)
    parser.add_argument("--images-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--cartography-run", type=Path, default=DEFAULT_CARTOGRAPHY_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--example-pattern",
        default="WhatsApp Image 2026-11-17*",
        help="Patrón de imágenes nuevas para la hoja de predicciones.",
    )
    parser.add_argument("--max-examples", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples debe ser positivo.")
    return args


def read_json(path: Path) -> dict[str, Any]:
    """Lee un JSON requerido y devuelve un objeto raíz."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el JSON requerido: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"El archivo no contiene JSON válido: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"El JSON debe contener un objeto raíz: {path}")
    return payload


def load_history(path: Path) -> list[dict[str, float]]:
    """Carga las tres columnas de pérdidas y valida sus valores numéricos."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el historial de pérdidas: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        expected = {"epoch", "train_loss", "validation_loss"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Columnas inesperadas en el historial: {path}")
        history = [
            {key: float(row[key]) for key in expected}
            for row in reader
        ]
    if not history:
        raise ValueError(f"El historial no contiene épocas: {path}")
    if any(not math.isfinite(value) for row in history for value in row.values()):
        raise ValueError(f"El historial contiene valores no finitos: {path}")
    return history


def nested_metric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Obtiene una métrica anidada y rechaza valores no finitos."""
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Falta la métrica {'/'.join(keys)}.")
        value = value[key]
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"La métrica {'/'.join(keys)} no es finita.")
    return number


def run_evaluator(args: argparse.Namespace, run_dir: Path) -> Path:
    """Ejecuta la evaluación de test con los mismos datos usados por el informe."""
    metrics_path = run_dir / "test_metrics.json"
    command = [
        sys.executable,
        "-m",
        "balsa_panoptica.training.evaluate_mask2former",
        "--run-dir",
        str(run_dir),
        "--annotations",
        str(args.annotations.resolve()),
        "--splits",
        str(args.splits.resolve()),
        "--images-root",
        str(args.images_root.resolve()),
        "--split",
        "test",
        "--device",
        args.device,
        "--output",
        str(metrics_path),
    ]
    if args.no_amp:
        command.append("--no-amp")
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)
    return metrics_path


def render_loss_curves(
    cartography_history: list[dict[str, float]],
    balsa_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Grafica train y validation en paneles separados para no mezclar escalas."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    for column, (name, history) in enumerate(
        (
            ("Mask2Former · Taller 3 Cartografía", cartography_history),
            ("Mask2Former · Proyecto de balsa", balsa_history),
        )
    ):
        epochs = [int(item["epoch"]) for item in history]
        axes[0, column].plot(
            epochs,
            [item["train_loss"] for item in history],
            color="#2878b5",
            marker=".",
        )
        axes[1, column].plot(
            epochs,
            [item["validation_loss"] for item in history],
            color="#d9534f",
            marker=".",
        )
        axes[0, column].set_title(f"{name} — entrenamiento")
        axes[0, column].set_ylabel("Pérdida train")
        axes[1, column].set_title(f"{name} — validación")
        axes[1, column].set_ylabel("Pérdida validation")
        axes[1, column].set_xlabel("Época")
        axes[0, column].grid(True, alpha=0.25)
        axes[1, column].grid(True, alpha=0.25)
    figure.suptitle("Curvas de pérdida por proyecto")
    figure.text(
        0.5,
        0.01,
        "Cada panel tiene escala propia; las pérdidas de tareas distintas no se comparan directamente.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def render_metric_chart(
    cartography_metrics: dict[str, Any],
    balsa_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Muestra métricas lado a lado y advierte que las tareas son distintas."""
    values = {
        "Taller 3 Cartografía": {
            name: nested_metric(cartography_metrics, path)
            for name, path in METRIC_KEYS.items()
        },
        "Proyecto de balsa": {
            name: nested_metric(balsa_metrics, path)
            for name, path in METRIC_KEYS.items()
        },
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    metric_names = list(METRIC_KEYS)
    x = np.arange(len(metric_names))
    width = 0.34
    for index, (model_name, model_values) in enumerate(values.items()):
        scores = [model_values[name] for name in metric_names]
        bars = axis.bar(
            x + (index - 0.5) * width,
            [score if score is not None else 0 for score in scores],
            width,
            label=model_name,
        )
        for bar, score in zip(bars, scores):
            axis.annotate(
                "N/D" if score is None else f"{100 * score:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(x, metric_names)
    axis.set_ylim(0, 1.12)
    axis.set_ylabel("Puntuación")
    axis.set_title("Métricas reportadas por proyecto — tareas y tests distintos")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def overlay_panoptic(
    image: np.ndarray,
    segment_ids: np.ndarray,
    segments_info: list[dict[str, Any]],
    id2label: dict[int, str],
) -> Image.Image:
    """Colorea segmentos y bordes para comparar visualmente etiquetas y predicción."""
    if segment_ids.shape != image.shape[:2]:
        raise ValueError("El mapa panóptico y la imagen tienen dimensiones distintas.")
    output = image.copy()
    for segment in segments_info:
        segment_id = int(segment["id"])
        label_id = int(segment["label_id"])
        if label_id not in id2label:
            raise ValueError(f"Clase del checkpoint desconocida: {label_id}.")
        class_name = id2label[label_id]
        mask = segment_ids == segment_id
        if mask.any():
            color = np.asarray(CLASS_COLORS[class_name], dtype=np.float32)
            output[mask] = (
                output[mask].astype(np.float32) * 0.55 + color * 0.45
            ).astype(np.uint8)

    boundary = (segment_ids >= 0) & (
        (segment_ids != np.roll(segment_ids, 1, axis=0))
        | (segment_ids != np.roll(segment_ids, -1, axis=0))
        | (segment_ids != np.roll(segment_ids, 1, axis=1))
        | (segment_ids != np.roll(segment_ids, -1, axis=1))
    )
    boundary[0, :] |= segment_ids[0, :] >= 0
    boundary[-1, :] |= segment_ids[-1, :] >= 0
    boundary[:, 0] |= segment_ids[:, 0] >= 0
    boundary[:, -1] |= segment_ids[:, -1] >= 0
    output[boundary] = (255, 255, 255)
    return Image.fromarray(output)


def render_new_image_predictions(
    args: argparse.Namespace,
    run_dir: Path,
    output_path: Path,
) -> list[str]:
    """Selecciona ejemplos espaciados del lote nuevo y muestra su split y máscaras."""
    dataset = CocoPanopticDataset(
        args.annotations,
        images_root=args.images_root,
        splits_path=None,
        split="all",
    )
    records = [
        index
        for index, record in enumerate(dataset.records)
        if fnmatch.fnmatchcase(Path(record["file_name"]).name, args.example_pattern)
    ]
    if not records:
        raise ValueError(
            f"No hay imágenes que coincidan con --example-pattern "
            f"{args.example_pattern!r} en el COCO fusionado."
        )
    selected_indices = np.linspace(
        0,
        len(records) - 1,
        min(args.max_examples, len(records)),
        dtype=int,
    ).tolist()
    selected_indices = [records[index] for index in selected_indices]
    split_by_name = {
        str(file_name): str(split_name)
        for split_name, file_names in read_json(args.splits).items()
        if isinstance(file_names, list)
        for file_name in file_names
    }

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    model_dir = run_dir / "best_model"
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
            f"Las etiquetas del modelo {actual_labels} no coinciden con COCO "
            f"{expected_labels}."
        )
    stuff_ids = {
        index
        for index, category in enumerate(dataset.categories)
        if category.get("supercategory") != "thing"
        and not bool(category.get("isthing", 0))
    }
    model.eval()

    figure, axes = plt.subplots(
        len(selected_indices),
        3,
        figsize=(15, 4.1 * len(selected_indices)),
        squeeze=False,
    )
    axes[0, 0].set_title("Foto nueva")
    axes[0, 1].set_title("Anotación revisada")
    axes[0, 2].set_title("Mask2Former balsa")
    selected_names: list[str] = []
    image_size = int(read_json(run_dir / "run_config.json").get("image_size", 256))
    for row, index in enumerate(selected_indices):
        sample = dataset[index]
        partition = split_by_name.get(sample["file_name"], "sin partición")
        selected_names.append(f"{sample['file_name']} — {partition}")
        height, width = int(sample["height"]), int(sample["width"])
        encoded = encode_for_mask2former(sample, processor, image_size)
        pixels = encoded["pixel_values"].unsqueeze(0).to(device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                output = model(pixel_values=pixels)
            prediction = processor.post_process_panoptic_segmentation(
                output,
                target_sizes=[(height, width)],
                label_ids_to_fuse=stuff_ids,
            )[0]

        gt_segments = [
            {"id": int(segment_id), "label_id": int(class_id)}
            for segment_id, class_id in sample["instance_id_to_semantic_id"].items()
        ]
        gt_image = overlay_panoptic(
            sample["image"],
            np.where(sample["panoptic_map"] == IGNORE_INDEX, -1, sample["panoptic_map"]),
            gt_segments,
            expected_labels,
        )
        prediction_image = overlay_panoptic(
            sample["image"],
            prediction["segmentation"].cpu().numpy().astype(np.int32),
            prediction["segments_info"],
            expected_labels,
        )
        panels = (
            Image.fromarray(sample["image"]),
            gt_image,
            prediction_image,
        )
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(
            f"{Path(sample['file_name']).name}\n[{partition}]",
            rotation=0,
            labelpad=115,
            va="center",
            fontsize=8,
        )

    figure.suptitle(
        "Ejemplos del lote nuevo — pueden pertenecer a train, validation o test",
        y=1,
        fontsize=13,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return selected_names


def format_percent(value: float | None) -> str:
    """Formatea una métrica entre cero y uno como porcentaje o N/D."""
    return "N/D" if value is None else f"{100 * value:.2f}%"


def model_row(
    name: str,
    task: str,
    image_count: int,
    metrics: dict[str, Any],
) -> dict[str, str]:
    """Prepara una fila común para comparar las métricas reportadas."""
    row = {
        "Modelo": name,
        "Tarea/clases": task,
        "Imágenes test": str(image_count),
    }
    row.update(
        {
            metric_name: format_percent(nested_metric(metrics, path))
            for metric_name, path in METRIC_KEYS.items()
        }
    )
    return row


def write_report(
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
    cartography_metrics: dict[str, Any],
    balsa_metrics: dict[str, Any],
    cartography_history: list[dict[str, float]],
    balsa_history: list[dict[str, float]],
) -> None:
    """Escribe CSV, gráficas y Markdown con métricas, advertencias y ejemplos."""
    report_rows = [
        model_row(
            "Mask2Former · Taller 3 Cartografía",
            "Cartografía: acera, manzana construida, zona verde, carretera",
            int(cartography_metrics["image_count"]),
            cartography_metrics,
        ),
        model_row(
            "Mask2Former · Proyecto de balsa",
            "Balsa: pieza, mano/brazo, superficie, entorno",
            int(balsa_metrics["image_count"]),
            balsa_metrics,
        ),
    ]
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    loss_chart = output_dir / "loss_curves.png"
    metrics_chart = output_dir / "metrics_comparison.png"
    prediction_chart = output_dir / "new_photo_predictions.png"
    render_loss_curves(cartography_history, balsa_history, loss_chart)
    render_metric_chart(cartography_metrics, balsa_metrics, metrics_chart)
    selected_names = render_new_image_predictions(
        args,
        run_dir,
        prediction_chart,
    )

    table_headers = list(report_rows[0])
    table = [
        "| " + " | ".join(table_headers) + " |",
        "| " + " | ".join("---" for _ in table_headers) + " |",
    ]
    table.extend(
        "| " + " | ".join(row[column] for column in table_headers) + " |"
        for row in report_rows
    )
    project_values = {
        metric_name: nested_metric(balsa_metrics, path)
        for metric_name, path in METRIC_KEYS.items()
    }
    threshold_rows = []
    for name, threshold in PROJECT_THRESHOLDS.items():
        value = project_values[name]
        result = "N/D" if value is None else ("Cumple" if value >= threshold else "No cumple")
        threshold_rows.append(
            f"| {name} | {threshold:.2f} | {format_percent(value)} | {result} |"
        )
    new_class_rows = []
    for class_name, values in balsa_metrics["pixel_metrics"]["per_class"].items():
        new_class_rows.append(
            f"| {class_name} | {format_percent(values['IoU'])} | "
            f"{format_percent(values['F1'])} | "
            f"{format_percent(values['precision'])} |"
        )
    old_class_rows = []
    for values in cartography_metrics["pixel_metrics"]["per_category"].values():
        old_class_rows.append(
            f"| {values['name']} | {format_percent(values['IoU'])} | "
            f"{format_percent(values['F1'])} | "
            f"{format_percent(values['precision'])} |"
        )
    example_list = "\n".join(f"- `{name}`" for name in selected_names)
    report = (
        "# Comparación de Mask2Former: Cartografía y Proyecto de balsa\n\n"
        "## Métricas reportadas\n\n"
        "La guía pide IoU, mAP, F1 y precisión. La tabla reúne las métricas "
        "históricas del test cartográfico y las métricas de test del modelo "
        "ajustado para balsa:\n\n"
        + "\n".join(table)
        + "\n\n**Advertencia de comparabilidad:** estos números no son una "
        "comparación controlada ni miden la misma tarea. El Mask2Former del "
        "Taller 3 fue evaluado en 15 imágenes cartográficas con cuatro clases "
        "diferentes; el modelo de balsa se evalúa en "
        f"{balsa_metrics['image_count']} imágenes del test de balsa. Además, "
        "los tests no comparten imágenes ni ground truth. Como indica la guía, "
        "el modelo antiguo no tiene etiquetas compatibles para calcular "
        "métricas de madera balsa; sus valores se muestran solo como referencia "
        "del resultado original del taller.\n\n"
        "## Umbrales de la guía del Proyecto de Corte\n\n"
        "| Métrica | Mínimo | Proyecto de balsa | Resultado |\n"
        "| --- | ---: | ---: | --- |\n"
        + "\n".join(threshold_rows)
        + "\n\nLa precisión también se reporta en las tablas, aunque la guía no "
        "especifica un umbral mínimo para ella. Los valores no implican "
        "generalización fuera de este pequeño test.\n\n"
        "## Métricas por clase\n\n"
        "### Proyecto de balsa\n\n"
        "| Clase | IoU | F1 | Precisión |\n"
        "| --- | ---: | ---: | ---: |\n"
        + "\n".join(new_class_rows)
        + "\n\n### Resultado original de Cartografía\n\n"
        "| Clase cartográfica | IoU | F1 | Precisión |\n"
        "| --- | ---: | ---: | ---: |\n"
        + "\n".join(old_class_rows)
        + "\n\n## Gráficas de entrenamiento y métricas\n\n"
        "![Métricas reportadas; tareas diferentes](./metrics_comparison.png)\n\n"
        "La gráfica pone las cifras lado a lado para consulta, no para afirmar "
        "superioridad entre tareas. Las pérdidas tienen escalas independientes "
        "y no deben compararse entre datasets ni arquitecturas.\n\n"
        "![Pérdidas train y validation](./loss_curves.png)\n\n"
        "## Predicciones de ejemplo en las fotos nuevas\n\n"
        "Se muestrean fotos nuevas del COCO fusionado sin importar su "
        "partición. Por defecto se muestran hasta seis; para incluir todas, "
        "genera el informe con `--max-examples 23`. La partición aparece junto "
        "al nombre de cada imagen. Las de train son ejemplos **in-sample**; "
        "las de validation y test son ejemplos de particiones retenidas. Estas "
        "visualizaciones son cualitativas; las métricas se calculan aparte "
        "sobre todo el split test.\n\n"
        "En cada fila se muestra foto, anotación revisada y predicción del "
        "Mask2Former de balsa. El Mask2Former cartográfico no se superpone aquí "
        "porque sus etiquetas no significan balsa, manos o superficies.\n\n"
        "![Predicciones sobre fotos nuevas](./new_photo_predictions.png)\n\n"
        "Fotos incluidas:\n\n"
        + example_list
        + "\n\n## Análisis\n\n"
        "Las métricas cartográficas describen el desempeño del modelo en su "
        "dataset original; las métricas del Proyecto de Corte describen el "
        "modelo específico de balsa en su propio test. No se debe concluir que "
        "un resultado supera al otro sin un mapeo de clases válido y un test "
        "común con las mismas imágenes y anotaciones. Los resultados del "
        "proyecto deben interpretarse considerando el tamaño del split test, "
        "el reducido split de validación y la resolución de solapamientos de "
        "máscaras documentada en el cargador.\n"
    )
    report_path = output_dir / "comparison.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Informe: {report_path}")
    print(f"CSV: {csv_path}")
    print(f"Gráfica de métricas: {metrics_chart}")
    print(f"Curvas de pérdida: {loss_chart}")
    print(f"Predicciones nuevas: {prediction_chart}")


def main() -> None:
    """Evalúa el checkpoint y produce todos los archivos del informe comparativo."""
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "best_model").is_dir():
        raise FileNotFoundError(f"No existe best_model en {run_dir}.")
    cartography_run = args.cartography_run.resolve()
    cartography_metrics_path = cartography_run / "test_metrics.json"
    cartography_history_path = cartography_run / "loss_history.csv"
    cartography_metrics = read_json(cartography_metrics_path)
    if cartography_metrics.get("split") != "test":
        raise ValueError(
            f"{cartography_metrics_path} no contiene métricas del test original."
        )
    balsa_metrics_path = run_evaluator(args, run_dir)
    balsa_metrics = read_json(balsa_metrics_path)
    if balsa_metrics.get("split") != "test":
        raise ValueError(f"{balsa_metrics_path} no contiene métricas de test.")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else run_dir / "comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(
        args,
        run_dir,
        output_dir,
        cartography_metrics,
        balsa_metrics,
        load_history(cartography_history_path),
        load_history(run_dir / "loss_history.csv"),
    )


if __name__ == "__main__":
    main()
