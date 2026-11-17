#!/usr/bin/env python3
"""Aplicación local de segmentación panóptica en vivo desde la cámara."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
from av import VideoFrame
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import webrtc_streamer
from transformers.models.mask2former.image_processing_mask2former import (
    Mask2FormerImageProcessor,
)
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentation,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "training" / "runs"
IMAGE_SIZE = 256
SCORE_THRESHOLD = 0.5
OVERLAY_ALPHA = 0.45
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

st.set_page_config(
    page_title="Detección en vivo · Balsa",
    page_icon="🪵",
    layout="wide",
)


@dataclass
class Detector:
    """Agrupa el modelo cargado y los recursos compartidos por el callback."""

    processor: Mask2FormerImageProcessor
    model: Mask2FormerForUniversalSegmentation
    device: torch.device
    id2label: dict[int, str]
    stuff_ids: set[int]
    image_size: int
    inference_lock: threading.Lock


def available_checkpoints() -> list[Path]:
    """Lista checkpoints completos y devuelve primero el run más reciente."""
    if not RUNS_ROOT.is_dir():
        return []
    return sorted(
        (
            path
            for path in RUNS_ROOT.glob("mask2former_balsa_*/best_model")
            if (path / "model.safetensors").is_file()
        ),
        key=lambda path: path.parent.name,
        reverse=True,
    )


@st.cache_resource(show_spinner="Cargando Mask2Former en el dispositivo...")
def load_detector(checkpoint: str, device_name: str) -> Detector:
    """Carga el modelo, verifica sus clases y prepara el dispositivo de inferencia."""
    checkpoint_path = Path(checkpoint)
    if not (checkpoint_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint_path)
    id2label = {
        int(index): str(label)
        for index, label in model.config.id2label.items()
    }
    expected_labels = set(CLASS_COLORS)
    if set(id2label.values()) != expected_labels:
        raise ValueError(
            f"Las clases del checkpoint {sorted(id2label.values())} no coinciden "
            f"con las clases esperadas {sorted(expected_labels)}."
        )
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    stuff_ids = {
        index
        for index, name in id2label.items()
        if name in {"work_surface", "background"}
    }
    run_config_path = checkpoint_path.parent.parent / "run_config.json"
    image_size = IMAGE_SIZE
    if run_config_path.is_file():
        image_size = int(
            json.loads(run_config_path.read_text(encoding="utf-8")).get(
                "image_size",
                IMAGE_SIZE,
            )
        )
    return Detector(
        processor=processor,
        model=model,
        device=device,
        id2label=id2label,
        stuff_ids=stuff_ids,
        image_size=image_size,
        inference_lock=threading.Lock(),
    )


def estimate_mask_dimensions(mask: np.ndarray) -> tuple[float, float]:
    """Estima largo y ancho de una máscara siguiendo sus ejes principales en píxeles."""
    if mask.ndim != 2:
        raise ValueError("La máscara debe ser bidimensional.")
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("No se pueden estimar dimensiones de una máscara vacía.")
    if xs.size == 1:
        return 1.0, 1.0

    coordinates = np.column_stack((xs, ys)).astype(np.float64)
    centered = coordinates - coordinates.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    projections = coordinates @ axes
    pixel_extent = np.abs(axes).sum(axis=0)
    dimensions = np.ptp(projections, axis=0) + pixel_extent
    length, width = sorted((float(value) for value in dimensions), reverse=True)
    return length, width


def overlay_panoptic(
    frame: np.ndarray,
    segmentation: np.ndarray,
    segments_info: list[dict[str, Any]],
    id2label: dict[int, str],
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Dibuja segmentos y puntajes; solo las piezas balsa reciben medidas."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("El frame debe tener forma alto × ancho × 3.")
    if segmentation.shape != frame.shape[:2]:
        raise ValueError("El mapa panóptico y el frame tienen dimensiones distintas.")
    if not 0 <= alpha <= 1:
        raise ValueError("La transparencia debe estar entre 0 y 1.")

    output = frame.copy()
    labels: list[tuple[int, int, str]] = []
    for segment in segments_info:
        segment_id = int(segment["id"])
        label_id = int(segment["label_id"])
        if label_id not in id2label:
            raise ValueError(f"El modelo devolvió una clase desconocida: {label_id}.")
        class_name = id2label[label_id]
        mask = segmentation == segment_id
        if not mask.any():
            continue
        color = np.asarray(CLASS_COLORS[class_name], dtype=np.float32)
        output[mask] = (
            output[mask].astype(np.float32) * (1 - alpha) + color * alpha
        ).astype(np.uint8)
        ys, xs = np.nonzero(mask)
        score = float(segment.get("score", 0.0))
        dimension_label = ""
        if class_name == "balsa_piece":
            length, width = estimate_mask_dimensions(mask)
            dimension_label = f" · {length:.0f} × {width:.0f} px"
        labels.append(
            (
                int(xs.min()),
                int(ys.min()),
                f"{DISPLAY_NAMES[class_name]} {score:.2f}{dimension_label}",
            )
        )

    boundary = (segmentation >= 0) & (
        (segmentation != np.roll(segmentation, 1, axis=0))
        | (segmentation != np.roll(segmentation, -1, axis=0))
        | (segmentation != np.roll(segmentation, 1, axis=1))
        | (segmentation != np.roll(segmentation, -1, axis=1))
    )
    boundary[0, :] |= segmentation[0, :] >= 0
    boundary[-1, :] |= segmentation[-1, :] >= 0
    boundary[:, 0] |= segmentation[:, 0] >= 0
    boundary[:, -1] |= segmentation[:, -1] >= 0
    output[boundary] = (255, 255, 255)

    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    if not labels:
        draw.text(
            (12, 12),
            "Sin detecciones sobre el umbral",
            font=font,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    for x, y, label in labels:
        draw.text(
            (x, y),
            label,
            font=font,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(image)


def make_frame_callback(detector: Detector):
    """Crea el callback WebRTC que predice y anota cada frame de cámara."""
    def process_frame(frame: VideoFrame) -> VideoFrame:
        rgb = frame.to_ndarray(format="rgb24")
        height, width = rgb.shape[:2]
        resized = Image.fromarray(rgb).resize(
            (detector.image_size, detector.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        pixel_values = detector.processor(
            images=np.asarray(resized),
            do_resize=False,
            return_tensors="pt",
        )["pixel_values"].to(detector.device)

        with detector.inference_lock, torch.inference_mode():
            with torch.autocast(
                device_type=detector.device.type,
                dtype=torch.float16,
                enabled=detector.device.type == "cuda",
            ):
                output = detector.model(pixel_values=pixel_values)
            prediction = detector.processor.post_process_panoptic_segmentation(
                output,
                target_sizes=[(height, width)],
                label_ids_to_fuse=detector.stuff_ids,
                threshold=SCORE_THRESHOLD,
                mask_threshold=0.5,
                overlap_mask_area_threshold=0.8,
            )[0]
        segmentation = prediction["segmentation"].cpu().numpy().astype(np.int32)
        annotated = overlay_panoptic(
            rgb,
            segmentation,
            prediction["segments_info"],
            detector.id2label,
        )
        return VideoFrame.from_ndarray(annotated, format="rgb24")

    return process_frame


def main() -> None:
    """Configura Streamlit y permite seleccionar el checkpoint para la cámara."""
    st.title("Segmentación panóptica en vivo")
    st.write(
        "Activa la cámara para detectar piezas de balsa, manos/brazos, "
        "superficie de trabajo y entorno. En cada pieza de balsa se estima "
        "largo × ancho en píxeles."
    )
    checkpoints = available_checkpoints()
    if not checkpoints:
        st.error(
            "No encontré un checkpoint entrenado. Ejecuta primero el "
            "entrenamiento para generar `training/runs/*/best_model/`."
        )
        st.stop()

    with st.sidebar:
        st.header("Configuración")
        selected_checkpoint = st.selectbox(
            "Checkpoint",
            options=checkpoints,
            format_func=lambda path: path.parent.name,
        )
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"Dispositivo: {device_name.upper()} · entrada: {IMAGE_SIZE}×{IMAGE_SIZE}")
        st.caption(f"Umbral de confianza: {SCORE_THRESHOLD:.2f}")

    detector = load_detector(str(selected_checkpoint), device_name)
    st.caption(
        "El video se procesa localmente; la aplicación no guarda ni sube "
        "fotogramas. Permite el acceso a la cámara cuando el navegador lo solicite."
    )
    webrtc_streamer(
        key="balsa-panoptic-live",
        video_frame_callback=make_frame_callback(detector),
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "facingMode": "environment",
            },
            "audio": False,
        },
        rtc_configuration={"iceServers": []},
        async_processing=True,
    )

    legend_columns = st.columns(4)
    for column, class_name in zip(legend_columns, CLASS_COLORS):
        red, green, blue = CLASS_COLORS[class_name]
        column.markdown(
            f"<span style='color:rgb({red},{green},{blue})'>■</span> "
            f"{DISPLAY_NAMES[class_name]}",
            unsafe_allow_html=True,
        )
    st.caption(
        "La velocidad depende del dispositivo. Esta demostración usa la entrada "
        "cuadrada de 256×256 del entrenamiento. Largo × ancho son estimaciones "
        "orientadas a partir de la máscara en píxeles; no son centímetros ni "
        "medidas físicas calibradas."
    )


if __name__ == "__main__":
    main()
