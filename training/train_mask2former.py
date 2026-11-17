#!/usr/bin/env python3
"""Entrena Mask2Former con las anotaciones COCO panópticas de balsa."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from balsa_panoptica.training.coco_dataset import (
    WORKSPACE_ROOT,
    CocoPanopticDataset,
    collate_samples,
    encode_for_mask2former,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = "facebook/mask2former-swin-tiny-coco-panoptic"
DEFAULT_TRAIN_ANNOTATIONS = PROJECT_ROOT / "data/annotations/balsa_coco_merged.json"
DEFAULT_TRAIN_SPLITS = PROJECT_ROOT / "data/splits_merged.json"


class EpochDataset(Protocol):
    """Interfaz mínima para actualizar aumentos reproducibles por época."""

    epoch: int

    def set_epoch(self, epoch: int) -> None: ...


def parse_args() -> argparse.Namespace:
    """Lee la configuración CLI y valida los parámetros de entrenamiento."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_TRAIN_ANNOTATIONS)
    parser.add_argument("--images-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_TRAIN_SPLITS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Límite opcional para comprobar el ciclo con pocos lotes.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=0,
        help="Límite opcional para comprobar la validación.",
    )
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.image_size < 128 or args.batch_size < 1:
        parser.error(
            "--epochs y --batch-size deben ser positivos; --image-size mínimo 128."
        )
    if args.patience < 1 or args.min_delta < 0:
        parser.error("--patience debe ser positivo y --min-delta no negativo.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("--learning-rate debe ser positivo y --weight-decay no negativo.")
    if args.max_train_batches < 0 or args.max_validation_batches < 0:
        parser.error("Los límites de lotes no pueden ser negativos.")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = PROJECT_ROOT / "training" / "runs" / f"mask2former_balsa_{timestamp}"
    return args


def seed_everything(seed: int) -> None:
    """Fija semillas de Python, NumPy y PyTorch para repetir el entrenamiento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_processor(
    checkpoint: str,
    categories: list[dict[str, Any]],
) -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
    """Adapta el checkpoint base al orden de categorías del COCO del proyecto."""
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
    id2label = {index: item["name"] for index, item in enumerate(categories)}
    label2id = {name: index for index, name in id2label.items()}
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.config.ignore_value = 255
    return processor, model


def loss_for_sample(
    model: torch.nn.Module,
    processor: Mask2FormerImageProcessor,
    sample: dict[str, Any],
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Codifica una imagen/target y exige que Mask2Former produzca una loss finita."""
    encoded = encode_for_mask2former(sample, processor, image_size)
    if not encoded["class_labels"].numel():
        raise ValueError(
            f"La imagen {sample['file_name']} no tiene segmentos tras el reescalado."
        )
    output = model(
        pixel_values=encoded["pixel_values"].unsqueeze(0).to(device),
        mask_labels=[encoded["mask_labels"].to(device)],
        class_labels=[encoded["class_labels"].to(device)],
    )
    if output.loss is None:
        raise RuntimeError("Mask2Former no devolvió loss con las etiquetas COCO.")
    if not torch.isfinite(output.loss):
        raise RuntimeError(
            f"Mask2Former devolvió una pérdida no finita para {sample['file_name']}."
        )
    return output.loss


def run_epoch(
    model: torch.nn.Module,
    processor: Mask2FormerImageProcessor,
    loader: Iterable[list[dict[str, Any]]],
    dataset: EpochDataset,
    image_size: int,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    max_batches: int = 0,
) -> tuple[float, int, int]:
    """Ejecuta train o validation y devuelve loss media, muestras y solapamientos."""
    training = optimizer is not None
    model.train(training)
    if training:
        dataset.set_epoch(dataset.epoch + 1)

    total_loss = 0.0
    sample_count = 0
    overlap_pixels = 0
    for batch_index, batch in enumerate(loader, start=1):
        if max_batches and batch_index > max_batches:
            break
        for sample in batch:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp,
                ):
                    loss = loss_for_sample(
                        model,
                        processor,
                        sample,
                        image_size,
                        device,
                    )
                if training:
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

            total_loss += float(loss.detach().item())
            sample_count += 1
            overlap_pixels += int(sample["overlap_pixels_resolved"])
            if sample_count % 10 == 0:
                print(
                    f"  {'train' if training else 'validation'} "
                    f"{sample_count} loss={total_loss / sample_count:.4f}",
                    flush=True,
                )
    if sample_count == 0:
        raise ValueError("No hay lotes para calcular la pérdida.")
    return total_loss / sample_count, sample_count, overlap_pixels


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    """Guarda el historial numérico y la curva de pérdidas de la ejecución."""
    if not history:
        raise ValueError("No hay historial de entrenamiento para guardar.")
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plt.figure(figsize=(8, 5))
    epochs = [int(item["epoch"]) for item in history]
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train")
    plt.plot(epochs, [item["validation_loss"] for item in history], label="Validation")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Mask2Former de balsa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()


def main() -> None:
    """Entrena usando train, selecciona por validation y reserva test para después."""
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA, pero PyTorch no detecta una GPU CUDA.")
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )
    device = torch.device("cuda" if use_cuda else "cpu")
    amp = device.type == "cuda" and not args.no_amp

    train_dataset = CocoPanopticDataset(
        args.annotations,
        images_root=args.images_root,
        splits_path=args.splits,
        split="train",
        augment=True,
        seed=args.seed,
    )
    validation_dataset = CocoPanopticDataset(
        args.annotations,
        images_root=args.images_root,
        splits_path=args.splits,
        split="validation",
    )
    if len(validation_dataset) < 10:
        print(
            f"ADVERTENCIA: solo hay {len(validation_dataset)} imágenes de validación; "
            "la pérdida de validación será muy variable.",
            flush=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_samples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"El directorio de salida ya existe: {output_dir}")
    processor, model = load_model_and_processor(
        args.checkpoint,
        train_dataset.categories,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "annotations": str(args.annotations.resolve()),
        "splits": str(args.splits.resolve()),
        "checkpoint": args.checkpoint,
        "device": str(device),
        "amp": amp,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Dispositivo: {device}; AMP: {amp}; checkpoint: {args.checkpoint}\n"
        f"Entrenamiento: {len(train_dataset)} imágenes; "
        f"validación: {len(validation_dataset)} imágenes\n"
        f"Salida: {output_dir}",
        flush=True,
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        print(f"Época {epoch + 1}/{args.epochs}", flush=True)
        train_loss, train_count, train_overlaps = run_epoch(
            model,
            processor,
            train_loader,
            train_dataset,
            args.image_size,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
            max_batches=args.max_train_batches,
        )
        validation_loss, validation_count, validation_overlaps = run_epoch(
            model,
            processor,
            validation_loader,
            validation_dataset,
            args.image_size,
            device,
            amp=amp,
            max_batches=args.max_validation_batches,
        )
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise RuntimeError("La pérdida de entrenamiento o validación no es finita.")
        scheduler.step()
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        save_history(history, output_dir)
        improved = validation_loss < best_validation_loss - args.min_delta
        if improved:
            # El mejor modelo se decide únicamente con validation; test no se carga aquí.
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            model.save_pretrained(output_dir / "best_model")
            processor.save_pretrained(output_dir / "best_model")
        else:
            epochs_without_improvement += 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            output_dir / "last_training.pt",
        )
        print(
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} "
            f"muestras={train_count}/{validation_count} "
            f"solapamientos={train_overlaps + validation_overlaps} "
            f"sin_mejora={epochs_without_improvement}/{args.patience}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            print("Early stopping: la validación dejó de mejorar.", flush=True)
            break

    print(f"Mejor checkpoint: {output_dir / 'best_model'}")
    print(f"Curva de pérdida: {output_dir / 'loss_curve.png'}")
    print(
        "El conjunto test no se usó durante este entrenamiento. "
        "Evalúalo cuando se haya seleccionado el checkpoint.",
        flush=True,
    )


if __name__ == "__main__":
    main()
