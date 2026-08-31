"""Reusable, test-set-independent inference for the local and Colab demos."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from augmentations import (
    apply_robustness_transform,
    build_eval_transform,
    build_tta_eval_transform,
)
from deployment import (
    DEFAULT_INFERENCE_AGGREGATION,
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_INFERENCE_TRANSFORMS,
    NEGATIVE_VERDICT,
    POSITIVE_VERDICT,
)
from inference_policy import aggregate_tensor_logits
from model import build_model, count_parameters, load_checkpoint
from utils import load_config, resolve_project_path, select_device


DEMO_TRANSFORMS = (
    "clean",
    "jpeg_70",
    "jpeg_30",
    "blur_1.0",
    "resize_0.5",
)
TRANSFORM_LABELS = {
    "clean": "Original",
    "jpeg_70": "JPEG 70",
    "jpeg_30": "JPEG 30",
    "blur_1.0": "Blur 1.0",
    "resize_0.5": "Downscale 50%",
}
MAX_INPUT_PIXELS = 40_000_000


@dataclass(frozen=True)
class PredictionResult:
    score: float
    threshold: float
    verdict: str
    runtime_ms: float
    width: int
    height: int
    image_format: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def prepare_image(value: Image.Image | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a Gradio/Pillow input into validated uint8 RGB pixels."""

    if value is None:
        raise ValueError("Upload, paste, or capture an image before analysing it.")

    if isinstance(value, Image.Image):
        image_format = (value.format or "unknown").upper()
        image = ImageOps.exif_transpose(value).convert("RGB")
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError("The supplied image is empty.")
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array).convert("RGB")
        image_format = "ARRAY"
    else:
        raise TypeError(f"Unsupported image input: {type(value).__name__}")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("The supplied image has invalid dimensions.")
    if width * height > MAX_INPUT_PIXELS:
        raise ValueError(
            f"Image is too large ({width}x{height}); limit is "
            f"{MAX_INPUT_PIXELS:,} pixels."
        )
    pixels = np.asarray(image, dtype=np.uint8).copy()
    return pixels, {
        "width": width,
        "height": height,
        "image_format": image_format,
    }


class DemoDetector:
    """Load one checkpoint and expose single-image and robustness inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        *,
        device: str = "auto",
        use_tta: bool = True,
    ) -> None:
        self.config_path = resolve_project_path(config_path)
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.config = load_config(self.config_path)
        self.device = select_device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available to PyTorch.")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available to PyTorch.")

        self.model = build_model(self.config, pretrained=False).to(self.device)
        checkpoint = load_checkpoint(
            self.model,
            self.checkpoint_path,
            map_location=self.device,
        )
        self.model.eval()
        self.checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
        self.use_tta = use_tta
        self.tta_aggregation = DEFAULT_INFERENCE_AGGREGATION if use_tta else None
        self.threshold = (
            DEFAULT_INFERENCE_THRESHOLD if use_tta else self.checkpoint_threshold
        )
        self.validation_metrics = dict(checkpoint.get("validation_metrics", {}))
        self.validation_robustness = dict(
            checkpoint.get("validation_robustness", {})
        )
        self.selected_epoch = int(checkpoint.get("epoch", -1)) + 1
        self.image_size = int(self.config["data"]["image_size"])
        if use_tta:
            self.transforms = {
                name: build_tta_eval_transform(
                    self.image_size,
                    DEFAULT_INFERENCE_TRANSFORMS,
                    base_transform=name,
                )
                for name in DEMO_TRANSFORMS
            }
        else:
            self.transforms = {
                name: build_eval_transform(self.image_size, name)
                for name in DEMO_TRANSFORMS
            }
        self.parameter_count = count_parameters(self.model)
        self._inference_lock = threading.Lock()

    def _score_pixels(self, pixels: np.ndarray, transform_name: str) -> float:
        if transform_name not in self.transforms:
            raise ValueError(f"Unsupported demo transform: {transform_name}")
        tensor = self.transforms[transform_name](image=pixels)["image"]
        tensor = (
            tensor.to(self.device)
            if self.use_tta
            else tensor.unsqueeze(0).to(self.device)
        )
        with self._inference_lock, torch.inference_mode():
            logits = self.model(tensor)
            logit = (
                aggregate_tensor_logits(logits, method=self.tta_aggregation)
                if self.tta_aggregation is not None
                else logits.squeeze(0)
            )
            return float(logit.sigmoid().item())

    def verdict(self, score: float) -> str:
        return POSITIVE_VERDICT if score >= self.threshold else NEGATIVE_VERDICT

    def predict(self, value: Image.Image | np.ndarray) -> PredictionResult:
        pixels, metadata = prepare_image(value)
        start = time.perf_counter()
        score = self._score_pixels(pixels, "clean")
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult(
            score=score,
            threshold=self.threshold,
            verdict=self.verdict(score),
            runtime_ms=runtime_ms,
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            image_format=str(metadata["image_format"]),
        )

    def analyse_robustness(
        self, value: Image.Image | np.ndarray
    ) -> dict[str, object]:
        pixels, _ = prepare_image(value)
        start = time.perf_counter()
        rows: list[dict[str, object]] = []
        previews: list[tuple[np.ndarray, str]] = []
        for transform_name in DEMO_TRANSFORMS:
            score = self._score_pixels(pixels, transform_name)
            label = TRANSFORM_LABELS[transform_name]
            rows.append(
                {
                    "condition": label,
                    "score": score,
                    "verdict": self.verdict(score),
                    "distance_from_threshold": score - self.threshold,
                }
            )
            preview = apply_robustness_transform(pixels, transform_name, seed=0)
            previews.append((preview, label))

        scores = [float(row["score"]) for row in rows]
        return {
            "rows": rows,
            "previews": previews,
            "score_range": max(scores) - min(scores),
            "all_verdicts_agree": len({str(row["verdict"]) for row in rows}) == 1,
            "runtime_ms": (time.perf_counter() - start) * 1000.0,
        }

    def model_metadata(self) -> dict[str, object]:
        return {
            "checkpoint": self.checkpoint_path.name,
            "config": self.config_path.name,
            "device": str(self.device),
            "parameters": self.parameter_count,
            "image_size": self.image_size,
            "decision_threshold": self.threshold,
            "checkpoint_threshold": self.checkpoint_threshold,
            "inference_policy": (
                f"{len(DEFAULT_INFERENCE_TRANSFORMS)}-view "
                f"{self.tta_aggregation} TTA"
                if self.use_tta
                else "single view"
            ),
            "inference_transforms": (
                list(DEFAULT_INFERENCE_TRANSFORMS) if self.use_tta else ["clean"]
            ),
            "selected_epoch": self.selected_epoch,
            "validation_auroc": self.validation_metrics.get("auroc"),
            "validation_f1": self.validation_metrics.get("f1"),
            "validation_selection_score": self.validation_robustness.get(
                "selection_score"
            ),
            "validation_worst_auroc": self.validation_robustness.get(
                "worst_auroc"
            ),
        }
