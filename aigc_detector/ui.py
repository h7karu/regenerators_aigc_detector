"""Inference service used by the Gradio demo."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps

# The CLIP files are downloaded during setup. Avoid network checks on every run.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MAX_IMAGE_SIDE = 2048


def confidence_labels(fake_probability: float) -> dict[str, float]:
    """Convert P(FAKE) into the mapping expected by gr.Label."""
    probability = min(1.0, max(0.0, float(fake_probability)))
    return {
        "AI-GENERATED": probability,
        "REAL": 1.0 - probability,
    }


def verdict_markdown(fake_probability: float, elapsed_seconds: float) -> str:
    """Create the human-readable result shown beside the confidence chart."""
    labels = confidence_labels(fake_probability)
    verdict = max(labels, key=labels.get)
    confidence = labels[verdict]
    return (
        f"## {verdict}\n\n"
        f"**Confidence: {confidence:.1%}**  \n"
        f"Processed in {elapsed_seconds:.2f} seconds.\n\n"
        "> This is a probabilistic research model, not definitive proof of an "
        "image's origin."
    )


def prepare_image(image: Image.Image) -> Image.Image:
    """Normalize orientation/mode and cap unusually large uploads."""
    if image is None:
        raise ValueError("Please upload an image before running the detector.")

    prepared = ImageOps.exif_transpose(image).convert("RGB")
    if max(prepared.size) > MAX_IMAGE_SIDE:
        prepared.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return prepared


class DetectorService:
    """Load one checkpoint once and serialize model inference calls."""

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)
        self._model = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if not self.checkpoint.is_file():
                        raise FileNotFoundError(
                            f"Model checkpoint not found: {self.checkpoint}"
                        )
                    from aigc_detector.models.dual_branch import DualBranchClassifier

                    self._model = DualBranchClassifier.load(self.checkpoint)
        return self._model

    def predict(self, image: Image.Image) -> tuple[dict[str, float], str]:
        prepared = prepare_image(image)
        started = time.perf_counter()
        with self._predict_lock:
            fake_probability = float(self.model.predict_proba([prepared], batch_size=1)[0])
        elapsed = time.perf_counter() - started
        return confidence_labels(fake_probability), verdict_markdown(
            fake_probability, elapsed
        )

