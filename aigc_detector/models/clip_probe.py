"""CLIP-backbone + linear-probe classifier for AIGC detection.

Why this architecture: a frozen, large-scale-pretrained CLIP vision encoder
gives features that are far more invariant to JPEG re-encoding, blur, resize,
noise, color jitter, and cropping than a small CNN trained from scratch would
be out of the box -- exactly the robustness axis this brief scores on. Only a
lightweight linear head is trained, which keeps the whole pipeline well under
the <2B parameter cap (CLIP ViT-B/32 is ~150M params) and fast to train/eval
on a laptop or a single hackathon GPU. Swap `model_name` for a larger CLIP
checkpoint (e.g. ViT-L/14, ~300M) if compute allows and more headroom is
wanted before hitting the parameter cap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class ClipFeatureExtractor:
    """Frozen CLIP vision encoder used purely as a feature extractor."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: "torch.device | None" = None):
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        from aigc_detector.utils import get_device

        self.model_name = model_name
        self.device = device or get_device()
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def extract(self, images: Iterable[Image.Image]) -> np.ndarray:
        torch = self._torch
        images = list(images)
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs).image_embeds
        return out.cpu().numpy()


class ClipLinearProbeClassifier:
    """CLIP features -> scikit-learn logistic regression head (0 = REAL, 1 = FAKE)."""

    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: "torch.device | None" = None):
        self.model_name = model_name
        self.device = device
        self._extractor: ClipFeatureExtractor | None = None
        self.head = None  # sklearn LogisticRegression, set by fit() or load()

    @property
    def extractor(self) -> ClipFeatureExtractor:
        if self._extractor is None:
            self._extractor = ClipFeatureExtractor(self.model_name, self.device)
        return self._extractor

    def fit(self, images: list[Image.Image], labels: list[int], batch_size: int = 64) -> None:
        from sklearn.linear_model import LogisticRegression

        features = self._extract_batched(images, batch_size)
        self.head = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.head.fit(features, labels)

    def predict_proba(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("Classifier head not trained/loaded. Call fit() or load() first.")
        features = self._extract_batched(images, batch_size)
        return self.head.predict_proba(features)[:, 1]  # P(FAKE)

    def _extract_batched(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        chunks = []
        for i in range(0, len(images), batch_size):
            chunks.append(self.extractor.extract(images[i : i + batch_size]))
        return np.concatenate(chunks, axis=0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model_name": self.model_name, "head": self.head}, path)

    @classmethod
    def load(cls, path: str | Path, device: "torch.device | None" = None) -> "ClipLinearProbeClassifier":
        payload = joblib.load(path)
        clf = cls(model_name=payload["model_name"], device=device)
        clf.head = payload["head"]
        return clf
