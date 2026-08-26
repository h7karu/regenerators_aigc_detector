"""Dual-branch AIGC detector: semantic (CLIP) + forensic (frequency/residual).

    image ──┬──> frozen CLIP ViT-B/32 ────> 512-d semantic embedding ──┐
            │                                                          ├─> scale -> head -> P(FAKE)
            └──> fixed forensic transforms -> 143-d artifact vector ───┘

The two branches are deliberately complementary:

  * CLIP sees *content* and is robust to post-processing, but was never trained
    to notice generator fingerprints and tends to smooth away high-frequency
    detail.
  * The forensic branch ignores content and looks only at spectral / noise
    statistics, which is where up-convolution artifacts live -- but it is more
    fragile under heavy blur or downscaling, which destroy those frequencies.

Fusing them means a transform that defeats one branch usually leaves the other
intact, which is exactly the robustness property the brief scores on.

Both branches are frozen feature extractors; only the small head is trained, so
the parameter count stays ~150M (CLIP) -- far under the 2B cap -- and training
is minutes on CPU rather than hours on a GPU.

`--branches` lets you train clip-only / forensic-only / both, so the fusion can
be A/B'd against each branch alone rather than assumed to help.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
VALID_BRANCHES = ("clip", "forensic")


class DualBranchClassifier:
    """Frozen CLIP + forensic features -> trained sklearn head (0 = REAL, 1 = FAKE)."""

    def __init__(
        self,
        branches: tuple[str, ...] | list[str] = VALID_BRANCHES,
        clip_model: str = DEFAULT_CLIP_MODEL,
        head: str = "logreg",
        device=None,
    ):
        bad = set(branches) - set(VALID_BRANCHES)
        if bad:
            raise ValueError(f"Unknown branch(es) {sorted(bad)}; valid: {VALID_BRANCHES}")
        if not branches:
            raise ValueError("At least one branch must be enabled")

        self.branches = tuple(branches)
        self.clip_model = clip_model
        self.head_kind = head
        self.device = device
        self._clip = None
        self.head = None  # sklearn Pipeline, set by fit() or load()

    # -- feature extraction -------------------------------------------------
    @property
    def clip(self):
        from aigc_detector.models.clip_probe import ClipFeatureExtractor

        if self._clip is None:
            self._clip = ClipFeatureExtractor(self.clip_model, self.device)
        return self._clip

    def _extract(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        from aigc_detector.features.forensic import extract_forensic_features_batch

        parts: list[np.ndarray] = []
        if "clip" in self.branches:
            chunks = [
                self.clip.extract(images[i : i + batch_size])
                for i in range(0, len(images), batch_size)
            ]
            parts.append(np.concatenate(chunks, axis=0))
        if "forensic" in self.branches:
            parts.append(extract_forensic_features_batch(images))
        return np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

    # -- training / inference ----------------------------------------------
    def _build_head(self):
        # Scaling matters here: CLIP embeddings and forensic statistics live on
        # very different numeric scales, and an unscaled linear head would let
        # whichever branch has larger magnitudes dominate the fit.
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if self.head_kind == "logreg":
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
        elif self.head_kind == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=600, early_stopping=True)
        else:
            raise ValueError(f"Unknown head {self.head_kind!r}; use 'logreg' or 'mlp'")
        return make_pipeline(StandardScaler(), model)

    def fit(self, images: list[Image.Image], labels: list[int], batch_size: int = 64) -> None:
        features = self._extract(images, batch_size)
        self.head = self._build_head()
        self.head.fit(features, labels)

    def predict_proba(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("Head not trained/loaded. Call fit() or load() first.")
        features = self._extract(images, batch_size)
        return self.head.predict_proba(features)[:, 1]  # P(FAKE)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "branches": self.branches,
                "clip_model": self.clip_model,
                "head_kind": self.head_kind,
                "head": self.head,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device=None) -> "DualBranchClassifier":
        payload = joblib.load(path)
        clf = cls(
            branches=payload["branches"],
            clip_model=payload["clip_model"],
            head=payload["head_kind"],
            device=device,
        )
        clf.head = payload["head"]
        return clf
