"""Canonical model artifacts used by inference and user-facing applications."""

from __future__ import annotations

from inference_policy import TTA_TRANSFORMS


DEFAULT_MODEL_CHECKPOINT = "checkpoints/sid_local_lora/sid_local_lora_best.pt"
DEFAULT_MODEL_CONFIG = "configs/sid_local_lora.yaml"
DEFAULT_INFERENCE_TRANSFORMS = TTA_TRANSFORMS
DEFAULT_INFERENCE_AGGREGATION = "trimmed_mean"
DEFAULT_INFERENCE_THRESHOLD = 0.4856400298408816
POSITIVE_VERDICT = "AI Generated or Manipulated"
NEGATIVE_VERDICT = "Real"
