"""Inference-time aggregation helpers shared by benchmarks and deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


TTA_TRANSFORMS = (
    "clean",
    "jpeg_70",
    "blur_1.0",
    "resize_0.5",
    "crop_0.8",
)
TTA_AGGREGATIONS = ("median", "trimmed_mean")


def _validate_aggregation(method: str, view_count: int) -> None:
    if method not in TTA_AGGREGATIONS:
        raise ValueError(
            f"Unknown TTA aggregation {method!r}; expected {TTA_AGGREGATIONS}."
        )
    if view_count < 3 or view_count % 2 == 0:
        raise ValueError("TTA aggregation requires an odd number of at least 3 views.")


def aggregate_numpy_logits(
    logits_by_view: Mapping[str, np.ndarray],
    *,
    method: str,
    view_order: Sequence[str] = TTA_TRANSFORMS,
) -> np.ndarray:
    """Aggregate aligned per-view logits into one logit per image."""

    missing = [name for name in view_order if name not in logits_by_view]
    if missing:
        raise ValueError(f"Missing TTA logits for: {', '.join(missing)}")
    _validate_aggregation(method, len(view_order))
    stacked = np.stack(
        [np.asarray(logits_by_view[name], dtype=np.float64) for name in view_order]
    )
    if stacked.ndim != 2:
        raise ValueError("TTA logits must contain one 1-D array per view.")
    if method == "median":
        return np.median(stacked, axis=0)
    ordered = np.sort(stacked, axis=0)
    return np.mean(ordered[1:-1], axis=0)


def aggregate_tensor_logits(logits: torch.Tensor, *, method: str) -> torch.Tensor:
    """Aggregate view logits along the last dimension."""

    if logits.ndim < 1:
        raise ValueError("Online TTA logits must have at least one dimension.")
    _validate_aggregation(method, int(logits.shape[-1]))
    if method == "median":
        return torch.median(logits, dim=-1).values
    ordered, _ = torch.sort(logits, dim=-1)
    return ordered[..., 1:-1].mean(dim=-1)


def aggregate_model_views(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    method: str,
) -> torch.Tensor:
    """Run and aggregate a ``[batch, views, channels, height, width]`` tensor."""

    if images.ndim != 5:
        raise ValueError("TTA model input must have shape [batch, views, C, H, W].")
    batch_size, view_count = images.shape[:2]
    _validate_aggregation(method, int(view_count))
    flat_images = images.reshape(-1, *images.shape[2:])
    view_logits = model(flat_images).reshape(batch_size, view_count)
    return aggregate_tensor_logits(view_logits, method=method)


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def logit_threshold(threshold: float) -> float:
    clipped = float(np.clip(threshold, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def standardized_evidence(
    logits: np.ndarray,
    *,
    threshold: float,
    scale: float,
) -> np.ndarray:
    """Center logits on a model decision boundary and normalize their spread."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Evidence scale must be a positive finite number.")
    return (np.asarray(logits, dtype=np.float64) - logit_threshold(threshold)) / scale
