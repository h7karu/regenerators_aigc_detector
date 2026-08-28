"""Binary classification metrics and inference collection helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def optimal_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities
    )
    finite = np.isfinite(thresholds)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def compute_binary_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "samples": int(labels.size),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(
            false_positive / max(false_positive + true_negative, 1)
        ),
        "confusion_matrix": matrix.tolist(),
    }
    if np.unique(labels).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["average_precision"] = float(
            average_precision_score(labels, probabilities)
        )
    else:
        metrics["auroc"] = None
        metrics["average_precision"] = None
    return metrics


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_paths: list[str] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        all_probabilities.append(logits.sigmoid().cpu().numpy())
        all_labels.append(batch["label"].cpu().numpy())
        all_paths.extend(batch["path"])
    if not all_labels:
        raise ValueError("No evaluation batches were produced.")
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        all_paths,
    )
