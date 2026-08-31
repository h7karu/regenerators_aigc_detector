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

from inference_policy import aggregate_model_views


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


def compute_robustness_metrics(
    predictions_by_transform: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Summarise validation predictions using one threshold across conditions.

    AUROC is calculated independently for every condition. The checkpoint
    selection score is their unweighted mean so a clean condition cannot
    dominate the transformed conditions merely by containing more samples.
    The decision threshold is fitted to the pooled validation predictions.
    """

    if not predictions_by_transform:
        raise ValueError("At least one validation transform is required.")

    pooled_labels: list[np.ndarray] = []
    pooled_probabilities: list[np.ndarray] = []
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for transform_name, (labels, probabilities) in predictions_by_transform.items():
        labels = np.asarray(labels, dtype=np.int64)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if labels.ndim != 1 or probabilities.ndim != 1:
            raise ValueError("Validation labels and probabilities must be 1-D.")
        if labels.size == 0 or labels.size != probabilities.size:
            raise ValueError(
                f"Invalid validation predictions for {transform_name!r}."
            )
        normalized[transform_name] = (labels, probabilities)
        pooled_labels.append(labels)
        pooled_probabilities.append(probabilities)

    all_labels = np.concatenate(pooled_labels)
    all_probabilities = np.concatenate(pooled_probabilities)
    threshold = optimal_threshold(all_labels, all_probabilities)
    transform_metrics = {
        transform_name: compute_binary_metrics(
            labels,
            probabilities,
            threshold=threshold,
        )
        for transform_name, (labels, probabilities) in normalized.items()
    }
    aucs = [
        float(metrics["auroc"])
        for metrics in transform_metrics.values()
        if metrics["auroc"] is not None
    ]
    if not aucs:
        raise ValueError(
            "Robustness validation requires both labels in at least one condition."
        )

    return {
        "threshold": threshold,
        "selection_metric": "mean_transform_auroc",
        "selection_score": float(np.mean(aucs)),
        "mean_auroc": float(np.mean(aucs)),
        "worst_auroc": float(np.min(aucs)),
        "aggregate": compute_binary_metrics(
            all_labels,
            all_probabilities,
            threshold=threshold,
        ),
        "transforms": transform_metrics,
    }


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    tta_aggregation: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_paths: list[str] = []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        if tta_aggregation is None:
            logits = model(images)
        else:
            logits = aggregate_model_views(
                model,
                images,
                method=tta_aggregation,
            )
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
