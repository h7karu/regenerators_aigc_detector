"""Evaluate a checkpoint on clean and transformed manifest images."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from augmentations import (
    ROBUSTNESS_TRANSFORMS,
    build_eval_transform,
    build_tta_eval_transform,
)
from data_pipeline import ManifestImageDataset
from deployment import (
    DEFAULT_INFERENCE_AGGREGATION,
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_INFERENCE_TRANSFORMS,
    DEFAULT_MODEL_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
)
from metrics import collect_predictions, compute_binary_metrics
from model import build_model, load_checkpoint
from utils import atomic_write_json, load_config, select_device, worker_seed


def evaluate_transform(
    model: torch.nn.Module,
    manifest: str,
    transform_name: str,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    limit: int | None,
    max_batches: int | None,
    tta_aggregation: str | None = None,
) -> dict[str, Any]:
    dataset = ManifestImageDataset(
        manifest,
        (
            build_tta_eval_transform(
                image_size,
                DEFAULT_INFERENCE_TRANSFORMS,
                base_transform=transform_name,
            )
            if tta_aggregation is not None
            else build_eval_transform(image_size, transform_name)
        ),
        limit=limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
    )
    labels, probabilities, _ = collect_predictions(
        model,
        loader,
        device,
        max_batches=max_batches,
        tta_aggregation=tta_aggregation,
    )
    metrics = compute_binary_metrics(labels, probabilities, threshold=threshold)
    metadata = dataset.frame.iloc[: len(labels)].reset_index(drop=True)
    predictions = (probabilities >= threshold).astype(np.int64)
    subgroup_metrics: dict[str, dict[str, Any]] = {}
    for original_label, indices in metadata.groupby("original_label").groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        subgroup_labels = labels[positions]
        subgroup_predictions = predictions[positions]
        subgroup_probabilities = probabilities[positions]
        subgroup_metrics[str(original_label)] = {
            "samples": int(len(positions)),
            "binary_label": int(subgroup_labels[0]),
            "generator": str(metadata.loc[positions[0], "generator"]),
            "accuracy": float(np.mean(subgroup_predictions == subgroup_labels)),
            "predicted_aigc_rate": float(np.mean(subgroup_predictions)),
            "mean_probability": float(np.mean(subgroup_probabilities)),
        }
    metrics["subgroups"] = subgroup_metrics
    return metrics


def summarize_transforms(
    transform_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not transform_metrics:
        raise ValueError("At least one transform result is required.")
    aucs = {
        name: float(metrics["auroc"])
        for name, metrics in transform_metrics.items()
        if metrics.get("auroc") is not None
    }
    balanced_accuracies = {
        name: float(metrics["balanced_accuracy"])
        for name, metrics in transform_metrics.items()
    }
    transformed_names = [name for name in transform_metrics if name != "clean"]
    summary: dict[str, Any] = {
        "mean_auroc": float(np.mean(list(aucs.values()))) if aucs else None,
        "worst_auroc": float(np.min(list(aucs.values()))) if aucs else None,
        "worst_auroc_transform": min(aucs, key=aucs.get) if aucs else None,
        "mean_balanced_accuracy": float(
            np.mean(list(balanced_accuracies.values()))
        ),
        "worst_balanced_accuracy": float(
            np.min(list(balanced_accuracies.values()))
        ),
        "worst_balanced_accuracy_transform": min(
            balanced_accuracies,
            key=balanced_accuracies.get,
        ),
    }
    if transformed_names:
        transformed_aucs = [aucs[name] for name in transformed_names if name in aucs]
        summary["mean_transformed_auroc"] = (
            float(np.mean(transformed_aucs)) if transformed_aucs else None
        )
        summary["mean_transformed_balanced_accuracy"] = float(
            np.mean(
                [balanced_accuracies[name] for name in transformed_names]
            )
        )
    if "clean" in aucs and summary["worst_auroc"] is not None:
        summary["clean_to_worst_auroc_drop"] = float(
            aucs["clean"] - summary["worst_auroc"]
        )
    if "clean" in balanced_accuracies:
        summary["clean_to_worst_balanced_accuracy_drop"] = float(
            balanced_accuracies["clean"] - summary["worst_balanced_accuracy"]
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--manifest",
        help="Evaluate an explicit manifest instead of the configured split.",
    )
    parser.add_argument("--transform", choices=ROBUSTNESS_TRANSFORMS, default="clean")
    parser.add_argument("--all-transforms", action="store_true")
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Evaluate the checkpoint directly instead of the deployed TTA policy.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override evaluation DataLoader workers.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the source-image batch size.",
    )
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", default="reports/metrics/evaluation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device(args.device)
    model = build_model(config, pretrained=False).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)
    threshold = (
        float(checkpoint.get("threshold", 0.5))
        if args.single_view
        else DEFAULT_INFERENCE_THRESHOLD
    )
    tta_aggregation = None if args.single_view else DEFAULT_INFERENCE_AGGREGATION
    data_config = config["data"]
    manifest = args.manifest or data_config[f"{args.split}_manifest"]
    transform_names = ROBUSTNESS_TRANSFORMS if args.all_transforms else (args.transform,)
    configured_batch_size = int(config["training"]["batch_size"])
    default_batch_size = (
        configured_batch_size
        if args.single_view
        else max(1, configured_batch_size // len(DEFAULT_INFERENCE_TRANSFORMS))
    )
    results: dict[str, Any] = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "manifest": manifest,
        "threshold": threshold,
        "inference_policy": (
            "single_view"
            if args.single_view
            else f"{DEFAULT_INFERENCE_AGGREGATION}_tta"
        ),
        "transforms": {},
    }
    for transform_name in transform_names:
        metrics = evaluate_transform(
            model,
            manifest,
            transform_name,
            image_size=int(data_config["image_size"]),
            batch_size=args.batch_size or default_batch_size,
            num_workers=(
                args.num_workers
                if args.num_workers is not None
                else int(
                    data_config.get(
                        "validation_num_workers",
                        data_config.get("num_workers", 0),
                    )
                )
            ),
            device=device,
            threshold=threshold,
            limit=args.limit_samples,
            max_batches=args.max_batches,
            tta_aggregation=tta_aggregation,
        )
        results["transforms"][transform_name] = metrics
        print(
            f"{transform_name:>12} | auroc={metrics['auroc']} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
    results["summary"] = summarize_transforms(results["transforms"])
    atomic_write_json(results, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
