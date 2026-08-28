"""Evaluate a checkpoint on clean and transformed CIFAKE images."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from augmentations import ROBUSTNESS_TRANSFORMS, build_eval_transform
from datasets import ManifestImageDataset
from metrics import collect_predictions, compute_binary_metrics
from model import build_model, load_checkpoint
from utils import atomic_write_json, load_config, select_device


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
) -> dict[str, Any]:
    dataset = ManifestImageDataset(
        manifest,
        build_eval_transform(image_size, transform_name),
        limit=limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    labels, probabilities, _ = collect_predictions(
        model, loader, device, max_batches=max_batches
    )
    return compute_binary_metrics(labels, probabilities, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rgb_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--transform", choices=ROBUSTNESS_TRANSFORMS, default="clean")
    parser.add_argument("--all-transforms", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output", default="reports/metrics/evaluation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device(args.device)
    model = build_model(config, pretrained=False).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)
    threshold = float(checkpoint.get("threshold", 0.5))
    data_config = config["data"]
    manifest = data_config[f"{args.split}_manifest"]
    transform_names = ROBUSTNESS_TRANSFORMS if args.all_transforms else (args.transform,)
    results: dict[str, Any] = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "threshold": threshold,
        "transforms": {},
    }
    for transform_name in transform_names:
        metrics = evaluate_transform(
            model,
            manifest,
            transform_name,
            image_size=int(data_config["image_size"]),
            batch_size=int(config["training"]["batch_size"]),
            num_workers=int(data_config.get("num_workers", 0)),
            device=device,
            threshold=threshold,
            limit=args.limit_samples,
            max_batches=args.max_batches,
        )
        results["transforms"][transform_name] = metrics
        print(
            f"{transform_name:>12} | auroc={metrics['auroc']} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f}"
        )
    atomic_write_json(results, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
