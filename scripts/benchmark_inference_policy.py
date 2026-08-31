"""Select TTA and optional checkpoint ensembling on validation data only."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from augmentations import build_eval_transform  # noqa: E402
from data_pipeline import ManifestImageDataset  # noqa: E402
from deployment import DEFAULT_MODEL_CHECKPOINT, DEFAULT_MODEL_CONFIG  # noqa: E402
from inference_policy import (  # noqa: E402
    TTA_AGGREGATIONS,
    TTA_TRANSFORMS,
    aggregate_numpy_logits,
    sigmoid_numpy,
    standardized_evidence,
)
from metrics import compute_binary_metrics, optimal_threshold  # noqa: E402
from model import build_model, load_checkpoint  # noqa: E402
from utils import (  # noqa: E402
    atomic_write_json,
    load_config,
    select_device,
    worker_seed,
)


CIFAKE_CHECKPOINT = "checkpoints/full_cifake_lora/full_cifake_lora_best.pt"
CIFAKE_CONFIG = "configs/full_cifake_lora.yaml"
ENSEMBLE_ALPHAS = tuple(round(value, 2) for value in np.arange(0.50, 1.001, 0.05))


def collect_view_logits(
    model: torch.nn.Module,
    *,
    manifest: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    limit: int | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    expected_labels: np.ndarray | None = None
    expected_paths: list[str] | None = None
    logits_by_view: dict[str, np.ndarray] = {}
    for transform_name in TTA_TRANSFORMS:
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
            worker_init_fn=worker_seed,
        )
        labels: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        paths: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                logits.append(model(images).float().cpu().numpy())
                labels.append(batch["label"].cpu().numpy())
                paths.extend(batch["path"])
        view_labels = np.concatenate(labels).astype(np.int64, copy=False)
        view_logits = np.concatenate(logits).astype(np.float64, copy=False)
        if expected_labels is None:
            expected_labels = view_labels
            expected_paths = paths
        elif not np.array_equal(view_labels, expected_labels) or paths != expected_paths:
            raise RuntimeError("TTA views did not preserve manifest ordering.")
        logits_by_view[transform_name] = view_logits
        print(
            f"  {transform_name}: {len(view_logits):,} images",
            flush=True,
        )
    assert expected_labels is not None
    return expected_labels, logits_by_view


def policy_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probabilities = sigmoid_numpy(logits)
    threshold = optimal_threshold(labels, probabilities)
    return compute_binary_metrics(labels, probabilities, threshold=threshold)


def weighted_domain_threshold(
    labels_by_domain: dict[str, np.ndarray],
    probabilities_by_domain: dict[str, np.ndarray],
) -> float:
    labels = np.concatenate(list(labels_by_domain.values()))
    probabilities = np.concatenate(list(probabilities_by_domain.values()))
    weights = np.concatenate(
        [
            np.full(len(domain_labels), 1.0 / len(domain_labels))
            for domain_labels in labels_by_domain.values()
        ]
    )
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels,
        probabilities,
        sample_weight=weights,
    )
    finite = np.isfinite(thresholds)
    scores = true_positive_rate[finite] - false_positive_rate[finite]
    return float(thresholds[finite][int(np.argmax(scores))])


def summarize_policy(
    labels_by_domain: dict[str, np.ndarray],
    logits_by_domain: dict[str, np.ndarray],
) -> dict[str, Any]:
    probabilities = {
        domain: sigmoid_numpy(logits)
        for domain, logits in logits_by_domain.items()
    }
    threshold = weighted_domain_threshold(labels_by_domain, probabilities)
    domains = {
        domain: compute_binary_metrics(
            labels_by_domain[domain],
            domain_probabilities,
            threshold=threshold,
        )
        for domain, domain_probabilities in probabilities.items()
    }
    aucs = [float(metrics["auroc"]) for metrics in domains.values()]
    return {
        "threshold": threshold,
        "mean_domain_auroc": float(np.mean(aucs)),
        "worst_domain_auroc": float(np.min(aucs)),
        "domains": domains,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sid-limit", type=int)
    parser.add_argument("--cifake-limit", type=int)
    parser.add_argument(
        "--output",
        default="reports/metrics/inference_policy_validation.json",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    sid_config = load_config(DEFAULT_MODEL_CONFIG)
    cifake_config = load_config(CIFAKE_CONFIG)
    domains = {
        "sid": {
            "manifest": sid_config["data"]["val_manifest"],
            "limit": args.sid_limit,
        },
        "cifake": {
            "manifest": cifake_config["data"]["val_manifest"],
            "limit": args.cifake_limit,
        },
    }
    labels_by_domain: dict[str, np.ndarray] = {}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    checkpoints = {
        "sid": DEFAULT_MODEL_CHECKPOINT,
        "cifake": CIFAKE_CHECKPOINT,
    }
    checkpoint_thresholds: dict[str, float] = {}
    for model_name, checkpoint_path in checkpoints.items():
        print(f"Loading {model_name} model: {checkpoint_path}", flush=True)
        model = build_model(sid_config, pretrained=False).to(device)
        checkpoint = load_checkpoint(model, checkpoint_path, map_location=device)
        model.eval()
        checkpoint_thresholds[model_name] = float(checkpoint.get("threshold", 0.5))
        predictions[model_name] = {}
        del checkpoint
        for domain_name, domain in domains.items():
            print(f"{model_name} model on {domain_name} validation:", flush=True)
            labels, view_logits = collect_view_logits(
                model,
                manifest=str(domain["manifest"]),
                image_size=int(sid_config["data"]["image_size"]),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                limit=domain["limit"],
            )
            if domain_name in labels_by_domain and not np.array_equal(
                labels, labels_by_domain[domain_name]
            ):
                raise RuntimeError("Models received differently ordered labels.")
            labels_by_domain[domain_name] = labels
            predictions[model_name][domain_name] = {
                "clean": view_logits["clean"],
                **{
                    aggregation: aggregate_numpy_logits(
                        view_logits,
                        method=aggregation,
                    )
                    for aggregation in TTA_AGGREGATIONS
                },
            }
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    single_model: dict[str, dict[str, Any]] = {}
    for model_name, domain_predictions in predictions.items():
        single_model[model_name] = {}
        for policy_name in ("clean", *TTA_AGGREGATIONS):
            single_model[model_name][policy_name] = {
                domain: policy_metrics(
                    labels_by_domain[domain],
                    domain_predictions[domain][policy_name],
                )
                for domain in domains
            }

    ensembles: dict[str, dict[str, Any]] = {}
    for policy_name in ("clean", *TTA_AGGREGATIONS):
        native_thresholds = {
            "sid": float(
                single_model["sid"][policy_name]["sid"]["threshold"]
            ),
            "cifake": float(
                single_model["cifake"][policy_name]["cifake"]["threshold"]
            ),
        }
        native_scales = {
            "sid": float(np.std(predictions["sid"]["sid"][policy_name])),
            "cifake": float(
                np.std(predictions["cifake"]["cifake"][policy_name])
            ),
        }
        policy_ensembles: dict[str, Any] = {}
        for alpha in ENSEMBLE_ALPHAS:
            combined_by_domain = {}
            for domain in domains:
                sid_evidence = standardized_evidence(
                    predictions["sid"][domain][policy_name],
                    threshold=native_thresholds["sid"],
                    scale=native_scales["sid"],
                )
                cifake_evidence = standardized_evidence(
                    predictions["cifake"][domain][policy_name],
                    threshold=native_thresholds["cifake"],
                    scale=native_scales["cifake"],
                )
                combined_by_domain[domain] = (
                    alpha * sid_evidence + (1.0 - alpha) * cifake_evidence
                )
            policy_ensembles[f"{alpha:.2f}"] = summarize_policy(
                labels_by_domain,
                combined_by_domain,
            )
        best_alpha, best_result = max(
            policy_ensembles.items(),
            key=lambda item: (
                item[1]["worst_domain_auroc"],
                item[1]["mean_domain_auroc"],
            ),
        )
        ensembles[policy_name] = {
            "native_thresholds": native_thresholds,
            "native_scales": native_scales,
            "best_alpha": float(best_alpha),
            "best": best_result,
            "candidates": policy_ensembles,
        }

    result = {
        "selection_data": {
            domain: {
                "manifest": str(domains[domain]["manifest"]),
                "samples": int(len(labels_by_domain[domain])),
            }
            for domain in domains
        },
        "checkpoints": checkpoints,
        "checkpoint_thresholds": checkpoint_thresholds,
        "tta_transforms": list(TTA_TRANSFORMS),
        "single_model": single_model,
        "ensembles": ensembles,
    }
    atomic_write_json(result, args.output)
    print(f"Wrote {args.output}", flush=True)
    for policy_name, policy in ensembles.items():
        best = policy["best"]
        print(
            f"{policy_name}: best_alpha={policy['best_alpha']:.2f} "
            f"worst_domain_auc={best['worst_domain_auroc']:.4f} "
            f"mean_domain_auc={best['mean_domain_auroc']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
