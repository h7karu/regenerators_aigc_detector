"""Verify a local installation without loading images or the reserved test set."""

from __future__ import annotations

import csv
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model, count_parameters  # noqa: E402
from demo_inference import DemoDetector  # noqa: E402
from deployment import (  # noqa: E402
    DEFAULT_INFERENCE_AGGREGATION,
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_INFERENCE_TRANSFORMS,
    DEFAULT_MODEL_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
)
from utils import load_config  # noqa: E402

EXPECTED_PYTHON = (3, 12, 14)
DIRECT_PACKAGES = (
    "albumentations",
    "datasets",
    "gradio",
    "matplotlib",
    "numpy",
    "pandas",
    "Pillow",
    "pytest",
    "PyYAML",
    "scikit-learn",
    "tensorboard",
    "timm",
    "torch",
    "torchvision",
)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> None:
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        raise RuntimeError(
            f"Expected Python {'.'.join(map(str, EXPECTED_PYTHON))}, "
            f"found {platform.python_version()}."
        )

    print(f"Python: {platform.python_version()}")
    for package in DIRECT_PACKAGES:
        print(f"{package}: {version(package)}")

    config = load_config(DEFAULT_MODEL_CONFIG)
    model = build_model(config, pretrained=False)
    parameter_count = count_parameters(model)
    if parameter_count >= 2_000_000_000:
        raise RuntimeError(f"Model exceeds parameter limit: {parameter_count:,}")
    print(f"Model construction: ok ({parameter_count:,} parameters)")
    print(f"Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    del model

    manifests = PROJECT_ROOT / "data" / "manifests"
    if manifests.is_dir():
        for name in ("cifake_train.csv", "cifake_val.csv", "cifake_test.csv"):
            path = manifests / name
            if path.is_file():
                print(f"Manifest {name}: {count_csv_rows(path):,} rows")

    checkpoint = PROJECT_ROOT / DEFAULT_MODEL_CHECKPOINT
    if checkpoint.is_file():
        print(f"Demo checkpoint: present ({checkpoint.stat().st_size:,} bytes)")
        detector = DemoDetector(
            DEFAULT_MODEL_CHECKPOINT,
            DEFAULT_MODEL_CONFIG,
            device="cpu",
        )
        metadata = detector.model_metadata()
        expected_policy = (
            f"{len(DEFAULT_INFERENCE_TRANSFORMS)}-view "
            f"{DEFAULT_INFERENCE_AGGREGATION} TTA"
        )
        if metadata["inference_policy"] != expected_policy:
            raise RuntimeError(
                "Demo inference policy mismatch: "
                f"{metadata['inference_policy']} != {expected_policy}"
            )
        if float(metadata["decision_threshold"]) != DEFAULT_INFERENCE_THRESHOLD:
            raise RuntimeError("Demo decision threshold does not match deployment.")
        synthetic_image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = detector.predict(synthetic_image)
        print(
            f"Demo inference: ok ({metadata['checkpoint']}, "
            f"{metadata['inference_policy']}, score={result.score:.4f})"
        )
    else:
        raise RuntimeError(
            "Deployed checkpoint is missing. Run `git lfs pull --include="
            '"checkpoints/sid_local_lora/sid_local_lora_best.pt"`.'
        )

    print("Environment verification passed. No test images were loaded.")


if __name__ == "__main__":
    main()
