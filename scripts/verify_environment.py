"""Verify a local installation without loading images or the reserved test set."""

from __future__ import annotations

import csv
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model, count_parameters  # noqa: E402
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

    config = load_config("configs/local_phase_experiment.yaml")
    model = build_model(config, pretrained=False)
    parameter_count = count_parameters(model)
    if parameter_count >= 2_000_000_000:
        raise RuntimeError(f"Model exceeds parameter limit: {parameter_count:,}")
    print(f"Model construction: ok ({parameter_count:,} parameters)")
    print(f"Torch device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    manifests = PROJECT_ROOT / "data" / "manifests"
    if manifests.is_dir():
        for name in ("cifake_train.csv", "cifake_val.csv", "cifake_test.csv"):
            path = manifests / name
            if path.is_file():
                print(f"Manifest {name}: {count_csv_rows(path):,} rows")

    checkpoint = (
        PROJECT_ROOT
        / "checkpoints"
        / "local_phase_experiment"
        / "local_phase_best.pt"
    )
    if checkpoint.is_file():
        print(f"Demo checkpoint: present ({checkpoint.stat().st_size:,} bytes)")
    else:
        print("Demo checkpoint: absent (expected in a fresh source-only clone)")

    print("Environment verification passed. No test images were loaded.")


if __name__ == "__main__":
    main()
