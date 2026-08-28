"""Audit manifests, cross-split leakage, local paths, and accelerator access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from manifest_schema import normalise_manifest, validate_manifest  # noqa: E402
from utils import load_config, resolve_project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multisource_phase_robust.yaml")
    parser.add_argument("--skip-path-check", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    frames: list[pd.DataFrame] = []
    for split, key in (("train", "train_manifest"), ("val", "val_manifest"), ("test", "test_manifest")):
        manifest_path = resolve_project_path(config["data"][key])
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing {split} manifest: {manifest_path}")
        frame = normalise_manifest(pd.read_csv(manifest_path))
        if set(frame["split"]) != {split}:
            raise ValueError(
                f"{manifest_path} should contain only split={split!r}; "
                f"found {sorted(frame['split'].unique())}"
            )
        validate_manifest(frame, check_paths=not args.skip_path_check)
        frames.append(frame)

    combined = normalise_manifest(pd.concat(frames, ignore_index=True))
    validate_manifest(combined, check_paths=not args.skip_path_check)
    print("Manifest counts:")
    print(combined.groupby(["split", "dataset", "label"]).size().to_string())
    print("\nGenerator counts:")
    print(
        combined.groupby(["dataset", "generator_family", "generator"])
        .size()
        .to_string()
    )

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        gib = properties.total_memory / (1024**3)
        print(f"\nCUDA ready: {properties.name} ({gib:.1f} GiB)")
    else:
        print("\nCUDA unavailable: use a GPU runtime for full training.")
        if args.require_cuda:
            raise RuntimeError("--require-cuda was set but PyTorch cannot access CUDA.")


if __name__ == "__main__":
    main()
