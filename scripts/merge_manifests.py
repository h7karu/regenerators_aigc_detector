"""Merge canonical dataset manifests and emit train/val/test CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from manifest_schema import normalise_manifest, validate_manifest, write_manifest  # noqa: E402
from utils import resolve_project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--name", default="combined")
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Require every referenced local image to exist.",
    )
    args = parser.parse_args()

    frames = []
    for input_path in args.inputs:
        path = resolve_project_path(input_path)
        frame = normalise_manifest(pd.read_csv(path))
        validate_manifest(frame, check_paths=args.check_paths)
        frames.append(frame)

    combined = normalise_manifest(pd.concat(frames, ignore_index=True))
    validate_manifest(combined, check_paths=args.check_paths)
    output_directory = resolve_project_path(args.output_dir)
    write_manifest(
        combined,
        output_directory / f"{args.name}_all.csv",
        check_paths=args.check_paths,
    )
    for split in ("train", "val", "test"):
        split_frame = combined[combined["split"] == split].reset_index(drop=True)
        if not split_frame.empty:
            write_manifest(
                split_frame,
                output_directory / f"{args.name}_{split}.csv",
                check_paths=args.check_paths,
            )

    print(f"Merged {len(combined):,} records from {len(frames)} manifests")
    print(combined.groupby(["split", "dataset", "label"]).size().to_string())


if __name__ == "__main__":
    main()
