"""Download the organisers' reserved WildFake demo benchmark.

The brief reserves a WildFake subset purely for demonstrating model performance
and tracking iterative improvement. It does NOT count toward the final score and
must NOT be used for training:

    Non-AIGC : COCO val2017        (4998 images)
    AIGC     : DALL-E "Advanced"   (8843 images)  -> DALLE/Advanced/DALLE3

It is written to a separate directory (default `data/wildfake_benchmark/`) with
no train/ split, so it can never be picked up accidentally by a training run
that globs `data/*/train`.

Usage:
    python -m aigc_detector.data.download_wildfake_benchmark --per-class 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aigc_detector.data.download_wildfake import collect_from_archive
from aigc_detector.data.prepare import summarize, target_dir, write_bytes

# (archive, member prefix, tag) for each side of the reserved benchmark.
BENCHMARK_SOURCES = {
    0: ("Images/Real/coco.zip", "coco/coco2017/val2017/", "coco_val2017"),
    1: ("Images/Diffusion_based/DALLE.zip", "DALLE/Advanced/DALLE3/", "dalle3_advanced"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="Images per class. Use 0 for the full benchmark (4998 real / 8843 fake).",
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", default="wildfake_benchmark")
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep original bytes instead of re-encoding (must match how the model was trained).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Fetching the RESERVED WildFake demo benchmark (COCO val2017 + DALL-E 3 Advanced).\n"
        "This is for evaluation/demo only -- do NOT train on it.\n"
    )

    for label, (archive, prefix, tag) in BENCHMARK_SOURCES.items():
        name = "REAL" if label == 0 else "FAKE"
        print(f"[{name}]")
        # 0 means "everything"; collect_from_archive caps at the member count.
        n_wanted = args.per_class if args.per_class > 0 else 10**9
        items = collect_from_archive(
            archive, prefix, tag, n_wanted, args.seed, not args.no_normalize
        )

        # No train/ split: this directory is evaluation-only by construction.
        dest = target_dir(args.data_root, args.dataset_name, "eval", label)
        for filename, payload in items:
            write_bytes(dest, filename, payload)
        print(f"  wrote {len(items)} {name} images\n")

    summarize(args.data_root, args.dataset_name)
    print(
        "\nEvaluate against it with:\n"
        f"  python -m aigc_detector.evaluate --data-dir {args.data_root}/{args.dataset_name}/eval \\\n"
        "      --checkpoint models/dual_branch.joblib --output-dir reports/benchmark"
    )


if __name__ == "__main__":
    main()
