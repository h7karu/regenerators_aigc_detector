"""Train the dual-branch AIGC detector.

Key choices this script implements:
  * Robustness is trained in, not just measured. Each source image contributes a
    clean view plus `--augment-copies` randomly degraded views (JPEG, blur,
    resize, noise, color jitter, crop at random severity), so the head learns
    how features move under post-processing.
  * Multiple datasets can be mixed with repeated `--data-dir`, which matters for
    generalising across generator families.

Usage:
    # CIFAKE only, with augmentation
    python -m aigc_detector.train --data-dir data/cifake/train --max-per-class 2000

    # Mix CIFAKE + WildFake + SID_Set
    python -m aigc_detector.train \
        --data-dir data/cifake/train --data-dir data/wildfake/train \
        --data-dir data/sid_set/train --max-per-class 1000

    # Ablation: forensic branch only, no augmentation
    python -m aigc_detector.train --branches forensic --augment-copies 0
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from aigc_detector.data.dataset import CIFAKEDataset
from aigc_detector.data.transforms import build_training_views
from aigc_detector.models.dual_branch import DEFAULT_CLIP_MODEL, VALID_BRANCHES, DualBranchClassifier
from aigc_detector.utils import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        dest="data_dirs",
        help="Train split with REAL/ and FAKE/ subfolders. Repeat to mix datasets.",
    )
    p.add_argument("--output", type=Path, default=Path("models/dual_branch.joblib"))
    p.add_argument(
        "--branches",
        nargs="+",
        default=list(VALID_BRANCHES),
        choices=list(VALID_BRANCHES),
        help="Which feature branches to use. Use one branch alone for ablations.",
    )
    p.add_argument("--head", default="logreg", choices=["logreg", "mlp"])
    p.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--augment-copies",
        type=int,
        default=2,
        help="Randomly degraded copies per source image (0 disables augmentation).",
    )
    p.add_argument(
        "--max-augment-ops",
        type=int,
        default=2,
        help="Max transforms stacked on a single augmented copy.",
    )
    p.add_argument(
        "--no-clean-view",
        action="store_true",
        help="Drop the un-augmented copy (not recommended: hurts clean accuracy).",
    )
    p.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Cap source images per REAL/FAKE class per dataset (before augmentation).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    data_dirs = args.data_dirs or [Path("data/cifake/train")]

    images: list = []
    labels: list[int] = []
    for data_dir in data_dirs:
        dataset = CIFAKEDataset(data_dir, max_per_class=args.max_per_class)
        print(f"Loaded {len(dataset)} source images from {data_dir}")
        for i in range(len(dataset)):
            img, label, _path = dataset[i]
            views = build_training_views(
                img,
                rng,
                n_augmented=args.augment_copies,
                keep_clean=not args.no_clean_view,
                max_ops=args.max_augment_ops,
            )
            images.extend(views)
            labels.extend([label] * len(views))

    if not images:
        raise SystemExit(f"No training images found in: {', '.join(str(d) for d in data_dirs)}")

    n_pos = sum(labels)
    print(
        f"\nTraining views: {len(images)} "
        f"({len(images) - n_pos} REAL / {n_pos} FAKE), "
        f"augment-copies={args.augment_copies}, branches={'+'.join(args.branches)}"
    )

    clf = DualBranchClassifier(
        branches=args.branches,
        clip_model=args.clip_model,
        head=args.head,
    )
    print("Extracting features and fitting head...")
    clf.fit(images, labels, batch_size=args.batch_size)

    clf.save(args.output)
    print(f"Saved classifier to {args.output}")


if __name__ == "__main__":
    main()
