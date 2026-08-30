"""Run AIGC detection over a directory of images and write JSON predictions.

Output is a JSON list of {"image_path": ..., "pred": ...} where `pred` is the
model's confidence (0-1) that the image is AI-generated, matching the
deliverable spec in the hackathon brief.

Usage:
    python -m aigc_detector.infer \
        --input-dir path/to/images \
        --checkpoint models/clip_linear_probe.joblib \
        --output predictions.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from aigc_detector.models.dual_branch import DualBranchClassifier
from aigc_detector.utils import list_images, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("models/dual_branch.joblib"))
    p.add_argument("--output", type=Path, default=Path("predictions.json"))
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    paths = list_images(args.input_dir)
    if not paths:
        raise SystemExit(f"No images found under {args.input_dir}")
    print(f"Found {len(paths)} images under {args.input_dir}")

    clf = DualBranchClassifier.load(args.checkpoint)

    results = []
    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i : i + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        preds = clf.predict_proba(images, batch_size=args.batch_size)
        for path, pred in zip(batch_paths, preds):
            results.append({"image_path": str(path), "pred": float(pred)})
        print(f"  scored {min(i + args.batch_size, len(paths))}/{len(paths)}")

    write_json(results, args.output)
    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
