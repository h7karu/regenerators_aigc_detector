"""Evaluate a trained detector on clean data and every robustness transform.

Produces the brief's evaluation deliverables:
  - reports/robustness_summary.csv : per-transform accuracy/precision/recall/f1/
    AUROC, plus the false-positive rate at a fixed operating threshold.
  - reports/robustness_summary.md  : the same table, ready to paste into a
    writeup.
  - reports/error_analysis.json    : the most confident false positives and
    false negatives on clean data, for the error-analysis note.

On thresholds: accuracy at 0.5 is a weak summary for a moderation-style system,
where wrongly flagging a real user photo is the costlier error. We therefore
also pick the threshold that holds the false-positive rate at or below
`--target-fpr` on clean data and report every transform at that same fixed
threshold, which is how the detector would actually be deployed.

Usage:
    python -m aigc_detector.evaluate --data-dir data/cifake/test \
        --checkpoint models/dual_branch.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from aigc_detector.data.dataset import CIFAKEDataset
from aigc_detector.data.transforms import ROBUSTNESS_TRANSFORMS
from aigc_detector.models.dual_branch import DualBranchClassifier
from aigc_detector.utils import set_seed, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/cifake/test"))
    p.add_argument("--checkpoint", type=Path, default=Path("models/dual_branch.joblib"))
    p.add_argument("--output-dir", type=Path, default=Path("reports"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--target-fpr",
        type=float,
        default=0.05,
        help="False-positive rate the fixed operating threshold is tuned to on clean data.",
    )
    p.add_argument("--n-error-examples", type=int, default=15)
    p.add_argument("--max-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def threshold_for_fpr(labels: np.ndarray, probs: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on REAL images is <= target."""
    real_scores = np.sort(probs[labels == 0])
    if real_scores.size == 0:
        return 0.5
    idx = int(np.ceil((1.0 - target_fpr) * real_scores.size)) - 1
    idx = min(max(idx, 0), real_scores.size - 1)
    return float(np.nextafter(real_scores[idx], 1.0))


def metrics_at(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    n_real = int((labels == 0).sum())
    false_pos = int(((labels == 0) & (preds == 1)).sum())
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "fpr": false_pos / n_real if n_real else 0.0,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset = CIFAKEDataset(args.data_dir, max_per_class=args.max_per_class)
    if len(dataset) == 0:
        raise SystemExit(f"No images found under {args.data_dir}")
    print(f"Loaded {len(dataset)} test images from {args.data_dir}")

    clean_images, label_list, paths = [], [], []
    for img, label, path in (dataset[i] for i in range(len(dataset))):
        clean_images.append(img)
        label_list.append(label)
        paths.append(path)
    labels = np.asarray(label_list)

    clf = DualBranchClassifier.load(args.checkpoint)
    print(f"Model branches: {'+'.join(clf.branches)}\n")

    # Clean pass first: it defines the fixed operating threshold used everywhere.
    clean_probs = clf.predict_proba(clean_images, batch_size=args.batch_size)
    fixed_threshold = threshold_for_fpr(labels, clean_probs, args.target_fpr)
    print(f"Fixed operating threshold @ {args.target_fpr:.0%} clean FPR: {fixed_threshold:.4f}\n")

    rows = []
    for name, transform in ROBUSTNESS_TRANSFORMS.items():
        probs = (
            clean_probs
            if name == "clean"
            else clf.predict_proba([transform(img) for img in clean_images], batch_size=args.batch_size)
        )
        at_half = metrics_at(labels, probs, 0.5)
        at_fixed = metrics_at(labels, probs, fixed_threshold)
        rows.append(
            {
                "variant": name,
                "n": len(labels),
                "accuracy": at_half["accuracy"],
                "precision": at_half["precision"],
                "recall": at_half["recall"],
                "f1": at_half["f1"],
                "auroc": roc_auc_score(labels, probs),
                "acc@fixed_thr": at_fixed["accuracy"],
                "recall@fixed_thr": at_fixed["recall"],
                "fpr@fixed_thr": at_fixed["fpr"],
            }
        )
        print(
            f"  {name:18s} acc={rows[-1]['accuracy']:.3f}  auroc={rows[-1]['auroc']:.3f}  "
            f"recall@fixed={rows[-1]['recall@fixed_thr']:.3f}"
        )

    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "robustness_summary.csv"
    summary.to_csv(csv_path, index=False)

    md_path = args.output_dir / "robustness_summary.md"
    clean_row = summary.iloc[0]
    with open(md_path, "w") as f:
        f.write("# Robustness evaluation summary\n\n")
        f.write(f"- Data: `{args.data_dir}` ({len(labels)} images)\n")
        f.write(f"- Model branches: `{'+'.join(clf.branches)}`\n")
        f.write(
            f"- Fixed operating threshold: `{fixed_threshold:.4f}` "
            f"(tuned for <= {args.target_fpr:.0%} FPR on clean data)\n\n"
        )
        f.write(summary.round(4).to_markdown(index=False))
        f.write("\n\n## Degradation vs clean\n\n")
        f.write("| variant | Δ accuracy | Δ AUROC |\n|---|---|---|\n")
        for _, r in summary.iloc[1:].iterrows():
            f.write(
                f"| {r['variant']} | {r['accuracy'] - clean_row['accuracy']:+.4f} "
                f"| {r['auroc'] - clean_row['auroc']:+.4f} |\n"
            )
    print(f"\nWrote {csv_path} and {md_path}")

    # Error analysis on clean data, ranked by model confidence (most egregious first).
    clean_preds = (clean_probs >= fixed_threshold).astype(int)
    fp_idx = np.where((labels == 0) & (clean_preds == 1))[0]
    fn_idx = np.where((labels == 1) & (clean_preds == 0))[0]
    fp_idx = fp_idx[np.argsort(-clean_probs[fp_idx])][: args.n_error_examples]
    fn_idx = fn_idx[np.argsort(clean_probs[fn_idx])][: args.n_error_examples]

    error_path = args.output_dir / "error_analysis.json"
    write_json(
        {
            "threshold": fixed_threshold,
            "target_fpr": args.target_fpr,
            "n_false_positives": int(((labels == 0) & (clean_preds == 1)).sum()),
            "n_false_negatives": int(((labels == 1) & (clean_preds == 0)).sum()),
            "false_positives": [
                {"image_path": paths[i], "true_label": "REAL", "pred_prob": float(clean_probs[i])}
                for i in fp_idx
            ],
            "false_negatives": [
                {"image_path": paths[i], "true_label": "FAKE", "pred_prob": float(clean_probs[i])}
                for i in fn_idx
            ],
        },
        error_path,
    )
    print(f"Wrote {error_path}")


if __name__ == "__main__":
    main()
