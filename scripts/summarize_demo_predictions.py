"""Summarize accuracy over reports/notebook/demo_predictions.json for the demo video.

Ground truth is inferred from the filename convention used in demo_images/
(e.g. "sid_set_real_3.jpg" -> real, "cifake_fake_0.jpg" -> fake).

Usage:
    python scripts/summarize_demo_predictions.py [predictions.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deployment import DEFAULT_INFERENCE_THRESHOLD


def true_label_is_fake(image_path: str) -> bool:
    stem = Path(image_path).stem.lower()
    if "_real_" in f"_{stem}_":
        return False
    if "_fake_" in f"_{stem}_":
        return True
    raise ValueError(f"Cannot infer ground truth from filename: {image_path}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/notebook/demo_predictions.json")
    predictions = json.loads(path.read_text())

    correct = 0
    real_total = real_correct = 0
    fake_total = fake_correct = 0
    for entry in predictions:
        threshold = entry.get("threshold", DEFAULT_INFERENCE_THRESHOLD)
        predicted_fake = entry["pred"] >= threshold
        actual_fake = true_label_is_fake(entry["image_path"])
        is_correct = predicted_fake == actual_fake
        correct += is_correct
        if actual_fake:
            fake_total += 1
            fake_correct += is_correct
        else:
            real_total += 1
            real_correct += is_correct

    total = len(predictions)
    print(f"Overall: {correct}/{total} correct ({correct / total:.1%})")
    print(f"Real images:  {real_correct}/{real_total} correct ({real_correct / real_total:.1%})")
    print(f"Fake images:  {fake_correct}/{fake_total} correct ({fake_correct / fake_total:.1%})")


if __name__ == "__main__":
    main()
