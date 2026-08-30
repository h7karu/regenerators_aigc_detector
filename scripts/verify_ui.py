"""Fail-fast verification for the local UI and its checkpoint."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gradio

from aigc_detector.ui import DetectorService


def main() -> None:
    checkpoint = PROJECT_ROOT / "models" / "notebook_dual_branch.joblib"
    service = DetectorService(checkpoint)
    model = service.model

    if model.branches != ("clip", "forensic"):
        raise RuntimeError(f"Expected clip+forensic checkpoint, got {model.branches!r}")

    print(f"Gradio: {gradio.__version__}")
    print(f"Checkpoint: {checkpoint.resolve()}")
    print(f"Model: {type(model).__module__}.{type(model).__name__}")
    print(f"Branches: {', '.join(model.branches)}")
    print("UI verification passed.")


if __name__ == "__main__":
    main()
