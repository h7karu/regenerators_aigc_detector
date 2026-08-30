#!/bin/bash
set -euo pipefail

# Mac/Linux counterpart to train_demo_model.cmd. Reproduces the small demo
# checkpoint the UI loads by default, with the same seed and settings as the
# Windows script so both platforms produce the same model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"

MAX_PER_CLASS=150
AUGMENT_COPIES=2
SEED=42
OUTPUT="models/notebook_dual_branch.joblib"
DATA_DIR="${PROJECT_ROOT}/data/cifake/train"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: virtual environment not found. Run ./scripts/setup_ui.sh first." >&2
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: CIFAKE training data not found at ${DATA_DIR}" >&2
  echo >&2
  echo "Download it with ./scripts/download_cifake.sh (needs a Kaggle token)," >&2
  echo "or train on the credential-free datasets instead:" >&2
  echo "  ./scripts/download_sid_set.sh && ./scripts/download_wildfake.sh" >&2
  echo "  ./venv/bin/python -m aigc_detector.train \\" >&2
  echo "      --data-dir data/sid_set/train --data-dir data/wildfake/train \\" >&2
  echo "      --max-per-class 150 --augment-copies 2 --output ${OUTPUT}" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -m aigc_detector.train \
  --data-dir "$DATA_DIR" \
  --max-per-class "$MAX_PER_CLASS" \
  --augment-copies "$AUGMENT_COPIES" \
  --seed "$SEED" \
  --output "$OUTPUT"
