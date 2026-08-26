#!/bin/bash
set -euo pipefail

# Download the organisers' RESERVED WildFake demo benchmark into
# data/wildfake_benchmark/eval/.
#
#   Non-AIGC : COCO val2017      (4998 images)
#   AIGC     : DALL-E "Advanced" (8843 images)
#
# This benchmark is for demonstrating performance and tracking iterative
# improvement only. It does NOT count toward the final score and must NEVER be
# used for training. It is written to its own directory with an `eval/` split
# (not `train/`) so a training run cannot pick it up by accident.
#
# Usage:
#   ./scripts/download_wildfake_benchmark.sh                # 300 images per class
#   ./scripts/download_wildfake_benchmark.sh --per-class 0  # the full benchmark

# Resolve the directory this script lives in, so it works regardless of
# where you run it from (e.g. `./download_wildfake_benchmark.sh` vs `bash path/to/it`)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

# Prefer the project venv if it exists, else fall back to whatever python is active.
PYTHON="${REPO_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"

echo "Downloading the reserved WildFake benchmark into ${REPO_ROOT}/data/wildfake_benchmark ..."
"$PYTHON" -m aigc_detector.data.download_wildfake_benchmark "$@"
