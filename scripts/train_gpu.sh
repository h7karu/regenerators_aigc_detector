#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/multisource_phase_robust.yaml}"

python scripts/check_training_readiness.py --config "$CONFIG" --require-cuda
python train.py --config "$CONFIG" --device cuda
