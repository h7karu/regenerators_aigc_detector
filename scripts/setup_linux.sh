#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEPLOYED_CHECKPOINT="checkpoints/sid_local_lora/sid_local_lora_best.pt"
if ! git lfs version >/dev/null 2>&1; then
  echo "Git LFS is required. Install it from https://git-lfs.com/ and rerun this script." >&2
  exit 1
fi

git lfs install --local
git lfs pull --include="$DEPLOYED_CHECKPOINT"
if [[ ! -f "$DEPLOYED_CHECKPOINT" ]]; then
  echo "Missing deployed checkpoint: $DEPLOYED_CHECKPOINT" >&2
  exit 1
fi

CHECKPOINT_SIZE="$(stat -c %s "$DEPLOYED_CHECKPOINT")"
if (( CHECKPOINT_SIZE < 100000000 )); then
  echo "$DEPLOYED_CHECKPOINT is too small ($CHECKPOINT_SIZE bytes); Git LFS may have left a pointer file." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv python install 3.12.14

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python 3.12.14 --seed
fi

PYTHON_VERSION="$(.venv/bin/python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")"
if [[ "$PYTHON_VERSION" != "3.12.14" ]]; then
  echo ".venv uses Python $PYTHON_VERSION, expected 3.12.14. Remove the disposable environment and rerun this script." >&2
  exit 1
fi

uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.13.0+cpu torchvision==0.28.0+cpu
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/verify_environment.py

echo "Setup complete. Activate with: source .venv/bin/activate"
