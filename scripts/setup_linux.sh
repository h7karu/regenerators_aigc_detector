#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

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
