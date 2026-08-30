#!/bin/bash
set -euo pipefail

# Mac/Linux counterpart to run_ui.cmd. Launches the Gradio detector UI using the
# repo's virtual environment, without needing it activated first.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: virtual environment not found at ${SCRIPT_DIR}/venv" >&2
  echo >&2
  echo "Create it first:" >&2
  echo "  ./scripts/setup_ui.sh" >&2
  echo >&2
  echo "Or manually:" >&2
  echo "  python3 -m venv venv" >&2
  echo "  ./venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" app.py
