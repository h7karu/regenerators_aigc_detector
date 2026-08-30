#!/bin/bash
set -euo pipefail

# Mac/Linux counterpart to setup_ui.cmd. Creates venv/ and installs the
# requirements needed to run the Gradio UI.
#
#   ./scripts/setup_ui.sh              # environment only
#   ./scripts/setup_ui.sh --train      # also train the demo checkpoint if missing

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/venv/bin/python"

TRAIN_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --train) TRAIN_MODEL=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

cd "$PROJECT_ROOT"

if [ ! -x "$PYTHON" ]; then
  echo "Creating virtual environment at venv/ ..."
  python3 -m venv venv
fi

echo "Installing requirements..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

CHECKPOINT="${PROJECT_ROOT}/models/notebook_dual_branch.joblib"
if [ "$TRAIN_MODEL" -eq 1 ] && [ ! -f "$CHECKPOINT" ]; then
  "${SCRIPT_DIR}/train_demo_model.sh"
fi

echo
echo "UI environment is ready."
if [ ! -f "$CHECKPOINT" ]; then
  echo "No checkpoint at models/notebook_dual_branch.joblib yet -- train one with:"
  echo "  ./scripts/train_demo_model.sh"
fi
echo "Launch with: ./run_ui.sh"
