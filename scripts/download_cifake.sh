#!/bin/bash
set -euo pipefail

# Resolve the directory this script lives in, so it works regardless of
# where you run it from (e.g. `./download_cifake.sh` vs `bash path/to/it`)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep this in sync with scripts/build_manifest.py and configs/*.yaml.
DEST_DIR="${SCRIPT_DIR}/../cifake-real-and-ai-generated-synthetic-images"

mkdir -p "$DEST_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: the Kaggle CLI is not installed." >&2
  echo "Install it with: python -m pip install kaggle" >&2
  exit 1
fi

echo "Downloading CIFAKE dataset to ${DEST_DIR}..."
kaggle datasets download \
  birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p "$DEST_DIR" \
  --unzip

echo "Done. Contents:"
ls -la "$DEST_DIR"

echo
echo "Next, build the training manifests with:"
echo "  python scripts/build_manifest.py"
