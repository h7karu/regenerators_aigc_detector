#!/bin/bash
set -euo pipefail

# Resolve the directory this script lives in, so it works regardless of
# where you run it from (e.g. `./download_cifake.sh` vs `bash path/to/it`)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the dataset should end up (change this if you want a different location)
DEST_DIR="${SCRIPT_DIR}/../data/cifake"

mkdir -p "$DEST_DIR"

echo "Downloading CIFAKE dataset to ${DEST_DIR}..."
kaggle datasets download \
  birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p "$DEST_DIR" \
  --unzip

echo "Done. Contents:"
ls -la "$DEST_DIR"
