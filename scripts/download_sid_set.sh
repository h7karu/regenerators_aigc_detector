#!/bin/bash
set -euo pipefail

# Download a SMALL subset of SID_Set (Hugging Face) into data/sid_set/.
#
# SID_Set is ~140 GB in total, so this does NOT download the dataset. It reads
# the parquet footers over HTTP range requests, pulls only the image+label
# columns of the row groups it needs, and stops once it has enough images.
# Expect a few hundred MB of transfer for the default subset.
#
# Usage:
#   ./scripts/download_sid_set.sh              # default: 150 images per class
#   ./scripts/download_sid_set.sh --per-class 300
#   ./scripts/download_sid_set.sh --include-tampered

# Resolve the directory this script lives in, so it works regardless of
# where you run it from (e.g. `./download_sid_set.sh` vs `bash path/to/it`)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

# Prefer the project venv if it exists, else fall back to whatever python is active.
PYTHON="${REPO_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"

echo "Downloading a SID_Set subset into ${REPO_ROOT}/data/sid_set ..."
"$PYTHON" -m aigc_detector.data.download_sid_set "$@"
