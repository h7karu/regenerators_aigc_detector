#!/bin/bash
set -euo pipefail

# Download a SMALL subset of WildFake (ModelScope) into data/wildfake/.
#
# WildFake is ~1.3 TB of monolithic ZIP archives (6-54 GB each), so this does
# NOT download any archive. It reads each ZIP's central directory over HTTP
# range requests and fetches only the individual image members it samples --
# a few MB of transfer for the default subset.
#
# The subset deliberately EXCLUDES COCO val2017 and DALL-E 3 "Advanced", which
# the organisers reserved as a demo-only benchmark. Fetch those separately with
# ./scripts/download_wildfake_benchmark.sh
#
# Usage:
#   ./scripts/download_wildfake.sh             # default: 300 images per class
#   ./scripts/download_wildfake.sh --per-class 500
#   ./scripts/download_wildfake.sh --fake-sources ddim ddpm vqdm adm

# Resolve the directory this script lives in, so it works regardless of
# where you run it from (e.g. `./download_wildfake.sh` vs `bash path/to/it`)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

# Prefer the project venv if it exists, else fall back to whatever python is active.
PYTHON="${REPO_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"

echo "Downloading a WildFake subset into ${REPO_ROOT}/data/wildfake ..."
"$PYTHON" -m aigc_detector.data.download_wildfake "$@"
