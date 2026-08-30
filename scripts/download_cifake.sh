#!/bin/bash
set -euo pipefail

# Resolve the directory this script lives in, so it works regardless of
# where you run it from (e.g. `./download_cifake.sh` vs `bash path/to/it`)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where the dataset should end up (change this if you want a different location)
DEST_DIR="${SCRIPT_DIR}/../data/cifake"

# Fail early with an actionable message rather than a raw API stack trace.
# The Kaggle CLI reads ~/.kaggle/kaggle.json (or $KAGGLE_CONFIG_DIR/kaggle.json),
# else KAGGLE_USERNAME/KAGGLE_KEY exported in the shell. A .env file in the repo
# root is NOT read by anything -- see the README.
CONFIG_DIR="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
if [ ! -f "${CONFIG_DIR}/kaggle.json" ] && [ -z "${KAGGLE_USERNAME:-}" -o -z "${KAGGLE_KEY:-}" ]; then
  echo "ERROR: no Kaggle API credentials found." >&2
  echo >&2
  echo "CIFAKE is downloaded through the Kaggle API, which needs a token of your own:" >&2
  echo "  1. kaggle.com -> your profile -> Settings -> API -> 'Create New Token'" >&2
  echo "  2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/" >&2
  echo "     chmod 600 ~/.kaggle/kaggle.json" >&2
  echo >&2
  echo "Or export KAGGLE_USERNAME and KAGGLE_KEY in this shell instead." >&2
  echo "(A .env file in the repo root does not work -- nothing reads it.)" >&2
  echo >&2
  echo "The SID_Set and WildFake scripts need no credentials and will work as-is." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

echo "Downloading CIFAKE dataset to ${DEST_DIR}..."
kaggle datasets download \
  birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p "$DEST_DIR" \
  --unzip

echo "Done. Contents:"
ls -la "$DEST_DIR"
