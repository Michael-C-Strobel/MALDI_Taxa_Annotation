#!/usr/bin/env bash
#
# Zip the data folder, excluding raw 16S input sequences.
# Usage: bash scripts/zip_results.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUTPUT="${PROJECT_DIR}/data.zip"

rm -f "${OUTPUT}"
cd "${PROJECT_DIR}"
zip -r "${OUTPUT}" data/ -x "data/16S/*"

echo ""
echo "Created: ${OUTPUT}"
