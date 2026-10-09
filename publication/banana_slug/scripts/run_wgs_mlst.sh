#!/usr/bin/env bash
#
# Run MLST typing on WGS assemblies.
# Usage: bash scripts/run_wgs_mlst.sh [--threads 4]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/WGS"
OUTPUT_DIR="${PROJECT_DIR}/data/wgs_mlst"
CONDA_ENV="${PROJECT_DIR}/conda_env"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== MLST Typing ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_DIR}/bin/wgs_mlst.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
