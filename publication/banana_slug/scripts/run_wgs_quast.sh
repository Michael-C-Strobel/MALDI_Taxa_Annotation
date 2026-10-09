#!/usr/bin/env bash
#
# Run QUAST assembly QC on all WGS assemblies.
# Usage: bash scripts/run_wgs_quast.sh [--threads 4]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/WGS"
OUTPUT_DIR="${PROJECT_DIR}/data/wgs_quast"
CONDA_ENV="${PROJECT_DIR}/conda_env"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== QUAST Assembly QC ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_DIR}/bin/wgs_quast.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
