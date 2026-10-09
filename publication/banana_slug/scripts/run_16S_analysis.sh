#!/usr/bin/env bash
#
# Analyze 16S BLAST results at genus and species identity thresholds.
# Usage: bash scripts/run_16S_analysis.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/16S_results"
OUTPUT_DIR="${PROJECT_DIR}/data/16S_analysis"
CONDA_ENV="${PROJECT_DIR}/conda_env"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== 16S Analysis Pipeline ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_DIR}/bin/16S_analysis.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
