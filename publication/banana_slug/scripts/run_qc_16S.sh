#!/usr/bin/env bash
#
# QC 16S copies: detect mixed/contaminated assemblies and length outliers.
# Usage: bash scripts/run_qc_16S.sh [--blast-reps]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/16S"
OUTPUT_DIR="${PROJECT_DIR}/data/16S_clusters"
CONDA_ENV="${PROJECT_DIR}/conda_env"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== 16S QC Pipeline ==="
echo "Input:    ${INPUT_DIR}"
echo "Clusters: ${OUTPUT_DIR}/clusters"
echo "Metrics:  ${OUTPUT_DIR}/metrics"
echo ""

python "${PROJECT_DIR}/bin/qc_16S.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
