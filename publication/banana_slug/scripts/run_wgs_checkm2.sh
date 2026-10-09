#!/usr/bin/env bash
#
# Run CheckM2 completeness/contamination analysis.
# Usage: bash scripts/run_wgs_checkm2.sh [--threads 4]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/WGS"
OUTPUT_DIR="${PROJECT_DIR}/data/wgs_checkm2"
DB_PATH="${PROJECT_DIR}/data/checkm2_db"
CONDA_ENV="${PROJECT_DIR}/conda_env_checkm2"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== CheckM2 Contamination/Completeness ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "DB:     ${DB_PATH}"
echo ""

python "${PROJECT_DIR}/bin/wgs_checkm2.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --db "${DB_PATH}" \
    --download-db \
    "$@"
