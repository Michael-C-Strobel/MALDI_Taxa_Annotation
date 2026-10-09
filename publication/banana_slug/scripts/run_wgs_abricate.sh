#!/usr/bin/env bash
#
# Screen assemblies for AMR and virulence genes with abricate.
# Usage: bash scripts/run_wgs_abricate.sh [--databases ncbi,card,resfinder,vfdb]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/WGS"
OUTPUT_DIR="${PROJECT_DIR}/data/wgs_abricate"
CONDA_ENV="${PROJECT_DIR}/conda_env"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== Abricate AMR/Virulence Screening ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_DIR}/bin/wgs_abricate.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
