#!/usr/bin/env bash
#
# Run BLAST on consensus 16S sequences against NCBI.
# Usage: bash scripts/run_blast_16S.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/consensus_16S"
OUTPUT_DIR="${PROJECT_DIR}/data/16S_results"
CONDA_ENV="${PROJECT_DIR}/conda_env"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== BLAST 16S Pipeline ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_DIR}/bin/blast_16S.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
