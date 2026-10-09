#!/usr/bin/env bash
#
# Run the consensus 16S pipeline.
# Usage: bash scripts/run_consensus_16S.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/16S"
OUTPUT_DIR="${PROJECT_DIR}/data/consensus_16S"
CONDA_ENV="${PROJECT_DIR}/conda_env"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== Consensus 16S Pipeline ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_DIR}/bin/make_consensus_16S.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"
