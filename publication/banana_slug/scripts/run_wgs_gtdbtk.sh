#!/usr/bin/env bash
#
# Run GTDB-tk taxonomy classification on WGS assemblies.
# Usage: bash scripts/run_wgs_gtdbtk.sh [--threads 4]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_DIR="${PROJECT_DIR}/data/WGS"
OUTPUT_DIR="${PROJECT_DIR}/data/wgs_gtdbtk"
DB_PATH="${PROJECT_DIR}/data/gtdbtk_db"
CONDA_ENV="${PROJECT_DIR}/conda_env_gtdbtk"

# Create conda env if it doesn't exist
if [ ! -d "${CONDA_ENV}" ]; then
    echo "Creating GTDB-tk conda environment at ${CONDA_ENV}..."
    echo "This only needs to happen once."
    conda create -p "${CONDA_ENV}" -c conda-forge -c bioconda gtdbtk=2.6.1 -y
fi

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== GTDB-tk Taxonomy Classification ==="
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "DB:     ${DB_PATH}"
echo ""

python "${PROJECT_DIR}/bin/wgs_gtdbtk.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --db "${DB_PATH}" \
    --download-db \
    "$@"
