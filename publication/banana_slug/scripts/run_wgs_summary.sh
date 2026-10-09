#!/usr/bin/env bash
#
# Merge all WGS analysis summaries into a master table.
# Usage: bash scripts/run_wgs_summary.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONDA_ENV="${PROJECT_DIR}/conda_env"

eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "=== WGS Master Summary ==="
echo ""

python "${PROJECT_DIR}/bin/wgs_summary.py" \
    --quast-dir "${PROJECT_DIR}/data/wgs_quast" \
    --checkm2-dir "${PROJECT_DIR}/data/wgs_checkm2" \
    --abricate-dir "${PROJECT_DIR}/data/wgs_abricate" \
    --mlst-dir "${PROJECT_DIR}/data/wgs_mlst" \
    --gtdbtk-dir "${PROJECT_DIR}/data/wgs_gtdbtk" \
    --qc-16s "${PROJECT_DIR}/data/16S_clusters/metrics/qc_report.tsv" \
    --output-dir "${PROJECT_DIR}/data/wgs_summary"
