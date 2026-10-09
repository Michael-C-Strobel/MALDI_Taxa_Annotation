#!/usr/bin/env bash
#
# Run the complete WGS analysis pipeline.
# Usage: bash scripts/run_wgs_all.sh [--threads 4]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================="
echo "  WGS Analysis Pipeline"
echo "==============================="
echo ""

echo "[Step 1/6] Assembly QC (QUAST)"
echo "-------------------------------"
bash "${SCRIPT_DIR}/run_wgs_quast.sh" "$@"
echo ""

echo "[Step 2/6] Contamination/Completeness (CheckM2)"
echo "------------------------------------------------"
bash "${SCRIPT_DIR}/run_wgs_checkm2.sh" "$@"
echo ""

echo "[Step 3/6] AMR/Virulence Screening (Abricate)"
echo "----------------------------------------------"
bash "${SCRIPT_DIR}/run_wgs_abricate.sh"
echo ""

echo "[Step 4/6] MLST Typing"
echo "----------------------"
bash "${SCRIPT_DIR}/run_wgs_mlst.sh" "$@"
echo ""

echo "[Step 5/6] GTDB-tk Taxonomy Classification"
echo "-------------------------------------------"
bash "${SCRIPT_DIR}/run_wgs_gtdbtk.sh" "$@"
echo ""

echo "[Step 6/6] Master Summary"
echo "-------------------------"
bash "${SCRIPT_DIR}/run_wgs_summary.sh"
echo ""

echo "==============================="
echo "  Pipeline Complete"
echo "==============================="
