#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="/home/mstro016/miniforge3/bin/conda run -p ${PROJECT_ROOT}/ml_maldi_nn_conda_env python"

cd "${PROJECT_ROOT}"
${PYTHON_BIN} scripts/publication_optuna/generate_optuna_publication_assets.py \
  --project-root . \
  --output-dir figures/publication_optuna/genera_species_vs_genera_genera \
  --use-defaults \
  --top-k 15
