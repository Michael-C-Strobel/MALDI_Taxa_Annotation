#!/bin/bash

set -e

cd ../../

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml
PYTHON_SCRIPT="compute_within_class_cosine.py"

# Params
metadata_path='../../data/driams/preprocessing/merged_metadata.csv'
processed_spectra_dir='../../data/driams/processed_data/spectra/'
target='species'
output_path='../../data/driams/processed_data/within_class_cosine_${target}.pt'

python3 "$PYTHON_SCRIPT" \
  --metadata_path "$metadata_path" \
  --processed_spectra_dir "$processed_spectra_dir" \
  --target "$target" \
  --output_path "$output_path"

# Params
metadata_path='../../data/driams/preprocessing/merged_metadata.csv'
processed_spectra_dir='../../data/driams/processed_data/spectra/'
target='genera'
output_path="../../data/driams/processed_data/within_class_cosine_${target}.pt"

echo "Output path: $output_path"

python3 "$PYTHON_SCRIPT" \
  --metadata_path "$metadata_path" \
  --processed_spectra_dir "$processed_spectra_dir" \
  --target "$target" \
  --output_path "$output_path"
