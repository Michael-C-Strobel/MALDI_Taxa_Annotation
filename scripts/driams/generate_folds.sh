#!/bin/bash

# This script generates folds for cross-validation using the provided dataset.

set -e

cd ../../

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml
PYTHON_SCRIPT="generate_k_fold_splits.py"

# species_even
# INPUT_FILE='../../data/driams/preprocessing/merged_metadata_code_accessions.csv'
OUTPUT_DIR='../../data/driams/processed_data/'
# SPLIT_STYLE='species_even'
N_FOLDS=7   # Approx 30% test and val combined | 70% train

# python3 "$PYTHON_SCRIPT" \
#   --input_file "$INPUT_FILE" \
#   --output_dir "$OUTPUT_DIR" \
#   --split_style "$SPLIT_STYLE" \
#   --min_group_size "$((N_FOLDS * 2))" \
#   -k "$N_FOLDS"

# # species disjoint
# INPUT_FILE='../../data/driams/preprocessing/merged_metadata.csv'
# SPLIT_STYLE='species'

# python3 "$PYTHON_SCRIPT" \
#   --input_file "$INPUT_FILE" \
#   --output_dir "$OUTPUT_DIR" \
#   --split_style "$SPLIT_STYLE" \
#   -k "$N_FOLDS"

# # genera disjoint
INPUT_FILE='../../data/driams/preprocessing/merged_metadata.csv'
SPLIT_STYLE='genera'

python3 "$PYTHON_SCRIPT" \
  --input_file "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --split_style "$SPLIT_STYLE" \
  -k "$N_FOLDS" \
  --min_group_size 7
