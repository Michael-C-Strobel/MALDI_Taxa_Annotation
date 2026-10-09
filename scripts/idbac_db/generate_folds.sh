#!/bin/bash

# This script generates folds for cross-validation using the provided dataset.

set -e

cd ../../

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml
PYTHON_SCRIPT="generate_k_fold_splits.py"

# species
# INPUT_FILE='../../data/idbac_db/raw/ammended_db.csv'
INPUT_FILE='../../data/idbac_db/preprocessing/db_with_taxonomy.csv'
OUTPUT_DIR='../../data/idbac_db/processed_data/'
SPLIT_STYLE='genera_holdout'
N_FOLDS=1   # Approx 40% test and val combined | 60% train (larger test to accomidate for small dataset)

# Old Min Group Size: "$((N_FOLDS * 2))" 

python3 "$PYTHON_SCRIPT" \
  --input_file "$INPUT_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --split_style "$SPLIT_STYLE" \
  --min_group_size 3 \
  -k "$N_FOLDS" \
  --test_genera "bacillus;micromonospora;solwaraspora;plantactinospora"


# Params
# INPUT_FILE='../../data/idbac_db/raw/ammended_db.csv'
# OUTPUT_DIR='../../data/idbac_db/processed_data/'
# SPLIT_STYLE='species_even'
# N_FOLDS=4   # Approx 40% test and val combined | 60% train (larger test to accomidate for small dataset)

# python3 "$PYTHON_SCRIPT" \
#   --input_file "$INPUT_FILE" \
#   --output_dir "$OUTPUT_DIR" \
#   --split_style "$SPLIT_STYLE" \
#   -k "$N_FOLDS" \
#   --min_group_size 3

