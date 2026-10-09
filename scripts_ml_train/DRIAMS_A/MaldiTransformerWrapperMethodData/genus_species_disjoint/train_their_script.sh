#!/bin/bash
# This script runs the Optuna hyperparameter tuning for the specified model.
# It assumes you have Python and Optuna installed.

set -e

cd ../../../../

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml
PYTHON_SCRIPT="abstracted_train_malditransformer.py"


# Params
# MODEL_TYPE="MaldiTransformerWrapper"
# DATASET="driams"  # "driams" or "idbac"
# TARGET="genera"
# SPLIT_METHOD="species_even"
# BATCH_SIZE=1024   # Congruent with original manuscript
# N_EPOCHS=4      # Congruent in terms of steps with the original manuscript

# If the first argument is not provided, raise an error
if [ -z "$1" ]; then
  echo "Error: No k-fold argument provided."
  echo "Usage: $0 <k-fold>"
  exit 1
fi


echo "K-fold: $1"

mkdir -p "./MALDI-Transformer_Reproduction/k=${1}"

# Fold 0 only
python3 $PYTHON_SCRIPT \
  "../../data/driams/processed_data/MaldiTransformer/species/maldi_transformer_fold_${1}.h5torch" \
  "./MALDI-Transformer_Reproduction/k=${1}" \
  "M" \
  --batch_size 1024