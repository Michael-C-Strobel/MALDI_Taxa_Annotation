#!/bin/bash
# This script runs the Optuna hyperparameter tuning for the specified model.
# It assumes you have Python and Optuna installed.

set -e

cd ../../../../

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml
PYTHON_SCRIPT="train.py"


# Params
MODEL_TYPE="MaldiTransformerWrapperMethodData"
DATASET="driams"  # "driams" or "idbac"
TARGET="genera"
SPLIT_METHOD="species"
BATCH_SIZE=1024   # Congruent with original manuscript
N_EPOCHS=488      # Congruent in terms of steps with the original manuscript

# If the first argument is not provided, raise an error
if [ -z "$1" ]; then
  echo "Error: No k-fold argument provided."
  echo "Usage: $0 <k-fold>"
  exit 1
fi

echo "Running Optuna hyperparameter tuning for $MODEL_TYPE on $DATASET..."
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $N_EPOCHS"
echo "Target: $TARGET"
echo "Split method: $SPLIT_METHOD"
echo "K-fold: $1"

python "$PYTHON_SCRIPT" \
  --model_type "$MODEL_TYPE" \
  --dataset "$DATASET" \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$N_EPOCHS" \
  --target "$TARGET" \
  --split_method "$SPLIT_METHOD" \
  --train_for_score \
  --hparam_dir None \
  --maldi_nn_preprocessing \
  -k "$1"
