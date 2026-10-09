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
MODEL_TYPE="CLIP_MALDI"
DATASET="driams"  # "driams" or "idbac"
TARGET="genera"
SPLIT_METHOD="genera"
N_TRIALS=250
BATCH_SIZE=32
N_EPOCHS=600


echo "Running Optuna hyperparameter tuning for $MODEL_TYPE on $DATASET..."
echo "Number of trials: $N_TRIALS"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $N_EPOCHS"
echo "Target: $TARGET"
echo "Split method: $SPLIT_METHOD"

python "$PYTHON_SCRIPT" \
  --model_type "$MODEL_TYPE" \
  --dataset "$DATASET" \
  --n_trials "$N_TRIALS" \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$N_EPOCHS" \
  --target "$TARGET" \
  --split_method "$SPLIT_METHOD" \
  -k 0

# Display a message when the script finishes
echo "Optuna hyperparameter tuning finished."
