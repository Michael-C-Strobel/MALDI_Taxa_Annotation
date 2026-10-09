#!/bin/bash

set -e

cd ../../../../

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml
PYTHON_SCRIPT="abstracted_train_malditransformer.py"

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
  "../../data/driams/processed_data/MaldiTransformer_Genus_Labels/species/maldi_transformer_fold_${1}.h5torch" \
  "./MALDI-Transformer_Reproduction/k=${1}" \
  "M" \
  --batch_size 1024