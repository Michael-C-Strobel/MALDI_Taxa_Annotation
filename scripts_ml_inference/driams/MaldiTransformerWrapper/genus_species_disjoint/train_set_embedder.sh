#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

for k in {0..6}; do
    python3 bare_inference.py   --model "MaldiTransformerWrapper" \
                                --version 0 \
                                --dataset "DIRAMS-A" \
                                --target "genera" \
                                --split_type "species" \
                                --inference_set "train" \
                                --run_for_score \
                                --new_paths \
                                -k $k
done
