#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

for version in {0..4}; do
    python3 bare_inference.py   --model "Prototyical_Transformer" \
                                --version $version \
                                --dataset "IDBac" \
                                --target "genera" \
                                --split_type "genera" \
                                --inference_set "test" \
                                --run_for_score \
                                --new_paths
done