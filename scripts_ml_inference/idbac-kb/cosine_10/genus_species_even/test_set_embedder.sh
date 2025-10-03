#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

for k in {0..3}; do
    python3 bare_inference.py   --model "cosine_10" \
                                --dataset "IDBac" \
                                --target "genera" \
                                --split_type "species_even" \
                                --inference_set "test" \
                                --run_for_score \
                                --new_paths \
                                -k $k
done
