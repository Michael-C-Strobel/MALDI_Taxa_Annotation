#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

# python3 bare_inference.py   --model "Prototyical_Transformer" \
#                             --checkpoint_path "./lightning_logs_DRIAMS_A/Prototyical_Transformer/species/version_9/checkpoints/epoch=99-step=100000.ckpt" \
#                             --dataset "DIRAMS-A" \
#                             --target "species" \
#                             --split_type "species" \
#                             --inference_set "train"

for version in {0..4}; do
    python3 bare_inference.py   --model "Prototyical_Transformer" \
                                --version $version \
                                --dataset "DIRAMS-A" \
                                --target "species" \
                                --split_type "species" \
                                --inference_set "train" \
                                --run_for_score \
                                --new_paths 
done