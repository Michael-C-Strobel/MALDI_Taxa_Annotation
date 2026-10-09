#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

# python3 bare_inference.py   --model "CLIP_Transformer_Classifier" \
#                             --checkpoint_path "./lightning_logs_DRIAMS_A/CLIP_Transformer_Classifier/version_7/checkpoints/epoch=1069-step=6420.ckpt" \
#                             --dataset "DIRAMS-A" \
#                             --target "species" \
#                             --split_type "species" \
#                             --inference_set "test"
for version in {0..4}; do
    python3 bare_inference.py   --model "CLIP_Transformer_Classifier" \
                                --version $version \
                                --dataset "DIRAMS-A" \
                                --target "species" \
                                --split_type "genera" \
                                --inference_set "test" \
                                --run_for_score \
                                --new_paths 
done