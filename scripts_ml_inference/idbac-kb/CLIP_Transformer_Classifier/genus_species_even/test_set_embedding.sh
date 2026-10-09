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
#                             --version 13 \
#                             --dataset "IDBac" \
#                             --target "genera" \
#                             --split_type "species" \
#                             --inference_set "test"

for version in {0..4}; do
    python3 bare_inference.py   --model "CLIP_Transformer_Classifier" \
                                --version $version \
                                --dataset "IDBac" \
                                --target "genera" \
                                --split_type "species_even" \
                                --inference_set "test" \
                                --run_for_score \
                                --new_paths
done

# Model Graveyard:
                            # --checkpoint_path "./lightning_logs/CLIP_Transformer_Classifier/version_1//checkpoints/epoch=329-step=6270.ckpt" \