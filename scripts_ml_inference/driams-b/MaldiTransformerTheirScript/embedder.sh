#!/bin/bash

# Enable exit on error
set -e

cd ../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

k=$1
version=0

echo "Running genus_species_disjoint test set inference with k=$k"

# Identify the checkpoint path for the specified k value (number of epochs varies based on k)
possible_checkpoint_paths="/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/MALDI-Transformer_Reproduction/k=$k/malditrfvanilla_M_200_0.15_0.01_0.0005/version_$version/checkpoints/epoch=*.ckpt"
# If more than 1, fail
checkpoint_path=$(ls $possible_checkpoint_paths 2>/dev/null || true)
if [ -z "$checkpoint_path" ]; then
    echo "Error: No checkpoint found for k=$k at $possible_checkpoint_paths"
    exit 1
elif [ $(echo "$checkpoint_path" | wc -l) -gt 1 ]; then
    echo "Error: Multiple checkpoints found for k=$k at $possible_checkpoint_paths"
    echo "$checkpoint_path"
    exit 1
fi

python3 bare_inference.py   --model "MaldiTransformerWrapper" \
                            --dataset "DRIAMS-B" \
                            --target "genera" \
                            --split_type "species" \
                            --inference_set "all" \
                            --run_for_score \
                            --new_paths \
                            --checkpoint_path $checkpoint_path \
                            --maldi_nn_preprocessing \
                            -k $k