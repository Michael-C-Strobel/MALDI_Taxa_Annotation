#!/bin/bash

# Enable exit on error
set -e

cd ../../../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./ml_maldi_nn_conda_env/

current_dir=$(pwd)

cd bin/ml

python3 bare_inference.py   --model "CLIP_Transformer" \
                            --version 3 \
                            --dataset "DIRAMS-A" \
                            --target "genera" \
                            --split_type "genera" \
                            --inference_set "test"

# """
# spectra_path ../../data/driams/preprocessing
# metadata_path ../../data/driams/preprocessing/merged_metadata.csv
# ml_processing_path ../../data/driams/processed_data
# batch_size 1
# num_workers 0
# transforms Compose(
#     <custom_transforms.SquareRootTransform object at 0x730d95b36c00>
#     <custom_transforms.SelectTopKPeaks object at 0x730d95af6c00>
#     <custom_transforms.NormalizeIntensity object at 0x730d95c153d0>
#     <custom_transforms.PadToLength object at 0x730d95cce1b0>
# )
# split_method genera
# inference_set_to_use test
# cast_to_classification False
# targets genera
# """