#!/bin/bash


cd ../..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd bin

# DRIAMS-B

# TODO change outputs to identical directory, and add other DRIAMS datasets (if desired)
python3 process_data.py --input_file "../data/driams/raw/DRIAMS-B/raw/2018/*.txt" \
                        --driams_csv "../data/driams/raw/DRIAMS-B/id/2018/2018_clean.csv" \
                        --output_mzML_dir "../data/driams/raw/converted_to_mzml/driams-b" \
                        --output_dir "../data/driams/preprocessing/baseline_corrected/driams-b"

# TODO add other metadata files
python3 driams/generate_metadata_file.py --input_csv "../data/driams/raw/DRIAMS-B/id/2018/2018_clean.csv" \
                                         --output_file "../data/driams/preprocessing/merged_metadata.csv" \