#!/bin/bash

# Exit on any failure
set -e

cd ../..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd bin

# DRIAMS-A

# TODO add other metadata files
python3 driams/generate_metadata_file.py --input_csv "../data/driams/downloads/DRIAMS-A/id/**/*_clean.csv" \
                                         --output_file "../data/driams/preprocessing/merged_metadata.csv"

# # TODO change outputs to identical directory, and add other DRIAMS datasets (if desired)
# python3 process_data.py --input_file "../data/driams/downloads/DRIAMS-A/raw/**/*.txt" \
#                         --driams_csv "../data/driams/preprocessing/merged_metadata.csv" \
#                         --output_mzML_dir "../data/driams/raw/converted_to_mzml/driams-a" \
#                         --output_dir "../data/driams/preprocessing/baseline_corrected/driams"


# # Merge JSON Spectra
# python3 driams/merge_json_files.py --json_files "../data/driams/preprocessing/baseline_corrected/driams.json" \
#                                    --output_file "../data/driams/preprocessing/baseline_corrected.json"