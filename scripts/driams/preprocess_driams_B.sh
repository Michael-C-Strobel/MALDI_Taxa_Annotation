#!/bin/bash

# Exit on any failure
set -e

cd ../..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd bin

# DRIAMS-B
mkdir -p ../data/driams-B/ preprocessing

echo "Generating metdata file for DRIAMS-B"
python3 driams/generate_metadata_file.py --input_csv "../data/driams/downloads/DRIAMS-B/id/**/*_clean.csv" \
                                         --output_file "../data/driams-B/preprocessing/merged_metadata.csv"

echo "Processing DRIAMS-B raw data"
python3 process_data.py --input_file "../data/driams/downloads/DRIAMS-B/raw/**/*.txt" \
                        --driams_csv "../data/driams-B/preprocessing/merged_metadata.csv" \
                        --output_mzML_dir "../data/driams-B/raw/converted_to_mzml/driams-b" \
                        --output_dir "../data/driams-B/preprocessing/baseline_corrected/driams"


echo "Merging JSON spectra for DRIAMS-B"
python3 driams/merge_json_files.py --json_files "../data/driams-B/preprocessing/baseline_corrected/driams.json" \
                                   --output_file "../data/driams-B/preprocessing/baseline_corrected.json"