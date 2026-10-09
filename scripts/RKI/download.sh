#!/bin/bash

set -e

cd ../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd ./data/


mkdir -p ./RKI/downloads
cd ./RKI/downloads

# wget https://zenodo.org/records/7702375/files/zenodo%20db%20230306.zip?download=1 -O RKI.4.0.zip

# Unzip it to an directory
# unzip RKI.4.0.zip -d ./RKI.4.0/

# Process (convert to mzML, extract metadata, run MALDIQuant)

cd ../../../bin/
python process_rki.py --input_dir ../data/RKI/downloads/RKI.4.0 \
    --output_dir ../data/RKI/processed/ \
    --output_csv_path ../data/RKI/processed/rki_metadata.csv \
    --output_mzML_dir ../data/RKI/processed/mzML/ \
    --output_dir ../data/RKI/processed/BaselineCorrected/ \
    --n_jobs 46
