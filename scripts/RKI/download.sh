#!/bin/bash

set -e

cd ../../

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd ./data/


mkdir -p ./RKI/downloads
cd ./RKI/downloads

# wget https://zenodo.org/records/7702375/files/230306_v4_RKI_DB_BSL3.btmsp?download=1 -O RKI.4.0.btmsp

# Unzip it to an .msp file
# unzip RKI.4.0.btmsp -d ./RKI.4.0/

# Process (convert to mzML, extract metadata, run MALDIQuant)

cd ../../../bin/
python process_rki.py --input_dir ../data/RKI/downloads/RKI.4.0 \
    --output_dir ../data/RKI/processed/ \
    --output_csv_path ../data/RKI/processed/rki_metadata.csv \
    --output_mzML_dir ../data/RKI/processed/mzML/ \
    --output_dir ../data/RKI/processed/BaselineCorrected/ \
    --n_jobs 46 \
