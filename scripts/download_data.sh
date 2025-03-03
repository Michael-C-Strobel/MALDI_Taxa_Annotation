#!/bin/bash

cd ..

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./conda_env

cd bin

mkdir -p ../data/idbac_db/raw

python3 download_db.py --json_output "../data/idbac_db/raw/db.json"  --csv_output "../data/idbac_db/raw/db.csv" 
