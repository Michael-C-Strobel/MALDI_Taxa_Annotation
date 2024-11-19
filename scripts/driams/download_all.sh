#!/bin/bash

# Set exit on fail
set -e

cd ../../data/


mkdir -p ./driams/downloads
cd ./driams/downloads

# Download all of the dataset from DRIAMS

# # DRIAMS-A
# echo "Downloading DRIAMS-A dataset..."
# wget --progress=dot:giga https://datadryad.org/api/v2/files/1158722/download -O DRIAMS_A.tar.gz &

# # DRIAMS-B
# echo "Downloading DRIAMS-B dataset..."
# wget --progress=dot:giga https://datadryad.org/api/v2/files/1144870/download -O DRIAMS_B.tar.gz &

# # DRIAMS-C
# echo "Downloading DRIAMS-C dataset..."
# wget --progress=dot:giga https://datadryad.org/api/v2/files/1144871/download -O DRIAMS_C.tar.gz &

# # DRIAMS-D
# echo "Downloading DRIAMS-D dataset..."
# wget --progress=dot:giga https://datadryad.org/api/v2/files/1158723/download -O DRIAMS_D.tar.gz &

# # Wait for all downloads to complete
# wait

# # Extract all
# echo "Extracting DRIAMS datasets..."
# tar -xvf DRIAMS_A.tar.gz
tar -xvf DRIAMS_B.tar.gz
tar -xvf DRIAMS_C.tar.gz
tar -xvf DRIAMS_D.tar.gz