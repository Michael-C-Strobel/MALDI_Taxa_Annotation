import argparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from tqdm import tqdm
import hashlib
import time
import json
from decimal import Decimal
import re
import ijson
import csv
import polars as pl
import logging

from utils import init_logging, convert_to_serializable

url = 'https://idbac.org/api/spectrum?database_id=ALL'
checksum_url = 'https://idbac.org/api/db-checksum'

def calculate_checksum(file_path, algorithm='sha256', chunk_size=65536):
    """
    Calculate the checksum of a file.

    Args:
        file_path (str): The path to the file.
        algorithm (str): The hashing algorithm to use. Defaults to 'sha256'.
        chunk_size (int): The size of the chunks to read from the file. Defaults to 65536.
    
    Returns:
        str: The checksum of the file
    """
    hash_function = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as file:
        while chunk := file.read(chunk_size):
            hash_function.update(chunk)
    
    return hash_function.hexdigest()


def get_remote_checksum(checksum_url: str):
    # Retry setup for resilience
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    # Attempt to get the checksum with retries
    for attempt in range(retries.total):
        try:
            response = session.get(checksum_url, timeout=60)
            response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
            return response.json()['checksum']
        except requests.exceptions.RequestException as e:
            wait_time = retries.backoff_factor * (2 ** attempt)  # Exponential backoff
            print(f"Checksum retrieval failed: {e}. Retrying {attempt + 1}/{retries.total} in {wait_time:.1f}s...")
            time.sleep(wait_time)  # Wait before retrying

    raise Exception(f"Failed to retrieve checksum after {retries.total} attempts.")

def download(output_path: str = 'all_spectra.json'):
    # First, let's see if our database is up to date
    local_checksum = None
    if Path(output_path).exists():
        local_checksum = calculate_checksum(output_path)
        print(f"Local checksum: {local_checksum}")
    
    remote_checksum = get_remote_checksum(checksum_url)

    if local_checksum == remote_checksum:
        print("Database is up to date")
        return False

    # Retry setup for resilience
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    # Stream the download
    with session.get(url, timeout=60*5, stream=True) as response:
        response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))  # In bytes

        # Write JSON to file in chunks, with a progress bar
        with open(output_path, 'wb') as f:
            # Set up progress bar
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Database:") as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks
                    if chunk:  # Filter out keep-alive new chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))  # Update the progress bar by the size of the chunk

    print(f"Downloaded successfully to {output_path}")

    return True

# TODO: Use this instead of pandas to clean the file, (waiting for DB refresh)
# def convert_to_csv(json_path:str, csv_path:str):
#     with open(csv_path, mode='w', newline='') as csvfile:
#         writer = csv.writer(csvfile)

#         with open(json_path, mode='r') as jsonfile:
#             # Use ijson to iterate over each object in the array
#             for record in ijson.items(jsonfile, 'item'):                # TODO: This doesn't work because of nan values. Update export to convert to string
#                 # Skip the unwanted key
#                 record.pop('spectrum', None)  # Skip the spectrum key
#                 # Write the remaining fields to CSV
#                 writer.writerow(record.values())

def sanitize_file(input_path:str, output_path:str):
    # Count total lines to give tqdm an accurate progress bar
    with open(input_path, 'r') as infile:
        total_lines = sum(1 for _ in infile)

    # Iterate over lines looking for NaN, make a string
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in tqdm(infile, total=total_lines, desc="Sanitizing file"):
            outfile.write(line.replace('NaN', '"NaN"'))

def split_spectrum_metadata(input_path: str, output_spec_path: str, output_meta_path: str):
    # Open input file and two output files
    with open(input_path, 'r') as input_file:
        # Use ijson to parse the input file one item at a time
        parser = ijson.items(input_file, 'item')
        
        # Open output files and write the opening brackets for JSON arrays
        with open(output_spec_path, 'w') as spec_file, open(output_meta_path, 'w') as meta_file:
            spec_file.write('[\n')  # Start of JSON array
            meta_file.write('[\n')  # Start of JSON array
            
            first_spec = True  # To handle comma placement
            first_meta = True  # To handle comma placement
            
            for obj in parser:
                obj = convert_to_serializable(obj)

                # Assume "Strain name" is the key for each object
                
                # Extract spectrum
                if "spectrum" in obj:
                    spectrum = obj.pop("spectrum")
                    # Write spectrum to output_spec_path
                    if not first_spec:
                        spec_file.write(',\n')  # Add comma before subsequent entries
                    spec_file.write(json.dumps({"Strain name": obj["Strain name"], "spectrum": spectrum}))
                    first_spec = False  # Mark that we have written the first entry
                
                # Write remaining metadata (without spectrum) to output_meta_path
                if not first_meta:
                    meta_file.write(',\n')  # Add comma before subsequent entries
                meta_file.write(json.dumps(obj))
                first_meta = False  # Mark that we have written the first entry

            spec_file.write('\n]')  # End of JSON array
            meta_file.write('\n]')  # End of JSON array

def convert_to_csv(json_path:str, csv_path:str):
    # Load the JSON file into a DataFrame
    df = pl.read_json(json_path)
    # Drop the 'spectrum' column
    if 'spectrum' in df.columns:
        df = df.drop('spectrum')
    # Write the DataFrame to a CSV file
    df.write_csv(csv_path)

def main():
    parser = argparse.ArgumentParser(description='Download all spectra from IDBac')
    parser.add_argument('-j', '--json_output', type=str, default='all_spectra.json', help='Output file path')
    parser.add_argument('-c', '--csv_output', type=str, default='all_spectra.csv', help='Convert json to csv')
    args = parser.parse_args()

    init_logging()

    db_json_path = Path(args.json_output)
    db_summary_path = db_json_path.with_suffix('.csv')

    if not db_json_path.suffix == '.json':
        raise ValueError('Output file must be a json file')
    if not db_json_path.parent.exists():
        db_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_summary_path.suffix == '.csv':
        raise ValueError('Output file must be a csv file')
    if not db_summary_path.parent.exists():
        db_summary_path.parent.mkdir(parents=True, exist_ok=True)
        
    # Download the database
    # fresh_download = download(db_json_path)
    # if not fresh_download:    # Data is not new, nothing to do!
    #    return 

    # Sanitize the file (Only temporarily needed)
    logging.info('Sanitizing file...')
    sanitized_path = db_json_path.with_name('sanitized.json')
    # sanitize_file(db_json_path, sanitized_path)

    # Split into spectra and metadata
    logging.info('Splitting file...')
    output_spectra_path = db_json_path.with_name('spectra.json')
    output_metadata_path = db_json_path.with_name('metadata.json')
    split_spectrum_metadata(sanitized_path, output_spectra_path, output_metadata_path)

    # Convert metadata to CSV
    logging.info('Converting to CSV...')
    convert_to_csv(output_metadata_path, db_summary_path)



if __name__ == '__main__':
    main()