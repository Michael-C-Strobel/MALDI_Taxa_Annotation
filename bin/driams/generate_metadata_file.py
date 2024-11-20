import argparse
from pathlib import Path
import pandas as pd
from typing import List

def generate_metadata_file(input_paths:List[Path], output_file:Path)->None:
    """
    Generate a metadata file based on one or more DRIAMS files.
    
    Args:
        input_dir (Path): Input directory containing DRIAMS files
        output_file (Path): Output file path

    Returns:
        None
    """
    # Read and merge all of the csvs
    metadata = None
    for input_path in input_paths:
        if metadata is None:
            metadata = pd.read_csv(input_path)
        else:
            metadata = pd.concat([metadata, pd.read_csv(input_path)])

    # Create required columns
    metadata['accession'] = metadata['species'] # species name
    metadata['Strain name'] = metadata['code']  # strain name

    metadata = metadata.loc[metadata['Strain name'].str.lower() != 'no peaks found']

    metadata.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Generate metadata file based on one or more DRIAMS files.')
    parser.add_argument('--input_csvs', type=str, help='Input directory containing DRIAMS files', required=True)
    parser.add_argument('--output_file', type=str, help='Output file path', required=True)
    args = parser.parse_args()

    input_csvs = args.input_csvs.split(';')
    input_csv_paths = [Path(input_csv) for input_csv in input_csvs]

    for input_csv_path in input_csv_paths:
        if not input_csv_path.exists():
            raise ValueError('Input CSV file does not exist')

    output_file = Path(args.output_file)

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    generate_metadata_file(input_csvs, output_file)

if __name__ == "__main__":
    main()