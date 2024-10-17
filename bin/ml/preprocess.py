import argparse
import pandas as pd
import ijson
import shutil
import numpy as np
import torch
from pathlib import Path

def read_blast_result(file):
    headers = ['query_id', 'subject_id', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    blast_result=pd.read_csv(file, sep='\t', header=None, names=headers)

    return blast_result[['query_id', 'subject_id', 'pident', 'length', 'evalue', 'bitscore']]

def convert_spectra_to_tensor(json_file: Path):
    f = open(json_file, 'rb')
    parser = ijson.items(f, 'item')
    for obj in parser:
        # Assume "Strain name" is the key for each object
        strain_name = obj["Strain name"]
        mz_array = np.array([float(x) for x in obj["m/z array"]])
        intensity_array = np.array([float(x) for x in obj["intensity array"]])

        # Get top 200 peaks by intensity
        indices = np.argsort(intensity_array)[::-1][:200]
        mz_array = mz_array[indices]
        intensity_array = intensity_array[indices]

        # TODO: Optionally add binning

        # Convert to tensor
        mz_tensor = torch.tensor(mz_array)
        intensity_tensor = torch.tensor(intensity_array)

        spectrum_as_tensor = torch.stack([mz_tensor, intensity_tensor], dim=1)

        yield strain_name, spectrum_as_tensor


def postprocess_files(input_dir: Path, output_dir: Path):

    # Should contain directory blast_results
    # should contain file baseline_corrected.json

    output_spectra_dir = output_dir / 'spectra'
    if not output_spectra_dir.exists():
        output_spectra_dir.mkdir(parents=True)

    for strain_name, spectrum_as_tensor in convert_spectra_to_tensor(input_dir / 'baseline_corrected.json'):
        torch.save(spectrum_as_tensor, output_spectra_dir / f'{strain_name}.pt')

    # Convert blast results to feather file
    all_blast_results = list((input_dir / 'blast_results').glob('*.txt'))
    blast_results = [read_blast_result(file) for file in all_blast_results]
    pident_matrix = pd.concat(blast_results)
    pident_matrix['query_genbank'] = pident_matrix['query_id'].str.split('.').str[0]
    pident_matrix['subject_genbank'] = pident_matrix['subject_id'].str.split('.').str[0]
    
    pident_matrix.to_feather(output_dir / 'similarities.feather')

def main():
    parser = argparse.ArgumentParser(description="Finalize processing of files")
    parser.add_argument('--input_dir', type=str, help="Directory containing the files to process")
    parser.add_argument('--metadata_file', type=str, help="File containing metadata")
    parser.add_argument('--output_dir', type=str, help="Directory to write the processed files to")
    args=parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    postprocess_files(input_dir, output_dir)

    shutil.copy(args.metadata_file, output_dir / 'metadata.json')

if __name__ == "__main__":
    main()