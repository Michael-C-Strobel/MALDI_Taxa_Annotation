import argparse
import sys
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import ijson
from glob import glob
from psims.mzml.writer import MzMLWriter
from pyteomics import mzml
import logging
from utils import init_logging, convert_to_serializable
from tqdm import tqdm
import pandas as pd
import warnings
from psims.utils import StateTransitionWarning
from psims.document import ReferentialIntegrityWarning
from joblib import Parallel, delayed
import numpy as np
from typing import List

GROUP_SIZE = 100

def write_mzML_files_from_json(json_input:Path, output_mzML_dir:Path)->None:
    with open(json_input, 'r') as input_file:
        parser = ijson.items(input_file, 'item')
        for obj in tqdm(parser, desc="Writing mzML files"):
            obj = convert_to_serializable(obj)
            
            # Assume "Strain name" is the key for each object
            strain_name = obj["Strain name"]
            all_scans = obj["spectrum"]
            with MzMLWriter(open(output_mzML_dir / f'{strain_name}.mzML', 'wb'), close=True) as writer:
                writer.controlled_vocabularies()
                with writer.run(id='my_analysis'):
                    with writer.spectrum_list(count=len(all_scans)):
                        for scan_idx, scan in enumerate(all_scans):
                            mz_array = [float(x[0]) for x in scan]
                            intensity_array = [float(x[1]) for x in scan]

                            writer.write_spectrum(
                                mz_array,
                                intensity_array,
                                id=f'scan={scan_idx}',
                                params=[
                                    "MS1 Spectrum",
                                    {"ms level": 1},
                                    {"total ion current": sum(intensity_array)}
                                ]
                            )

def parse_driams_txt_to_dict(txt_file:Path, species_dict:dict)->dict:
    data = pd.read_csv( txt_file,
                        sep=r"\s+", 
                        skiprows=3,
                        names=["m/z array", "intensity array"])
    
    sample_hash = txt_file.stem

    species = species_dict.get(sample_hash, None)
    if species is None:
        # print("No species found for", sample_hash, flush=True)
        return None

    species = str(species).strip()

    if species.lower().strip() == "no peaks":
        return None

    output_dict = {
        'sample hash': sample_hash,
        'species': species,
        'm/z array': data['m/z array'].values,
        'intensity array': data['intensity array'].values
    }
    
    return output_dict

def parallel_parse_driams_txt_to_dict(txt_files:List[Path], species_dict:dict, output_mzML_dir:Path)->dict:
    """Helper function to write to mzML. This function is used in parallel processing"""

    for txt_file in txt_files:
        data = parse_driams_txt_to_dict(txt_file, species_dict)
        if data is None:
            continue
        
        sample_hash = data['sample hash']   # Dangerous if sample hash is repeated
        species     = data['species']
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=StateTransitionWarning)
            warnings.filterwarnings("ignore", category=ReferentialIntegrityWarning)
            warnings.filterwarnings("ignore", category=UserWarning, message="No Data Processing method found. mzML file may not be fully standard-compliant")

            with MzMLWriter(open(output_mzML_dir / f'{sample_hash}.mzML', 'wb'), close=True) as writer:
                # print("Writing to", output_mzML_dir / f'{sample_hash}.mzML')
                writer.file_description()
                writer.controlled_vocabularies()
                with writer.run(id='my_analysis'):
                    with writer.spectrum_list(count=1):
                        writer.write_spectrum(
                            data['m/z array'],
                            data['intensity array'],
                            id=f'{sample_hash}',
                            params=[
                                "MS1 Spectrum",
                                {"ms level": 1},
                                {"species": species}
                            ]
                        )


def write_mzML_files_from_txt(txt_glob:Path, driams_csv: Path, output_mzML_dir:Path, n_jobs:int=-1)->None:
    # Glob back from path
    txt_glob = str(txt_glob)
    all_txt_files = [Path(x) for x in glob(txt_glob)]
    num_files = np.unique(all_txt_files).shape[0]
    print(f"Found {num_files} txt files")
    
    metadata_csv = pd.read_csv(driams_csv)
    species_dict = dict(zip(metadata_csv['code'], metadata_csv['species']))

    # Split into groups to reduce overhead
    print(f"Splitting into groups of size {GROUP_SIZE}")
    all_txt_files = np.array_split(all_txt_files, num_files // GROUP_SIZE + 1)

    # Parallel processing
    Parallel(n_jobs=n_jobs)(delayed(parallel_parse_driams_txt_to_dict)(txt_files, species_dict, output_mzML_dir) for txt_files in tqdm(all_txt_files, desc="Writing mzML files"))
    final_num_files = len(list(output_mzML_dir.glob("*.mzML")))
    print(f"Finished writing mzML files. Wrote {final_num_files} files. Lost {num_files - final_num_files} files.")

def process_with_maldi_quant(input_path: Path, output_path: Path, n_jobs:int=-1):
    debug = logging.getLogger().getEffectiveLevel() == logging.DEBUG
    subprocess_output_path = subprocess.DEVNULL
    subprocess_check = False

    if debug:
        n_jobs = 1
        subprocess_output_path = sys.stdout
        subprocess_check = True

    
    def _run_rscript(spectrum_paths: List[Path],):
        for spectrum_path in spectrum_paths:
            _output_path = output_path / f"{spectrum_path.stem}.mzML"
            subprocess.run(['Rscript', 'preprocess_data.R', str(spectrum_path), str(_output_path)], 
                           stdout = subprocess_output_path, 
                           stderr = subprocess_output_path,
                           check  = subprocess_check)

    # Performs peak picking, baseline correction, binning, and merging
    all_spectrum_paths = list(input_path.glob("*.mzML"))

    # Split into groups to reduce overhead
    all_spectrum_paths = np.array_split(all_spectrum_paths, len(all_spectrum_paths) // GROUP_SIZE + 1)

    # Run the R script in parallel
    logging.info("Running MaldiQuant in Parallel")
    logging.info("Outputting files to %s", output_path)
    Parallel(n_jobs=n_jobs)(delayed(_run_rscript)(spectrum_paths) for spectrum_paths in tqdm(all_spectrum_paths, desc="Running MALDIquant"))

def convert_to_json(path: Path):
    # Read each mzML file, store to a single JSON file
    all_spectrum_paths = list(path.glob("*.mzML"))
    
    output_path = path.parent / f"{path.stem}.json"
    spectra_list = []  # List to hold all spectrum data
    
    for spectrum_path in tqdm(all_spectrum_paths, desc="Converting to JSON"):
        mzml_file = mzml.read(str(spectrum_path))
        # Average the scans
        num_scans = 0
        spectrum_dict = defaultdict(float)
        for scan in mzml_file:
            num_scans += 1
            for mz, intensity in zip(scan["m/z array"], scan["intensity array"]):
                spectrum_dict[mz] += intensity

        output_mz_array = []
        output_intensity_array = []
        for mz in spectrum_dict:
            output_mz_array.append(mz)
            output_intensity_array.append(spectrum_dict[mz] / num_scans)

        spectrum = {}
        spectrum["Strain name"] = spectrum_path.stem
        spectrum["m/z array"] = output_mz_array
        spectrum["intensity array"] = output_intensity_array
        
        spectra_list.append(spectrum)  # Add the spectrum dictionary to the list

    # Write the complete list to a JSON file
    with open(output_path, 'w') as output_file:
        json.dump(spectra_list, output_file, indent=4)  # Use indent for pretty printing



def main():
    parser = argparse.ArgumentParser(description='Process Spectra')
    parser.add_argument('--input_file', type=str, help='Input file path', required=True)
    parser.add_argument('--driams_csv', type=str, help='DRIAMS CSV file path', required=False, default=None)
    parser.add_argument('--output_mzML_dir', type=str, default='mzML_dir', help='Output directory for mzML files')
    parser.add_argument('--output_dir', type=str, help='Output file path', required=True)
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    init_logging(args.debug)

    print("args.csv", args.driams_csv)
    print("args.input_file", args.input_file)
    print("input_file suffix", Path(args.input_file).suffix)

    # Write mzML files for each object.
    # Each scan in spectrum becomes a scan in the output
    output_mzML_dir = Path(args.output_mzML_dir)
    input_file = Path(args.input_file)
    if not input_file.suffix == '.json' and \
        not (args.driams_csv and input_file.suffix == '.txt'):
        raise ValueError('Input file must be a JSON file or a DRIAMS txt glob with associated csv')
    if not output_mzML_dir.exists():
        output_mzML_dir.mkdir(parents=True, exist_ok=True)
    

    # if input_file.suffix == '.json':
    #     logging.info("Writing mzML files from JSON to %s", output_mzML_dir)
    #     write_mzML_files_from_json(input_file, output_mzML_dir)

    # elif input_file.suffix == '.txt' and args.driams_csv:
    #     if "*" not in str(input_file):
    #         raise ValueError("Expected a glob pattern in the input file")
    #     write_mzML_files_from_txt(input_file, Path(args.driams_csv), output_mzML_dir, args.n_jobs)
    # else:
    #     raise ValueError('Input file must be a JSON or DRIAMS txt file')

    # Process the mzML files with MALDIquant
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    process_with_maldi_quant(output_mzML_dir, output_path, args.n_jobs)

    convert_to_json(output_path)

if __name__ == "__main__":
    main()