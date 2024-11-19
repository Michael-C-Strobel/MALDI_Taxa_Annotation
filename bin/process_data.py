import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import ijson
from glob import glob
from psims.mzml.writer import MzMLWriter
from pyteomics import mzml
from utils import init_logging, convert_to_serializable
from tqdm import tqdm
import pandas as pd
import warnings
from psims.utils import StateTransitionWarning
from psims.document import ReferentialIntegrityWarning

def write_mzML_files_from_json(json_input:Path, output_mzML_dir:Path)->None:
    with open(json_input, 'r') as input_file:
        parser = ijson.items(input_file, 'item')
        for obj in tqdm(parser):
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


def write_mzML_files_from_txt(txt_glob:Path, driams_csv: Path, output_mzML_dir:Path)->None:
    # Glob back from path
    txt_glob = str(txt_glob)
    all_txt_files = [Path(x) for x in glob(txt_glob)]
    
    metadata_csv = pd.read_csv(driams_csv)
    species_dict = dict(zip(metadata_csv['code'], metadata_csv['species']))

    for txt_file in tqdm(all_txt_files):
        data = parse_driams_txt_to_dict(txt_file, species_dict)
        if data is None:
            continue

        sample_hash = data['sample hash']
        species     = data['species']
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=StateTransitionWarning)
            warnings.filterwarnings("ignore", category=ReferentialIntegrityWarning)
            warnings.filterwarnings("ignore", category=UserWarning, message="No Data Processing method found. mzML file may not be fully standard-compliant")

            with MzMLWriter(open(output_mzML_dir / f'{sample_hash}.mzML', 'wb'), close=True) as writer:
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


def process_with_maldi_quant(input_path: Path, output_path: Path):
    # Performs peak picking, baseline correction, binning, and merging
    all_spectrum_paths = list(input_path.glob("*.mzML"))

    for spectrum_path in tqdm(all_spectrum_paths):
        _output_path = output_path / f"{spectrum_path.stem}.mzML"
        subprocess.run(['Rscript', 'preprocess_data.R', str(spectrum_path), str(_output_path)],
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL)

def convert_to_json(path: Path):
    # Read each mzML file, store to a single JSON file
    all_spectrum_paths = list(path.glob("*.mzML"))
    
    output_path = path.parent / f"{path.stem}.json"
    spectra_list = []  # List to hold all spectrum data
    
    for spectrum_path in tqdm(all_spectrum_paths):
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
    args = parser.parse_args()

    init_logging()

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

    if input_file.suffix == '.json':
        write_mzML_files_from_json(input_file, output_mzML_dir)
    elif input_file.suffix == '.txt' and args.driams_csv:
        if "*" not in str(input_file):
            raise ValueError("Expected a glob pattern in the input file")
        write_mzML_files_from_txt(input_file, Path(args.driams_csv), output_mzML_dir)
    else:
        raise ValueError('Input file must be a JSON or DRIAMS txt file')

    # Process the mzML files with MALDIquant
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    process_with_maldi_quant(output_mzML_dir, output_path)

    convert_to_json(output_path)

if __name__ == "__main__":
    main()