import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import ijson
from psims.mzml.writer import MzMLWriter
from pyteomics import mzml
from utils import init_logging, convert_to_serializable
from tqdm import tqdm

def write_mzML_files(json_input, output_mzML_dir):
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
    parser.add_argument('--json_input', type=str, help='Input file path', required=True)
    parser.add_argument('--output_mzML_dir', type=str, default='mzML_dir', help='Output directory for mzML files')
    parser.add_argument('--output_dir', type=str, help='Output file path', required=True)
    args = parser.parse_args()

    init_logging()

    # Write mzML files for each object.
    # Each scan in spectrum becomes a scan in the output
    output_mzML_dir = Path(args.output_mzML_dir)
    json_input = Path(args.json_input)
    if not json_input.suffix == '.json':
        raise ValueError('Input file must be a JSON file')
    if not json_input.exists():
        raise FileNotFoundError('Input file does not exist')
    if not output_mzML_dir.exists():
        output_mzML_dir.mkdir(parents=True, exist_ok=True)

    # write_mzML_files(json_input, output_mzML_dir)

    # Process the mzML files with MALDIquant
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    process_with_maldi_quant(output_mzML_dir, output_path)

    convert_to_json(output_path)

if __name__ == "__main__":
    main()