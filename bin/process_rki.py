import argparse
from pathlib import Path
from psims.mzml.writer import MzMLWriter
import lxml
import logging
from utils import init_logging
from tqdm import tqdm
import pandas as pd
import warnings
from psims.utils import StateTransitionWarning
from psims.document import ReferentialIntegrityWarning
from joblib import Parallel, delayed
import numpy as np
from typing import List, Dict

from process_data import process_with_maldi_quant
from ete3 import NCBITaxa


def parse_msp_file(msp_path: Path) -> List[Dict]:
    """ Parse a single MSP file and extract metadata and peaklists
    Args:
        msp_path (Path): Path to the input MSP file
    Returns:
        List[Dict]: List of spectra with metadata and peaklists
    """
    ncbi = NCBITaxa()

    tree = lxml.etree.parse(str(msp_path))
    root = tree.getroot()

    main_spectrum = root.find('.//mainSpectrum')

    taxonomy_info_parent = root.find('.//taxonomyTreeInfos')
    most_specific_taxonomy_info = taxonomy_info_parent.findall('.//node')[-1]
    logging.debug('Most specific taxonomy info:', most_specific_taxonomy_info)
    most_specific_taxonomy_name = most_specific_taxonomy_info.attrib['name']
    most_specific_taxonomy_id = ncbi.get_name_translator([most_specific_taxonomy_name]).get(most_specific_taxonomy_name)
    if most_specific_taxonomy_id is None:
        raise ValueError(f"Could not find taxonomy ID for name: {most_specific_taxonomy_name}")
    else:
        most_specific_taxonomy_id = most_specific_taxonomy_id[0]
    logging.debug(f"Most specific taxonomy name: {most_specific_taxonomy_name}")
    logging.debug(f"Most specific taxonomy ID: {most_specific_taxonomy_id}")

    lineage = ncbi.get_lineage(most_specific_taxonomy_id)
    names = ncbi.get_taxid_translator(lineage)
    ranks = ncbi.get_rank(lineage)
    lineage_info= {
            ranks[taxid]: {'name': names[taxid], 'taxid': taxid}
            for taxid in lineage
        }
    logging.debug('Lineage info:', lineage_info)


    raw_spectra_metadata_parent = root.find('.//spectraMetadata')
    raw_spectra_metadata = raw_spectra_metadata_parent.findall('.//spectrum')
    logging.debug(f"Found {len(raw_spectra_metadata)} spectraMetadata elements")

    spectra_peak_lists = root.find('.//spectraPeaklists').findall('.//peaklist')
    logging.debug(f"Found {len(spectra_peak_lists)} spectraPeaklists elements")

    assert len(raw_spectra_metadata) == len(spectra_peak_lists), "Mismatch between number of spectraMetadata and spectraPeaklists"

    output_data = []

    for idx, mdata in enumerate(raw_spectra_metadata):
        genus = lineage_info.get('genus', None)
        genus_name = genus.get('name', None) if genus else None
        genus_taxid = genus.get('taxid', None) if genus else None
        species = lineage_info.get('species', None)
        species_name = species.get('name', None) if species else None
        species_taxid = species.get('taxid', None) if species else None

        peaklist = spectra_peak_lists[idx]

        # Get att <peak mass="3027.3" intensity="0.24" profile="1.0" sigma="10.0"/>
        peaks = []
        for peak in peaklist.findall('.//peak'):
            mass = float(peak.attrib['mass'])
            intensity = float(peak.attrib['intensity'])
            peaks.append((mass, intensity))

        peaks = np.array(peaks)
        # Sort by mass
        peaks = peaks[np.argsort(peaks[:, 0])]

        output_data.append(
            {
                'spectrum_id': mdata.attrib['uuid'],
                'name': mdata.attrib.get('name', None),
                'genus': genus_name,
                'species': species_name,
                'genus_taxid': genus_taxid,
                'species_taxid': species_taxid,
                'accession': lineage_info.get('species', {}).get('taxid', None),    # Used to balance species seen during trainnig (likely to be unused)
                'peaklist': peaks,
            }
        )
    return output_data

def write_dict_to_mzML(entry: Dict, output_path: Path):
    """ Write a single spectrum entry to an mzML file
    Args:
        entry (Dict): Spectrum entry with metadata and peaklist
        output_path (Path): Path to the output mzML file
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=StateTransitionWarning)
        warnings.filterwarnings("ignore", category=ReferentialIntegrityWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message="No Data Processing method found. mzML file may not be fully standard-compliant")

        with MzMLWriter(open(output_path, 'wb'), close=True) as writer:
            writer.file_description()
            writer.controlled_vocabularies()
            with writer.run(id='my_analysis'):
                with writer.spectrum_list(count=1):
                    writer.write_spectrum(
                        entry['peaklist'][:, 0],
                        entry['peaklist'][:, 1],
                        id=f"{entry['spectrum_id']}",
                        params=[
                            "MS1 Spectrum",
                            {"ms level": 1},
                            {"species": entry['species']},
                            {"genus": entry['genus']}
                        ]
                    )


def convert_msp_to_mzml(msp_path: Path, output_mzML_dir: Path):
    """
    Convert a single MSP file to mzML format using lxml
    
    Args:
        msp_path (Path): Path to the input MSP file
        output_mzML_dir (Path): Directory to save the output mzML file
    """
    try:
        parsed_spectra = parse_msp_file(msp_path)
    except ValueError as e:
        logging.warning(f"Error parsing {msp_path}: {e}, skipping.")
        return None

    if len(parsed_spectra) == 0:
        logging.warning(f"No spectra found in {msp_path}, skipping.")
        return

    for entry in parsed_spectra:
        # We're going to write each spectrum to its own mzML file 
        # Practically, this means we're going to write technical replicates (4) and biological replicates (3) to 12 files
        # This is consistent with how DRIAMS data is structured and used in literature
        
        output_mzML_path = output_mzML_dir / f"{entry['spectrum_id']}.mzML"
        write_dict_to_mzML(entry, output_mzML_path)

    # Drop peak info and return
    for entry in parsed_spectra:
        entry.pop('peaklist', None)
    return parsed_spectra

def main():
    parser = argparse.ArgumentParser(description='Process Spectra')
    parser.add_argument('--input_dir', type=str, help='Input unzipped rki directory', required=True)
    parser.add_argument('--output_csv_path', type=str, help='Path to output csv metadata file', required=True)
    parser.add_argument('--output_mzML_dir', type=str, help='Output directory for intermediate mzML files', required=True)
    parser.add_argument('--output_dir', type=str, help='Output directory for MALDIQuant processed mzML files', required=True)
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of jobs to run in parallel')
    parser.add_argument('--debug', action='store_true', help='Debug mode')

    args = parser.parse_args()

    # Init logging
    init_logging(debug=args.debug)

    # Dump all args
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    input_dir = Path(args.input_dir)
    output_csv_path = Path(args.output_csv_path)
    output_mzML_dir = Path(args.output_mzML_dir)
    output_dir = Path(args.output_dir)
    n_jobs = int(args.n_jobs)
    if args.debug:
        n_jobs = 1

    assert input_dir.exists(), f"Input directory {input_dir} does not exist"
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_mzML_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write all MSP files to mzML files

    msp_files = list(input_dir.glob('**/*'))
    logging.info(f"Found {len(msp_files)} MSP files in {input_dir}")

    metadata = Parallel(n_jobs=n_jobs)(
        delayed(convert_msp_to_mzml)(msp_path, output_mzML_dir) for msp_path in tqdm(msp_files, desc="Converting MSP to mzML")
    )

    # Write metadata to csv
    metadata = [item for sublist in metadata if sublist is not None for item in sublist]
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_csv_path, index=False)
    logging.info(f"Wrote metadata for {len(metadata_df)} spectra to {output_csv_path}")

    # Now we have all mzML files, process them with MALDIQuant
    process_with_maldi_quant(output_mzML_dir, output_dir, n_jobs=n_jobs)


if __name__ == "__main__":
    main()