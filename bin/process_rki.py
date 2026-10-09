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

from process_data import process_with_maldi_quant, convert_to_json
from ete3 import NCBITaxa

#### Code abstracted and modified from https://github.com/gdewael/maldi-nn/blob/afccd3c2fd4ff1e71701f397db6858077831f8ee/maldi_nn/spectrum.py#L78 #####
def tof2mass(ML1, ML2, ML3, TOF):
    A = ML3
    B = np.sqrt(1e12 / ML1)
    C = ML2 - TOF

    if A == 0:
        return (C * C) / (B * B)
    else:
        return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

def load_bruker_spectra(acqu_file, fid_file):
    """Read a spectrum from Bruker's format

    Parameters
    ----------
    acqu_file : str
        "acqu" file bruker folder
    fid_file : str
        "fid" file in bruker folder

    Returns
    -------
    SpectrumObject
    """
    with open(acqu_file, "rb") as f:
        lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]
    for l in lines:
        if l.startswith("##$TD"):
            TD = int(l.split("= ")[1])
        if l.startswith("##$DELAY"):
            DELAY = int(l.split("= ")[1])
        if l.startswith("##$DW"):
            DW = float(l.split("= ")[1])
        if l.startswith("##$ML1"):
            ML1 = float(l.split("= ")[1])
        if l.startswith("##$ML2"):
            ML2 = float(l.split("= ")[1])
        if l.startswith("##$ML3"):
            ML3 = float(l.split("= ")[1])
        if l.startswith("##$BYTORDA"):
            BYTORDA = int(l.split("= ")[1])
        if l.startswith("##$NTBCal"):
            NTBCal = l.split("= ")[1]

    intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

    if len(intensity) < TD:
        TD = len(intensity)
    TOF = DELAY + np.arange(TD) * DW

    mass = tof2mass(ML1, ML2, ML3, TOF)

    intensity[intensity < 0] = 0

    return np.array(list(zip(mass, intensity)))
####        ####

def get_most_specific_node_parsable(taxonomy_info_parent, ncbi: NCBITaxa):
    all_taxonomy_info = taxonomy_info_parent.findall('.//node')
    # Reverse the list since they seem to be least to most specific
    all_taxonomy_info.reverse()
    most_specific_taxonomy_id = None

    for taxonomy_info in all_taxonomy_info:
        taxid = None
        try:
            name = taxonomy_info.attrib['name']
            taxid = ncbi.get_name_translator([name]).get(name)
        except Exception as e:
            logging.error(f"Error getting taxid for name {name}: {e}")
            continue
        if taxid is not None:
            most_specific_taxonomy_id = taxid[0]
            break

    return most_specific_taxonomy_id


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
    if taxonomy_info_parent is None:
        raise ValueError(f"No taxonomyTreeInfos found in {msp_path}")


    most_specific_taxonomy_id = get_most_specific_node_parsable(taxonomy_info_parent, ncbi)

    if most_specific_taxonomy_id is None:
        raise ValueError(f"Could not find taxonomy ID for: {msp_path}")

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

        print("Number of peaks:", len(peaks))

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

def parse_raw_directory(input_dir: Path) -> List[Dict]:
    """ Parse a directory of raw files and extract metadata and peaklists.

    Args:
        input_dir (Path): Path to the input directory.

    Returns:
        List[Dict]: List of spectra with metadata and peaklists.
    """
    input_dir = Path(input_dir)
    data_paths = list(input_dir.glob('*/*/*/*/'))
    logging.info(f"Found {len(data_paths)} strains in {input_dir}")

    outputs = []

    def _helper(data_path):
        o = []
        genus = data_path.parts[-4]
        species = data_path.parts[-3]
        subspecies = data_path.parts[-2]
        strain_name = data_path.parts[-1]

        spectra_dirs = list(data_path.glob('**/pdata/'))
        spectra_dirs = [p.parent for p in spectra_dirs]

        peak_lists = [load_bruker_spectra(s / 'acqu', s / 'fid') for s in spectra_dirs]

        for i, peaklist in enumerate(peak_lists):
            if len(peaklist) == 0:
                continue
            o.append({
                'genus': genus,
                'species': species,
                'subspecies': subspecies,
                'strain_name': strain_name,
                'peaklist': peaklist,
                'spectrum_id': f"{genus}_{species}_{subspecies}_{strain_name}_rep{i}".replace(' ', '_').replace('/', '_'),
            })
        return o

    outputs = Parallel(n_jobs=-1)(
        delayed(_helper)(data_path) for data_path in tqdm(data_paths, desc="Parsing raw directories")
    )
    return outputs
    
def convert_raw_directory_to_mzml(input_dir: Path, output_mzML_dir: Path):
    """ Convert a directory of raw files to mzML format.

    Args:
        input_dir (Path): Path to the input directory.
        output_mzML_dir (Path): Path to the output mzML directory.
    """
    parsed_spectra = parse_raw_directory(input_dir)

    # Flatten list of lists
    parsed_spectra = [item for sublist in parsed_spectra for item in sublist]
    parsed_spectra = [item for item in parsed_spectra if item is not None]

    def _write_entry(entry):
        output_mzML_path = output_mzML_dir / f"{entry['spectrum_id']}.mzML"
        write_dict_to_mzML(entry, output_mzML_path)

    Parallel(n_jobs=-1)(
        delayed(_write_entry)(entry) for entry in tqdm(parsed_spectra, desc="Writing mzML files")
    )

    # Drop peak info and return
    for entry in parsed_spectra:
        entry.pop('peaklist', None)
    return parsed_spectra


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

        if output_path.exists():
            logging.warning(f"Output file {output_path} already exists, overwritting.")

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

    if input_dir.name.endswith('.msp'):
        # Single MSP file
        msp_files = [input_dir]
        msp_files = list(input_dir.glob('**/*'))
        logging.info(f"Found {len(msp_files)} MSP files in {input_dir}")

        metadata = Parallel(n_jobs=n_jobs)(
            delayed(convert_msp_to_mzml)(msp_path, output_mzML_dir) for msp_path in tqdm(msp_files, desc="Converting MSP to mzML")
        )
        metadata = [item for sublist in metadata if sublist is not None for item in sublist]
        metadata_df = pd.DataFrame(metadata)

    else:
        metadata = convert_raw_directory_to_mzml(input_dir, output_mzML_dir)
        metadata_df = pd.DataFrame(metadata)

        # Assign pseudo-accessions to subspecies
        metadata_df['accession'] = metadata_df['subspecies'].astype('category').cat.codes

    # Report number of failed files
    logging.warning("Number of failed files: {}".format(sum([1 for item in metadata if item is None])))

    # Rename spectrum_id to Strain name
    metadata_df = metadata_df.rename(columns={'spectrum_id': 'Strain name'})    # Now we have a strain_name and a Strain name that are different, 'Strain name' should really be called spectrum_id or something

    # Write metadata to csv
    metadata_df.to_csv(output_csv_path, index=False)
    logging.info(f"Wrote metadata for {len(metadata_df)} spectra to {output_csv_path}")

    # Now we have all mzML files, process them with MALDIQuant
    process_with_maldi_quant(output_mzML_dir, output_dir, n_jobs=n_jobs)

    convert_to_json(output_dir)

if __name__ == "__main__":
    main()