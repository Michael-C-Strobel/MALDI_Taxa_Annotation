import argparse
from pathlib import Path
import pandas as pd
from typing import List
from glob import glob
from ete3 import NCBITaxa


def enrich_with_ncbi_taxonomy(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Use ete3 to convert species names to taxonomy IDs, add them as the 'accession' column.
    Additionally, add 'genus' annotations to the metadata.
    Args:
        metadata (pd.DataFrame): DataFrame containing metadata with species names
    Returns:
        pd.DataFrame: DataFrame with enriched metadata including taxonomy IDs and genus
    """
    ncbi = NCBITaxa()
    
    # Get unique species names
    unique_species = metadata['species'].unique()
    
    # Create a mapping of species to taxonomy IDs
    species_to_taxid = ncbi.get_name_translator(unique_species)
    for key, value in species_to_taxid.items():
        # Get the first taxonomy ID for each species
        species_to_taxid[key] = value[0]

    # Canonicalize species names
    all_taxids = species_to_taxid.values()
    tax_id_to_species = {}
    for taxid in all_taxids:
        # Get the species name for each taxonomy ID
        tax_id_to_species[taxid] = ncbi.get_taxid_translator([taxid])[taxid]


    # Slightly overkill, we could just take the first word, but this is robust
    species_to_genus = {}
    for species, taxid in species_to_taxid.items():
        # Get the lineage for each species
        lineage = ncbi.get_lineage(taxid)
        value = ncbi.get_taxid_translator(lineage)
        rank = ncbi.get_rank(lineage)
        # Figure out which key has a value 'genus'
        for k, v in rank.items():
            if v == 'genus':
                species_to_genus[species] = value[k]
                break

    # Map the taxonomy IDs to the metadata DataFrame
    metadata['accession'] = metadata['species'].map(species_to_taxid)

    # Canonicalize the species names
    metadata['species'] = metadata['accession'].map(tax_id_to_species)
    
    # Add genus annotations
    metadata['genus'] = metadata['species'].map(species_to_genus)

    return metadata

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
            # Add the original file path as a column
            metadata['file_path'] = input_path.as_posix()
        else:
            _new_data = pd.read_csv(input_path)
            _new_data['file_path'] = input_path.as_posix()
            metadata = pd.concat([metadata, _new_data])

    # Create required columns
    metadata['Strain name'] = metadata['code']
    metadata['database_id'] = metadata['code']

    metadata = metadata.loc[metadata['Strain name'].str.lower() != 'no peaks found']

    metadata = metadata.loc[~ metadata['species'].str.lower().str.contains('mix!')]

    metadata = enrich_with_ncbi_taxonomy(metadata)

    # Some sanity checks
    # assert "2c6d74a0-068f-4b62-876b-c09e6ec283a3_MALDI1" in metadata['code'].values, "Expected strain '2c6d74a0-068f-4b62-876b-c09e6ec283a3_MALDI1' not found in metadata."

    metadata.to_csv(output_file, index=False)

def main():
    parser = argparse.ArgumentParser(description='Generate metadata file based on one or more DRIAMS files.')
    parser.add_argument('--input_csvs', type=str, help='Input directory containing DRIAMS files', required=True)
    parser.add_argument('--output_file', type=str, help='Output file path', required=True)
    args = parser.parse_args()

    # Semi-colon seperated list of globs
    input_csvs = args.input_csvs.split(';')
    input_csv_paths = [Path(input_csv) for input_csv in input_csvs]
    # Glob each one and flatten
    input_csv_paths = [Path(path) for input_csv_path in input_csv_paths for path in glob(str(input_csv_path))]

    print(input_csv_paths)

    for input_csv_path in input_csv_paths:
        if not input_csv_path.exists():
            raise ValueError('Input CSV file does not exist')

    output_file = Path(args.output_file)

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    generate_metadata_file(input_csv_paths, output_file)

if __name__ == "__main__":
    main()