import argparse
from pathlib import Path
from Bio import Entrez, SeqIO
import polars as pl
import os
from utils import init_logging
import logging
from tqdm import tqdm

# Function to fetch FASTA files for given taxids
def fetch_fasta_by_taxids(taxids, output_dir):  # Defunct
    for taxid in tqdm(taxids):
        logging.debug("Fetching sequences for TaxID: %s", taxid)
        
        # Step 1: Search for sequences by taxid in the nucleotide database
        search_handle = Entrez.esearch(db="nucleotide", term="txid{}[Organism]".format(taxid), retmax=1000)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        sequence_ids = search_results["IdList"]
        logging.debug("Found %d sequences for TaxID %s", len(sequence_ids), taxid)

        # Step 2: Fetch the corresponding FASTA sequences
        if sequence_ids:
            logging.debug("Fetching FASTA sequences for TaxID %s", taxid)
            fetch_handle = Entrez.efetch(db="nucleotide", id=sequence_ids, rettype="fasta", retmode="text")
            fasta_data = fetch_handle.read()
            fetch_handle.close()

            # Step 3: Save FASTA data to file
            output_file = os.path.join(output_dir, "{}.fasta".format(taxid))
            logging.debug("Saving FASTA sequences for TaxID %s to %s", taxid, output_file)
            with open(output_file, "w") as f:
                f.write(fasta_data)
            logging.debug("Saved FASTA sequences for TaxID %s to %s", taxid, output_file)
        else:
            logging.debug("No sequences found for TaxID %s", taxid)

# Function to fetch FASTA files for given GenBank accessions
def fetch_fasta_by_accessions(accessions, output_dir="fasta_files"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for accession in tqdm(accessions):
        logging.debug("Fetching sequence for accession: %s", accession)
        
        # Step 1: Fetch the FASTA sequence using the accession number
        try:
            fetch_handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
            fasta_data = fetch_handle.read()
            fetch_handle.close()

            # Step 2: Save FASTA data to file
            output_file = os.path.join(output_dir, f"{accession}.fasta")
            with open(output_file, "w") as f:
                f.write(fasta_data)
            logging.debug("Saved FASTA sequence for accession %s to %s", accession, output_file)

            # Step 3: Check if the file was empty, if so, delete it
            with open(output_file, "r") as f:
                if not f.read().strip():
                    os.remove(output_file)
                    logging.debug("Deleted empty FASTA file for accession %s", accession)
        
        except Exception as e:
            logging.error("Error fetching data for accession %s: %s", accession, e)

def create_fasta_from_df(df, output_dir, output_csv):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    assert '16S Sequence' in df.columns, "16S Sequence column not found in DataFrame"

    missing_accessions = df.filter(df['Genbank accession'] == 'NaN')
    missing_accessions = df.filter(df['16S Sequence']      != 'NaN')

    for row in tqdm(missing_accessions.iter_rows(named=True)):
        strain_name = row['Strain name'].replace(' ', '_')

        accession_standin = f"strain_{strain_name}"

        sequence = row['16S Sequence']
       
        output_file = os.path.join(output_dir, f"{accession_standin}.fasta")
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(f">{accession_standin} : user-uploaded 16S \n{sequence}\n")
        logging.debug("Saved FASTA sequence for accession %s to %s", accession_standin, output_file)

    # Replace missing accessions with the new accession standins
    df = df.with_columns(
        pl.when((pl.col('Genbank accession') == 'NaN') & (pl.col('16S Sequence') != 'Nan'))
        .then(
            pl.col('Strain name').map_elements(lambda x: f"strain_{x.replace(' ', '_')}" if x is not None else x)
        )
        .otherwise(pl.col('Genbank accession'))
        .alias('Genbank accession')
    )
    output_csv = Path(output_csv)
    if not output_csv.parent.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_csv)

def main():
    parser = argparse.ArgumentParser(description='Collect fasta files')
    parser.add_argument('--input_csv', type=str, help='Input CSV file path', required=True)
    parser.add_argument('--output_csv', type=str, help='Output CSV file path', required=False)
    parser.add_argument('--output_dir', type=str, help='Output directory', required=True)
    parser.add_argument('--email', type=str, help='Email address for Entrez', required=False)
    args = parser.parse_args()

    if args.email:
        Entrez.email = args.email

    init_logging()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_csv = Path(args.output_csv)

    if not input_csv.suffix == '.csv':
        raise ValueError('Input file must be a CSV file')
    if not input_csv.exists():
        raise FileNotFoundError('Input file does not exist')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    df = pl.read_csv(input_csv, infer_schema_length=None)
    
    genbank_accessions = df.filter(df['Genbank accession'] != 'NaN')['Genbank accession'].unique()
    fetch_fasta_by_accessions(list(genbank_accessions), output_dir)


    create_fasta_from_df(df, output_dir, output_csv)

if __name__ == "__main__":
    main()