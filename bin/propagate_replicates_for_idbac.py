import argparse
from pathlib import Path
import pandas as pd
import logging

def main():
    parser = argparse.ArgumentParser(description='Propagate metadata for IDBac to replicates.')
    parser.add_argument('--input_file', type=Path, required=True, help='Path to the input CSV file with taxonomy.')
    parser.add_argument('--spectrum_dir', type=Path, required=True, help='Directory containing baseline corrected spectra.')
    parser.add_argument('--output_file', type=Path, required=True, help='Path to the output CSV file with taxonomy and replicates.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for var, path in vars(args).items():
        logging.info(f"{var}: {path}")

    input_file = Path(args.input_file)
    spectrum_dir = Path(args.spectrum_dir)
    output_file = Path(args.output_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if not spectrum_dir.exists() or not spectrum_dir.is_dir():
        raise NotADirectoryError(f"Spectrum directory {spectrum_dir} does not exist or is not a directory.")
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file)

    # List all files in spectrum_dir
    spectrum_paths = [f for f in spectrum_dir.glob("*.mzML")]

    rep_df = pd.DataFrame(spectrum_paths, columns=['filepaths'])
    rep_df['Strain name']   = rep_df['filepaths'].apply(lambda x: x.stem)
    rep_df['db_name']       = rep_df['Strain name'].apply(lambda x: x.split('_rep_')[0])

    rep_df = rep_df.sort_values(by=['db_name', 'Strain name']).reset_index(drop=True)

    # Repeat metadata for each replicate
    new_metadata = []
    for _, row in rep_df.iterrows():
        db_name = row['db_name']
        matched_rows = df[df['Strain name'] == db_name]
        if matched_rows.empty:
            logging.warning(f"No metadata found for db_name '{db_name}'. Skipping (likely no genbank accession).")
            continue
        elif len(matched_rows) > 1:
            logging.warning(f"Multiple metadata entries found for db_name '{db_name}'. Using the first match.")
        
        metadata = matched_rows.iloc[0].to_dict()
        metadata['filepaths'] = row['filepaths']
        metadata['Strain name'] = row['Strain name']
        new_metadata.append(metadata)

    logging.info(f"Propagated metadata for {len(new_metadata)} replicates.")

    new_df = pd.DataFrame(new_metadata)
    new_df.to_csv(output_file, index=False)
    logging.info(f"Output written to {output_file}")

if __name__ == "__main__":
    main()