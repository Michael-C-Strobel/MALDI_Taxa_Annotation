import argparse
from pathlib import Path
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from custom_transforms import BinSpectrum
from tqdm import tqdm

def compute_similarity_dict(metadata:pd.DataFrame,
                            processed_spectra_dir:Path,
                           )->Dict[str, Tuple[List[str], torch.Tensor]]:
    """
    Creates the cosine similarity dict for a given metadata dataframe group.
    """
    # Load all spectra into memory
    all_names = metadata["Strain name"].unique()

    all_spectra_paths = [
        processed_spectra_dir / f"{name}.pt" for name in all_names
    ]

    binner = BinSpectrum(10, 3_000, 20_000)

    all_spectra = [
        binner(torch.load(path, weights_only=False)) for path in all_spectra_paths
    ]

    all_spectra = torch.stack(all_spectra, dim=0)
    all_spectra = all_spectra.squeeze(1)

    # Compute all pairs cosine similarity
    pairwise_matrix = torch.tensor(cosine_similarity(all_spectra))

    # Convert to distance
    pairwise_matrix = 1 - pairwise_matrix

    # Sigmoid to smooth out higher values and devalue lower values
    pairwise_matrix = torch.sigmoid((10*pairwise_matrix)) # At 10, everything with dissimilarity > 0.5 is ~ 1 (equally weighted in probability space)

    # Clamp
    pairwise_matrix = torch.clamp(pairwise_matrix, min=0.0, max=1.0)

    # Set diagonal to 0
    pairwise_matrix.fill_diagonal_(0.0)

    # Normalize by row
    row_sums = pairwise_matrix.sum(axis=1, keepdim=True)
    pairwise_matrix = pairwise_matrix / row_sums

    # Save as Dict[str, Tuple[List[str], torch.Tensor]]
    similarity_dict = {}
    for i, name in enumerate(all_names):
        similarity_dict[name] = (
            all_names.tolist(),
            pairwise_matrix[i]
        )

    return similarity_dict

def generate_similarity_dicts(metadata:pd.DataFrame,
                              processed_spectra_dir:Path,
                              output_path:Path,
                              target:str):
    """
    Load all spectra into memory
    """
    if target == "genera":
        target_col = "genus"
    elif target == "species":
        target_col = "species"
    else:
        raise ValueError(f"Invalid target: {target}. Must be 'genera' or 'species'.")

    metadata = metadata.groupby(target_col)

    all_dicts = {}

    # Call a function to do what we implemented below
    for group in tqdm(metadata.groups):
        # Caclulate pairwise sims for that group
        group_df = metadata.get_group(group)
        similarity_dict = compute_similarity_dict(
            group_df,
            processed_spectra_dir
        )
        all_dicts.update(similarity_dict)

    # Save the similarity dict to a file
    torch.save(all_dicts, output_path)

def main():
    parser = argparse.ArgumentParser(description="Compute within-class cosine similarity.")
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--processed_spectra_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        '--target',
        choices=['genera', 'species'],
        required=True,
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    processed_spectra_dir = Path(args.processed_spectra_dir)
    output_path = Path(args.output_path)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if not processed_spectra_dir.exists():
        raise FileNotFoundError(f"Processed spectra directory not found: {processed_spectra_dir}")
    
    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory not found: {output_path.parent}")
    
    metadata = pd.read_csv(metadata_path)

    generate_similarity_dicts(
        metadata,
        processed_spectra_dir,
        output_path,
        args.target
    )

if __name__ == "__main__":
    main()
