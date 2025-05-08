import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def generate_k_fold_splits(metadata, k, split_style):
    """ Generate a list of 'accession' values that correspont to the k-fold splits. """
    # Create a dictionary to hold the train/val/test accessions for each fold
    fold_indices = {i: {'train': [], 'val': [], 'test':[]} for i in range(k)}

    # Group by the specified split style
    if split_style == "genera":
        groups = metadata.groupby("genus")
        split_col = "genus"
    elif split_style == "species":
        groups = metadata.groupby("species")
        split_col = "species"
    elif split_style == "species_even":
        groups = metadata.groupby("species_even")
        split_col = "species"
    else:
        raise ValueError(f"Unknown split style: {split_style}")
    
    # Automatically assign any group with fewer than max(5, k) samples to the train set
    small_groups = set()
    for name, group in groups:
        if len(group) <= max(5, k):
            small_groups.add(name)
            for i in range(k):
                fold_indices[i]['train'].extend(group['accession'].tolist())
    # Remove small groups from further processing
    org_num_groups = len(groups)
    groups = {name: group for name, group in groups if name not in small_groups}
    if len(small_groups) > 0:
        logging.info(f"Removed {len(small_groups)} small groups from {org_num_groups} total groups.")
        assert len(groups) > 0, "All groups are small, no folds can be generated."
        assert len(groups) != org_num_groups, "No groups were removed, but some should have been."

    # Shuffle the groups
    shuffled_groups = list(groups.keys())
    np.random.shuffle(shuffled_groups)

    # Assign the remaining groups to folds in a round-robin fashion
    fold_groups = [[] for _ in range(k)]
    for i, group_name in enumerate(shuffled_groups):
        fold_groups[i % k].append(group_name)

    print(f"Fold groups: {fold_groups}")

    # Now assign accessions to val/test/train for each fold
    for i in range(k):
        val_groups = fold_groups[i]
        test_groups = fold_groups[(i + 1) % k]
        train_groups = [g for j in range(k) if j not in {i, (i + 1) % k} for g in fold_groups[j]]

        print(test_groups)

        val_accessions = metadata[metadata[split_col].isin(val_groups)]['accession'].tolist()
        test_accessions = metadata[metadata[split_col].isin(test_groups)]['accession'].tolist()
        train_accessions = metadata[metadata[split_col].isin(train_groups)]['accession'].tolist()
        print("Test accessions: ", test_accessions)

        # Assert no overlap
        assert len(set(val_accessions) & set(test_accessions)) == 0, "Validation and test sets overlap."
        assert len(set(val_accessions) & set(train_accessions)) == 0, "Validation and train sets overlap."
        assert len(set(test_accessions) & set(train_accessions)) == 0, "Test and train sets overlap."

        fold_indices[i]['val'].extend(val_accessions)
        fold_indices[i]['test'].extend(test_accessions)
        fold_indices[i]['train'].extend(train_accessions)


        # Convert to numpy array of str
        fold_indices[i]['val'] = np.array(fold_indices[i]['val'], dtype=str)
        fold_indices[i]['test'] = np.array(fold_indices[i]['test'], dtype=str)
        fold_indices[i]['train'] = np.array(fold_indices[i]['train'], dtype=str)


    return fold_indices

def main():
    parser = argparse.ArgumentParser(description="Generate k-fold splits for a dataset.")
    parser.add_argument("--input_file", type=str, help="Path to the input file containing the dataset.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save the k-fold splits.", required=True)
    parser.add_argument("--split_style", type=str, choices=["genera", "species", "species_even"], help="Criterion for splitting the strata.", required=True)
    parser.add_argument("-k", type=int, help="Number of folds for k-fold cross-validation.", default=7)
    args = parser.parse_args()

    if args.split_style == "species_even":
        raise NotImplementedError("species_even split style is not implemented yet.")

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    # Dump all args
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Set a global random seed for reproducibility
    np.random.seed(42)

    # Read the metadata
    metadata = pd.read_csv(input_file)
    logging.info(f"Read {len(metadata)} rows from {input_file}")

    logging.info(f"Generating {args.k}-fold splits for {args.split_style}...")
    fold_indices = generate_k_fold_splits(metadata, args.k, args.split_style)
    logging.info(f"Generated {args.k}-fold splits.")

    output_dir = output_dir / str(args.split_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving k-fold splits to %s ...", output_dir)
    for i in range(args.k):
        for key in ['train', 'val', 'test']:
            fold_file = output_dir / f"{key}_fold_{i}.pt"
            torch.save(fold_indices[i][key], fold_file)
            logging.info(f"Saved fold {i} to {fold_file}")

    logging.info("All folds saved.")
    logging.info("Done.")   

if __name__ == "__main__":
    main()