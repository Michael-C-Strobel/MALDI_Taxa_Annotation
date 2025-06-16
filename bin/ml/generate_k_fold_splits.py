import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def split_by_genera(metadata, k, min_group_size):
    fold_indices = {i: {'train': [], 'val': [], 'test': []} for i in range(k)}
    groups = metadata.groupby("genus")

    small_groups = set()
    for name, group in groups:
        if len(group) <= max(min_group_size, k):
            small_groups.add(name)
            for i in range(k):
                fold_indices[i]['train'].extend(group['accession'].tolist())

    groups = {name: group for name, group in groups if name not in small_groups}
    logging.info(f"Removed {len(small_groups)} small groups from {len(groups) + len(small_groups)} total groups.")
    assert len(groups) > 0, "All groups are small, no folds can be generated."

    shuffled_groups = list(groups.keys())
    np.random.shuffle(shuffled_groups)
    fold_groups = [[] for _ in range(k)]
    for i, group_name in enumerate(shuffled_groups):
        fold_groups[i % k].append(group_name)

    for i in range(k):
        val_groups = fold_groups[i]
        test_groups = fold_groups[(i + 1) % k]
        train_groups = [g for j in range(k) if j not in {i, (i + 1) % k} for g in fold_groups[j]]

        val_accessions = metadata[metadata["genus"].isin(val_groups)]['accession'].tolist()
        test_accessions = metadata[metadata["genus"].isin(test_groups)]['accession'].tolist()
        train_accessions = metadata[metadata["genus"].isin(train_groups)]['accession'].tolist()

        fold_indices[i]['val'] = np.array(val_accessions, dtype=str)
        fold_indices[i]['test'] = np.array(test_accessions, dtype=str)
        fold_indices[i]['train'] = np.array(train_accessions, dtype=str)

    return fold_indices


def split_by_species(metadata, k, min_group_size):
    fold_indices = {i: {'train': [], 'val': [], 'test': []} for i in range(k)}
    genus_groups = metadata.groupby("genus")

    for genus, genus_df in genus_groups:
        species_groups = genus_df.groupby("species")

        large_species = [name for name, grp in species_groups if len(grp) > max(min_group_size, k)]
        if len(large_species) < k:
            for i in range(k):
                fold_indices[i]['train'].extend(genus_df['accession'].tolist())
            continue

        shuffled_species = large_species.copy()
        np.random.shuffle(shuffled_species)
        fold_species = [[] for _ in range(k)]
        for i, sp in enumerate(shuffled_species):
            fold_species[i % k].append(sp)

        for i in range(k):
            val_species = fold_species[i]
            test_species = fold_species[(i + 1) % k]
            train_species = [s for j in range(k) if j not in {i, (i + 1) % k} for s in fold_species[j]]

            val_acc = genus_df[genus_df['species'].isin(val_species)]['accession'].tolist()
            test_acc = genus_df[genus_df['species'].isin(test_species)]['accession'].tolist()
            train_acc = genus_df[genus_df['species'].isin(train_species)]['accession'].tolist()

            fold_indices[i]['val'].extend(val_acc)
            fold_indices[i]['test'].extend(test_acc)
            fold_indices[i]['train'].extend(train_acc)

    for i in range(k):
        fold_indices[i]['val'] = np.array(fold_indices[i]['val'], dtype=str)
        fold_indices[i]['test'] = np.array(fold_indices[i]['test'], dtype=str)
        fold_indices[i]['train'] = np.array(fold_indices[i]['train'], dtype=str)

    return fold_indices


def split_by_species_even(metadata, k, min_group_size):
    fold_indices = {i: {'train': [], 'val': [], 'test': []} for i in range(k)}
    # Make accessions unique
    metadata = metadata.drop_duplicates(subset='accession')

    groups = metadata.groupby("species")

    small_groups = set()
    small_accessions = set()
    for name, group in groups:
        if len(group) <= max(min_group_size, k):
            small_groups.add(name)
            for i in range(k):
                small_accessions.update(group['accession'].tolist())

    groups = {name: group for name, group in groups if name not in small_groups}
    logging.info(f"Removed {len(small_groups)} small groups from {len(groups) + len(small_groups)} total groups.")
    assert len(groups) > 0, "All groups are small, no folds can be generated."

    shuffled_groups = list(groups.keys())
    np.random.shuffle(shuffled_groups)
    fold_groups = [[] for _ in range(k)]
    for i, group_name in enumerate(shuffled_groups):
        # Split each group evenly across folds
        group = groups[group_name]
        group_members = group['accession'].tolist()
        np.random.shuffle(group_members)
        for j, member in enumerate(group_members):
            fold_groups[j % k].append(member)

    # Generate folds
    for i in range(k):
        val_accessions = fold_groups[i]
        test_accessions = fold_groups[(i + 1) % k]
        train_accessions = [acc for j in range(k) if j not in {i, (i + 1) % k} for acc in fold_groups[j]]

        fold_indices[i]['val'] = np.array(val_accessions, dtype=str)
        fold_indices[i]['test'] = np.array(test_accessions, dtype=str)
        fold_indices[i]['train'] = np.array(train_accessions, dtype=str)
        # Add small groups to train
        fold_indices[i]['train'] = np.concatenate((fold_indices[i]['train'], list(small_accessions)), axis=0)

        # Ensure all folds have unique accessions
        print(set(fold_indices[i]['train']) & set(fold_indices[i]['val']))
        assert len(set(fold_indices[i]['train']) & set(fold_indices[i]['val'])) == 0, "Train and Val folds overlap."
        assert len(set(fold_indices[i]['train']) & set(fold_indices[i]['test'])) == 0, "Train and Test folds overlap."
        assert len(set(fold_indices[i]['val']) & set(fold_indices[i]['test'])) == 0, "Val and Test folds overlap."
    
    # Convert to numpy arrays
    for i in range(k):
        fold_indices[i]['val'] = np.array(fold_indices[i]['val'], dtype=str)
        fold_indices[i]['test'] = np.array(fold_indices[i]['test'], dtype=str)
        fold_indices[i]['train'] = np.array(fold_indices[i]['train'], dtype=str)

    return fold_indices

def generate_k_fold_splits(metadata, k, split_style, min_group_size=5):
    if split_style == "genera":
        return split_by_genera(metadata, k, min_group_size)
    elif split_style == "species":
        return split_by_species(metadata, k, min_group_size)
    elif split_style == "species_even":
        return split_by_species_even(metadata, k, min_group_size)
    else:
        raise ValueError(f"Unknown split style: {split_style}")

def main():
    parser = argparse.ArgumentParser(description="Generate k-fold splits for a dataset.")
    parser.add_argument("--input_file", type=str, help="Path to the input file containing the dataset.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save the k-fold splits.", required=True)
    parser.add_argument("--split_style", type=str, choices=["genera", "species", "species_even"], help="Criterion for splitting the strata.", required=True)
    parser.add_argument("-k", type=int, help="Number of folds for k-fold cross-validation.", default=7)
    parser.add_argument("--min_group_size", type=int, help="Minimum group size for splitting.", default=5)
    args = parser.parse_args()

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
    if "accession" not in metadata.columns:
        metadata['accession'] = metadata["Genbank accession"]
        metadata['accession'] = metadata['accession'].astype(str).str.strip().str.split('.').str[0]

    logging.info(f"Generating {args.k}-fold splits for {args.split_style}...")
    fold_indices = generate_k_fold_splits(metadata, args.k, args.split_style, args.min_group_size)
    logging.info(f"Generated {args.k}-fold splits.")

    output_dir = output_dir / str(args.split_style)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving k-fold splits to %s ...", output_dir)
    for i in range(args.k):
        for key in ['train', 'val', 'test']:
            fold_file = output_dir / f"{key}_fold_{i}.pt"
            torch.save(fold_indices[i][key], fold_file)
            # Based on accessions, get the number of spectra
            num_spectra = metadata[metadata['accession'].isin(fold_indices[i][key])].shape[0]
            num_genera = metadata[metadata['accession'].isin(fold_indices[i][key])]['genus'].nunique()
            num_species = metadata[metadata['accession'].isin(fold_indices[i][key])]['species'].nunique()
            logging.info(f"Saved fold {i} to {fold_file} with {num_spectra} spectra, {num_species} species, and {num_genera} genera.")

    logging.info("All folds saved.")
    logging.info("Done.")   

if __name__ == "__main__":
    main()