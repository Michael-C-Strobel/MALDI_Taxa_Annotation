import argparse
from pathlib import Path
import shutil
import torch
import h5torch
import pandas as pd
import numpy as np
from tqdm import tqdm

def gather_k_folds(split_dir):


    """
    Format:
    test_fold_i.pt
    train_fold_i.pt
    val_fold_i.pt
    """
    
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory {split_dir} does not exist.")
    
    folds = split_dir.glob("*.pt")

    if not folds:
        raise ValueError(f"No split files found in {split_dir}. Ensure the files are named correctly.")
    
    # Group as tuples of (fold_number, train, val, test)
    max_fold_number = max(int(f.stem.split("_")[-1]) for f in folds)
    folds_as_tuples = []
    for i in range(max_fold_number + 1):
        train_file = split_dir / f"train_fold_{i}.pt"
        val_file = split_dir / f"val_fold_{i}.pt"
        test_file = split_dir / f"test_fold_{i}.pt"

        if train_file.exists() and val_file.exists() and test_file.exists():
            folds_as_tuples.append((i, str(train_file), str(val_file), str(test_file)))
        else:
            raise ValueError(f"Missing files for fold {i}: train: {train_file.exists()}, val: {val_file.exists()}, test: {test_file.exists()}")
        
    k_fold_accessions = {}
    for fold_number, train_file, val_file, test_file in folds_as_tuples:
        k_fold_accessions[fold_number] = {
            "train": torch.load(train_file, weights_only=False).tolist(),
            "val": torch.load(val_file, weights_only=False).tolist(),
            "test": torch.load(test_file, weights_only=False).tolist()
        }

    return k_fold_accessions

def restructure_splits(k_fold_accessions,):
    """
    Restructure the k_fold_accessions dictionary to match the expected format.
    """
    splits = {}
    for fold_number, accessions in k_fold_accessions.items():
        train_ids = set(accessions["train"])
        val_ids = set(accessions["val"])
        test_ids = set(accessions["test"])
        splits[fold_number] = (train_ids, val_ids, test_ids)
    
    return splits

def resplit_maldi_transformer_data(input_h5torch_path, output_h5torch_dir, splits):
    fold_iter = 1
    f = None
    old_f = h5torch.File(input_h5torch_path)

    try:
        for fold, (train_ids, val_ids, test_ids) in splits.items():
            print(f"Processing fold {fold_iter} of {len(splits)} folds...")
            output_path = Path(output_h5torch_dir) / f"maldi_transformer_fold_{fold}.h5torch"
            
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Copying input file to {output_path}...")
            
            print(f"Resplitting data for fold {fold_iter}...")
            f = h5torch.File(output_path, 'w')
           
            ids = np.array(old_f["0/loc"])
            ids = [(str(id_).split("/")[-1]).split(".")[0] for id_ in ids]
            valid_indices = []
            splits = []

            for idx, id_ in tqdm(enumerate(ids)):
                if id_ in train_ids:
                    valid_indices.append(idx)
                    splits.append(b'A_train')
                elif id_ in val_ids:
                    valid_indices.append(idx)
                    splits.append(b'A_val')
                elif id_ in test_ids:
                    valid_indices.append(idx)
                    splits.append(b'A_test')
                else:
                    pass
                    # raise ValueError(f"ID {id_} not found in any split: train, val, or test.")

            # Debug, print the shape of all old_f arrays
            print(f"Old file has {len(old_f['central'])} central species, {len(old_f['0/mz'])} mz arrays, {len(old_f['0/intensity'])} intensity arrays, and {len(old_f['0/loc'])} loc arrays.")
            print(f"Old file has {len(np.array(old_f['unstructured/species_labels']))} species labels and {len(old_f['unstructured/split'])} split arrays.")

            mz_array = [np.array(x, dtype=float) for x in old_f["0/mz"][valid_indices]]
            intensity_array = [np.array(x, dtype=float) for x in old_f["0/intensity"][valid_indices]]

            f.register(old_f["central"][valid_indices], "central")  # Central is a ref to species
            f.register(mz_array, 0, name="mz", mode='vlen')
            f.register(intensity_array, 0, name="intensity", mode='vlen')
            f.register(old_f["0/loc"][valid_indices], 0, name="loc")
            f.register(old_f["unstructured/species_labels"], "unstructured", name="species_labels") # Leave species unmodified (index is used as mapping for central)
            # Add new splits
            f.register(np.array(splits), "unstructured", name="split")
            fold_iter += 1
            f.close()
    finally:
        if f is not None:
            f.close()
        if old_f is not None:
            old_f.close()

def main():
    parser = argparse.ArgumentParser(
        description="Resplit MALDI Transformer data into training, validation, and test sets based on CV folds."
    )
    parser.add_argument(
        "--input_h5torch_path",
        type=str,
        help="Path to the input h5torch file containing the MALDI Transformer data.",
    )
    parser.add_argument(
        "--output_h5torch_dir",
        type=str,
        help="Directory where the output h5torch files will be saved.",
    )
    parser.add_argument(
        "--split_dir",
        type=str,
        help="Directory containing the split files for training, validation, and test sets.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to the metadata file used to map accessions to strain IDs.",
    )
    args = parser.parse_args()


    k_folds_accessions = gather_k_folds(args.split_dir)
    # Print some examples from k_folds_accessions
    # k_folds_strain_ids = convert_accessions_to_strain_ids(k_folds_accessions, args.metadata_path)
    k_folds_accessions = restructure_splits(k_folds_accessions)


    resplit_maldi_transformer_data(
        input_h5torch_path=args.input_h5torch_path,
        output_h5torch_dir=args.output_h5torch_dir,
        splits=k_folds_accessions,
    )


if __name__ == "__main__":
    main()