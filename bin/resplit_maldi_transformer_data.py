import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    
    folds = [f for f in split_dir.glob("*.pt") if "fold" in f.name]
    if not folds:
        raise ValueError(f"No split files found in {split_dir}. Ensure the files are named correctly.")

    print(f"Found fold files: {[f.name for f in sorted(folds)]}")

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

def _normalize_spectrum_id(value):
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    return Path(str(value)).stem


def _process_single_fold(
    fold,
    split_tuple,
    input_h5torch_path,
    output_h5torch_dir,
    accessions,
    accession_to_codes,
):
    train_ids, val_ids, test_ids = split_tuple
    output_path = Path(output_h5torch_dir) / f"maldi_transformer_fold_{fold}.h5torch"

    # Convert train_ids to codes from accessions if needed
    if accessions:
        _train_ids = [accession_to_codes.get(str(acc), 'NOT_AN_ACCESSION') for acc in train_ids]
        _val_ids = [accession_to_codes.get(str(acc), 'NOT_AN_ACCESSION') for acc in val_ids]
        _test_ids = [accession_to_codes.get(str(acc), 'NOT_AN_ACCESSION') for acc in test_ids]
        # Flatten
        _train_ids = [item for sublist in _train_ids for item in sublist]
        _val_ids = [item for sublist in _val_ids for item in sublist]
        _test_ids = [item for sublist in _test_ids for item in sublist]

        print(f"Fold {fold}: converted {len(train_ids)} train, {len(val_ids)} val, and {len(test_ids)} test accessions to codes.")

        print(
            f"Fold {fold}: failed to convert {_train_ids.count('NOT_AN_ACCESSION')} train, "
            f"{_val_ids.count('NOT_AN_ACCESSION')} val, and {_test_ids.count('NOT_AN_ACCESSION')} test accessions to codes."
        )
        # N.B. If it's missing here, it's probably a mixed species/genus example
        if _train_ids.count('NOT_AN_ACCESSION') > 0:
            print("Example failed train accessions:", [acc for acc, code in zip(train_ids, _train_ids) if code == 'NOT_AN_ACCESSION'][:5])
        if _val_ids.count('NOT_AN_ACCESSION') > 0:
            print("Example failed val accessions:", [acc for acc, code in zip(val_ids, _val_ids) if code == 'NOT_AN_ACCESSION'][:5])
        if _test_ids.count('NOT_AN_ACCESSION') > 0:
            print("Example failed test accessions:", [acc for acc, code in zip(test_ids, _test_ids) if code == 'NOT_AN_ACCESSION'][:5])

        train_ids = {code for code in _train_ids if code != 'NOT_AN_ACCESSION'}
        val_ids = {code for code in _val_ids if code != 'NOT_AN_ACCESSION'}
        test_ids = {code for code in _test_ids if code != 'NOT_AN_ACCESSION'}
    else:
        train_ids = {_normalize_spectrum_id(x) for x in train_ids}
        val_ids = {_normalize_spectrum_id(x) for x in val_ids}
        test_ids = {_normalize_spectrum_id(x) for x in test_ids}

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fold {fold}: resplitting data...")
    f = None
    old_f = None
    try:
        f = h5torch.File(output_path, 'w')
        old_f = h5torch.File(input_h5torch_path)

        ids = np.array(old_f["0/loc"])
        ids = [_normalize_spectrum_id(id_) for id_ in ids]
        valid_indices = []
        tt_splits = []

        print(f"Fold {fold}: example train IDs:", list(train_ids)[:5])
        print(f"Fold {fold}: example ids:", list(ids)[:5])

        for idx, id_ in tqdm(enumerate(ids), desc=f"fold {fold}", total=len(ids)):
            if id_ in train_ids:
                valid_indices.append(idx)
                tt_splits.append(b'A_train')
            elif id_ in val_ids:
                valid_indices.append(idx)
                tt_splits.append(b'A_val')
            elif id_ in test_ids:
                valid_indices.append(idx)
                tt_splits.append(b'A_test')

        mz_array = [np.array(x, dtype=float) for x in old_f["0/mz"][valid_indices]]
        intensity_array = [np.array(x, dtype=float) for x in old_f["0/intensity"][valid_indices]]

        print(f"Fold {fold}: found a total of {len(mz_array)} valid spectra.")

        species_labels = np.array(old_f["unstructured/species_labels"])[old_f["central"][valid_indices]]
        # Decode from bytes to str if needed
        species_labels = [s.decode('utf-8') if isinstance(s, bytes) else s for s in species_labels]

        # Use split species labels as genus labels
        genus_labels = [x.split(' ')[0].encode('utf-8') for x in species_labels]

        unique_genera = np.array(sorted(list(set(genus_labels))))
        genera_as_int = {genus: i for i, genus in enumerate(unique_genera)}
        # Ensure 'nan' has a home
        if b'nan' not in genera_as_int:
            genera_as_int[b'nan'] = len(genera_as_int)
            unique_genera = np.append(unique_genera, b'nan')
        genus_indices = np.array([genera_as_int[genus] for genus in genus_labels])

        f.register(genus_indices, "central")  # Central is a ref to genus (was species) by numerical id
        f.register(mz_array, 0, name="mz", mode='vlen')
        f.register(intensity_array, 0, name="intensity", mode='vlen')
        f.register(old_f["0/loc"][valid_indices], 0, name="loc")
        f.register(old_f["central"][valid_indices], 0, name="species_central")
        f.register(old_f["unstructured/species_labels"], "unstructured", name="species_labels") # Leave species unmodified (index is used as mapping for species_central)
        f.register(unique_genera, "unstructured", name="genus_labels")
        # Add new splits
        f.register(np.array(tt_splits), "unstructured", name="split")

        print(f"Fold {fold}: wrote {len(f['central'])} spectra to {output_path}.")
    finally:
        if f is not None:
            f.close()
        if old_f is not None:
            old_f.close()


def resplit_maldi_transformer_data(input_h5torch_path, output_h5torch_dir, metadata, splits, accessions, num_workers=1):
    # DEBUG: temporarily restrict resplitting to beyond fold 0 only.
    splits = {fold: split for fold, split in splits.items() if fold == 0}
    if not splits:
        raise ValueError("Fold 0 was not found in the provided splits.")

    if accessions and metadata is None:
        raise ValueError("--accessions requires --metadata_path to be provided.")

    accession_to_codes = {}
    if metadata is not None:
        metadata = metadata.copy()
        metadata['id'] = metadata['code'].map(_normalize_spectrum_id)
        metadata['accession'] = metadata['accession'].astype(str)
        accession_to_codes = metadata.groupby('accession')['id'].apply(list).to_dict()

    fold_items = sorted(splits.items(), key=lambda x: x[0])
    print(f"Processing {len(fold_items)} folds with num_workers={num_workers}...")

    if num_workers <= 1 or len(fold_items) <= 1:
        for fold, split_tuple in fold_items:
            _process_single_fold(
                fold=fold,
                split_tuple=split_tuple,
                input_h5torch_path=input_h5torch_path,
                output_h5torch_dir=output_h5torch_dir,
                accessions=accessions,
                accession_to_codes=accession_to_codes,
            )
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _process_single_fold,
                fold,
                split_tuple,
                input_h5torch_path,
                output_h5torch_dir,
                accessions,
                accession_to_codes,
            )
            for fold, split_tuple in fold_items
        ]
        for future in as_completed(futures):
            future.result()

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
    parser.add_argument(
        "--accessions",
        action="store_true",
        help="If set, the split files contain accessions instead of strain IDs.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes to use for fold-level parallel resplitting.",
    )
    args = parser.parse_args()


    k_folds_accessions = gather_k_folds(args.split_dir)
    # Print some examples from k_folds_accessions
    # k_folds_strain_ids = convert_accessions_to_strain_ids(k_folds_accessions, args.metadata_path)
    k_folds_accessions = restructure_splits(k_folds_accessions)

    metadata=pd.read_csv(args.metadata_path) if args.metadata_path else None

    resplit_maldi_transformer_data(
        input_h5torch_path=args.input_h5torch_path,
        output_h5torch_dir=args.output_h5torch_dir,
        metadata=metadata,
        splits=k_folds_accessions,
        accessions=args.accessions,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()