import argparse
import os
from pathlib import Path
from maldi_nn.scripts.process_DRIAMS import DRIAMS_raw_to_peaks
import pandas as pd
import numpy as np
import json
import h5torch
from importlib.resources import files
import logging

try:
    from pyteomics import mzml as pyteomics_mzml
except ImportError:
    pyteomics_mzml = None


def DRIAMS_raw_spectra_to_h5torch(DRIAMS_ROOT, outfile):
    """ This is a very lightly modified version of the original DRIAMS script
    all changes are marked with "MODIFIED"
    """
    print("(1) gathering all spectra files ...")
    ids = []
    for ix, (root, _dirs, files_) in enumerate(os.walk(DRIAMS_ROOT, followlinks=True)):  # MODIFIED (follow symlinks)
        relative_to_driams_root = root.replace(str(DRIAMS_ROOT), "").lstrip("/")
        
        if "raw" in relative_to_driams_root:    # MODIFIED (ignore "raw" in higher directories)
            for f in files_:  # walk through all raw files
                if f.endswith(".txt"):
                    k = root + "/" + f
                    ids.append("/".join(k.split("/")[-4:]))

    print("(2) gathering species data for all spectra files ...")
    alldata = []
    for root, _dirs, files_ in os.walk(DRIAMS_ROOT, followlinks=True):                   # MODIFIED (follow symlinks)
        if "id" in root:
            for f in files_:
                if f.endswith("_clean.csv"):
                    print("... reading amr data for " + root + "/" + f)
                    k = root + "/" + f
                    data = pd.read_csv(k)

                    data["code"] = (
                        k.split("/")[-4]
                        + "/raw/"
                        + k.split("/")[-2]
                        + "/"
                        + data["code"]
                        + ".txt"
                    )
                    ids_driams_year = [
                        f
                        for f in ids
                        if (f.startswith(k.split("/")[-4]))
                        and (k.split("/")[-2] == f.split("/")[2])
                    ]
                    # only keep amr data for files that exist
                    keepers = [d in ids_driams_year for d in data["code"]]
                    alldata.append(data.iloc[keepers, :])

    if len(alldata) > 1:    # MODIFIED
        alldata = pd.concat(alldata)
    else:
        alldata = alldata[0]
    # cleanup data MODIFIED (only drop present cols)
    columns_to_drop = [
        "laboratory_species", "Unnamed: 0.1", "Unnamed: 0", "combined_code", "genus"
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in alldata.columns]
    alldata = alldata.drop(existing_columns_to_drop, axis=1)[["code", "species"]]

    # putting unidentified species to nans
    alldata.loc[
        alldata["species"].str.contains("not reliable identification"), ["species"]
    ] = np.nan

    alldata = pd.concat(
        [
            pd.DataFrame({"code": list(set(ids).difference(set(alldata["code"])))}),
            alldata,
        ]
    )
    dataraw = alldata

    species_labels = np.unique(dataraw["species"].astype(bytes))
    _, species = np.where(
        dataraw["species"].astype(bytes).values.reshape(-1, 1) == species_labels
    )

    loc = dataraw["code"].values.astype(bytes)

    data_path = files("maldi_nn.utils").joinpath("driams_split.json")
    with open(data_path, encoding="utf-8") as fh:
        split = json.load(fh)

    spectrum_split = np.array(
        [
            split[l]
            for l in pd.Series(loc.astype(str))
            .str.split("/", expand=True)[[0, 2, 3]]
            .apply("/".join, axis=1)
            .values
        ]
    )

    # MODIFIED
    print(f"Writing a total of {len(loc)} spectra to {outfile}")

    f = h5torch.File(outfile, "w")

    f.register(species, "central")  # Note to self: species is central
    f.register(loc, 0, name="loc")
    f.register(species_labels, "unstructured", name="species_labels")
    f.register(spectrum_split.astype(bytes), "unstructured", name="split")


    ints = []
    mzs = []
    for ix, k in enumerate(loc):
        spectrum = pd.read_table(
            os.path.join(DRIAMS_ROOT, k.astype(str)),
            comment="#",
            sep=" ",
            index_col=None,
            header=0,
        ).values
        spectrum = spectrum[~np.isnan(spectrum).any(1)]

        mz = spectrum[:, 0]
        intensities = spectrum[:, 1]
        ints.append(intensities.astype(np.uint32))
        mzs.append(mz.astype(np.float32))
        if (ix + 1) % 1000 == 0:
            print(ix, end=" ", flush=True)
            if (ix + 1) == 1000:
                f.register(ints, 0, name="intensity", mode="vlen", length=len(species))
                f.register(mzs, 0, name="mz", mode="vlen", length=len(species))
                ints = []
                mzs = []
            else:
                f.append(ints, "0/intensity")
                f.append(mzs, "0/mz")
                ints = []
                mzs = []

    f.append(ints, "0/intensity")
    f.append(mzs, "0/mz")
    f.close()
    print("done")
    return None


def _read_mzml_profile_spectrum(mzml_path: Path):
    if pyteomics_mzml is None:
        raise ImportError(
            "pyteomics is required to read mzML files. Install it in your environment before running RKI processing."
        )

    mzml_file = pyteomics_mzml.read(str(mzml_path))

    selected_mz = None
    selected_intensity = None
    valid_scans = 0
    for scan in mzml_file:
        mz_array = scan.get("m/z array")
        intensity_array = scan.get("intensity array")
        if mz_array is None or intensity_array is None:
            continue
        valid_scans += 1
        if selected_mz is None:
            selected_mz = mz_array
            selected_intensity = intensity_array

    if selected_mz is None or selected_intensity is None:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    if valid_scans > 1:
        logging.info(
            "mzML file %s contains %d scans; using first valid scan only (no scan aggregation).",
            mzml_path,
            valid_scans,
        )

    mz_vals = np.asarray(selected_mz, dtype=np.float32)
    intensities = np.asarray(selected_intensity, dtype=np.float32)
    return mz_vals, intensities


def RKI_mzml_spectra_to_h5torch(input_mzml_dir, outfile, metadata_csv=None):
    input_mzml_dir = Path(input_mzml_dir)
    mzml_paths = sorted(input_mzml_dir.rglob("*.mzML"))
    if len(mzml_paths) == 0:
        raise ValueError(f"No .mzML files found under {input_mzml_dir}")

    inferred_metadata = input_mzml_dir.parent / "rki_metadata.csv"
    metadata_path = Path(metadata_csv) if metadata_csv else inferred_metadata

    species_by_stem = {}
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if "Strain name" in metadata.columns and "species" in metadata.columns:
            species_by_stem = dict(zip(metadata["Strain name"].astype(str), metadata["species"].astype(str)))
        else:
            logging.warning(
                "Metadata file %s is missing required columns ('Strain name', 'species'); falling back to filename parsing.",
                metadata_path,
            )
    else:
        logging.warning(
            "No metadata file found at %s. Falling back to filename parsing for species labels.",
            metadata_path,
        )

    loc_strings = [str(p.relative_to(input_mzml_dir)) for p in mzml_paths]
    species_names = []
    for p in mzml_paths:
        stem = p.stem
        species = species_by_stem.get(stem)
        if species is None:
            parts = stem.split("_")
            species = " ".join(parts[1:3]) if len(parts) >= 3 else "unknown"
        species_names.append(species)

    species_array = np.array(species_names, dtype=object)
    species_labels = np.unique(species_array.astype(bytes))
    _, species = np.where(species_array.astype(bytes).reshape(-1, 1) == species_labels)

    loc = np.array(loc_strings, dtype=object).astype(bytes)
    spectrum_split = np.array(["train"] * len(loc), dtype=object).astype(bytes)

    print(f"Writing a total of {len(loc)} mzML spectra to {outfile}")
    f = h5torch.File(outfile, "w")

    f.register(species, "central")
    f.register(loc, 0, name="loc")
    f.register(species_labels, "unstructured", name="species_labels")
    f.register(spectrum_split, "unstructured", name="split")

    ints = []
    mzs = []
    initialized = False
    for ix, mzml_path in enumerate(mzml_paths):
        mz, intensities = _read_mzml_profile_spectrum(mzml_path)
        ints.append(intensities.astype(np.float32))
        mzs.append(mz.astype(np.float32))

        if (ix + 1) % 1000 == 0:
            print(ix, end=" ", flush=True)
            if not initialized:
                f.register(ints, 0, name="intensity", mode="vlen", length=len(species))
                f.register(mzs, 0, name="mz", mode="vlen", length=len(species))
                initialized = True
            else:
                f.append(ints, "0/intensity")
                f.append(mzs, "0/mz")
            ints = []
            mzs = []

    if not initialized:
        f.register(ints, 0, name="intensity", mode="vlen", length=len(species))
        f.register(mzs, 0, name="mz", mode="vlen", length=len(species))
    else:
        f.append(ints, "0/intensity")
        f.append(mzs, "0/mz")

    f.close()
    print("done")
    return None


def process_data(input_path, output_dir, dataset, metadata_csv=None):
    raw_h5torch_path = output_dir / "raw_spectra.h5torch"
    peaks_h5torch_path = output_dir / "MaldiTransformer_peaks.h5torch"

    # Convert spectra to h5torch format
    logging.info("Converting spectra to h5torch format...")
    if dataset == "driams":
        DRIAMS_raw_spectra_to_h5torch(input_path, raw_h5torch_path)
    elif dataset == "rki":
        RKI_mzml_spectra_to_h5torch(input_path, raw_h5torch_path, metadata_csv=metadata_csv)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset}")

    # Convert raw data to peaks
    logging.info("Converting raw spectra to peaks...")
    DRIAMS_raw_to_peaks(raw_h5torch_path, peaks_h5torch_path)


def main():
    parser = argparse.ArgumentParser(
        description="Process spectra for MALDI Transformer (DRIAMS txt or RKI mzML).")
    parser.add_argument('--dataset', type=str, choices=['driams', 'rki'], default='driams',
                        help='Dataset mode: driams (raw txt layout) or rki (mzML files).')
    parser.add_argument('--input_raw_path', type=str, required=False,
                        help='Legacy input path argument. For DRIAMS, raw root. For RKI, mzML directory.')
    parser.add_argument('--input_path', type=str, required=False,
                        help='Input path. For DRIAMS, raw root. For RKI, mzML directory.')
    parser.add_argument('--input_mzml_path', type=str, required=False,
                        help='Input mzML directory for RKI mode.')
    parser.add_argument('--metadata_csv', type=str, required=False, default=None,
                        help='Optional RKI metadata CSV (defaults to <input_mzml_path>/../rki_metadata.csv).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the processed data.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Dump all args
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info("%s: %s", arg, getattr(args, arg))

    input_arg = args.input_path or args.input_mzml_path or args.input_raw_path
    if input_arg is None:
        raise ValueError("One of --input_path, --input_mzml_path, or --input_raw_path must be provided.")

    input_path = Path(input_arg)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)

    process_data(
        input_path=input_path,
        output_dir=output_dir,
        dataset=args.dataset,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()