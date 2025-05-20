import argparse
import os
import sys
from pathlib import Path
from maldi_nn.scripts.process_DRIAMS import DRIAMS_raw_to_peaks
from maldi_nn.spectrum import *
import pandas as pd
import numpy as np
import json
import h5torch
import h5py
from importlib.resources import files
import logging


def DRIAMS_raw_spectra_to_h5torch(DRIAMS_ROOT, outfile):
    """ This is a very lightly modified version of the original DRIAMS script
    all changes are marked with "MODIFIED"
    """
    print("(1) gathering all spectra files ...")
    ids = []
    for ix, (root, dirs, files_) in enumerate(os.walk(DRIAMS_ROOT, followlinks=True)):  # MODIFIED (follow symlinks)
        if "raw" in root:
            for f in files_:  # walk through all raw files
                if f.endswith(".txt"):
                    k = root + "/" + f
                    ids.append("/".join(k.split("/")[-4:]))

    print("(2) gathering species data for all spectra files ...")
    alldata = []
    for root, dirs, files_ in os.walk(DRIAMS_ROOT, followlinks=True):                   # MODIFIED (follow symlinks)
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
    split = json.load(open(data_path))

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

    f.register(species, "central")
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



def process_data(input_raw_path, output_dir):
    # Convert raw spectra to h5torch format
    logging.info("Converting raw spectra to h5torch format...")
    # DRIAMS_raw_spectra_to_h5torch(input_raw_path, output_dir / "raw_spectra.h5torch")

    # Convert raw data to peaks
    logging.info("Converting raw spectra to peaks...")
    DRIAMS_raw_to_peaks(output_dir / "raw_spectra.h5torch", output_dir / "MaldiTransformer_peaks.h5torch")

def main():
    parser = argparse.ArgumentParser(
        description="Process DRIAMS data consistent with MALDI Transformer.")
    parser.add_argument('--input_raw_path', type=str, required=True,
                        help='Path to the raw DRIAMS data directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for the processed data.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Dump all args
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")

    input_raw_path = Path(args.input_raw_path)
    if not input_raw_path.exists():
        raise FileNotFoundError(f"Input path {input_raw_path} does not exist.")
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)

    process_data(input_raw_path, output_dir)


if __name__ == "__main__":
    main()