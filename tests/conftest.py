import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRATCH_DIR = REPO_ROOT / "scratch"

for path in (REPO_ROOT, SCRATCH_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


import importlib

import numpy as np
import pandas as pd
import pytest


def _make_dataframe(rows):
    frame = pd.DataFrame(rows)
    frame["embedding"] = frame["embedding"].apply(lambda value: np.asarray(value, dtype=np.float32))
    return frame


@pytest.fixture
def metrics_module():
    return importlib.import_module("publication_figures_metrics")


@pytest.fixture
def balanced_pair_frames():
    rows = [
        {
            "strain_name": "strain_a",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_1",
            "embedding": [1.0, 0.0],
        },
        {
            "strain_name": "strain_b",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_2",
            "embedding": [0.9, 0.1],
        },
        {
            "strain_name": "strain_c",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_3",
            "embedding": [0.0, 1.0],
        },
        {
            "strain_name": "strain_d",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_4",
            "embedding": [0.1, 0.9],
        },
    ]
    frame = _make_dataframe(rows)
    return frame.copy(deep=True), frame.copy(deep=True)


@pytest.fixture
def singleton_genera_frames():
    rows = [
        {
            "strain_name": "strain_a",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_1",
            "embedding": [1.0, 0.0],
        },
        {
            "strain_name": "strain_b",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_2",
            "embedding": [0.9, 0.1],
        },
        {
            "strain_name": "strain_c",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_3",
            "embedding": [0.0, 1.0],
        },
    ]
    frame = _make_dataframe(rows)
    return frame.copy(deep=True), frame.copy(deep=True)


@pytest.fixture
def between_species_frames():
    train_rows = [
        {
            "strain_name": "train_a",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_1",
            "embedding": [1.0, 0.0],
        },
        {
            "strain_name": "train_b",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_2",
            "embedding": [0.8, 0.2],
        },
        {
            "strain_name": "train_c",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_3",
            "embedding": [0.0, 1.0],
        },
        {
            "strain_name": "train_d",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_4",
            "embedding": [0.2, 0.8],
        },
    ]
    test_rows = [
        {
            "strain_name": "test_a",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_1",
            "embedding": [1.0, 0.0],
        },
        {
            "strain_name": "test_b",
            "true_label": "A",
            "genus": "Genus_1",
            "species": "Species_5",
            "embedding": [0.75, 0.25],
        },
        {
            "strain_name": "test_c",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_3",
            "embedding": [0.0, 1.0],
        },
        {
            "strain_name": "test_d",
            "true_label": "B",
            "genus": "Genus_2",
            "species": "Species_6",
            "embedding": [0.25, 0.75],
        },
    ]
    return _make_dataframe(train_rows), _make_dataframe(test_rows)
