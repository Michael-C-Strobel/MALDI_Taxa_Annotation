from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from preprocess import postprocess_files, convert_spectra_to_tensor
import os 
import sys
import numpy as np
import torch
from torch import default_generator
from torch.utils.data import Subset, Sampler, IterableDataset
from torch.utils.data import random_split, DataLoader

import scipy
import pytest
import copy
from tqdm import tqdm
from dataset import single_MALDI_TOF_DS
from collections import defaultdict
import random
from clip_datamodule import CLIP_DataModule

import time

from typing import (
    cast,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

"""
Broad goals:
Produce batches of data that contain Nc classes. 

For each class sample Ns points (Sk) that are used as the support examples.
For each class sample Nq points (Qk) that are used as the query examples.
"""

class EpisodicDatamodule(CLIP_DataModule):
    def __init__(self, n_classes, n_support_samples, n_query_samples, episodes_per_epoch, **kwargs):
        # Modify kwargs to force sampling_mode = None
        kwargs['sampling_mode'] = None
        kwargs['num_turns'] = 1
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.n_support_samples = n_support_samples
        self.n_query_samples = n_query_samples
        self.episodes_per_epoch = episodes_per_epoch
        self.targets=kwargs.get('targets', 'genera')

    def train_dataloader(self):
        sampler = EpisodicBatchSampler(
                            subset=self.train_set,
                            n_classes=self.n_classes,
                            n_support_samples=self.n_support_samples,
                            n_query_samples=self.n_query_samples,
                            episodes_per_epoch=self.episodes_per_epoch,
                            allow_oversampling=True, # Allowable for train and val, but not test
                            targets=self.targets,
                        )
        return DataLoader(
                            self.train_set, 
                            num_workers=7,#self.num_workers,
                            # collate_fn=self.collate_fn,
                            persistent_workers=True,
                            prefetch_factor=80,
                            pin_memory=True,
                            batch_sampler=sampler,
                        )
    
    def val_dataloader(self):
        sampler = EpisodicBatchSampler(
                            subset=self.val_set,
                            n_classes=self.n_classes,
                            n_support_samples=self.n_support_samples,
                            n_query_samples=self.n_query_samples,
                            episodes_per_epoch=10,
                            allow_oversampling=True, # Allowable for train and val, but not test
                            targets=self.targets,
                        )
        return DataLoader(
                            self.val_set, 
                            shuffle=False,
                            num_workers=2,#self.num_workers,
                            # collate_fn=self.collate_fn,
                            persistent_workers=True,
                            prefetch_factor=80,
                            pin_memory=True,
                            batch_sampler =sampler
                        )
        
    
class EpisodicBatchSampler(Sampler):
    def __init__(self,
                 subset:single_MALDI_TOF_DS,
                 n_classes:int,
                 n_support_samples:int,
                 n_query_samples:int,
                 episodes_per_epoch:int,
                 allow_oversampling:bool=False,
                 targets:str='genera'):
        self.dataset_obj = subset.dataset  # Datset is technically a subset of a larger dataset
        self.n_classes = n_classes
        self.n_support_samples = n_support_samples
        self.n_query_samples = n_query_samples
        self.allow_oversampling = allow_oversampling
        self.episodes_per_epoch = episodes_per_epoch

        self.SKIP_SINGLETON_CLASSES = False

        if not targets in ['genera', 'species']:
            raise ValueError(f"Invalid target type: {targets}. Must be 'genera' or 'species'")
        if targets == 'genera':
            self.targets = 'genus'
        elif targets == 'species':
            self.targets = 'species'
               

        all_accessions = np.array(self.dataset_obj.relevant_accessions)#metadata_table['accession'].values

        self.classes = self.dataset_obj.metadata_table[self.targets].unique().tolist()
        if self.SKIP_SINGLETON_CLASSES:
            # Remove classes with only one accession
            self.classes = [clss for clss in self.classes if len(self.dataset_obj.metadata_table[self.dataset_obj.metadata_table[self.targets] == clss]['accession'].unique()) > 1]

        # Collect accessions for the classes in our subset
        self.classes_to_accessions = defaultdict(list)
        for clss in self.classes:
            accessions = self.dataset_obj.metadata_table[self.dataset_obj.metadata_table[self.targets] == clss]['accession'].unique()
            if len(accessions) == 0:
                continue    # Not in this subset
            self.classes_to_accessions[clss] = accessions

        # Convert accessions to indices using self.dataset.all_accessions
        self.accession_to_indices = defaultdict(list)
        for clss in self.classes:
            for accession in self.classes_to_accessions[clss]:
                indices = np.where(all_accessions == accession)[0].tolist()
                self.accession_to_indices[accession].extend(indices)

        # Convert class names to indices
        self.classes_to_indices = defaultdict(list)
        for clss in self.classes:
            for accession in self.classes_to_accessions[clss]:
                indices = self.accession_to_indices[accession]
                self.classes_to_indices[clss].extend(indices)

        # Remove empty classes
        self.classes = [clss for clss in self.classes if len(self.classes_to_indices[clss]) > 0]
        if len(self.classes) == 0:
            raise ValueError("No classes found in the subset. Please check your dataset and class labels.")

    def __len__(self):
        return self.episodes_per_epoch
    
    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            selected_classes = random.sample(self.classes, self.n_classes)
            # print("Selected classes: ", selected_classes)
            batch = []
            for clss in selected_classes:
                indices = self.classes_to_indices[clss]
                # Take as many as possible without replacement, then get the rest with replacement
                if len(indices) >= self.n_support_samples + self.n_query_samples:
                    sampled_idx = np.random.choice(indices, self.n_support_samples + self.n_query_samples, replace=False).tolist()
                else:
                    if self.allow_oversampling:
                        remaining = self.n_support_samples + self.n_query_samples - len(indices)
                        sampled_idx = np.random.choice(indices, remaining, replace=True).tolist()
                        sampled_idx.extend(np.random.choice(indices.copy(), len(indices), replace=False).tolist())
                    else:
                        raise NotImplementedError("TODO")
                if len(sampled_idx) == 0:
                    raise ValueError(f"Not enough samples for class {clss}. Expected {self.n_support_samples + self.n_query_samples}, got {len(sampled_idx)}")
                batch.extend(sampled_idx)

            # print(batch)
            yield batch


def test_for_leakage():
    """
    Test for data leakage in the dataloader.
    """
    from torchvision import transforms
    from custom_transforms import SelectTopKPeaks, PadToLength

    SPECTRA_PATH = '../../data/idbac_db/preprocessing'
    METADATA_PATH = '../../data/idbac_db/raw/ammended_db.csv'
    ML_PROCESSING_PATH = '../../data/idbac_db/processed_data'
    log_dir = './lightning_logs'
    BATCH_SIZE = 32
    N_EPOCHS = 15

    TEST_ACCESSION_PATH = '../../data/idbac_db/processed_data/genera/test_accessions.pt'

    trans =  transforms.Compose([
                        SelectTopKPeaks(150),
                        PadToLength(150, padding_value=-1.0),
                        ])
    

    fit_datamodule = EpisodicDatamodule(n_classes=5,
                                        n_support_samples=5,
                                        n_query_samples=5,
                                        episodes_per_epoch=1000,
                                        preprocessing_dir=SPECTRA_PATH,
                                        metadata_table=METADATA_PATH,
                                        root_dir=ML_PROCESSING_PATH,
                                        num_workers=7,
                                        transforms=trans,
                                        batch_size=BATCH_SIZE,
                                        # split_method='species')
                                        split_method='genera',
                                        cast_to_classification=False)   # False so we get metadata
    
    fit_datamodule.setup('fit')

    train_accessions = set()
    val_accessions = set()
    test_accessions = torch.load(TEST_ACCESSION_PATH, weights_only=False)
    test_accessions = set([str(x) for x in test_accessions])

    for _ in range(N_EPOCHS):
        for batch in fit_datamodule.train_dataloader():
            _, metadata = batch
            accessions = metadata['accession']
            accessions = set([str(x) for x in accessions])
            train_accessions.update(accessions)
            

    for _ in range(N_EPOCHS):
        for batch in fit_datamodule.val_dataloader():
            _, metadata = batch
            accessions = metadata['accession']
            accessions = set([str(x) for x in accessions])
            val_accessions.update(accessions)
            


    print("Sample Train Accessions: ", list(train_accessions)[:5])
    print("Sample Val Accessions: ", list(val_accessions)[:5])
    print("Sample Test Accessions: ", list(test_accessions)[:5])

    assert set(train_accessions).isdisjoint(set(val_accessions)), "Train and val accessions overlap"
    assert set(train_accessions).isdisjoint(set(test_accessions)), "Train and test accessions overlap"
    assert set(val_accessions).isdisjoint(set(test_accessions)), "Val and test accessions overlap"
    print("No data leakage detected between train, val and test sets.")
