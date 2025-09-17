import lightning as L
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
from dataset import Paired_MALDI_TOF_DS, ExhaustiveMALDI_TOF_DS, single_MALDI_TOF_DS
from pathlib import Path
import torch
from torchvision import transforms
from custom_transforms import *
import time

class CLIP_DataModule(L.LightningDataModule):
    """
    TODO: Implement train test splits
    """
    def __init__(self, preprocessing_dir:str, 
                 metadata_table:str, 
                 root_dir:str, 
                 transforms=None, 
                 num_workers:int=4,
                 batch_size:int=32,
                 split_method='genera',
                 sampling_mode='nce',
                 cast_to_classification=False,
                 targets='genera',
                 num_turns=1,
                 k=None,
                 prefer_hard=False,
                 singletons_as_anchors=True,
                 ):
        super().__init__()
        self.preprocessing_dir = preprocessing_dir
        self.metadata_table = metadata_table
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.split_method = split_method
        self.sampling_mode = sampling_mode
        self.targets = targets
        self.k_fold_split = k

        if not self.targets in ['genera', 'species']:
            raise ValueError(f"Expected targets to be 'genera' or 'species', but got {self.targets}")

        self.transform = transforms

        self.full_dataset = single_MALDI_TOF_DS(self.preprocessing_dir, self.metadata_table, self.root_dir, process=False,
                                                transform=self.transform,
                                                sampling_mode=self.sampling_mode,
                                                cast_to_classification=cast_to_classification,
                                                num_turns=num_turns,
                                                targets=self.targets,
                                                prefer_hard=prefer_hard,
                                                singletons_as_anchors=singletons_as_anchors,
                                                )
        
        self.batch_size = batch_size

        accessions_path = Path(self.root_dir)/f'{self.split_method}'
        train_accessions_path = accessions_path / 'train_accessions.pt'
        val_accessions_path = accessions_path / 'val_accessions.pt'
        test_accessions_path = accessions_path / 'test_accessions.pt'

        if k is not None:
            print(f"Using k-fold split with k={k}")
            train_accessions_path = accessions_path / f'train_fold_{k}.pt'
            val_accessions_path = accessions_path / f'val_fold_{k}.pt'
            test_accessions_path = accessions_path / f'test_fold_{k}.pt'

        # Get train/val/test accessions
        if train_accessions_path:
            self.train_accessions = torch.load(train_accessions_path, weights_only=False)
        if val_accessions_path:
            self.val_accessions = torch.load(val_accessions_path, weights_only=False)
        if test_accessions_path:
            self.test_accessions = torch.load(test_accessions_path, weights_only=False)

    def prepare_data(self):
        pass

    def setup(self, stage:str):
        if stage == 'fit':
           if self.train_accessions is None or self.val_accessions is None:
               raise ValueError("Expected train_accessions and val_accessions to be set")
           
           self.train_set = self.full_dataset.subset(self.train_accessions)
           self.val_set   = self.full_dataset.subset(self.val_accessions)

           print("Train set size: ", len(self.train_set))
           print("Val set size: ", len(self.val_set))
        elif stage == 'test':
           raise NotImplementedError("Test set not implemented yet")
        
        elif stage == 'all':
            self.predict_set = self.full_dataset.subset(self.test_accessions)

    def collate_fn(self, batch):
        return batch
        return [x for x in batch]

    def train_dataloader(self):
        return DataLoader(
                            self.train_set, 
                            batch_size=self.batch_size, 
                            shuffle=True,
                            num_workers=7,#self.num_workers,
                            # collate_fn=self.collate_fn,
                            drop_last=True,
                            persistent_workers=True,
                            prefetch_factor=10,
                            pin_memory=True
                        )
    
    def val_dataloader(self):
        return DataLoader(
                            self.val_set, 
                            batch_size=self.batch_size, 
                            shuffle=False,
                            num_workers=2,#self.num_workers,
                            # collate_fn=self.collate_fn,
                            persistent_workers=True,
                            prefetch_factor=10,
                            pin_memory=True
                        )
    
    def test_dataloader(self):
        return DataLoader(
                            self.predict_set, 
                            batch_size=self.batch_size, 
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn,
                            persistent_workers=True,
                            prefetch_factor=5,
                            pin_memory=True
                        )
    
def test_triplet():
    dm = Triplet_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/db.csv',
                            '../../data/idbac_db/preprocessed',
                            num_workers=1,)
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    # Get one batch from each
    train_loader_iter = iter(train_loader)
    print(next(train_loader_iter))


def test_triplet_tt_split():
    dm = Triplet_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/db.csv',
                            '../../data/idbac_db/preprocessed',
                            num_workers=1,)
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader   = dm.val_dataloader()
    # Get one batch from each
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)
    print(next(train_loader_iter))
    print(next(val_loader_iter))
