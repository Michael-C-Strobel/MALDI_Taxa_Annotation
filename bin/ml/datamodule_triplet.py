import lightning as L
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
from dataset import Paired_MALDI_TOF_DS, ExhaustiveMALDI_TOF_DS, single_MALDI_TOF_DS
from pathlib import Path
import torch
from torchvision import transforms
from custom_transforms import *

class Triplet_DataModule(L.LightningDataModule):
    """
    TODO: Implement train test splits
    """
    def __init__(self, preprocessing_dir:str, 
                 metadata_table:str, 
                 root_dir:str, 
                 transforms=None, 
                 num_workers:int=4,
                 batch_size:int=32,
                 **kwargs):
        super().__init__()
        self.preprocessing_dir = preprocessing_dir
        self.metadata_table = metadata_table
        self.root_dir = root_dir
        self.num_workers = num_workers

        self.transform = transforms

        self.full_dataset = single_MALDI_TOF_DS(self.preprocessing_dir, self.metadata_table, self.root_dir, process=False,
                                                transform=self.transform,
                                                **kwargs)
        
        self.batch_size = batch_size

    def prepare_data(self):
       pass

    def setup(self, stage:str):
        pass

    def collate_fn(self, batch):
        # Note that the collate_fn is not parallelized, so this is less efficent than a __getitem__ implementation
        triplets = [self.full_dataset.generate_triplets(spectrum, metadata) for (spectrum, metadata) in batch]
        return triplets


    def train_dataloader(self):
        return DataLoader(
                            self.full_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn
                        )
    
def test_single_dataloader():
    dm = Triplet_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/db.csv',
                            '../../data/idbac_db/preprocessed',
                            num_workers=1,
                            require_genus=True)
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    # Get one batch from each
    train_loader_iter = iter(train_loader)
    print(next(train_loader_iter))
