import lightning as L
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Subset
from dataset import Paired_MALDI_TOF_DS, ExhaustiveMALDI_TOF_DS, single_MALDI_TOF_DS
from pathlib import Path
import torch
from torchvision import transforms
from custom_transforms import *

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
                 split_method='genera'):
        super().__init__()
        self.preprocessing_dir = preprocessing_dir
        self.metadata_table = metadata_table
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.split_method = split_method

        self.transform = transforms

        self.full_dataset = single_MALDI_TOF_DS(self.preprocessing_dir, self.metadata_table, self.root_dir, process=False,
                                                transform=self.transform,
                                                require_genus=True, 
                                                sampling_mode='nce')
        
        self.batch_size = batch_size

        accessions_path = Path(self.root_dir)/f'{self.split_method}'

        # Get train/val/test accessions
        self.train_accessions = None
        self.val_accessions = None
        self.test_accessions = None
        if accessions_path /'train_accessions.pt':
            self.train_accessions = torch.load(accessions_path /'train_accessions.pt', weights_only=False)
        if accessions_path /'val_accessions.pt':
            self.val_accessions = torch.load(accessions_path /'val_accessions.pt', weights_only=False)
        if accessions_path /'test_accessions.pt':
            self.test_accessions = torch.load(accessions_path /'test_accessions.pt', weights_only=False)

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
        return [x for x in batch]

    def train_dataloader(self):
        return DataLoader(
                            self.train_set, 
                            batch_size=self.batch_size, 
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn,
                            drop_last=True
                        )
    
    def val_dataloader(self):
        return DataLoader(
                            self.val_set, 
                            batch_size=self.batch_size, 
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn
                        )
    
    def test_dataloader(self):
        return DataLoader(
                            self.predict_set, 
                            batch_size=self.batch_size, 
                            shuffle=False,
                            num_workers=self.num_workers,
                            collate_fn=self.collate_fn
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
