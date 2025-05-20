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
    def __init__(self,
                 preprocessing_dir:str, 
                 metadata_table:str,
                 root_dir:str,
                 transforms=None,
                 num_workers:int=4,
                 batch_size:int=32,
                 split_method:str='species',
                 inference_set_to_use:str='train',
                 cast_to_classification:bool=False,
                 targets:str='genera',
                 k:int=None,
                 num_turns:int=1,):
        super().__init__()
        self.preprocessing_dir = preprocessing_dir
        self.metadata_table = metadata_table
        self.root_dir = root_dir
        self.num_workers = num_workers
        self.split_method = split_method
        self.inference_set_to_use = inference_set_to_use
        self.cast_to_classification = cast_to_classification
        self.targets = targets
        self.k_fold_split = k
        self.num_turns = num_turns

        if not self.targets in ['genera', 'species']:
            raise ValueError(f"Expected targets to be 'genera' or 'species', but got {self.targets}")
        
        
        self.transform = transforms
    
        self.train_accessions = None
        self.val_accessions = None
        self.test_accessions = None
       
        wipe_test_sets=False # Hardcoded 

        accessions_path = Path(self.root_dir)/f'{self.split_method}'
        train_accessions_path = accessions_path / 'train_accessions.pt'
        val_accessions_path = accessions_path / 'val_accessions.pt'
        test_accessions_path = accessions_path / 'test_accessions.pt'

        if k is not None:
            print(f"Using k-fold split with k={k}")
            train_accessions_path = accessions_path / f'train_fold_{k}.pt'
            val_accessions_path = accessions_path / f'val_fold_{k}.pt'
            test_accessions_path = accessions_path / f'test_fold_{k}.pt'

        if train_accessions_path:
            if wipe_test_sets:
                (train_accessions_path).unlink(missing_ok=True)
            else:
                self.train_accessions = torch.load(train_accessions_path, weights_only=False)
        if val_accessions_path:
            if wipe_test_sets:
                (val_accessions_path).unlink(missing_ok=True)
            else:
                self.val_accessions = torch.load(val_accessions_path, weights_only=False)
        if test_accessions_path:
            if wipe_test_sets:
                (test_accessions_path).unlink(missing_ok=True)
            else:
                self.test_accessions = torch.load(test_accessions_path, weights_only=False)
        
        self.batch_size = batch_size
        self.train_test_stats = None

    def prepare_data(self):
        pass

    def setup(self, stage:str):
        self.full_dataset = single_MALDI_TOF_DS(self.preprocessing_dir,
                                                self.metadata_table,
                                                self.root_dir,
                                                process=False,
                                                transform=self.transform,
                                                cast_to_classification=self.cast_to_classification,
                                                targets=self.targets,
                                                num_turns=self.num_turns,
                                                sampling_mode='triplets',
                                                )
        if stage == 'fit':
            # self.train_set, self.val_set = random_split(
            #     full_dataset, [int(len(full_dataset) * 0.8), len(full_dataset) - int(len(full_dataset) * 0.8)], generator=torch.Generator().manual_seed(42)
            # )
            if self.train_accessions is None or self.val_accessions is None:
                raise ValueError("Training and validation accessions must be provided.")
            else:
                self.train_set = self.full_dataset.subset(self.train_accessions)
                self.val_set = self.full_dataset.subset(self.val_accessions)
        elif stage == 'test' or stage == 'predict':
            if self.inference_set_to_use == "test":
                self.predict_set = ExhaustiveMALDI_TOF_DS(self.full_dataset, self.test_accessions, paired=False)
            elif self.inference_set_to_use == "val":
                self.predict_set = ExhaustiveMALDI_TOF_DS(self.full_dataset, self.val_accessions, paired=False)
            elif self.inference_set_to_use == "train": 
                self.predict_set = ExhaustiveMALDI_TOF_DS(self.full_dataset, self.train_accessions, paired=False)
            elif self.inference_set_to_use == "all": 
                self.predict_set = ExhaustiveMALDI_TOF_DS(self.full_dataset, self.full_dataset.all_accessions, paired=False)
        elif stage == 'all':
            self.predict_set = self.full_dataset
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, shuffle=False, num_workers=1)

    
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
