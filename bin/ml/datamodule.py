import lightning as L
from torch.utils.data import random_split, DataLoader
from dataset import MALDI_TOF_DS
import torch
from torchvision import transforms
from custom_transforms import PadSequence

from torchvision.transforms import Pad

class Spectrum_DataModule(L.LightningDataModule):
    def __init__(self, preprocessing_dir:str, metadata_table:str, root_dir:str):
        super().__init__()
        self.preprocessing_dir = preprocessing_dir
        self.metadata_table = metadata_table
        self.root_dir = root_dir

        padding_transform = PadSequence((200,2))
        self.transform = transforms.Compose([padding_transform])

    def prepare_data(self):
        raise NotImplementedError()
    
    def setup(self, stage:str):
        if stage == 'fit':
            full_dataset = MALDI_TOF_DS(self.preprocessing_dir, self.metadata_table, self.root_dir, process=False, transform=self.transform)
            self.train_set, self.val_set = random_split(
                full_dataset, [int(len(full_dataset) * 0.8), len(full_dataset) - int(len(full_dataset) * 0.8)], generator=torch.Generator().manual_seed(42)
            )
        if stage == 'test':
            raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)
    

def test_dataloader():
    dm = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/db.csv',
                            '../../data/idbac_db/preprocessed')
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    # Get one batch from each
    train_loader_iter = iter(train_loader)
    val_loader_iter = iter(val_loader)
    print(next(train_loader_iter))
    print(next(val_loader_iter))