import lightning as L
from torch.utils.data import random_split, DataLoader
from dataset import MALDI_TOF_DS
import torch
from torchvision import transforms
from custom_transforms import PadSequence, BinSpectrum, NormalizeIntensity

from torchvision.transforms import Pad

class Spectrum_DataModule(L.LightningDataModule):
    def __init__(self, preprocessing_dir:str, metadata_table:str, root_dir:str, num_workers:int=4):
        super().__init__()
        self.preprocessing_dir = preprocessing_dir
        self.metadata_table = metadata_table
        self.root_dir = root_dir
        self.num_workers = num_workers

        binning_transform = BinSpectrum(10, 2_000, 20_000)
        eucliden_norm     = NormalizeIntensity()
        self.transform = transforms.Compose([binning_transform, eucliden_norm])

    def prepare_data(self):
       pass
    
    def setup(self, stage:str):
        if stage == 'fit':
            full_dataset = MALDI_TOF_DS(self.preprocessing_dir, self.metadata_table, self.root_dir, process=False, transform=self.transform)
            self.train_set, self.val_set = random_split(
                full_dataset, [int(len(full_dataset) * 0.8), len(full_dataset) - int(len(full_dataset) * 0.8)], generator=torch.Generator().manual_seed(42)
            )
        if stage == 'test':
            raise NotImplementedError()
        if stage == 'all':
            full_dataset = MALDI_TOF_DS(self.preprocessing_dir, self.metadata_table, self.root_dir, process=False, transform=self.transform)
            self.predict_set = full_dataset

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=32, shuffle=False, num_workers=self.num_workers)

    def plot(self, index: int, dataset: str = 'train'):
        import matplotlib.pyplot as plt

        spectrum_a, spectrum_b, similarity = self.train_set[index]
        if len(spectrum_a.shape) > 1:
            raise NotImplementedError("Only 'intensity vectors' are supported for plotting.")
        
        fig = plt.figure(figsize=(10, 6))
        
        # Stick plot with spectrum_a on top and spectrum_b on bottom, removing dots at ends
        plt.stem(spectrum_a, linefmt='b-', markerfmt=' ', basefmt=' ')  # No markers
        plt.stem(spectrum_b * -1, linefmt='r-', markerfmt=' ', basefmt=' ')  # No markers
        
        # Add title with similarity score
        plt.title(f'Sequence Similarity: {similarity:.4f}')
        
        plt.savefig(f'{dataset}_{index}.png')

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