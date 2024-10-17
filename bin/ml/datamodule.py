import lightning as L
from torch.utils.data import random_split, DataLoader

class MALDI_TOF_DS(L.LightningDataModule):
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir

    def prepare_data(self):
        raise NotImplementedError()
    
    def setup(self, stage:str):
        if stage == 'fit':
            full_dataset = MALDI_TOF_Dataset(self.data_dir)
            