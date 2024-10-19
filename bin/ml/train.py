from models import MLP
# from dataset import MALDI_TOF_DS
from datamodule import Spectrum_DataModule
import lightning as L
import torch

def main():
    model = MLP(6000, 300, 250, 3)  # input_dim, output_dim, hidden_dim, hidden_layers
    
    torch.set_float32_matmul_precision('medium')    # medium | high

    # dataset = MALDI_TOF_DS('../../data/idbac_db/preprocessing',
    #                         '../../data/idbac_db/raw/db.csv',
    #                         '../../data/idbac_db/preprocessed',)
    
    datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                    '../../data/idbac_db/raw/db.csv',
                                    '../../data/idbac_db/preprocessed')

    trainer = L.Trainer(max_epochs=10, log_every_n_steps=10)
    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()