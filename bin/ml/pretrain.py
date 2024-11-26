from models.autoencoder import Autoencoder
from datamodule import SingleSpectrum_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch

def main():

    hyperparameters = {
        'input_dim': 1800,
        'output_dim': 1800,
        'hidden_dim': 600,
        'bottleneck_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
    }
    logger = TensorBoardLogger('lightning_logs', name='autoencoder_model')

    model = Autoencoder(hyperparameters)
    
    torch.set_float32_matmul_precision('medium')    # medium | high
    
    datamodule = SingleSpectrum_DataModule('../../data/driams/preprocessing/',
                                  '../../data/driams/preprocessing/merged_metadata.csv',
                                  '../../data/driams/preprocessed',
                                    num_workers=7,)
    datamodule.setup('fit')
    
    trainer = L.Trainer(max_epochs=150, log_every_n_steps=10, logger=logger)
    tuner = L.pytorch.tuner.Tuner(trainer)
    
    lr_find_results = tuner.lr_find(model,
                                    datamodule,
                                    min_lr=0.001,
                                    max_lr=1.0,
                                    early_stop_threshold=None)
    model.lr = lr_find_results.suggestion()
    print("Best learning rate: ", model.lr)

    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()