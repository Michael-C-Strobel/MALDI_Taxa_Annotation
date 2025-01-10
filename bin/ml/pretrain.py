from models.autoencoder import Autoencoder
from models.maldi_transformer import MaldiTransformer
from datamodule import SingleSpectrum_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from custom_transforms import *
import lightning as L
import torch

def main():

    autoencoder_hyperparameters = {
        'input_dim': 1800,
        'output_dim': 1800,
        'hidden_dim': 600,
        'bottleneck_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
    }

    MaldiTransformer_hyperparameters = {
        'depth': 10, # XL
        'dim': 304,  # XL
        'n_classes': 64,
        'n_heads': 8,
        'dropout': 0.2,
        'p': 0.15,
        'lmbda': 1.0,
        'proportional': False
    }

    logger = TensorBoardLogger('lightning_logs', name='maldi_transformer_model')

    # model = Autoencoder(autoencoder_hyperparameters)
    model = MaldiTransformer(MaldiTransformer_hyperparameters)
    
    torch.set_float32_matmul_precision('medium')    # medium | high

    if isinstance(model, Autoencoder):
        trans = transforms.Compose([BinSpectrum(10, 2_000, 20_000), SquareRootTransform(), NormalizeIntensity()])
        batch_size = 32
    elif isinstance(model, MaldiTransformer):
        trans = transforms.Compose([SelectMassRange(2_000, 20_000),
                                    # topf(),
                                    NormalizeIntensity(),
                                    SelectTopKPeaks(150),
                                    PadToLength(150),# We're actually truncating a bit here to 150
                                    toTensor()])
        batch_size = 128
    else:
        raise ValueError("Model type not recognized.")
    
    datamodule = SingleSpectrum_DataModule('../../data/driams/preprocessing/',
                                  '../../data/driams/preprocessing/merged_metadata.csv',
                                  '../../data/driams/preprocessed',
                                    num_workers=7,
                                    transforms=trans,
                                    batch_size=batch_size)
    datamodule.setup('fit')
    
    trainer = L.Trainer(max_epochs=150*66, log_every_n_steps=10, logger=logger, devices=1,
                        gradient_clip_val=1.0,)
    
    if isinstance(model, Autoencoder):
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