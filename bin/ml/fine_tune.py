from models.autoencoder import Autoencoder
from models.transformer_embedding_prediction_head import TransformerPredictionHead
from models.binary_transformer_embedding_prediction_head import BinaryTransformerPredictionHead
from datamodule import Spectrum_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from custom_transforms import *
import lightning as L
import torch
from datamodule_triplet import Triplet_DataModule


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

    transformer_prediction_head_hyperparameters = {
        'hidden_dim': 300,
        'latent_dim': 300,
        'hidden_layers': 3,
        'dropout': 0.2,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'freeze_encoder': True,
        'encoder_path': './lightning_logs/maldi_transformer_model/version_204/checkpoints/epoch=9899-step=198000.ckpt'
    }

    binary_transformer_prediction_head_hyperparameters = {
        'hidden_dim': 300,
        'latent_dim': 300,
        'hidden_layers': 0,
        'dropout': 0.2,
        'lr': 5e-6,
        'weight_decay': 1e-5,
        'freeze_encoder': True,
        'output_bin_edges': torch.Tensor([1.0]),
        'encoder_path': './lightning_logs/maldi_transformer_model/version_204/checkpoints/epoch=9899-step=198000.ckpt'
    }

    # logger = TensorBoardLogger('lightning_logs', name='autoencoder_contrastive_model')
    # model = Autoencoder(autoencoder_hyperparameters)

    # model = TransformerPredictionHead(transformer_prediction_head_hyperparameters)
    model = BinaryTransformerPredictionHead(binary_transformer_prediction_head_hyperparameters)

    # Load an autoencoder model
    # model = Autoencoder.load_from_checkpoint('./lightning_logs/autoencoder_model/version_15/checkpoints/epoch=149-step=3300.ckpt')
    # model = Autoencoder.load_from_checkpoint('./lightning_logs/autoencoder_model/version_21/checkpoints/epoch=149-step=11550.ckpt')
    print(model)
    model.convert() # TODO: disable backprop on first layers if needed
    
    torch.set_float32_matmul_precision('medium')    # medium | high
    
    if isinstance(model, Autoencoder):
        logger = TensorBoardLogger('lightning_logs', name='autoencoder_contrastive_model')
        trans = transforms.Compose([BinSpectrum(10, 2_000, 20_000), SquareRootTransform(), NormalizeIntensity()])
        batch_size = 32

        datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/ammended_db.csv',
                            '../../data/idbac_db/processed_data',
                            num_workers=7,
                            wipe_test_sets=False,
                            transforms=trans,
                            batch_size=batch_size)
                            # wipe_test_sets=True)  #DEBUG
    elif isinstance(model, TransformerPredictionHead):
        logger = TensorBoardLogger('lightning_logs', name='transformer_embedding_prediction_model')
        trans = transforms.Compose([SelectMassRange(2_000, 20_000), 
                                    NormalizeIntensity(),
                                    SelectTopKPeaks(150),
                                    PadToLength(150),])
        batch_size = 128
        
        datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                    '../../data/idbac_db/raw/ammended_db.csv',
                                    '../../data/idbac_db/processed_data',
                                    num_workers=7,
                                    wipe_test_sets=False,
                                    transforms=trans,
                                    batch_size=batch_size)
                                    # wipe_test_sets=True)  #DEBUG
    elif isinstance(model, BinaryTransformerPredictionHead):
        logger = TensorBoardLogger('lightning_logs', name='binary_transformer_embedding_prediction_model')
        trans = transforms.Compose([SelectMassRange(3_000, 20_000),
                                    NormalizeIntensity(),
                                    SelectTopKPeaks(150),
                                    PadToLength(150),])
        batch_size = 32
        datamodule = Triplet_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/ammended_db.csv',
                            '../../data/idbac_db/processed_data',
                            num_workers=7,
                            transforms=trans)
    else:
        raise ValueError("Model type not recognized.")


    
    
    datamodule.setup('fit')
    try:
        datamodule.plot(0)
    except Exception as e:
        print("Error plotting")
        print(e)

    # Plot the train/test split
    # datamodule.full_dataset.plot_split('./train_test_split.png')

    trainer = L.Trainer(max_epochs=150, log_every_n_steps=10, logger=logger, devices=1)
    tuner = L.pytorch.tuner.Tuner(trainer)
    
    # lr_find_results = tuner.lr_find(model,
    #                                 datamodule,
    #                                 min_lr=0.00001,   # 0.001
    #                                 max_lr=0.001,    # 1.0
    #                                 early_stop_threshold=None)
    # model.lr = lr_find_results.suggestion()
    # print("Best learning rate: ", model.lr)

    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()