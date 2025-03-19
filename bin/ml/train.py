from models.mlp import MLP
from models.mlp_classifier import MLPClassifier
from models.Sentence_MALDI import Sentence_MALDI
from models.CLIP_MALDI import CLIP_MALDI
from models.mlp_binary_classifier import MLPBinaryClassifier
from datamodule import Spectrum_DataModule, SingleSpectrum_DataModule
from datamodule_triplet import Triplet_DataModule
from clip_datamodule import CLIP_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import Trainer


from torchvision import transforms
from custom_transforms import *

def main():

    hyperparameters = {
        'input_dim': 1800,
        'output_dim': 250,
        'hidden_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
    }
    Sentence_MALDI_hyperparameters = {
            'input_dim': 1700,
            # 'output_bin_edges': torch.Tensor([0.95, 0.97, 0.99, 1.0]),
            'output_bin_edges': torch.Tensor([1.0]),
            'hidden_dim': 300,
            'hidden_layers': 3,
            'weight_decay': 1e-5,
            'dropout': 0.2,
            'tau': 1.0,
    }
    mlp_binary_classifier_hyperparameters = {
        'input_dim': 1700,
        'output_dim': 2,
        'hidden_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
        'lr': 5e-6,
    }
    clip_maldi_hyperparameters = {
            'input_dim': 1700,
            'output_bin_edges': torch.Tensor([1.0]),
            'hidden_dim': 300,
            'hidden_layers': 3,
            'weight_decay': 1e-5,
            'dropout': 0.2,
            'tau': 0.07, # Default from paper (0.07)
            'padding_value': -1.0,
            'encoder': 'transformer', # 'transformer' | 'mlp'
    }

    # model = MLP(hyperparameters)  
    # model = MLPClassifier(hyperparameters)
    # model = Sentence_MALDI(Sentence_MALDI_hyperparameters)
    # model = MLPBinaryClassifier(mlp_binary_classifier_hyperparameters)
    # model = CLIP_MALDI(clip_maldi_hyperparameters)
    model = CLIP_MALDI(clip_maldi_hyperparameters, )
    
    torch.set_float32_matmul_precision('medium')    # medium | high
    
    # trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity(), NoiseInjection(noise_factor=7e-2), NormalizeIntensity()])
    # trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

    if isinstance(model, MLP):
        logger = TensorBoardLogger('lightning_logs', name='MLP_model')

        datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                        '../../data/idbac_db/raw/ammended_db.csv',
                                        '../../data/idbac_db/processed_data',
                                        num_workers=7,
                                        wipe_test_sets=False,
                                        transforms=trans,
                                        split_method='species')
                                        # wipe_test_sets=True)  #DEBUG
        datamodule.setup('fit')
        datamodule.plot(0)

    elif isinstance(model, MLPClassifier):
        logger = TensorBoardLogger('lightning_logs', name='MLP_Classifier')

        datamodule = SingleSpectrum_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/ammended_db.csv',
                            '../../data/idbac_db/processed_data',
                            num_workers=7,
                            transforms=trans,
                            )
        
        datamodule.setup('fit')

        model.one_hot_encoder = datamodule.full_dataset.get_one_hot_encoded_classes()   # Note, would be better to do this in __getitem__
    elif isinstance(model, Sentence_MALDI):
        logger = TensorBoardLogger('lightning_logs', name='Sentence_MALDI')

        datamodule = Triplet_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/ammended_db.csv',
                            '../../data/idbac_db/processed_data',
                            num_workers=7,
                            transforms=trans)
        
        datamodule.setup('fit')
    elif isinstance(model, MLPBinaryClassifier):
        logger = TensorBoardLogger('lightning_logs', name='MLPBinaryClassifier')

        datamodule = Triplet_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/ammended_db.csv',
                            '../../data/idbac_db/processed_data',
                            num_workers=7,
                            transforms=trans)
        
        datamodule.setup('fit')
    elif isinstance(model, CLIP_MALDI) and \
        clip_maldi_hyperparameters['encoder'] == 'transformer':
        trans =  transforms.Compose([
                                        SquareRootTransform(),
                                        NormalizeIntensity(),
                                        SelectTopKPeaks(150),
                                        PadToLength(150, padding_value=-1.0),
                                     ])
        logger = TensorBoardLogger('lightning_logs', name='CLIP_Transformer')

        datamodule = CLIP_DataModule('../../data/idbac_db/preprocessing',
                    '../../data/idbac_db/raw/ammended_db.csv',
                    '../../data/idbac_db/processed_data',
                    num_workers=7,
                    transforms=trans,
                    batch_size=32,
                    split_method='species')
                    # split_method='genera')
        
        datamodule.setup('fit')


    elif isinstance(model, CLIP_MALDI):
        logger = TensorBoardLogger('lightning_logs', name='CLIP_MLP')
        trans =  transforms.Compose([
                                        BinSpectrum(10, 3_000, 20_000),
                                        SquareRootTransform(),
                                        NormalizeIntensity(),
                                        NoiseInjection(noise_factor=7e-2),
                                        NormalizeIntensity(),
                                    ])

        datamodule = CLIP_DataModule('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/ammended_db.csv',
                            '../../data/idbac_db/processed_data',
                            num_workers=7,
                            transforms=trans,
                            batch_size=32,
                            split_method='species')

        datamodule.setup('fit')

    else:
        raise ValueError("Model type not recognized")

    # Plot the train/test split
    # datamodule.full_dataset.plot_split('./train_test_split.png')

    early_stop_callback = EarlyStopping(
                            monitor="val_loss",  # Metric to monitor
                            patience=5,          # Number of epochs with no improvement before stopping
                            verbose=True,
                            mode="min"           # "min" because lower validation loss is better
                        )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=40,
        verbose=True,
        min_delta=0.00,)

    trainer = Trainer(
        max_epochs=330, 
        log_every_n_steps=5, 
        logger=logger, 
        devices=[0],
        # callbacks=[early_stop_callback]
    )

    
    # lr_find_results = tuner.lr_find(model,
    #                                 datamodule,
    #                                 min_lr=0.00001,
    #                                 max_lr=0.001,
    #                                 early_stop_threshold=None)
    # model.lr = lr_find_results.suggestion()
    print("Best learning rate: ", model.lr)

    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()