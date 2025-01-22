from models.mlp import MLP
from models.mlp_classifier import MLPClassifier
from models.Sentence_MALDI import Sentence_MALDI
from datamodule import Spectrum_DataModule, SingleSpectrum_DataModule
from datamodule_triplet import Triplet_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch

from torchvision import transforms
from custom_transforms import *

def main():

    # hyperparameters = {
    #     'input_dim': 1800,
    #     'output_dim': 250,
    #     'hidden_dim': 300,
    #     'hidden_layers': 3,
    #     'weight_decay': 1e-5,
    #     'dropout': 0.2,
    # }
    Sentence_MALDI_hyperparameters = {
            'input_dim': 1800,
            'output_bin_edges': torch.Tensor([0.95, 0.97, 0.99, 1.0]),
            'hidden_dim': 300,
            'hidden_layers': 3,
            'weight_decay': 1e-5,
            'dropout': 0.2,
    }

    # model = MLP(hyperparameters)
    # model = MLPClassifier(hyperparameters)
    model = Sentence_MALDI(Sentence_MALDI_hyperparameters)
    
    torch.set_float32_matmul_precision('medium')    # medium | high
    
    trans =  transforms.Compose([BinSpectrum(10, 2_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

    if isinstance(model, MLP):
        logger = TensorBoardLogger('lightning_logs', name='MLP_model')

        datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                        '../../data/idbac_db/raw/ammended_db.csv',
                                        '../../data/idbac_db/processed_data',
                                        num_workers=7,
                                        wipe_test_sets=False,
                                        transforms=trans)
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


    else:
        raise ValueError("Model type not recognized")

    # Plot the train/test split
    # datamodule.full_dataset.plot_split('./train_test_split.png')
    
    trainer = L.Trainer(max_epochs=100, log_every_n_steps=10, logger=logger, devices=[0])
    tuner = L.pytorch.tuner.Tuner(trainer)
    
    lr_find_results = tuner.lr_find(model,
                                    datamodule,
                                    min_lr=0.00001,
                                    max_lr=0.001,
                                    early_stop_threshold=None)
    model.lr = lr_find_results.suggestion()
    print("Best learning rate: ", model.lr)

    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()