from models.mlp import MLP
from models.mlp_classifier import MLPClassifier
from models.Sentence_MALDI import Sentence_MALDI
from models.CLIP_MALDI import CLIP_MALDI
from models.mlp_binary_classifier import MLPBinaryClassifier
from models.CLIP_MALDI_classifier import CLIP_MALDI_Classifier
from models.logistic_regression_classifier import MultinomialLogisticClassifier
from models.prototypical_transformer import PrototyicalTransformer
from datamodule import Spectrum_DataModule, SingleSpectrum_DataModule
from datamodule_triplet import Triplet_DataModule
from prototypical_datamodule import EpisodicDatamodule
from clip_datamodule import CLIP_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import Trainer

from torchvision import transforms
from custom_transforms import *

# IDBac Data
# SPECTRA_PATH = '../../data/idbac_db/preprocessing'
# METADATA_PATH = '../../data/idbac_db/raw/ammended_db.csv'
# ML_PROCESSING_PATH = '../../data/idbac_db/processed_data'
# log_dir = './lightning_logs'
# BATCH_SIZE = 32
# N_EPOCHS = 330
#### DRIAMS (-A, for Now) Data
SPECTRA_PATH = '../../data/driams/preprocessing'
METADATA_PATH = '../../data/driams/preprocessing/merged_metadata.csv'
ML_PROCESSING_PATH = '../../data/driams/processed_data'
log_dir = './lightning_logs_DRIAMS_A'
BATCH_SIZE = 64
N_EPOCHS = 2_000 # 100
#### Other Params
TARGET='genera' # 'genera' | 'species'
SPLIT_METHOD='species' # 'genera' | 'species'

def main():

    hyperparameters = {
        'input_dim': 1800,
        'output_dim': 250,
        'hidden_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
        'TARGET': TARGET,
        'N_EPOCHS': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'METADATA_PATH': METADATA_PATH,
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
            'TARGET': TARGET,
            'N_EPOCHS': N_EPOCHS,
            'batch_size': BATCH_SIZE,
            'METADATA_PATH': METADATA_PATH,
    }
    mlp_binary_classifier_hyperparameters = {
        'input_dim': 1700,
        'output_dim': 2,
        'hidden_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
        'lr': 5e-6,
        'TARGET': TARGET,
        'N_EPOCHS': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'METADATA_PATH': METADATA_PATH,
    }
    clip_maldi_hyperparameters = {
            'input_dim': 1700,
            'output_bin_edges': torch.Tensor([1.0]),
            'hidden_dim': 300,
            'hidden_layers': 3,
            'weight_decay': 1e-2,
            'dropout': 0.2,
            'tau': 0.07, # Default from paper (0.07)
            'padding_value': None, #-10.0,
            'encoder': 'transformer', # 'transformer' | 'mlp'
            'ss_task': None,    # 'recon' | 'mlm' (not implemented yet)
            'rcon_head_dim': 1700,
            'warmup_steps': 2000,
            'lr': 1e-6,
            'TARGET': TARGET,
            'N_EPOCHS': N_EPOCHS,
            'batch_size': BATCH_SIZE,
            'METADATA_PATH': METADATA_PATH,
    }
    multinomial_logistic_classifier_hyperparameters = {
        'input_dim': 1700,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'TARGET': TARGET,
        'N_EPOCHS': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'METADATA_PATH': METADATA_PATH,
    }
    if TARGET == 'genera' and 'idbac' in SPECTRA_PATH:
        multinomial_logistic_classifier_hyperparameters.update({'n_classes': 96})
    elif TARGET == 'genera' and 'driams' in SPECTRA_PATH:
        multinomial_logistic_classifier_hyperparameters.update({'n_classes': 182})
    elif TARGET == 'species' and 'driams' in SPECTRA_PATH:
        multinomial_logistic_classifier_hyperparameters.update({'n_classes': 723})


    clip_maldi_classifier_hyperparameters = clip_maldi_hyperparameters.copy()
    if TARGET == 'genera' and 'idbac' in SPECTRA_PATH:
        clip_maldi_classifier_hyperparameters.update({'n_classes': 96})
    elif TARGET == 'genera' and 'driams' in SPECTRA_PATH:
        clip_maldi_classifier_hyperparameters.update({'n_classes': 182})
    elif TARGET == 'species' and 'driams' in SPECTRA_PATH:
        clip_maldi_classifier_hyperparameters.update({'n_classes': 723}) # For species on Driams
    else:
        raise ValueError("TARGET must be 'genera' or 'species'")
    clip_maldi_classifier_hyperparameters.update({'embed_dim': 128})
    clip_maldi_classifier_hyperparameters.update({'lr': 5e-4})
    clip_maldi_classifier_hyperparameters.update({'weight_decay': 1e-5})

    prototypical_transformer_hyperparameters = clip_maldi_hyperparameters.copy()
    prototypical_transformer_hyperparameters.update({'n_classes': 30})
    prototypical_transformer_hyperparameters.update({'n_support_samples': 1}) # First model: 5,5, second model: 3, 2
    prototypical_transformer_hyperparameters.update({'n_query_samples': 5})     # Third model: 1, 5 # Fourth, same but skipping singletons in training

    # model = MLP(hyperparameters)  
    # model = MLPClassifier(hyperparameters)
    # model = Sentence_MALDI(Sentence_MALDI_hyperparameters)
    # model = MLPBinaryClassifier(mlp_binary_classifier_hyperparameters)
    # model = CLIP_MALDI(clip_maldi_hyperparameters, )
    # model = CLIP_MALDI_Classifier(clip_maldi_classifier_hyperparameters, )
    # model = MultinomialLogisticClassifier(**multinomial_logistic_classifier_hyperparameters)
    model = PrototyicalTransformer( prototypical_transformer_hyperparameters['n_classes'],
                                    prototypical_transformer_hyperparameters['n_support_samples'],
                                    prototypical_transformer_hyperparameters['n_query_samples'],
                                    prototypical_transformer_hyperparameters)

    torch.set_float32_matmul_precision('medium')    # medium | high
    
    # trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity(), NoiseInjection(noise_factor=7e-2), NormalizeIntensity()])
    # trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

    if isinstance(model, MLP):
        logger = TensorBoardLogger(log_dir, name='MLP_model')
        
        trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

        datamodule = Spectrum_DataModule(SPECTRA_PATH,
                                        METADATA_PATH,
                                        ML_PROCESSING_PATH,
                                        num_workers=7,
                                        wipe_test_sets=False,
                                        # wipe_test_sets=True, # DEBUG
                                        transforms=trans,
                                        split_method=SPLIT_METHOD,
                                        targets=TARGET,
                                        )
        datamodule.setup('fit')
        datamodule.plot(0)

    elif isinstance(model, MLPClassifier):
        logger = TensorBoardLogger(log_dir, name='MLP_Classifier')

        trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

        datamodule = SingleSpectrum_DataModule(SPECTRA_PATH,
                            METADATA_PATH,
                            ML_PROCESSING_PATH,
                            num_workers=7,
                            transforms=trans,
                            split_method=SPLIT_METHOD,
                            targets=TARGET,
                            )
        
        datamodule.setup('fit')

        model.one_hot_encoder = datamodule.full_dataset.get_one_hot_encoded_classes()   # Note, would be better to do this in __getitem__
    elif isinstance(model, Sentence_MALDI):
        logger = TensorBoardLogger(log_dir, name='Sentence_MALDI')

        trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

        datamodule = Triplet_DataModule(SPECTRA_PATH,
                            METADATA_PATH,
                            ML_PROCESSING_PATH,
                            num_workers=7,
                            transforms=trans,
                            split_method=SPLIT_METHOD,
                            targets=TARGET,)
        
        datamodule.setup('fit')
    elif isinstance(model, MLPBinaryClassifier):
        logger = TensorBoardLogger(log_dir, name='MLPBinaryClassifier')

        datamodule = Triplet_DataModule(SPECTRA_PATH,
                            METADATA_PATH,
                            ML_PROCESSING_PATH,
                            num_workers=7,
                            transforms=trans,
                            split_method=SPLIT_METHOD,
                            targets=TARGET,)
        
        datamodule.setup('fit')
    elif type(model) is CLIP_MALDI and \
        clip_maldi_hyperparameters['encoder'] == 'transformer':
        trans =  transforms.Compose([
                                        SquareRootTransform(),
                                        SelectTopKPeaks(150),
                                        NormalizeIntensity(),
                                        PadToLength(150, padding_value=-1.0),#clip_maldi_hyperparameters['padding_value']),
                                     ])
        logger = TensorBoardLogger(log_dir, name=f'CLIP_Transformer/{TARGET}')

        datamodule = CLIP_DataModule(SPECTRA_PATH,
                                    METADATA_PATH,
                                    ML_PROCESSING_PATH,
                                    num_workers=7,
                                    transforms=trans,
                                    batch_size=BATCH_SIZE,
                                    # split_method='species')
                                    split_method=SPLIT_METHOD,
                                    targets=TARGET,)
        
        datamodule.setup('fit')


    elif type(model) is CLIP_MALDI:
        logger = TensorBoardLogger(log_dir, name='CLIP_MLP')
        trans =  transforms.Compose([
                                        BinSpectrum(10, 3_000, 20_000),
                                        SquareRootTransform(),
                                        NormalizeIntensity(),
                                        NoiseInjection(noise_factor=7e-2),
                                        NormalizeIntensity(),
                                    ])

        datamodule = CLIP_DataModule(SPECTRA_PATH,
                            METADATA_PATH,
                            ML_PROCESSING_PATH,
                            num_workers=7,
                            transforms=trans,
                            batch_size=BATCH_SIZE,
                            targets=TARGET,
                            split_method=SPLIT_METHOD)

        datamodule.setup('fit')

    elif type(model) is CLIP_MALDI_Classifier:
        if clip_maldi_classifier_hyperparameters['encoder'] != 'transformer':
            raise ValueError("CLIP_MALDI_Classifier only supports transformer encoder")
        trans =  transforms.Compose([
                                SelectMassRange(3_000, 20_000),
                                SquareRootTransform(),
                                NormalizeIntensity(),
                                SelectTopKPeaks(150),
                                PadToLength(150, padding_value=-1.0),
                                ])
        logger = TensorBoardLogger(log_dir, name=f'CLIP_Transformer_Classifier/{TARGET}')

        datamodule = SingleSpectrum_DataModule(SPECTRA_PATH,
                                                METADATA_PATH,
                                                ML_PROCESSING_PATH,
                                                num_workers=7,
                                                transforms=trans,
                                                split_method=SPLIT_METHOD,
                                                cast_to_classification=True,
                                                targets=TARGET,
                                                )
        
        datamodule.setup('fit')
    elif type(model) is MultinomialLogisticClassifier:
        trans =  transforms.Compose([
                                    BinSpectrum(10, 3_000, 20_000),
                                    SquareRootTransform(),
                                    NormalizeIntensity(),
                        ])
        logger = TensorBoardLogger(log_dir, name=f'Multinomial_Logistic_Classifier/{TARGET}')
        datamodule = SingleSpectrum_DataModule(SPECTRA_PATH,
                                                METADATA_PATH,
                                                ML_PROCESSING_PATH,
                                                num_workers=7,
                                                transforms=trans,
                                                # split_method='species',
                                                split_method=SPLIT_METHOD,
                                                cast_to_classification=True,
                                                targets=TARGET,
                                                )
        datamodule.setup('fit')
        # train_set_stats = datamodule.calculate_transformed_train_stats
        # train_set_stats = { # Trainer moves model to GPU, so we need to move the stats too
        #     'mean': train_set_stats['mean'].float().cuda(),
        #     'std': train_set_stats['std'].float().cuda(),
        # }
        # model.set_train_set_stats(train_set_stats)
    elif type(model) is PrototyicalTransformer:
        if prototypical_transformer_hyperparameters['encoder'] != 'transformer':
            raise ValueError("PrototyicalTransformer only supports transformer encoder")
        trans =  transforms.Compose([
                                SelectMassRange(3_000, 20_000),
                                SquareRootTransform(),
                                SelectTopKPeaks(150),
                                NormalizeIntensity(),
                                PadToLength(150, padding_value=-1.0),
                                ])
        datamodule = EpisodicDatamodule(n_classes=5,
                                        n_support_samples=5,
                                        n_query_samples=5,
                                        episodes_per_epoch=1000,
                                        preprocessing_dir=SPECTRA_PATH,
                                        metadata_table=METADATA_PATH,
                                        root_dir=ML_PROCESSING_PATH,
                                        num_workers=7,
                                        transforms=trans,
                                        batch_size=BATCH_SIZE,
                                        split_method=SPLIT_METHOD,
                                        targets=TARGET,
                                        cast_to_classification=True)
        
        logger = TensorBoardLogger(log_dir, name=f'Prototyical_Transformer/{TARGET}')
        datamodule.setup('fit')
                    
                                               
    else:
        raise ValueError("Model type not recognized")

    # Add global hyperparameters (e.g., n_epochs, )
    # logger.log_hyperparams({
    #     'n_epochs': N_EPOCHS,
    #     'batch_size': BATCH_SIZE,
    #     'target': TARGET,
    #     'split_method': SPLIT_METHOD,
    #     'METADATA_PATH': METADATA_PATH,
    #     'ML_PROCESSING_PATH': ML_PROCESSING_PATH,
    #     'SPECTRA_PATH': SPECTRA_PATH,
    # })

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
        max_epochs=N_EPOCHS, 
        log_every_n_steps=1, 
        logger=logger, 
        devices=[0],
        # callbacks=[early_stop_callback]
        profiler=None#PyTorchProfiler(),
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