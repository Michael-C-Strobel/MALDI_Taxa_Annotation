from models.mlp import MLP
from models.mlp_classifier import MLPClassifier
from models.Sentence_MALDI import Sentence_MALDI
from models.CLIP_MALDI import CLIP_MALDI
from models.mlp_binary_classifier import MLPBinaryClassifier
from models.CLIP_MALDI_classifier import CLIP_MALDI_Classifier
from models.logistic_regression_classifier import MultinomialLogisticClassifier
from models.prototypical_transformer import PrototyicalTransformer as PrototypicalTransformer
from datamodule import Spectrum_DataModule, SingleSpectrum_DataModule
from datamodule_triplet import Triplet_DataModule
from prototypical_datamodule import EpisodicDatamodule
from clip_datamodule import CLIP_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from optuna.pruners import MedianPruner
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import Trainer
import shutil
import os
from typing import Dict, Any, Union

from torchvision import transforms
from custom_transforms import *
import tempfile
import optuna
import argparse

from train import DelayedCheckpoint

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Defines the objective function for Optuna to optimize."""
    GRAD_CLIP_VAL = None
    trial_hyperparameters: Dict[str, Any] = {
        'TARGET': args.target,
        'N_EPOCHS': args.n_epochs,
        'batch_size': args.batch_size,
    }

    if args.dataset == 'idbac':
        SPECTRA_PATH = '../../data/idbac_db/preprocessing'
        METADATA_PATH = '../../data/idbac_db/raw/ammended_db.csv'
        ML_PROCESSING_PATH = '../../data/idbac_db/processed_data'
        log_dir_base = './lightning_logs_idbac_optuna'
    elif args.dataset == 'driams':
        SPECTRA_PATH = '../../data/driams/preprocessing'
        METADATA_PATH = '../../data/driams/preprocessing/merged_metadata.csv'
        if args.split_method == 'species_even':
            METADATA_PATH = '../../data/driams/preprocessing/merged_metadata_code_accessions.csv'
        ML_PROCESSING_PATH = '../../data/driams/processed_data'
        log_dir_base = './lightning_logs_DRIAMS_A_optuna'
    else:
        raise ValueError(f"Dataset '{args.dataset}' not recognized.")

    log_dir = os.path.join(log_dir_base, f'{args.target}/{args.split_method}/{args.model_type}')

    model = None
    datamodule = None
    logger_name = None
    trans = None

    clip_common_params = {
            'input_dim': 1700,  # Fixed
            'output_bin_edges': torch.Tensor([1.0]),  # Fixed
            'padding_value': -1.0,  # Fixed
            'encoder': 'transformer',  # Fixed for classifier
            'ss_task': None,
            'rcon_head_dim': 1700,  # Fixed
            'warmup_steps': 2000,  # Fixed
        }

    if args.model_type == 'MLPClassifier':
        trial_hyperparameters.update(
            {
                'input_dim': 1800,  # Fixed for now
                'output_dim': 250,  # Fixed for now, depends on the target
                'hidden_dim': trial.suggest_int('mlp_hidden_dim', 100, 500),
                'hidden_layers': trial.suggest_int('mlp_hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('mlp_weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('mlp_dropout', 0.1, 0.5),
                'lr': trial.suggest_float('mlp_lr', 1e-6, 1e-3, log=True),
            }
        )
        model = MLPClassifier(trial_hyperparameters)
        trans = transforms.Compose(
            [BinSpectrum(10, 3000, 20000), SquareRootTransform(), NormalizeIntensity()]
        )
        datamodule = SingleSpectrum_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            split_method=args.split_method,
            targets=args.target,
            batch_size=args.batch_size,
        )
        datamodule.setup('fit')
        if hasattr(datamodule, 'full_dataset') and hasattr(datamodule.full_dataset, 'get_one_hot_encoded_classes'):
            model.one_hot_encoder = datamodule.full_dataset.get_one_hot_encoded_classes()
        logger_name = 'MLP_Classifier_Optuna'

    elif args.model_type == 'Sentence_MALDI':
        trial_hyperparameters.update(
            {
                'input_dim': 1700,  # Fixed
                'output_bin_edges': torch.Tensor([1.0]),  # Fixed
                'hidden_dim': trial.suggest_int('sentence_hidden_dim', 100, 500),
                'hidden_layers': trial.suggest_int('sentence_hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float(
                    'sentence_weight_decay', 1e-6, 1e-3, log=True
                ),
                'dropout': trial.suggest_float('sentence_dropout', 0.1, 0.5),
                'tau': trial.suggest_float('sentence_tau', 0.5, 2.0),
            }
        )
        model = Sentence_MALDI(trial_hyperparameters)
        trans = transforms.Compose(
            [BinSpectrum(10, 3000, 20000), SquareRootTransform(), NormalizeIntensity()]
        )
        datamodule = Triplet_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            split_method=args.split_method,
            targets=args.target,
            batch_size=args.batch_size,
        )
        datamodule.setup('fit')
        logger_name = 'Sentence_MALDI_Optuna'

    elif args.model_type == 'CLIP_MALDI_Classifier':
        trial_hyperparameters.update(clip_common_params)
        
        trial_hyperparameters.update({
                'hidden_dim': trial.suggest_int('clip_hidden_dim', 100, 500, step=10),
                'hidden_layers': trial.suggest_int('clip_hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('clip_weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('clip_dropout', 0.1, 0.5),
                'tau': trial.suggest_float('clip_tau', 0.05, 0.15),
                'lr': trial.suggest_float('clip_lr', 1e-5, 1e-3, log=True),
            })

        if args.target == 'genera' and 'idbac' in SPECTRA_PATH:
            trial_hyperparameters['n_classes'] = 96
        elif args.target == 'genera' and 'driams' in SPECTRA_PATH:
            trial_hyperparameters['n_classes'] = 182
        elif args.target == 'species' and 'driams' in SPECTRA_PATH:
            trial_hyperparameters['n_classes'] = 723
        else:
            raise ValueError(f"TARGET must be 'genera' or 'species.' Got {args.target} instead.")

        model = CLIP_MALDI_Classifier(trial_hyperparameters)
        trans = transforms.Compose(
            [
                SelectMassRange(3000, 20000),
                SquareRootTransform(),
                NormalizeIntensity(),
                SelectTopKPeaks(150),
                PadToLength(150, padding_value=-1.0),
            ]
        )
        datamodule = SingleSpectrum_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            split_method=args.split_method,
            cast_to_classification=True,
            targets=args.target,
            batch_size=args.batch_size,
        )
        datamodule.setup('fit')
        GRAD_CLIP_VAL = 0.5
        logger_name = 'CLIP_Transformer_Classifier_Optuna'

    elif args.model_type == 'MultinomialLogisticClassifier':
        trial_hyperparameters.update(
            {
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'spectrum_bin_width': trial.suggest_int('spectrum_bin_width', 1,10, step=1),
            }
        )
        trial_hyperparameters.update({
                'input_dim': (20_000 - 3000) // trial_hyperparameters['spectrum_bin_width'],
        })
        if args.target == 'genera' and 'idbac' in SPECTRA_PATH:
            trial_hyperparameters['n_classes'] = 96
        elif args.target == 'genera' and 'driams' in SPECTRA_PATH:
            trial_hyperparameters['n_classes'] = 182
        elif args.target == 'species' and 'driams' in SPECTRA_PATH:
            trial_hyperparameters['n_classes'] = 723
        else:
            raise ValueError(f"TARGET must be 'genera' or 'species.' Got {args.target} instead.")

        model = MultinomialLogisticClassifier(**trial_hyperparameters)
        trans = transforms.Compose(
            [BinSpectrum(trial_hyperparameters['spectrum_bin_width'], 3000, 20_000), 
             SquareRootTransform(), 
             NormalizeIntensity()]
        )
        datamodule = SingleSpectrum_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            split_method=args.split_method,
            cast_to_classification=True,
            targets=args.target,
            batch_size=args.batch_size,
        )
        datamodule.setup('fit')
        logger_name = 'Multinomial_Logistic_Classifier_Optuna'
    elif args.model_type == 'PrototypicalTransformer':
        trial_hyperparameters.update(clip_common_params)
        proto_hyperparams = {
            'hidden_dim': trial.suggest_int('clip_hidden_dim', 100, 500, step=10),
            'hidden_layers': trial.suggest_int('clip_hidden_layers', 2, 5),
            'weight_decay': trial.suggest_float('clip_weight_decay', 1e-6, 1e-3, log=True),
            'dropout': trial.suggest_float('clip_dropout', 0.1, 0.5),
            'tau': trial.suggest_float('clip_tau', 0.05, 0.15),
            'lr': trial.suggest_float('clip_lr', 1e-5, 1e-3, log=True),
            'n_classes': trial.suggest_int('proto_n_classes', 3, 5, step=1),    # Must be very low to accomidate IDBac
            'n_support_samples': 1,  # Fixed to mimic single-shot learning
            'n_query_samples': 5,    # Fixed
            'episodes_per_epoch': 1000,  # Fixed for now
        }
        trial_hyperparameters.update(proto_hyperparams)
        model = PrototypicalTransformer(trial_hyperparameters['n_classes'],
                                        trial_hyperparameters['n_support_samples'],
                                        trial_hyperparameters['n_query_samples'],
                                        trial_hyperparameters)
        trans = transforms.Compose([
            SelectMassRange(3_000, 20_000),
            SquareRootTransform(),
            SelectTopKPeaks(150),
            NormalizeIntensity(),
            PadToLength(150, padding_value=-1.0),
        ])
        datamodule = EpisodicDatamodule(
            n_classes=trial_hyperparameters['n_classes'],
            n_support_samples=trial_hyperparameters['n_support_samples'],
            n_query_samples=trial_hyperparameters['n_query_samples'],
            episodes_per_epoch=trial_hyperparameters['episodes_per_epoch'],
            preprocessing_dir=SPECTRA_PATH,
            metadata_table=METADATA_PATH,
            root_dir=ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            batch_size=args.batch_size,
            split_method=args.split_method,
            targets=args.target,
            cast_to_classification=True,
        )
        datamodule.setup('fit')
        logger_name = 'Prototyical_Transformer_Optuna'

    elif args.model_type == 'CLIP_MALDI':
        trial_hyperparameters.update(clip_common_params)

        trial_hyperparameters.update({
                'hidden_dim': trial.suggest_int('clip_hidden_dim', 100, 500, step=10),
                'hidden_layers': trial.suggest_int('clip_hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('clip_weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('clip_dropout', 0.1, 0.5),
                'tau': trial.suggest_float('clip_tau', 0.05, 0.15),
                'lr': trial.suggest_float('clip_lr', 1e-5, 1e-3, log=True),
            })
        
        trial_hyperparameters.update(trial_hyperparameters)
        model = CLIP_MALDI(trial_hyperparameters)
        trans = transforms.Compose([
            SquareRootTransform(),
            SelectTopKPeaks(150),
            NormalizeIntensity(),
            PadToLength(150, padding_value=-1.0),
        ])
        datamodule = CLIP_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            batch_size=args.batch_size,
            split_method=args.split_method,
            targets=args.target,
        )
        datamodule.setup('fit')
        logger_name = f'CLIP_Transformer_Optuna'

    else:
        raise ValueError(f"Model type '{args.model_type}' is not supported for tuning.")

    logger = TensorBoardLogger(log_dir, name=logger_name, version=trial.number)
    checkpoint_dir = os.path.join(
            logger.log_dir,
            "checkpoints"
        )
    checkpoint_callback = DelayedCheckpoint(
                            delay_epochs=int(0.33 * args.n_epochs),
                            monitor='val_loss_epoch',
                            save_top_k=1,
                            mode='min',
                            dirpath=checkpoint_dir, #log_dir + f'/CLIP_Transformer_Classifier/' '/checkpoints',
                            filename='best-checkpoint'
                        )
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss_epoch")

    callbacks = [checkpoint_callback, pruning_callback]

    trainer = Trainer(
        max_epochs=args.n_epochs,
        log_every_n_steps=1,
        logger=logger,
        devices=[0],
        callbacks=callbacks,
        gradient_clip_val=GRAD_CLIP_VAL,
    )

    trainer.fit(model, datamodule, ckpt_path=None) # Train for one epoch


    return trainer.callback_metrics['val_loss_epoch'].item()


def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for a specific model and dataset using Optuna.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            'MLPClassifier',
            'Sentence_MALDI',
            'CLIP_MALDI_Classifier',
            'MultinomialLogisticClassifier',
            'PrototypicalTransformer',
            'CLIP_MALDI',
        ],
        help="The type of model to tune.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=['idbac', 'driams'],
        help="The dataset to use for tuning ('idbac' or 'driams').",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="The number of Optuna trials to run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size to use for training.",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=2000,
        help="The maximum number of epochs for training each trial.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default='genera',
        choices=['genera', 'species'],
        help="The target variable for classification ('genera' or 'species').",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default='species',
        choices=['genera', 'species', 'species_even'],
        required=True,
        help="The method used to split the data ('genera', 'species', or 'species_even').",
    )
    args = parser.parse_args()

    # Save sqlite of study to log folder
    log_dir = os.path.join(
        './lightning_logs_idbac_optuna',
        f'{args.target}/{args.split_method}/{args.model_type}',
    )
    os.makedirs(log_dir, exist_ok=True)
    sqlite_path = f"sqlite:///{os.path.abspath(os.path.join(log_dir, 'optuna_study.db'))}"
    


    study = optuna.create_study(direction='minimize', 
                                study_name="optuna_study", 
                                pruner=MedianPruner(), 
                                storage=sqlite_path, 
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print(f"Best trial for {args.model_type} on {args.dataset} (Target: {args.target}, Split: {args.split_method}):")
    trial = study.best_trial
    print("  Value (Validation Loss): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Now you can use the best hyperparameters from trial.params to train your final model
    # of the specified type on the specified dataset with the chosen target and split method.


if __name__ == "__main__":
    main()