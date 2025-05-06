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
import json

class DelayedCheckpoint(ModelCheckpoint):
    def __init__(self, delay_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_epochs = delay_epochs

    def on_validation_end(self, trainer, pl_module):
        # Save only after the delay
        if trainer.current_epoch >= self.delay_epochs:
            super().on_validation_end(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # Always save final model at end of training
        filepath = self.format_checkpoint_name(
            metrics=trainer.callback_metrics,
            filename=f"last_epoch={trainer.current_epoch}"
        )

        # Ensure directory exists
        self._fs.makedirs(self.dirpath, exist_ok=True)

        # Save the model
        self._save_checkpoint(trainer, filepath)


def get_path_info(dataset: str, args: argparse.Namespace) -> Union[str, str, str, str]:
    """Returns the paths for the dataset and logs based on the dataset name."""
    if dataset == 'idbac':
        SPECTRA_PATH = '../../data/idbac_db/preprocessing'
        METADATA_PATH = '../../data/idbac_db/raw/ammended_db.csv'
        ML_PROCESSING_PATH = '../../data/idbac_db/processed_data'
        if args.train_for_score:
            log_dir = os.path.join('./lightning_logs_idbac_for_score', f'{args.target}/{args.split_method}') # /{args.model_type}?
        else:
            log_dir=  os.path.join('./lightning_logs_idbac_optuna', f'{args.target}/{args.split_method}') # /{args.model_type}?
    elif dataset == 'driams':
        SPECTRA_PATH = '../../data/driams/preprocessing'
        METADATA_PATH = '../../data/driams/preprocessing/merged_metadata.csv'
        if args.split_method == 'species_even':
            METADATA_PATH = '../../data/driams/preprocessing/merged_metadata_code_accessions.csv'
        ML_PROCESSING_PATH = '../../data/driams/processed_data'
        if args.train_for_score:
            log_dir = os.path.join('./lightning_logs_DRIAMS_A_for_score', f'{args.target}/{args.split_method}') # /{args.model_type}?
        else:
            log_dir = os.path.join('./lightning_logs_DRIAMS_A_optuna', f'{args.target}/{args.split_method}') # /{args.model_type}?
    else:
        raise ValueError(f"Dataset '{dataset}' not recognized.")
    
    return SPECTRA_PATH, METADATA_PATH, ML_PROCESSING_PATH, log_dir

def get_trial_hyperparameters(trial: optuna.Trial, args: argparse.Namespace, trial_hyperparameters) -> Dict[str, Any]:
    """Suggests hyperparameters for the trial."""
    if args.model_type == 'MLPClassifier':
        trial_hyperparameters.update(
            {
                'input_dim': 1800,  # Fixed for now
                'output_dim': 250,  # Fixed for now, depends on the target
                'hidden_dim': trial.suggest_int('hidden_dim', 100, 500),
                'hidden_layers': trial.suggest_int('hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'lr': trial.suggest_float('lr', 1e-6, 1e-3, log=True),
            }
        )
    elif args.model_type == 'Sentence_MALDI':
        trial_hyperparameters.update(
            {
                'input_dim': 1700,  # Fixed
                'output_bin_edges': torch.Tensor([1.0]),  # Fixed
                'hidden_dim': trial.suggest_int('hidden_dim', 100, 500),
                'hidden_layers': trial.suggest_int('hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float(
                    'weight_decay', 1e-6, 1e-3, log=True
                ),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'tau': trial.suggest_float('tau', 0.5, 2.0),
            }
        )
    elif args.model_type == 'CLIP_MALDI_Classifier':
        trial_hyperparameters.update({
                'hidden_dim': trial.suggest_int('hidden_dim', 100, 500, step=10),
                'hidden_layers': trial.suggest_int('hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'tau': trial.suggest_float('tau', 0.05, 0.15),
                'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            })
        if args.target == 'genera' and 'idbac' in args.dataset:
            trial_hyperparameters['n_classes'] = 96
        elif args.target == 'genera' and 'driams' in args.dataset:
            trial_hyperparameters['n_classes'] = 182
        elif args.target == 'species' and 'driams' in args.dataset:
            trial_hyperparameters['n_classes'] = 723
        else:
            raise ValueError(f"TARGET must be 'genera' or 'species.' Got {args.target} instead.")
    elif args.model_type == 'MultinomialLogisticClassifier':
        trial_hyperparameters.update(
            {
                'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'spectrum_bin_width': trial.suggest_int('spectrum_bin_width', 1,10, step=1),
            }
        )
        return trial_hyperparameters
    elif args.model_type == 'PrototypicalTransformer':
        trial_hyperparameters.update(
            {
                'hidden_dim': trial.suggest_int('hidden_dim', 100, 500, step=10),
                'hidden_layers': trial.suggest_int('hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'tau': trial.suggest_float('tau', 0.05, 0.15),
                'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
                'n_classes': trial.suggest_int('proto_n_classes', 3, 5, step=1),    # Must be very low to accomidate IDBac
            }
        )
    elif args.model_type == 'CLIP_MALDI':
        trial_hyperparameters.update(
            {
                'hidden_dim': trial.suggest_int('hidden_dim', 100, 500, step=10),
                'hidden_layers': trial.suggest_int('hidden_layers', 2, 5),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'tau': trial.suggest_float('tau', 0.05, 0.15),
                'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            }
        )
    else:
        raise ValueError(f"Model type '{args.model_type}' is not supported for tuning.")
    
    return trial_hyperparameters

def initialize_model(SPECTRA_PATH: str,
                     METADATA_PATH: str,
                     ML_PROCESSING_PATH: str,
                     args: argparse.Namespace,
                     trial: optuna.Trial=None,
                     hparam_path: str=None
                     ) -> L.LightningModule:
    model = None
    datamodule = None
    logger_name = None
    trans = None
    trainer_args = {}

    trial_hyperparameters: Dict[str, Any] = {
        'TARGET': args.target,
        'N_EPOCHS': args.n_epochs,
        'batch_size': args.batch_size,
    }

    clip_common_params = {
            'input_dim': 1700,  # Fixed
            'output_bin_edges': torch.Tensor([1.0]),  # Fixed
            'padding_value': -1.0,  # Fixed
            'encoder': 'transformer',  # Fixed for classifier
            'ss_task': None,
            'rcon_head_dim': 1700,  # Fixed
            'warmup_steps': 2000,  # Fixed
        }

    # If we're training for score, load them and don't change them
    if args.train_for_score:
        if not os.path.exists(hparam_path):
            raise FileNotFoundError(f"No hyperparameters found at {hparam_path}")
        print("Loading hyperparameters from file:", hparam_path)
        with open(hparam_path, 'r', encoding='utf-8') as f:
            loaded_hparams = json.load(f)
            print("Loaded hyperparameters:", loaded_hparams)
        trial_hyperparameters.update(loaded_hparams)

    if args.model_type == 'MLPClassifier':
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)
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
        logger_name = 'MLP_Classifier'

    elif args.model_type == 'Sentence_MALDI':
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)
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
        logger_name = 'Sentence_MALDI'

    elif args.model_type == 'CLIP_MALDI_Classifier':
        trial_hyperparameters.update(clip_common_params)
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)

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
        logger_name = 'CLIP_Transformer_Classifier'
        trainer_args = {
            'gradient_clip_val': 0.5,
        }

    elif args.model_type == 'MultinomialLogisticClassifier':
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)
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
        logger_name = 'Multinomial_Logistic_Classifier'
    elif args.model_type == 'PrototypicalTransformer':
        trial_hyperparameters.update(clip_common_params)
        # Set defaults for n_support_samples and n_query_samples
        trial_hyperparameters['n_support_samples'] = 1
        trial_hyperparameters['n_query_samples'] = 5
        trial_hyperparameters['episodes_per_epoch'] = 1000
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)
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
        logger_name = 'Prototyical_Transformer'

    elif args.model_type == 'CLIP_MALDI':
        trial_hyperparameters.update(clip_common_params)
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)
        
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
        logger_name = f'CLIP_Transformer'

    else:
        raise ValueError(f"Model type '{args.model_type}' is not supported for tuning.")
    
    return model, datamodule, logger_name, trans, trainer_args

def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """Defines the objective function for Optuna to optimize."""
    if args.train_for_score:
        raise ValueError("train_for_score must be False for Optuna optimization.")
    spectra_path, metadata_path, ml_processing_path, log_dir = get_path_info(args.dataset, args)

    model = None
    datamodule = None
    logger_name = None
    trans = None

    model, datamodule, logger_name, trans, traininer_args = initialize_model(
        spectra_path,
        metadata_path,
        ml_processing_path,
        args,
        trial
    )

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
        **traininer_args,
    )

    trainer.fit(model, datamodule, ckpt_path=None) # Train for one epoch


    return trainer.callback_metrics['val_loss_epoch'].item()

def train_for_score(args: argparse.Namespace, hparam_path: str) -> None:
    """Train the model for the best hyperparameters found by Optuna."""
    if not args.train_for_score:
        raise ValueError("train_for_score must be True to train for final score.")

    spectra_path, metadata_path, ml_processing_path, log_dir = get_path_info(args.dataset, args)
    
    model, datamodule, logger_name, trans, trainer_args = initialize_model(
        spectra_path,
        metadata_path,
        ml_processing_path,
        args,
        None ,   # No trial needed here
        hparam_path,
    )
    logger = TensorBoardLogger(log_dir, name=logger_name)
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
    
    callbacks = [checkpoint_callback]

    trainer = Trainer(
        max_epochs=args.n_epochs,
        log_every_n_steps=1,
        logger=logger,
        devices=[0],
        callbacks=callbacks,
        **trainer_args,
    )

    trainer.fit(model, datamodule, ckpt_path=None) # Train for one epoch


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
    parser.add_argument('--train_for_score',
                        action='store_true',
                        help='Use best hyperparams to train for final score'
    )
    parser.add_argument('--hparam_dir',
                        type=str,
                        default='./optuna_results',
                        help='Directory to store/load hyperparameters'
    )
    parser.add_argument(
        "--dump_best"
        , action='store_true',
        help="Dump the best hyperparameters to a file."
    )

    args = parser.parse_args()

    # Save sqlite of study to log folder
    log_dir = os.path.join(
        './lightning_logs_idbac_optuna',
        f'{args.target}/{args.split_method}/{args.model_type}',
    )
    os.makedirs(log_dir, exist_ok=True)
    sqlite_path = f"sqlite:///{os.path.abspath(os.path.join(log_dir, 'optuna_study.db'))}"

    # For optimal params
    if os.path.isfile(args.hparam_dir):
        hparam_path = args.hparam_dir
    else:
        hparam_path = os.path.join(args.hparam_dir, args.dataset, args.target, args.split_method, f"{args.model_type}.json")

    if not args.train_for_score:
        print(f"Saving best hyperparameters to {hparam_path}")

        study = optuna.create_study(direction='minimize', 
                                    study_name="optuna_study", 
                                    pruner=MedianPruner(), 
                                    storage=sqlite_path, 
                                    load_if_exists=True)
        if not args.dump_best:
            study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

        print("Number of finished trials: ", len(study.trials))
        print(f"Best trial for {args.model_type} on {args.dataset} (Target: {args.target}, Split: {args.split_method}):")
        trial = study.best_trial
        print("  Value (Validation Loss): ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        best_params = study.best_trial.params
        best_params.update({
            'TARGET': args.target,
            'N_EPOCHS': args.n_epochs,
            'batch_size': args.batch_size,
        })

        os.makedirs(os.path.dirname(hparam_path), exist_ok=True)

        with open(hparam_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2)
    else:
        train_for_score(args, hparam_path)


if __name__ == "__main__":
    main()