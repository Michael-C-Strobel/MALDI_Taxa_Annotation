from models.mlp import MLP
from models.mlp_classifier import MLPClassifier
from models.Sentence_MALDI import Sentence_MALDI
from models.CLIP_MALDI import CLIP_MALDI
from models.mlp_binary_classifier import MLPBinaryClassifier
from models.CLIP_MALDI_classifier import CLIP_MALDI_Classifier
from models.Cross_Encoder import Cross_Encoder
from models.logistic_regression_classifier import MultinomialLogisticClassifier
from models.prototypical_transformer import PrototyicalTransformer as PrototypicalTransformer
from models.Transformer_MultiLoss import Transformer_MulitLoss
from models.MaldiTransformer.MaldiTransformerWrapper import MaldiTransformerWrapper
from models.binary_transformer_embedding_prediction_head import BinaryTransformerPredictionHead
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
from pathlib import Path

from torchvision import transforms
from custom_transforms import *
import optuna
from optuna.trial import TrialState
import argparse
import json

import time

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
        if args.maldi_nn_preprocessing:
            raise ValueError("maldi_nn_preprocessing is not supported for IDBac dataset.")
        SPECTRA_PATH = '../../data/idbac_db/preprocessing/spectra'
        METADATA_PATH = '../../data/idbac_db/raw/ammended_db.csv'
        METADATA_PATH = '../../data/idbac_db/preprocessing/db_with_taxonomy.csv'
        ML_PROCESSING_PATH = '../../data/idbac_db/processed_data'
        if args.train_for_score:
            log_dir = os.path.join('./lightning_logs_idbac_for_score', f'{args.target}/{args.split_method}') # /{args.model_type}?
        else:
            log_dir=  os.path.join('./lightning_logs_idbac_optuna', f'{args.target}/{args.split_method}') # /{args.model_type}?
    elif dataset == 'driams':            
        SPECTRA_PATH = '../../data/driams/processed_data/spectra'
        if args.maldi_nn_preprocessing:
            print("**Using MALDI-Transformer preprocessing for DRIAMS dataset.**")
            SPECTRA_PATH = '../../data/driams/processed_data/MaldiTransformer/spectra'
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
    elif args.model_type == 'Transformer_MulitLoss':
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
    elif args.model_type == 'BinaryTransformerPredictionHead':
        raise ValueError("BinaryTransformerPredictionHead is not supported for tuning.")
    elif args.model_type == 'Cross_Encoder':
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
                     log_dir: str,
                     trial: optuna.Trial=None,
                     hparam_path: str=None
                     ) -> L.LightningModule:
    model = None
    datamodule = None
    logger_name = None
    trans = None
    trainer_args = {}

    parameter_free_methods = ('MaldiTransformerWrapper', 'BinaryTransformerPredictionHead', 'MaldiTransformerWrapperMethodData')

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
            'concat_pos': True,
        }

    # If we're training for score, load them and don't change them
    if args.train_for_score:
        if args.model_type not in parameter_free_methods:
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
            k=args.k
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
            trial_hyperparameters['n_classes'] = 233
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
            k=args.k
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
        if args.pretrained_model_path: trial_hyperparameters['lr'] = 1e-6

        model = CLIP_MALDI(trial_hyperparameters)
        if args.pretrained_model_path:
            if not os.path.exists(args.pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {args.pretrained_model_path}")
            print("Loading pretrained model from:", args.pretrained_model_path)
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=model.device)['state_dict'])
            # Freeze the first half of layers
            params = list(model.named_parameters())
            # print(params)
            # num_to_freeze = int(0.75 * len(params))
            num_to_freeze = len(params)-4
            # print('num_to_freeze', num_to_freeze)

            for name, param in params[:num_to_freeze]:
                # If it's a norm or bias term, don't freeze it
                # if 'norm' in name.lower() or 'bias' in name.lower():
                #     print(f"Not freezing (norm/bias): {name}")
                #     continue
                param.requires_grad = False
                print(f"Froze: {name}")

            for name, param in params[num_to_freeze:]:
                print(f"Trainable: {name}")


        # print("*******************************")
        # print("*******************************")
        # print("*******************************")
        # print("Warning: Binarizing Intensities")
        # print("*******************************")
        # print("*******************************")
        # print("*******************************")
        # time.sleep(5)
        # trans = transforms.Compose([
        #         SelectMassRange(3000, 20000),
        #         L1NormalizeIntensity(),
        #         SelectTopKPeaks(150),
        #         PadToLength(150, padding_value=-1.0),
        #     ]
        # )
        trans = transforms.Compose([
                SelectMassRange(3000, 20000),
                NormalizeIntensity(),
                SelectTopKPeaks(150),
                PadToLength(150, padding_value=-1.0),
            ])
        print("*******************************")
        print("Got k = ", args.k)
        print("*******************************")
        datamodule = CLIP_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            batch_size=args.batch_size,
            split_method=args.split_method,
            targets=args.target,
            k=args.k,
            prefer_hard=True,
        )
        datamodule.setup('fit')
        logger_name = f'CLIP_Transformer'
    elif args.model_type == 'Transformer_MulitLoss':
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
        
        model = Transformer_MulitLoss(trial_hyperparameters)

        print("*******************************")
        print("*******************************")
        print("*******************************")
        print("Warning: Binarizing Intensities")
        print("*******************************")
        print("*******************************")
        print("*******************************")
        # time.sleep(5)

        trans = transforms.Compose([
            SquareRootTransform(),
            SelectTopKPeaks(150),
            BinarizeIntensity(), # *************
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
            k=args.k,
            cast_to_classification=True,
            )
    
        trainer_args = {
            'gradient_clip_val': 0.5,
        }

        datamodule.setup('fit')
        logger_name = f'Transformer_MulitLoss'
    elif args.model_type == "MaldiTransformerWrapper" or \
            args.model_type == "MaldiTransformerWrapperMethodData":
        modeltype = MaldiTransformerWrapper

        if args.target == 'genera' and 'idbac' in SPECTRA_PATH:
            n_classes = 96
        elif args.target == 'genera' and 'driams' in SPECTRA_PATH:
            n_classes = 182
        elif args.target == 'species' and 'driams' in SPECTRA_PATH:
            n_classes = 723
        else:
            raise ValueError(f"TARGET must be 'genera' or 'species.' Got {args.target} instead.")
        
        size_to_layer_dims = {
            "S": [160, 4],
            "M": [184, 6],
            "L": [232, 8],
            "XL": [304, 10],
        }
            
        model_kwargs = {
            "n_classes": n_classes,
            "n_heads": 8,
            "dropout": 0.2,
            "p": 0.15,                  # depends on model size
            "clf": True,
            "clf_train_p": 1 / 100,        # depends on model size
            "lmbda": 1.0,               # depends on model size
            "lr": 0.0005,               # depends on model size
            "weight_decay": 0,
            "lr_decay_factor": 1,
            "warmup_steps": 2500,
            "proportional": False,      # Disabled by default
        }
        
        spectrum_embedder_size =  "M"   # "S", "M", "L", "XL", "M" metrics are reported in manuscript
        model = modeltype(
            size_to_layer_dims[spectrum_embedder_size][1],
            size_to_layer_dims[spectrum_embedder_size][0],
            **model_kwargs,
        )

        trainer_args = {
            'gradient_clip_val': 1.0,
            'precision': "bf16-mixed",
            'val_check_interval': 5_000,
            'check_val_every_n_epoch': None
        }

        trans = transforms.Compose([
                    L1NormalizeIntensity(),
                    SelectTopKPeaks(200),
                    PadToLength(200, padding_value=-1.0),
                ])

        datamodule = SingleSpectrum_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            split_method=args.split_method,
            targets=args.target,
            batch_size=args.batch_size,
            cast_to_classification=True,
            k=args.k,  # k-fold
            balance='strain',
        )
        datamodule.setup('fit')

        logger_name = 'MaldiTransformerWrapper'
        if args.model_type == "MaldiTransformerWrapperMethodData":
            logger_name = 'MaldiTransformerWrapperMethodData'
            assert args.maldi_nn_preprocessing, "maldi_nn_preprocessing must be True for MaldiTransformerWrapperMethodData"
    elif args.model_type == 'BinaryTransformerPredictionHead':
        trial_hyperparameters.update(clip_common_params)
        if not args.train_for_score:
            raise ValueError("BinaryTransformerPredictionHead is not supported for tuning.")
        
        trial_hyperparameters.update(trial_hyperparameters)
        # Use the most releveant CLIP_MALDI weights based on log path
        trial_hyperparameters['encoder_path'] = os.path.join(
            log_dir,
            'CLIP_Transformer',
            'version_0',    # Hardcoded for now, but should be the same for all trials
            'checkpoints',
            'best-checkpoint.ckpt'
        )

        print("Got log_dir = ", log_dir)
        print("Got encoder_path = ", trial_hyperparameters['encoder_path'])

        if not os.path.exists(trial_hyperparameters['encoder_path']):
            raise FileNotFoundError(f"No encoder found at {trial_hyperparameters['encoder_path']}")
        
        # Fixed hyperparameters
        trial_hyperparameters['mlp_hidden_dim'] = 512
        trial_hyperparameters['mlp_hidden_layers'] = 2

        model = BinaryTransformerPredictionHead(trial_hyperparameters)

        print("*******************************")
        print("*******************************")
        print("*******************************")
        print("Warning: Binarizing Intensities")
        print("*******************************")
        print("*******************************")
        print("*******************************")
        time.sleep(5)
        trans = transforms.Compose([
            SquareRootTransform(),
            SelectTopKPeaks(150),
            BinarizeIntensity(),  # *************
            NormalizeIntensity(),
            PadToLength(150, padding_value=-1.0),
        ])

        datamodule = Triplet_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            split_method=args.split_method,
            cast_to_classification=False,   # Not required
            targets=args.target,
            batch_size=args.batch_size,
            k=args.k,
        )
        datamodule.setup('fit')
        logger_name = f'BinaryTransformerPredictionHead'

    elif args.model_type == "Cross_Encoder":
        trial_hyperparameters.update(clip_common_params)
        if not args.train_for_score:
            trial_hyperparameters = get_trial_hyperparameters(trial, args, trial_hyperparameters)
        
        trial_hyperparameters.update(trial_hyperparameters)
        if args.pretrained_model_path: trial_hyperparameters['lr'] = 1e-6

        model = Cross_Encoder(trial_hyperparameters)
        if args.pretrained_model_path:
            if not os.path.exists(args.pretrained_model_path):
                raise FileNotFoundError(f"No pretrained model found at {args.pretrained_model_path}")
            print("Loading pretrained model from:", args.pretrained_model_path)
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location=model.device)['state_dict'])
            # Freeze the first half of layers
            params = list(model.named_parameters())
            # print(params)
            num_to_freeze = int(0.75 * len(params))
            # print('num_to_freeze', num_to_freeze)

            for name, param in params[:num_to_freeze]:
                param.requires_grad = False
                print(f"Froze: {name}")

        trans = transforms.Compose([
                SelectMassRange(3000, 20000),
                NormalizeIntensity(),
                SelectTopKPeaks(150),
                PadToLength(150, padding_value=-1.0),
            ])
        print("*******************************")
        print("Got k = ", args.k)
        print("*******************************")
        datamodule = CLIP_DataModule(
            SPECTRA_PATH,
            METADATA_PATH,
            ML_PROCESSING_PATH,
            num_workers=7,
            transforms=trans,
            batch_size=args.batch_size,
            split_method=args.split_method,
            targets=args.target,
            k=args.k,
            prefer_hard=True,
        )
        datamodule.setup('fit')
        logger_name = f'Cross_Encoder'
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
        log_dir,
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

    if args.k is not None:
        log_dir = os.path.join(log_dir, f'k={args.k}')
    
    model, datamodule, logger_name, trans, trainer_args = initialize_model(
        spectra_path,
        metadata_path,
        ml_processing_path,
        args,
        log_dir,
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

    if args.cpu:
        devices = 1
        accelerator = 'cpu'
    else:
        devices = [0]
        accelerator = 'gpu'

    trainer = Trainer(
        max_epochs=args.n_epochs,
        max_steps=args.n_steps,
        log_every_n_steps=1,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
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
            'Transformer_MulitLoss',
            'BinaryTransformerPredictionHead',
            'MaldiTransformerWrapper',
            'MaldiTransformerWrapperMethodData',
            'Cross_Encoder',
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
        help="The maximum number of epochs for training each trial. Specify either n_epochs or n_steps (and set n_epochs=-1)",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=-1,
        help="The number of steps to train each trial. If -1, automatically calculated."
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
        choices=['genera', 'species', 'species_even', 'genera_holdout'],
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
    parser.add_argument(
        "--k",
        "-k",
        type=int,
        help="Which k-fold to use for training.",
        default=None,
        required=False
    )
    parser.add_argument(
        "--maldi_nn_preprocessing",
        help="Use MALDI-Transformer preprocessing for DRIAMS dataset.",
        action='store_true',
    )
    parser.add_argument(
        "--pretrained_model_path",
        help="Path to a pretrained model to use for fine-tuning.",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        action='store_true',
        help="Use CPU for training instead of GPU.",
    )

    args = parser.parse_args()

    for var, val in vars(args).items():
        print(f"{var}: {val}")

    if args.n_epochs != -1 and args.n_steps != -1:
        raise ValueError("Cannot specify both n_epochs and n_steps. Use n_epochs=-1 to specify n_steps.")

    # Save sqlite of study to log folder
    log_dir = os.path.join(
        f'./lightning_logs_{args.dataset}_optuna',
        f'{args.target}/{args.split_method}/{args.model_type}',
    )
    os.makedirs(log_dir, exist_ok=True)
    sqlite_path = f"sqlite:///{os.path.abspath(os.path.join(log_dir, 'optuna_study.db'))}"

    # For optimal params
    if os.path.isfile(args.hparam_dir):
        hparam_path = args.hparam_dir
    else:
        hparam_path = Path(log_dir) / f"{args.model_type}.json" #os.path.join(args.hparam_dir, args.dataset, args.target, args.split_method, f"{args.model_type}.json")
    print(f"Hyperparameter path: {hparam_path}")

    if not args.train_for_score:
        print(f"Saving best hyperparameters to {hparam_path}")

        study = optuna.create_study(direction='minimize', 
                                    study_name="optuna_study", 
                                    pruner=MedianPruner(n_warmup_steps=int(0.1 * args.n_epochs)), 
                                    storage=sqlite_path, 
                                    load_if_exists=True)
        curr_num_trials = len(study.get_trials(states=[TrialState.COMPLETE, TrialState.PRUNED]))

        remaining_trials = args.n_trials-curr_num_trials

        print(f"Number of existing trials in the study: {curr_num_trials}")
        print(f"Number of remaining trials to run: {remaining_trials}")
        
        if not args.dump_best:
            study.optimize(lambda trial: objective(trial, args), n_trials=remaining_trials)

        print("Number of finished trials: ", len(study.trials))
        print(f"Best trial for {args.model_type} on {args.dataset} (Target: {args.target}, Split: {args.split_method}):")
        all_trials = sorted(study.get_trials(deepcopy=False), key=lambda t: t.number)
        first_complete = [
            t for t in all_trials if t.state in [TrialState.COMPLETE]   # No need to check pruned
        ]
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            best_from_subset = min(first_complete, key=lambda t: t.value)
        else:
            best_from_subset = max(first_complete, key=lambda t: t.value)
        best_trial = best_from_subset
        print("  Value (Validation Loss): ", best_trial.value)
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        best_params = best_trial.params.copy()
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