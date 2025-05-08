import argparse
from pathlib import Path
from typing import List, Tuple

from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from torchvision import transforms
from custom_transforms import *
import numpy as np
from scipy.stats import mode as mode_fn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import logging
import os 

from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.transformer_embedding_prediction_head import TransformerPredictionHead
from models.binary_transformer_embedding_prediction_head import BinaryTransformerPredictionHead
from models.cosine import RawCosine
from models.Sentence_MALDI import Sentence_MALDI
from models.mlp_binary_classifier import MLPBinaryClassifier
from models.CLIP_MALDI import CLIP_MALDI
from models.CLIP_MALDI_classifier import CLIP_MALDI_Classifier
from models.prototypical_transformer import PrototyicalTransformer
from models.logistic_regression_classifier import MultinomialLogisticClassifier

from datamodule import SingleSpectrum_DataModule
from tqdm import tqdm

IMPLEMENTED_MODELS = {'Prototyical_Transformer',    # TODO: Spell it right once the models are done training
                      'CLIP_Transformer_Classifier',
                      'Multinomial_Logistic_Classifier',
                      'CLIP_Transformer',
                      'cosine_1', 'cosine_3', 'cosine_5', 'cosine_7', 'cosine_10',
                      }

def setup_model(model_name: str,
                checkpoint_path: Path,
) -> Tuple[L.LightningDataModule, transforms.Compose]:
    """
    Setup the model and the transforms for the model
    Args:
        model_name (str): Name of the model
        checkpoint_path (Path): Path to the checkpoint  
    Returns:
        model (L.LightningDataModule): Model
        transforms (transforms.Compose): Transforms for the model
    """
    if model_name.split('_')[0] != 'cosine':
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")
    
    model, trans = None, None
    if model_name == 'Prototyical_Transformer':
        model = PrototyicalTransformer.load_from_checkpoint(checkpoint_path=checkpoint_path)
        trans =  transforms.Compose([
                                SelectMassRange(3_000, 20_000),
                                SquareRootTransform(),
                                SelectTopKPeaks(150),
                                NormalizeIntensity(),
                                ])
    elif model_name == 'CLIP_Transformer_Classifier':
        model = CLIP_MALDI_Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path)
        if model.hparams.encoder != 'transformer':
            raise ValueError(f"Model {model_name} is not a transformer model.")
        trans =  transforms.Compose([
                                SelectMassRange(3_000, 20_000),
                                SquareRootTransform(),
                                NormalizeIntensity(),
                                SelectTopKPeaks(150),
                                # PadToLength(150, padding_value=-1.0),
                                ])
    elif model_name == 'Multinomial_Logistic_Classifier':
        raise ValueError(f"Model {model_name} does not generate embeddigs!")
        model = MultinomialLogisticClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path)
        trans =  transforms.Compose([
                                    BinSpectrum(10, 3_000, 20_000),
                                    SquareRootTransform(),
                                    NormalizeIntensity(),
                        ])
    elif model_name == 'CLIP_Transformer':
        model = CLIP_MALDI.load_from_checkpoint(checkpoint_path=checkpoint_path)
        if model.hparams.encoder != 'transformer':
            raise ValueError(f"Model {model_name} is not a transformer model.")
        

        trans =  transforms.Compose([
                                        SquareRootTransform(),
                                        SelectTopKPeaks(150),
                                        NormalizeIntensity(),
                                     ])
    elif model_name.split('_')[0] == 'cosine':
        class _IdentityModel:
            def __init__(self):
                self.device = 'cpu'
                pass
            def embed_step(self, spectra):
                return spectra
            def eval(self):
                pass
        model = _IdentityModel()
        cosine_bin_size = int(model_name.split('_')[1])
        trans = transforms.Compose([
                                        BinSpectrum(cosine_bin_size, 3_000, 20_000),
                                        SquareRootTransform(),
                                        NormalizeIntensity(),
                                    ])


    else:
        raise ValueError(f"Model {model_name} not implemented, please check the model name")
    
    return model, trans



def get_inference_df(model, datamodule, inference_set: str, target: str) -> pd.DataFrame:
    model.eval()
    inference_lst = []
    with torch.no_grad():
        for batch in tqdm(datamodule.predict_dataloader()):
            spectra, metadata = batch
            spectra = spectra.to(model.device)
            if len(spectra.shape) == 3:
                if spectra.shape[1] == 0:
                    continue
            if len(spectra.shape) == 2:
                if spectra.shape[1] == 0:
                    continue
            try:

                embedding = model.embed_step(spectra)
                embedding = embedding.cpu().numpy().copy()  # Wihtout copy, runs into memory issues
            except Exception as e:
                print(f"Error in embedding: {e}")
                print(f"Batch: {batch}")
                print(spectra)
                continue

            embedding_dict = {
                                'accession':metadata['accession'], 
                                'strain_name':metadata['strain_name'],
                                'embedding': np.squeeze(embedding),
                              }
            inference_lst.append(embedding_dict)

    embedding_df = pd.DataFrame(inference_lst)

    print("Total number of spectra: ", len(embedding_df))
    return embedding_df

def main():
    parser = argparse.ArgumentParser(description="General inference script for all models")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--version", type=int, required=False, help="Model version")
    parser.add_argument("--dataset", "-ds", type=str, required=True, help="Dataset name", choices=['DIRAMS-A', 'IDBac'])
    parser.add_argument("--target", type=str, required=True, help="Target to predict", choices=['genera', 'species'])
    parser.add_argument("--split_type", type=str, required=False, help="Split type", choices=['genera', 'species', 'species_even'], default='genera')
    parser.add_argument("--inference_set", type=str, required=True, help="Inference set name", choices=['train', 'val', 'test', 'all'])
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to the model checkpoint")
    parser.add_argument("--new_paths", action='store_true', help="Use new paths for the data")
    parser.add_argument("--run_for_score", action='store_true', help="Run for score")
    parser.add_argument("--k", "-k", type=int, required=False, help="CV Fold", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.model in IMPLEMENTED_MODELS:
        raise ValueError(f"Model {args.model} not implemented, please check the model name")
    
    if args.checkpoint_path is not None and args.k is not None:
        raise ValueError("Either checkpoint_path or k must be specified, not both")

    checkpoint_path = None

    if args.model.split('_')[0] != 'cosine':
        # Either version must be specified or checkpoint_path  must be specified
        if args.checkpoint_path is None:
            if args.model is None or args.version is None:
                raise ValueError("Either version or checkpoint_path  must be specified")
            
        if args.checkpoint_path is not None:
            if args.version is not None:
                raise ValueError("Either version or checkpoint_path  must be specified")
        
   
    if args.dataset == 'DIRAMS-A':
        _checkpoint_root_dir = Path('./lightning_logs_DRIAMS_A/')
        if args.run_for_score:
            _checkpoint_root_dir = Path('./lightning_logs_DRIAMS_A_for_score/')
    elif args.dataset == 'IDBac':
        _checkpoint_root_dir = Path('./lightning_logs/')
        if args.run_for_score:
            _checkpoint_root_dir = Path('./lightning_logs_idbac_for_score/')
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist")
        output_dir = checkpoint_path.parent / f"../inference/{args.target}/{args.split_type}/"

    else:
        k_fold_insert = ""
        if args.k is not None:
            k_fold_insert = f"k={args.k}/"
        if args.model.split('_')[0] != 'cosine':
            if not args.new_paths:
                checkpoint = _checkpoint_root_dir / f"{args.model}/{args.target}/{k_fold_insert}/version_{args.version}/checkpoints/"
            else:
                checkpoint = _checkpoint_root_dir / f"{args.target}/{args.split_type}/{k_fold_insert}/{args.model}/version_{args.version}/checkpoints/"
            checkpoint_path = list(checkpoint.glob("best*.ckpt"))
            if len(checkpoint_path) == 0:
                checkpoint_path = list(checkpoint.glob("*.ckpt"))
            if len(checkpoint_path) == 0:
                raise ValueError(f"No checkpoint found in {checkpoint}")
            elif len(checkpoint_path) > 1:
                raise ValueError(f"Multiple checkpoints found in {checkpoint}, please specify the exact path")
            else:
                checkpoint_path = checkpoint_path[0]

            output_dir = checkpoint_path.parent / f"../inference/{args.target}/{args.split_type}/"

        else:
            output_dir = _checkpoint_root_dir / f"{args.model}/{args.target}/{args.split_type}/{k_fold_insert}"

    if args.inference_set.lower().strip() == 'all':
        output_dir = output_dir / '../../all'

    output_dir.mkdir(parents=True, exist_ok=True)
   
    if args.dataset == 'DIRAMS-A':
        spectra_path = '../../data/driams/preprocessing'
        metadata_path = '../../data/driams/preprocessing/merged_metadata.csv'
        if args.split_type == 'species_even':
            metadata_path = '../../data/driams/preprocessing/merged_metadata_code_accessions.csv'
        ml_processing_path = '../../data/driams/processed_data'
    elif args.dataset == 'IDBac':
        spectra_path = '../../data/idbac_db/preprocessing'
        metadata_path = '../../data/idbac_db/raw/ammended_db.csv'
        ml_processing_path = '../../data/idbac_db/processed_data'
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
        
    logging.info("Output directory: %s", output_dir)
    logging.info("Checkpoint path: %s", checkpoint_path)
    logging.info("Model: %s", args.model)
    logging.info("Version: %s", args.version)
    logging.info("Dataset: %s", args.dataset)
    logging.info("Target: %s", args.target)
    logging.info("Inference set: %s", args.inference_set)
    logging.info("spectra_path: %s", spectra_path)
    logging.info("metadata_path: %s", metadata_path)
    logging.info("ml_processing_path: %s", ml_processing_path)

    model, model_specific_transforms = setup_model(args.model, checkpoint_path)

    # DEBUG WRITE ALL ARGS to datamodule
    print('spectra_path', spectra_path)
    print('metadata_path', metadata_path)
    print('ml_processing_path', ml_processing_path)
    print('batch_size', 1)
    print('num_workers', 0)
    print('transforms', model_specific_transforms)
    print('split_method', args.split_type)
    print('inference_set_to_use', args.inference_set)
    print('cast_to_classification', False)
    print('targets', args.target)
    

    datamodule = SingleSpectrum_DataModule(
        spectra_path,
        metadata_path,
        ml_processing_path,
        batch_size=1,
        num_workers=0,
        transforms=model_specific_transforms,
        split_method=args.split_type,
        inference_set_to_use=args.inference_set,
        cast_to_classification=False,
        targets=args.target,
        k=args.k,
    )
    datamodule.setup('test')

    inferred_df = get_inference_df(model, datamodule, args.inference_set, args.target)

    # inferred_df.to_feather(output_dir / f"{args.inference_set}_{args.target}_inference.feather")
    inferred_df.to_feather(output_dir / f"{args.inference_set}_inference.feather")

if __name__ == "__main__":
    main()