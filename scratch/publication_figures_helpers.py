import functools

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


NAME_MAPPINGS = {
    'clip_transformer': "Contrastive Transformer",
    'clip_transformer_euclidean': "Contrastive Transformer (Euclidean)",
    'clip_transformer_intensity_agnostic': "Contrastive Transformer (Int. Agn.)",
    'clip_transformer_classifier': "Classifier Embeddings",
    'cross_encoder': 'Cross Encoder',
    'cosine': "Cosine Similarity",
    'prototypical_transformer': "Prototypical Transformer",
    'cosine_intensity_agnostic': "Cosine Similarity (Int. Agn.)",
    'multinomial_classifier': "Multinomial Classifier",
    'clip_transformer_genus_genus': 'Contrastive Transformer (Task-Specific HParams)',
    'BinaryTransformerPredictionHead': 'Binary Transformer Prediction Head',
    'Euclidean': 'Euclidean Distance',
    'maldi_transformer': 'MALDI Transformer',
    'cosine_1': 'Cosine Similarity (1 Da)',
    'cosine_3': 'Cosine Similarity (3 Da)',
    'cosine_5': 'Cosine Similarity (5 Da)',
    'cosine_7': 'Cosine Similarity (7 Da)',
    'cosine_10': 'Cosine Similarity (10 Da)',
    'cosine_1_intensity_agnostic': 'Cosine Similarity (1 Da, Int. Agn.)',
    'cosine_3_intensity_agnostic': 'Cosine Similarity (3 Da, Int. Agn.)',
    'cosine_5_intensity_agnostic': 'Cosine Similarity (5 Da, Int. Agn.)',
    'cosine_7_intensity_agnostic': 'Cosine Similarity (7 Da, Int. Agn.)',
    'cosine_10_intensity_agnostic': 'Cosine Similarity (10 Da, Int. Agn.)',
    'cosine_intensity_agnostic_between_species': 'Cosine Similarity (10 Da, Int. Agn., Between Species)',
    'maldi_transformer_ts': 'MALDI Transformer',
}

PAIRED_MODELS = {
    'cross_encoder',
}

colors = {
    'cosine': '#fc8d62',
    'cosine_intensity_agnostic': '#e78ac3',
    'clip_transformer': '#66c2a5',
    'multinomial_classifier': '#8da0cb',
    'cosine_10': '#fc8d62',
    'cosine_7': '#e78ac3',
    'cosine_5': '#66c2a5',
    'cosine_3': '#8da0cb',
    'cosine_1': "#ffc82f",
    'cosine_10_intensity_agnostic': '#fc8d62',
    'cosine_7_intensity_agnostic': '#e78ac3',
    'cosine_5_intensity_agnostic': '#66c2a5',
    'cosine_3_intensity_agnostic': '#8da0cb',
    'cosine_1_intensity_agnostic': "#ffc82f",
    'Theoretical Max': "#727272",
    'maldi_transformer_ts': '#a6d854',
    'cross_encoder': "#ff2f2f",
}

level_heirarchy = {
    'genera': 3,
    'species': 2,
    'species_even': 1,
}

@functools.lru_cache(maxsize=10)
def gather_embeddings_helper(dataset:str, target:str, split_type:str, rki_disjoint:bool=False):
    base_dir = (REPO_ROOT / 'bin/ml/')

    if dataset.lower() == 'driams-a':
        base_dir = base_dir / 'lightning_logs_DRIAMS_A'
    elif dataset.lower() == 'idbac-all':
        base_dir = base_dir / 'lightning_logs_idbac_for_score'
    elif dataset.lower() == 'driams-b':
        base_dir = base_dir / 'lightning_logs_DRIAMS_B_for_score'
    elif dataset.lower() == 'driams-c':
        base_dir = base_dir / 'lightning_logs_DRIAMS_C_for_score'
    elif dataset.lower() == 'driams-d':
        base_dir = base_dir / 'lightning_logs_DRIAMS_D_for_score'
    elif dataset.lower() == 'idbac-kb':
        base_dir = base_dir / 'lightning_logs'
    elif dataset.lower() == 'rki':
        pass
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    metadata_path = None
    clip_transformer_genus_genus_path = None
    clip_transformer_path = None
    clip_transformer_classifer_path = None
    cosine_path = None
    prototypical_transformer_path = None
    clip_transformer_intensity_agnostic_path = None
    multinomial_classifier_path = None
    binary_transformer_prediction_head_path = None
    maldi_transformer_path = None
    maldi_transformer_ts_path = None # "their script"
    cross_encoder_path = None

    DRIAMS_MAX_INDEX=6 # Exclude 7th (index=6) fold, as it's used for parameter tuning
    IDBAC_MAX_INDEX=1   # Used to be 3, only using 1 for eval since this is applicaitonn-focused

    # inference/{args.target}/{args.split_type}/"
    if target == 'genera':
        if split_type == 'genera':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,DRIAMS_MAX_INDEX)]
                prototypical_transformer_path = None
                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction_Genus_Disjoint' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'DRIAMS-A' / target / split_type for i in range(0, DRIAMS_MAX_INDEX)
                ]

            elif dataset.lower() == 'idbac-kb':
                raise NotImplementedError("Genera-Disjoint Genera prediction is a todo")
                metadata_path = (REPO_ROOT / 'data/idbac_db/raw/ammended_db.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,IDBAC_MAX_INDEX)
                ]
                prototypical_transformer_path = None
                clip_transformer_intensity_agnostic_path = None

            elif dataset.lower() == 'rki':
                metadata_path = (REPO_ROOT / 'data/RKI/processed/rki_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')    # That's right, we're using the DRIMAS_A mdoels
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / 'RKI' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / '../lightning_logs_RKI_for_score' / 'cosine_10' / 'all' ]
                prototypical_transformer_path = None
                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction_Genus_Disjoint' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'RKI' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
            
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)

                ]
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}'  for i in range(0,DRIAMS_MAX_INDEX)
                ]
                # multinomial_classifier_path = [
                #     _base_dir / target / split_type / f'k={i}' / 'Multinomial_Logistic_Classifier' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                # ]
                # maldi_transformer_path = [
                #     _base_dir / target / split_type / f'k={i}' / 'MaldiTransformerWrapperMethodData' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                # ]
                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'DRIAMS-A' / target / split_type for i in range(0, DRIAMS_MAX_INDEX)
                ]


            elif dataset.lower() == 'idbac-kb':
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_idbac_for_score')
                metadata_path = (REPO_ROOT / 'data/idbac_db/preprocessing/db_with_taxonomy.csv')

                clip_transformer_path = [
                     _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_3' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,IDBAC_MAX_INDEX)]
               
            elif dataset.lower() == 'idbac-all':
                metadata_path = (REPO_ROOT / 'data/idbac_db/preprocessing/db_with_taxonomy.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_11' / 'inference' / 'IDBac' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                # Different root for cosine
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_idbac_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']
            elif dataset.lower() == 'driams-b':
                print("Gathering driams-b test set")
                metadata_path = (REPO_ROOT / 'data/driams-B/preprocessing/merged_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / 'DRIAMS-B' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]

                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'DRIAMS-B' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]

                # Different root for cosine
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_B_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']
            elif dataset.lower() == 'driams-c':
                metadata_path = (REPO_ROOT / 'data/driams-C/preprocessing/merged_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / 'DRIAMS-C' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]

                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'DRIAMS-C' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]

                # Different root for cosine
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_C_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']

            elif dataset.lower() == 'driams-d':
                metadata_path = (REPO_ROOT / 'data/driams-D/preprocessing/merged_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / 'DRIAMS-D' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'DRIAMS-D' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]

                # Different root for cosine
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_D_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']

            elif dataset.lower() == 'rki':
                metadata_path = (REPO_ROOT / 'data/RKI/processed/rki_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')    # That's right, we're using the DRIMAS_A mdoels
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / 'RKI' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_0' / 'inference' / 'RKI' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]

                # Different root for cosine
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_RKI_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata_code_accessions.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                   _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_1' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,DRIAMS_MAX_INDEX)
                ]
                multinomial_classifier_path = [
                    _base_dir / target / split_type / f'k={i}' / 'Multinomial_Logistic_Classifier' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                ]
                # maldi_transformer_path = [
                #     _base_dir / target / split_type / f'k={i}' / 'MaldiTransformerWrapperMethodData' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                # ]
            elif dataset.lower() == 'idbac-kb':
                metadata_path = (REPO_ROOT / 'data/idbac_db/preprocessing/db_with_taxonomy.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = [
                     _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,IDBAC_MAX_INDEX)]
                prototypical_transformer_path = None
                multinomial_classifier_path = [
                    _base_dir / target / split_type / f'k={i}' / 'Multinomial_Logistic_Classifier' / 'version_0' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]


    elif target == 'species':
        if split_type == 'genera':
            metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
            if dataset.lower() == 'driams-a':
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None

            elif dataset.lower() == 'idbac-kb':
               raise NotImplementedError("Add metadata path of idbac-kb")
    
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
                _base_dir = (REPO_ROOT / 'bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,7)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,7)
                ]
                prototypical_transformer_path = None
            elif dataset.lower() == 'idbac-kb':
                metadata_path = (REPO_ROOT / 'data/idbac_db/preprocessing/db_with_taxonomy.csv')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata_code_accessions.csv')
            elif dataset.lower() == 'idbac-kb':
                raise NotImplementedError("Add metadata path of idbac-kb")
            raise NotImplementedError(f"Strain-Disjoint Species prediction is a todo")
            # clip_transformer_path = base_dir /
            # clip_transformer_classifer_path = base_dir /
            # cosine_path = base_dir / 'cosine' / target / split_type
            # prototypical_transformer_path = base_dir /
        else: 
            raise ValueError(f"Unknown split_type: {split_type} for target: {target}")

    else:
        raise ValueError(f"Unknown target: {target}")
    
    output_dict = {}
    # cosine_path = None

    if dataset.lower() in ['driams-b', 'driams-c', 'driams-d', 'rki', 'idbac-all']:
        if clip_transformer_path:
            output_dict['clip_transformer'] = {}
            output_dict['clip_transformer']['train'] = [pd.read_feather(x / 'all_inference.feather') for x in clip_transformer_path]
            output_dict['clip_transformer']['test'] = [pd.read_feather(x / 'all_inference.feather') for x in clip_transformer_path]
            print('clip_transformer_path', clip_transformer_path)
        if cosine_path:
            output_dict['cosine'] = {}
            output_dict['cosine']['train'] = [pd.read_feather(x / 'all_inference.feather') for x in cosine_path]
            output_dict['cosine']['test'] = [pd.read_feather(x / 'all_inference.feather') for x in cosine_path]
            print('cosine_path', cosine_path)
        if cross_encoder_path:
            output_dict['cross_encoder'] = {}
            output_dict['cross_encoder']['train'] = [pd.read_feather(x / 'all_inference.feather') for x in cross_encoder_path]
            output_dict['cross_encoder']['test'] = [pd.read_feather(x / 'all_inference.feather') for x in cross_encoder_path]
            print('cross_encoder_path', cross_encoder_path)

        if maldi_transformer_ts_path:
            output_dict['maldi_transformer_ts'] = {}
            output_dict['maldi_transformer_ts']['train'] = [pd.read_feather(x / 'all_inference.feather') for x in maldi_transformer_ts_path]
            output_dict['maldi_transformer_ts']['test'] = [pd.read_feather(x / 'all_inference.feather') for x in maldi_transformer_ts_path]
            print('maldi_transformer_ts_path', maldi_transformer_ts_path)
    else:
        if clip_transformer_path:
            output_dict['clip_transformer'] = {}
            output_dict['clip_transformer']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in clip_transformer_path]
            output_dict['clip_transformer']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in clip_transformer_path]
        if clip_transformer_classifer_path:
            output_dict['clip_transformer_classifier'] = {}
            output_dict['clip_transformer_classifier']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in clip_transformer_classifer_path]
            output_dict['clip_transformer_classifier']['test'] =[ pd.read_feather(x / 'test_inference.feather') for x in clip_transformer_classifer_path]
        if clip_transformer_genus_genus_path:
            output_dict['clip_transformer_genus_genus'] = {}
            output_dict['clip_transformer_genus_genus']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in clip_transformer_genus_genus_path]
            output_dict['clip_transformer_genus_genus']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in clip_transformer_genus_genus_path]
        if cosine_path:
            output_dict['cosine'] = {}
            output_dict['cosine']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_path]
            output_dict['cosine']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_path] 
        if prototypical_transformer_path:
            output_dict['prototypical_transformer'] = {}
            output_dict['prototypical_transformer']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in prototypical_transformer_path]
            output_dict['prototypical_transformer']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in prototypical_transformer_path]
        if clip_transformer_intensity_agnostic_path:
            output_dict['clip_transformer_intensity_agnostic'] = {}
            output_dict['clip_transformer_intensity_agnostic']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in clip_transformer_intensity_agnostic_path]
            output_dict['clip_transformer_intensity_agnostic']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in clip_transformer_intensity_agnostic_path]
        if multinomial_classifier_path:
            output_dict['multinomial_classifier'] = {}
            output_dict['multinomial_classifier']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in multinomial_classifier_path]
            output_dict['multinomial_classifier']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in multinomial_classifier_path]
        if binary_transformer_prediction_head_path:
            output_dict['binary_transformer_prediction_head'] = {}
            output_dict['binary_transformer_prediction_head']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in binary_transformer_prediction_head_path]
            output_dict['binary_transformer_prediction_head']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in binary_transformer_prediction_head_path]
        if maldi_transformer_path:
            output_dict['maldi_transformer'] = {}
            output_dict['maldi_transformer']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in maldi_transformer_path]
            output_dict['maldi_transformer']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in maldi_transformer_path]
        if maldi_transformer_ts_path:
            output_dict['maldi_transformer_ts'] = {}
            output_dict['maldi_transformer_ts']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in maldi_transformer_ts_path]
            output_dict['maldi_transformer_ts']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in maldi_transformer_ts_path]
        if cross_encoder_path:
            output_dict['cross_encoder'] = {}
            print("Warning -- Cross-Encoder is returning test for it's train data")
            output_dict['cross_encoder']['train'] = [pd.read_feather(x / 'test_inference.feather') for x in cross_encoder_path]
            output_dict['cross_encoder']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cross_encoder_path]

    # Augment the metadata with the true labels
    if target == 'genera':
        target_col = 'genus'
    elif target == 'species':
        target_col = 'species'
    else:
        raise ValueError(f"Unknown target: {target}")
    metadata_table = pd.read_csv(metadata_path)
    if 'accession' not in metadata_table.columns:
        metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
    else:
        metadata_table['accession'] = metadata_table['accession'].astype(str).str.strip()
    
    metadata_table.dropna(subset=[target_col], inplace=True)
    accession_to_label = metadata_table.set_index('accession')[target_col].to_dict()
    accession_to_genus = metadata_table.set_index('accession')['genus'].to_dict()
    accession_to_species = metadata_table.set_index('accession')['species'].to_dict()

    for key, train_test_dict in output_dict.items():
        print("getting key", key)
        if key not in PAIRED_MODELS:
            for i in range(len(output_dict[key]['test'])):
                output_dict[key]['train'][i]['accession'] = output_dict[key]['train'][i]['accession'].apply(lambda x: x[0]).astype(str)   # For some reason it's a list of a single string
                output_dict[key]['train'][i]['true_label'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))
                output_dict[key]['test'][i]['accession'] = output_dict[key]['test'][i]['accession'].apply(lambda x: x[0]).astype(str)   # For some reason it's a list of a single string
                output_dict[key]['test'][i]['true_label'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))

                # Precast embedding to np.array for convenience
                output_dict[key]['train'][i]['embedding'] = output_dict[key]['train'][i]['embedding'].apply(lambda x: np.array(x))
                output_dict[key]['test'][i]['embedding'] = output_dict[key]['test'][i]['embedding'].apply(lambda x: np.array(x))


                # Fix strain_name so it's no longer a list
                output_dict[key]['train'][i]['strain_name'] = output_dict[key]['train'][i]['strain_name'].apply(lambda x: x[0])
                output_dict[key]['test'][i]['strain_name'] = output_dict[key]['test'][i]['strain_name'].apply(lambda x: x[0])

                # Manually annotate with 'genus' and 'species' columns
                output_dict[key]['train'][i]['genus'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_genus.get(x, None))
                output_dict[key]['train'][i]['species'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_species.get(x, None))
                output_dict[key]['test'][i]['genus'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_genus.get(x, None))
                output_dict[key]['test'][i]['species'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_species.get(x, None))

                # Convert genus, species, accession, strain_name to categorical to save memory
                for col in ['genus', 'species', 'accession', 'strain_name']:
                    output_dict[key]['train'][i][col] = output_dict[key]['train'][i][col].astype('category')
                    output_dict[key]['test'][i][col] = output_dict[key]['test'][i][col].astype('category')

                # Add metadata to the output dataframes
                output_dict[key]['test'][i].target = target
                output_dict[key]['train'][i].target = target
                output_dict[key]['test'][i]._metadata += ('target',)
                output_dict[key]['train'][i]._metadata += ('target',)

            if rki_disjoint is not None:
                rki_disjoint = str(rki_disjoint).lower()
                if rki_disjoint not in ['false', 'genus', 'species']:
                    raise ValueError(f"Invalid value for rki_disjoint: '{rki_disjoint}'. Must be one of [false, 'genus', 'species']")

                if dataset.lower() == 'rki' and rki_disjoint != 'false':
                    # Remove all species/genus that overlap in DRIAMS-A metadata from test df

                    print(f"REMOVING ALL COMMON {rki_disjoint} WITH DRIAMS-A", flush=True)

                    driams_a = pd.read_csv(REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')

                    for key in output_dict.keys():
                        for i in range(len(output_dict[key]['test'])):
                            df = output_dict[key]['test'][i]
                            if rki_disjoint not in df.columns:
                                continue
                            inital_len = len(df)
                            output_dict[key]['test'][i] = df[~df[rki_disjoint].str.lower().isin(driams_a[rki_disjoint].str.lower())]      
                            print(f"Started with {inital_len} spectra, now we have {len(output_dict[key]['test'][i])} spectra after removing DRIAMS-A overlaps", flush=True)
       
        else: # Model is paired
            if rki_disjoint is not None:
                 raise NotImplementedError("RKI disjoint is not implemented for paired models yet")

            output_dict[key]['test'][i].test = True
            output_dict[key]['test'][i]._metadata += ('test',)

            for cond in ['train', 'test']:
                output_dict[key][cond][i].attrs.update({'test': cond == 'test'})
                output_dict[key][cond][i].attrs.update({'paired': True})

                for col in ['accession_a', 'accession_b']:
                    output_dict[key][cond][i][col] = output_dict[key][cond][i][col].astype(str)   # For some reason it's a list of a single string
                    output_dict[key][cond][i][f'true_label_{col[-1]}'] = output_dict[key][cond][i][col].apply(lambda x: accession_to_label.get(x, None))
    

                output_dict[key][cond][i]['genus_a'] = output_dict[key][cond][i]['accession_a'].apply(lambda x: accession_to_genus.get(x, None))
                output_dict[key][cond][i]['species_a'] = output_dict[key][cond][i]['accession_a'].apply(lambda x: accession_to_species.get(x, None))
                output_dict[key][cond][i]['genus_b'] = output_dict[key][cond][i]['accession_b'].apply(lambda x: accession_to_genus.get(x, None))
                output_dict[key][cond][i]['species_b'] = output_dict[key][cond][i]['accession_b'].apply(lambda x: accession_to_species.get(x, None))

                # Generate true_label column
                if target == 'genera':
                    output_dict[key][cond][i]['true_label'] = (output_dict[key][cond][i]['genus_a'] == output_dict[key][cond][i]['genus_b']).astype(int)
                elif target == 'species':
                    output_dict[key][cond][i]['true_label'] = (output_dict[key][cond][i]['species_a'] == output_dict[key][cond][i]['species_b']).astype(int)

                # Convert genus, species, accession, strain_name to categorical to save memory
                for col in ['genus_a', 'species_a', 'accession_a', 'genus_b', 'species_b', 'accession_b', 'strain_name_a', 'strain_name_b']:
                    output_dict[key][cond][i][col] = output_dict[key][cond][i][col].astype('category')

                output_dict[key][cond][i].attrs.update({'target': target})


    output_dict['metadata'] = metadata_table
    
    return output_dict

import copy 
def gather_embeddings(dataset:str, target:str, split_type:str, rki_disjoint:bool=False):
    return copy.deepcopy(gather_embeddings_helper(dataset, target, split_type, rki_disjoint=rki_disjoint))


def gather_embeddings_cosine_only(dataset:str, target:str, split_type:str):
    base_dir = (REPO_ROOT / 'bin/ml/')

    if dataset.lower() == 'driams-a':
        base_dir = base_dir / 'lightning_logs_DRIAMS_A_for_score'
    elif dataset.lower() == 'idbac-kb':
        base_dir = base_dir / 'lightning_logs'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    metadata_path = None
    cosine_1_path = None
    cosine_3_path = None
    cosine_5_path = None
    cosine_7_path = None
    cosine_10_path = None

    # inference/{args.target}/{args.split_type}/"
    if target == 'genera':
        if split_type == 'genera':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                metadata_path = (REPO_ROOT / 'data/idbac_db/raw/ammended_db.csv')
                raise NotImplementedError("This configuration is not implemented")
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
                cosine_1_path = [base_dir / 'cosine_1' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_3_path = [base_dir / 'cosine_3' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_5_path = [base_dir / 'cosine_5' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_7_path = [base_dir / 'cosine_7' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_10_path = [base_dir / 'cosine_10' / target / split_type / f"k={i}" for i in range(0, 6)]
            elif dataset.lower() == 'idbac-kb':
                metadata_path = (REPO_ROOT / 'data/idbac_db/raw/ammended_db.csv')
                cosine_1_path = [base_dir / 'cosine_1' / target / split_type]
                cosine_3_path = [base_dir / 'cosine_3' / target / split_type]
                cosine_5_path = [base_dir / 'cosine_5' / target / split_type]
                cosine_7_path = [base_dir / 'cosine_7' / target / split_type]
                cosine_10_path = [base_dir / 'cosine_10' / target / split_type]

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata_code_accessions.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                raise NotImplementedError("Add metadata path of idbac-kb")
            
    elif target == 'species':
        if split_type == 'genera':
            # I guess we could do this, but it feels like a bit of a stretch
            raise ValueError(f"Unknown split_type: {split_type} for target: {target}")
    
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                metadata_path = (REPO_ROOT / 'data/idbac_db/raw/ammended_db.csv')
                raise NotImplementedError("This configuration is not implemented")

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = (REPO_ROOT / 'data/driams/preprocessing/merged_metadata_code_accessions.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                raise NotImplementedError("Add metadata path of idbac-kb")
                raise NotImplementedError(f"Strain-Disjoint Species prediction is a todo")

    else:
        raise ValueError(f"Unknown target: {target}")
    
    output_dict = {}
    # cosine_path = None

    if cosine_1_path:
        output_dict['cosine_1'] = {}
        output_dict['cosine_1']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_1_path]
        output_dict['cosine_1']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_1_path]
    if cosine_3_path:
        output_dict['cosine_3'] = {}
        output_dict['cosine_3']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_3_path]
        output_dict['cosine_3']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_3_path]
    if cosine_5_path:
        output_dict['cosine_5'] = {}
        output_dict['cosine_5']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_5_path]
        output_dict['cosine_5']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_5_path]
    if cosine_7_path:
        output_dict['cosine_7'] = {}
        output_dict['cosine_7']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_7_path]
        output_dict['cosine_7']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_7_path]
    if cosine_10_path:
        output_dict['cosine_10'] = {}
        output_dict['cosine_10']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_10_path]
        output_dict['cosine_10']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_10_path]

    # Augment the metadata with the true labels
    if target == 'genera':
        target_col = 'genus'
    elif target == 'species':
        target_col = 'species'
    else:
        raise ValueError(f"Unknown target: {target}")
    metadata_table = pd.read_csv(metadata_path)
    if 'accession' not in metadata_table.columns:
        metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
    else:
        metadata_table['accession'] = metadata_table['accession'].astype(str).str.strip()
    
    accession_to_label = metadata_table.set_index('accession')[target_col].to_dict()
    accession_to_genus = metadata_table.set_index('accession')['genus'].to_dict()
    accession_to_species = metadata_table.set_index('accession')['species'].to_dict()

    for key, train_test_dict in output_dict.items():
        for i in range(len(output_dict[key]['train'])):

            output_dict[key]['train'][i]['accession'] = output_dict[key]['train'][i]['accession'].apply(lambda x: x[0])   # For some reason it's a list of a single string
            output_dict[key]['train'][i]['genus'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_genus.get(x, None))
            output_dict[key]['train'][i]['species'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_species.get(x, None))
            output_dict[key]['train'][i]['true_label'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))
            output_dict[key]['test'][i]['accession'] = output_dict[key]['test'][i]['accession'].apply(lambda x: x[0])   # For some reason it's a list of a single string
            output_dict[key]['test'][i]['genus'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_genus.get(x, None))
            output_dict[key]['test'][i]['species'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_species.get(x, None))
            output_dict[key]['test'][i]['true_label'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))

            # Precast embedding to np.array for convenience
            output_dict[key]['train'][i]['embedding'] = output_dict[key]['train'][i]['embedding'].apply(lambda x: np.array(x))
            output_dict[key]['test'][i]['embedding'] = output_dict[key]['test'][i]['embedding'].apply(lambda x: np.array(x))
    
            # Fix strain_name so it's no longer a list
            output_dict[key]['train'][i]['strain_name'] = output_dict[key]['train'][i]['strain_name'].apply(lambda x: x[0])
            output_dict[key]['test'][i]['strain_name'] = output_dict[key]['test'][i]['strain_name'].apply(lambda x: x[0])

    output_dict['metadata'] = metadata_table
    
    return output_dict
        

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist

def dendrogram_from_embeddings(
    dataset, target, split_type, model,distance_metric='cosine'
):
    """
    Generates a dendrogram from pairwise distances between embeddings.

    Args:
        dataset (str): Dataset name.
        target (str): Column with class labels.
        split_type (str): Split strategy (e.g., 'species_even').
        model (str): The embedding model to use.
        distance_metric (str): 'cosine' or 'euclidean'.

    Returns:
        matplotlib.figure.Figure: The dendrogram figure.
    """
    embeddings = gather_embeddings(dataset, target, split_type)

    if model == 'cosine_intensity_agnostic':
        embeddings['cosine_intensity_agnostic'] = {
            'train': [None for _ in range(len(embeddings['cosine']['train']))],
            'test': [None for _ in range(len(embeddings['cosine']['test']))]
        }
        for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
            embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
            embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
            embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
            embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))



    if model not in embeddings:
        raise ValueError(f"Model '{model}' not found in embeddings.")

    data = embeddings[model]
    print("Warning: Taking first CV fold only")

    df1 = data['test'][0].copy()

    X = np.stack(df1['embedding'].values)
    print(df1.head())
    labels = np.array(df1['strain_name'] + ' - ' + df1['true_label'])

    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = np.nan_to_num(X)

    if distance_metric == 'cosine':
        # Avoid subtractive cancellation by using scipy's cosine distance directly
        distances = squareform(pdist(X, metric='cosine'))
        xlabel = 'Cosine Distance'
    elif distance_metric == 'euclidean':
        distances = euclidean_distances(X)
        xlabel = 'Euclidean Distance'
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    # Create a condensed distance matrix
    condensed_distances = squareform(distances)
    # Create a linkage matrix
    linkage_matrix = linkage(condensed_distances, method='average')
    # Create a dendrogram
    fig, ax = plt.subplots(figsize=(10, 8))
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90, color_threshold=0.5)
    ax.set_title(f'Dendrogram for Model: {model} ({distance_metric})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Distance')
    plt.tight_layout()
    return fig