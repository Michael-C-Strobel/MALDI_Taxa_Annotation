# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import traceback

# %%
# Set the style of seaborn
sns.set(style="white")
sns.set_palette("Set2")

# Set font size to large
plt.rcParams.update({
    'font.size': 20,           # Base font size
    'axes.labelsize': 22,      # x/y label font size
    'axes.titlesize': 24,      # title font size
    'legend.fontsize': 20,     # legend font size
    'xtick.labelsize': 20,     # x-axis tick labels
    'ytick.labelsize': 20      # y-axis tick labels
})

# %%
os.getcwd()

# %%
NAME_MAPPINGS = {
    'clip_transformer': "Contrastive Transformer",
    'clip_transformer_euclidean': "Contrastive Transformer (Euclidean)",
    'clip_transformer_intensity_agnostic': "Contrastive Transformer (Int. Agn.)",
    'clip_transformer_classifier': "Classifier Embeddings",
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
    'maldi_transformer_ts': 'MALDI Transformer Reproduced',
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
}

level_heirarchy = {
    'genera': 3,
    'species': 2,
    'species_even': 1,
}

def gather_embeddings(dataset:str, target:str, split_type:str):
    base_dir = Path('../bin/ml/')

    if dataset.lower() == 'driams-a':
        base_dir = base_dir / 'lightning_logs_DRIAMS_A'
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

    DRIAMS_MAX_INDEX=6 # Exclude 7th (index=6) fold, as it's used for parameter tuning
    IDBAC_MAX_INDEX=3

    # inference/{args.target}/{args.split_type}/"
    if target == 'genera':
        if split_type == 'genera':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,DRIAMS_MAX_INDEX)]
                prototypical_transformer_path = None
                # maldi_transformer_path = [
                #     _base_dir / target / split_type / f'k={i}' / 'MaldiTransformerWrapper' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                # ]

            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,IDBAC_MAX_INDEX)
                ]
                prototypical_transformer_path = None
                clip_transformer_intensity_agnostic_path = None
            
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_11' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)

                ]
                clip_transformer_classifer_path = None
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}'  for i in range(0,DRIAMS_MAX_INDEX)
                ]
                prototypical_transformer_path = None
                multinomial_classifier_path = [
                    _base_dir / target / split_type / f'k={i}' / 'Multinomial_Logistic_Classifier' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                ]
                maldi_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'MaldiTransformerWrapperMethodData' / 'version_0' / 'inference' / target / split_type for i in range(0,DRIAMS_MAX_INDEX)
                ]
                # Will look like this: /data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/MALDI-Transformer_Reproduction/k=5/malditrfvanilla_M_200_0.15_0.01_0.0005/version_0/inference/DRIAMS-A/genera/species/test_inference.feather
                print("Warning - Using MALDI-Transformer Reproduction Split 0 Only")
                maldi_transformer_ts_path = [
                    _base_dir / '../MALDI-Transformer_Reproduction' / f'k={i}' / 'malditrfvanilla_M_200_0.15_0.01_0.0005' / 'version_1' / 'inference' / 'DRIAMS-A' / target / split_type for i in [0,]
                ]


            elif dataset.lower() == 'idbac-kb':
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')

                clip_transformer_path = [
                     _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_2' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,IDBAC_MAX_INDEX)]
                prototypical_transformer_path = None
                multinomial_classifier_path = [
                    _base_dir / target / split_type / f'k={i}' / 'Multinomial_Logistic_Classifier' / 'version_0' / 'inference' / target / split_type for i in range(0,IDBAC_MAX_INDEX)
                ]
                # binary_transformer_prediction_head_path = [
                #     _base_dir / target / split_type / f'k={i}' / 'Binary_Transformer_Prediction_Head' / 'version_0' / 'inference' / target / split_type for i in range(0,3)
                # ]
            elif dataset.lower() == 'driams-b':
                metadata_path = Path('../data/driams-B/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_11' / 'inference' / 'DRIAMS-B' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                # Different root for cosine
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_B_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']
            elif dataset.lower() == 'driams-c':
                metadata_path = Path('../data/driams-C/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_11' / 'inference' / 'DRIAMS-C' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                # Different root for cosine
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_C_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']

            elif dataset.lower() == 'driams-d':
                metadata_path = Path('../data/driams-D/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_11' / 'inference' / 'DRIAMS-D' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                # Different root for cosine
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_D_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']

            elif dataset.lower() == 'rki':
                metadata_path = Path('../data/RKI/processed/rki_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')    # That's right, we're using the DRIMAS_A mdoels
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_11' / 'inference' / 'RKI' / 'all' for i in range(0, DRIAMS_MAX_INDEX)
                ]
                clip_transformer_genus_genus_path = None
                clip_transformer_classifer_path = None
                prototypical_transformer_path = None

                # Different root for cosine
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_RKI_for_score')
                cosine_path = [_base_dir / 'cosine_10' / 'all']

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata_code_accessions.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
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
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
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
            metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
            if dataset.lower() == 'driams-a':
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None

            elif dataset.lower() == 'idbac-kb':
               raise NotImplementedError("Add metadata path of idbac-kb")
    
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / target / split_type / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type for i in range(0,7)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [
                    _base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(0,7)
                ]
                prototypical_transformer_path = None
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata_code_accessions.csv')
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

    if dataset.lower() in ['driams-b', 'driams-c', 'driams-d', 'rki']:
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
            print("*** DANGEROUS USING TRIANING SET")
            output_dict['maldi_transformer']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in maldi_transformer_path]
            output_dict['maldi_transformer']['test'] = [pd.read_feather(x / 'train_inference.feather') for x in maldi_transformer_path]
        if maldi_transformer_ts_path:
            output_dict['maldi_transformer_ts'] = {}
            print("*** DANGEROUS USING TRIANING SET")
            output_dict['maldi_transformer_ts']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in maldi_transformer_ts_path]
            output_dict['maldi_transformer_ts']['test'] = [pd.read_feather(x / 'train_inference.feather') for x in maldi_transformer_ts_path]

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
        for i in range(len(output_dict[key]['test'])):
            output_dict[key]['train'][i]['accession'] = output_dict[key]['train'][i]['accession'].apply(lambda x: x[0])   # For some reason it's a list of a single string
            output_dict[key]['train'][i]['true_label'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))
            output_dict[key]['test'][i]['accession'] = output_dict[key]['test'][i]['accession'].apply(lambda x: x[0])   # For some reason it's a list of a single string
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

            # Add metadata to the output dataframes
            output_dict[key]['test'][i].target = target
            output_dict[key]['train'][i].target = target
            output_dict[key]['test'][i]._metadata += ('target',)
            output_dict[key]['train'][i]._metadata += ('target',)

        if dataset.lower() == 'rki':
            # Remove all species that overlap in DRIAMS-A metadata from test df

            print("REMOVING ALL COMMON SPECTRA WITH DRIAMS-A", flush=True)

            driams_a = pd.read_csv('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/data/driams/preprocessing/merged_metadata.csv')

            for key in output_dict.keys():
                for i in range(len(output_dict[key]['test'])):
                    df = output_dict[key]['test'][i]
                    if 'species' not in df.columns:
                        continue
                    inital_len = len(df)
                    output_dict[key]['test'][i] = df[~df['species'].str.lower().isin(driams_a['species'].str.lower())]      
                    print(f"Started with {inital_len} spectra, now we have {len(output_dict[key]['test'][i])} spectra after removing DRIAMS-A overlaps", flush=True)
        else:
            print(f"NOT REMOVING ANYTHING BECAUSE WE ARE NOT USING RKI. GOT '{dataset.lower()}'", flush=True)

    output_dict['metadata'] = metadata_table
    
    return output_dict
        

# %%
def gather_embeddings_cosine_only(dataset:str, target:str, split_type:str):
    base_dir = Path('../bin/ml/')

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
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                raise NotImplementedError("This configuration is not implemented")
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                cosine_1_path = [base_dir / 'cosine_1' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_3_path = [base_dir / 'cosine_3' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_5_path = [base_dir / 'cosine_5' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_7_path = [base_dir / 'cosine_7' / target / split_type / f"k={i}" for i in range(0, 6)]
                cosine_10_path = [base_dir / 'cosine_10' / target / split_type / f"k={i}" for i in range(0, 6)]
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                cosine_1_path = [base_dir / 'cosine_1' / target / split_type]
                cosine_3_path = [base_dir / 'cosine_3' / target / split_type]
                cosine_5_path = [base_dir / 'cosine_5' / target / split_type]
                cosine_7_path = [base_dir / 'cosine_7' / target / split_type]
                cosine_10_path = [base_dir / 'cosine_10' / target / split_type]

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata_code_accessions.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                raise NotImplementedError("Add metadata path of idbac-kb")
            
    elif target == 'species':
        if split_type == 'genera':
            # I guess we could do this, but it feels like a bit of a stretch
            raise ValueError(f"Unknown split_type: {split_type} for target: {target}")
    
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                raise NotImplementedError("This configuration is not implemented")
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                raise NotImplementedError("This configuration is not implemented")

        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata_code_accessions.csv')
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

    for key, train_test_dict in output_dict.items():
        for i in range(len(output_dict[key]['train'])):
            output_dict[key]['train'][i]['accession'] = output_dict[key]['train'][i]['accession'].apply(lambda x: x[0])   # For some reason it's a list of a single string
            output_dict[key]['train'][i]['true_label'] = output_dict[key]['train'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))
            output_dict[key]['test'][i]['accession'] = output_dict[key]['test'][i]['accession'].apply(lambda x: x[0])   # For some reason it's a list of a single string
            output_dict[key]['test'][i]['true_label'] = output_dict[key]['test'][i]['accession'].apply(lambda x: accession_to_label.get(x, None))

            # Precast embedding to np.array for convenience
            output_dict[key]['train'][i]['embedding'] = output_dict[key]['train'][i]['embedding'].apply(lambda x: np.array(x))
            output_dict[key]['test'][i]['embedding'] = output_dict[key]['test'][i]['embedding'].apply(lambda x: np.array(x))
    
            # Fix strain_name so it's no longer a list
            output_dict[key]['train'][i]['strain_name'] = output_dict[key]['train'][i]['strain_name'].apply(lambda x: x[0])
            output_dict[key]['test'][i]['strain_name'] = output_dict[key]['test'][i]['strain_name'].apply(lambda x: x[0])

    output_dict['metadata'] = metadata_table
    
    return output_dict
        

# %%
def get_within_test_prototype_accuracy( 
                            test_df:pd.DataFrame,
                            k:int,
                            random_seed:int=42,
                            distance_metric:str='euclidean',
                            normalize:bool=True, 
                          )-> float:
    df = test_df.copy(deep=True)

    label_counts = df['true_label'].value_counts()
    singletons = label_counts[label_counts == 1].index.tolist()

    # Singletons are not approriate for this analysis
    df = df[~df['true_label'].isin(singletons)]


    if normalize:
        # L2 Norm
        df['embedding'] = df['embedding'].apply(lambda x: x / np.linalg.norm(x))
        # Check if any are nan, set to 0
        df['embedding'] = df['embedding'].apply(lambda x: np.nan_to_num(x))

    np.random.seed(random_seed)

    taxa_indices = {}
    for idx, row in df.iterrows():
        if row['true_label'] not in taxa_indices:
            taxa_indices[row['true_label']] = []
        taxa_indices[row['true_label']].append(idx)

    # For each taxa, sample k spectra (up to 50% of test set)
    prototype_components = dict()
    sampled_indices_set = set()
    for taxa, indices in taxa_indices.items():
        _k = min(k, len(indices)//2)
        sampled_indices = np.random.choice(indices, size=_k, replace=False)
        for i in sampled_indices:
            sampled_indices_set.add(i)
        prototype_components[taxa] = df.loc[sampled_indices, 'embedding'].values

    prototypes = dict()
    for taxa, components in prototype_components.items():
        # Take the mean of the components
        prototypes[taxa] = np.squeeze(np.mean(components, axis=0))
        if normalize:
            prototypes[taxa] = prototypes[taxa] / np.linalg.norm(prototypes[taxa])

    correct = 0
    total   = 0
    failure_cases = []

          
    for i, row in df.iterrows():
      if i in sampled_indices_set:
          continue  # Skip sampled indices
      
      embedding = row['embedding']
      strain_name = row['strain_name']
      true_label = row['true_label']

      similarities = []
      for prototype_genus, prototype_embedding in prototypes.items():
          if distance_metric == 'cosine':
                similarity = np.dot(embedding, prototype_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(prototype_embedding))
          elif distance_metric == 'euclidean':
                similarity = -1 * np.linalg.norm(embedding - prototype_embedding)
          else:
                raise ValueError("Unknown distance metric: {}".format(distance_metric))
          similarities.append((prototype_genus, similarity))
    
      predicted_genus = max(similarities, key=lambda x: x[1])[0]
      if predicted_genus == true_label:
          correct += 1
      else:
          failure_cases.append({
              'strain_name': strain_name,
              'true_label': true_label, 
              'predicted_genus': predicted_genus,
              })
      total += 1

    acc = correct / total
    return acc, failure_cases
        

        

# %%
def evaluate_top_k_recall_old(
    train_df,
    test_df,
    method,
    distance_metric='euclidean',
    normalize=True,
    max_k=10,
    average='macro',
    within_test=False,
    return_failure_cases=False,
    cross_species=False,
    metadata=None,
) -> Tuple[List[float], List[Dict[str, str]]]:

    require_cross_species = cross_species
    if within_test:
        train_df = test_df

    # Remove any rows with nan true_label
    train_df = train_df[train_df['true_label'].notna()]
    test_df = test_df[test_df['true_label'].notna()]

    # For consistency, sort the dataframes by strain_name
    train_df = train_df.sort_values(by='strain_name')
    test_df = test_df.sort_values(by='strain_name')

    train_embeddings = np.vstack(train_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)

    if normalize:
        train_embeddings = np.nan_to_num(train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True))
        test_embeddings = np.nan_to_num(test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True))

    if require_cross_species:
        cross_species_mask = train_df['species']
        raise NotImplementedError("Cross-species evaluation is not implemented yet")

    if distance_metric == 'cosine':
        scores = cosine_similarity(test_embeddings, train_embeddings)
    elif distance_metric == 'euclidean':
        scores = -euclidean_distances(test_embeddings, train_embeddings)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    assert len(scores) == len(test_df), "Scores length does not match test_df length"
    assert len(scores[0]) == len(train_df), "Scores length does not match train_df length"

    sorted_indices = np.argsort(-scores, axis=1, kind='stable')[:, :max_k]

    train_labels = train_df['true_label'].values
    test_labels = test_df['true_label'].values
    strain_names = test_df['strain_name'].values
    predicted_k_labels = train_labels[sorted_indices]
    
    # Accuracy per k
    accuracies = []
    genera_counts = []
    failure_indices_per_k = []
    label_to_counts_k = None

    if average == 'macro':
        label_to_counts_k = [defaultdict(lambda: {'correct': 0, 'total': 0}) for _ in range(max_k)]
        # label_to_counts_k = [{l: {'correct': 0, 'total': 0}for l in np.unique(test_labels)} for _ in range(max_k) ]
        for i, true_label in enumerate(test_labels):
            for k in range(1, max_k + 1):
                label_to_counts_k[k - 1][true_label]['total'] += 1
                if true_label in predicted_k_labels[i, :k]:
                    label_to_counts_k[k - 1][true_label]['correct'] += 1

        for label_to_counts in label_to_counts_k:
            recalls = [v['correct'] / v['total'] for v in label_to_counts.values() if v['total'] > 0]
            accuracies.append(np.mean(recalls))
            genera_counts.append(len(label_to_counts))
    
    elif average == 'micro':
        match_matrix = predicted_k_labels == test_labels[:, None]
        for k in range(1, max_k + 1):
            hits = match_matrix[:, :k].any(axis=1)
            accuracies.append(hits.mean())
            failure_indices_per_k.append(np.where(~hits)[0])
    elif average == 'none':
        # Calculate accuracy within top k per class
        accuracies_per_class = {}

        if max_k != 1:
            raise NotImplementedError("Top-k accuracy is not implemented for k > 1 with 'average' == 'none'")

        match = predicted_k_labels[:, 0] == test_labels

        unique_labels = np.unique(test_labels)
        accuracies_per_class = {}

        for label in unique_labels:
            label_mask = (test_labels == label)
            if len(label_mask) == 0:
                accuracies_per_class[label] = {'mean_accuracy': 0.0}
            else:
                accuracies_per_class[label] = {'mean_accuracy': match[label_mask].mean(),
                                               'correct': match[label_mask].sum(),
                                                'total': label_mask.sum(),}
        return accuracies_per_class


    else:
        raise ValueError(f"Unknown average method: {average}")
    

    if len(failure_indices_per_k) == 0:
        failure_cases = [np.array([]) for _ in range(max_k)]
    else:
        failure_cases = [
            {
                'strain_name': strain_names[i],
                'true_label': test_labels[i],
                'predicted_labels': predicted_k_labels[i].tolist(),
                'genera_counts': genera_counts
            }
            for i in failure_indices_per_k[-1]  # only keep failures at max_k
        ]

    output_dict = {
        'model': method,
        'accuracies': accuracies,
        'genera_counts': genera_counts,
        # 'accuracies_per_class': accuracies_per_class,
    }
    if label_to_counts_k:
        output_dict['label_to_counts_k'] = label_to_counts_k
    # Add metadata
    if metadata is not None:
        if type(metadata) is dict:
            output_dict.update(metadata)
        else:
            output_dict['metadata'] = metadata
    if return_failure_cases:
        return output_dict, failure_cases
    else:
        return output_dict

# %%
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import recall_score
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

def _compute_top_k_recall(
    scores,
    train_labels,
    test_labels,
    strain_names=None,
    max_k=10,
    average='macro',
    return_failure_cases=False
):

    # Determine if we are in a "within-test" scenario
    within_test = np.array_equal(train_labels, test_labels)

    sorted_indices = np.argsort(-scores, axis=1)[:, :max_k]
    # predicted_k_labels = train_labels[sorted_indices]
    predicted_k_labels = []
    for i in range(len(test_labels)):
        valid_mask = ~np.isnan(scores[i])
        # invalid_mask_sum = sum(np.isnan(scores[i]))
        # if invalid_mask_sum >0:
        #     print(f"Encountered {invalid_mask_sum} invalid scores")
        valid_sorted = [idx for idx in sorted_indices[i] if valid_mask[idx]]
        # Pad with a sentinel (e.g., None) if fewer than k valid predictions
        # if (max_k - len(valid_sorted)) > 0:
        #     print(f"Appending {(max_k - len(valid_sorted))} padding values")
        padded = valid_sorted[:max_k] + [None] * (max_k - len(valid_sorted))
        predicted_k_labels.append([train_labels[idx] if idx is not None else None for idx in padded])
    predicted_k_labels = np.array(predicted_k_labels, dtype=object)

    accuracies = []
    failure_indices_per_k = []
    genera_counts = []
    label_to_counts_k = None

    # --- compute top-k recall ---
    if average == 'macro':
        label_to_counts_k = [defaultdict(lambda: {'correct': 0, 'total': 0}) for _ in range(max_k)]
        for i, true_label in enumerate(test_labels):
            for k in range(1, max_k+1):
                label_to_counts_k[k-1][true_label]['total'] += 1
                if true_label in predicted_k_labels[i, :k]:
                    label_to_counts_k[k-1][true_label]['correct'] += 1

        for label_to_counts in label_to_counts_k:
            recalls = [v['correct']/v['total'] for v in label_to_counts.values() if v['total'] > 0]
            accuracies.append(np.mean(recalls))
            genera_counts.append(len(label_to_counts))

    elif average == 'micro':
        match_matrix = predicted_k_labels == test_labels[:, None]
        for k in range(1, max_k+1):
            hits = match_matrix[:, :k].any(axis=1)
            accuracies.append(hits.mean())
            failure_indices_per_k.append(np.where(~hits)[0])

    elif average == 'none':
        if max_k != 1:
            raise NotImplementedError("Top-k accuracy is not implemented for k > 1 with 'average' == 'none'")
        match = predicted_k_labels[:, 0] == test_labels
        unique_labels = np.unique(test_labels)
        accuracies_per_class = {}
        for label in unique_labels:
            mask = (test_labels == label)
            accuracies_per_class[label] = {
                'mean_accuracy': match[mask].mean() if mask.any() else 0.0,
                'correct': match[mask].sum(),
                'total': mask.sum()
            }

        # --- theoretical max for 'none' ---
        return accuracies_per_class, [], {'theoretical_max': None}

    else:
        raise ValueError(f"Unknown average method: {average}")

    # --- compute theoretical maximum recall ---
    is_in_train = []
    for i, true_label in enumerate(test_labels):
        valid_scores = ~np.isnan(scores[i])
        valid_labels = train_labels[valid_scores]
        is_in_train.append(true_label in valid_labels)
    is_in_train = np.array(is_in_train)

    if average == 'micro':
        theoretical_max = np.ones(max_k) * is_in_train.mean()
    elif average == 'macro':
        label_counts = defaultdict(list)
        for lbl, present in zip(test_labels, is_in_train):
            label_counts[lbl].append(present)
        theoretical_max = []
        for _ in range(max_k):
            recalls = [np.mean(presence) for presence in label_counts.values()]
            theoretical_max.append(np.mean(recalls))

    # --- failure cases ---
    failure_cases = []
    if return_failure_cases and len(failure_indices_per_k) > 0 and strain_names is not None:
        failure_cases = [
            {
                'strain_name': strain_names[i],
                'true_label': test_labels[i],
                'predicted_labels': predicted_k_labels[i].tolist(),
                'genera_counts': genera_counts
            }
            for i in failure_indices_per_k[-1]
        ]

    extras = {
        'genera_counts': genera_counts,
        'label_to_counts_k': label_to_counts_k,
        'theoretical_max': theoretical_max
    }

    return accuracies, failure_cases, extras


def compute_scores(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    distance_metric: str = 'euclidean',
    normalize: bool = True,
    within_test: bool = False,
    cross_species: bool = False,
):
    if within_test:
        train_df = test_df

    train_df = train_df[train_df['true_label'].notna()].sort_values('strain_name')
    test_df = test_df[test_df['true_label'].notna()].sort_values('strain_name')

    train_embeddings = np.vstack(train_df['embedding'].values)
    test_embeddings = np.vstack(test_df['embedding'].values)

    if normalize:
        train_embeddings = np.nan_to_num(train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True))
        test_embeddings = np.nan_to_num(test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True))

    if distance_metric == 'cosine':
        scores = cosine_similarity(test_embeddings, train_embeddings)
    elif distance_metric == 'euclidean':
        scores = -euclidean_distances(test_embeddings, train_embeddings)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    train_species = train_df['species'].values
    test_species = test_df['species'].values

    if cross_species:
        # Set values to np.inf where species are the same
        species_mask = test_species[:, None] == train_species[None, :]
        assert species_mask.shape == scores.shape
        print(f"Setting {sum(species_mask.flatten())} scores to -inf due to cross-species filtering")
        scores[species_mask] = -np.inf

    if within_test:
        # Diagonal to inf
        # print("Filling diagonal.")
        assert scores.shape[0] == scores.shape[1], "Expected scores to be square when running with within_test=True"
        np.fill_diagonal(scores, -np.inf)

    return scores, train_df['true_label'].values, test_df['true_label'].values, test_df['strain_name'].values


def evaluate_top_k_recall(
    train_df=None,
    test_df=None,
    scores=None,
    train_labels=None,
    test_labels=None,
    strain_names=None,
    method=None,
    distance_metric='euclidean',
    normalize=True,
    max_k=10,
    average='macro',
    within_test=False,
    return_failure_cases=False,
    metadata=None,
    cross_species=False,
):
    if scores is None:
        if train_df is None or test_df is None:
            raise ValueError("Must provide either scores or both train_df and test_df")
        
        scores, train_labels, test_labels, strain_names = compute_scores(
            train_df, test_df,
            distance_metric=distance_metric,
            normalize=normalize,
            within_test=within_test,
            cross_species=cross_species,
        )

    accuracies, failure_cases, extras = _compute_top_k_recall(
        scores, train_labels, test_labels,
        strain_names, max_k, average, return_failure_cases
    )

    if average == 'none':
        output_dict = accuracies  # per-class dict
    else:
        output_dict = {
            'model': method,
            'accuracies': accuracies,
            **extras  # include genera_counts + label_to_counts_k
        }

    if metadata is not None:
        for key in metadata.keys():
            if key in output_dict:
                raise ValueError(f"Metadata key {key} already exists in output_dict")
        output_dict.update(metadata)

    if return_failure_cases:
        return output_dict, failure_cases
    return output_dict


def test_evaluate_top_k_recall_1():
    df1 = pd.DataFrame({
        'embedding': [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
        'true_label': ['A', 'B', 'A'],
        'strain_name': ['strain1', 'strain2', 'strain3'],
    })
    df2 = pd.DataFrame({
        'embedding': [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
        'true_label': ['A', 'B', 'A'],
        'strain_name': ['strain4', 'strain5', 'strain6'],
    })
    result = evaluate_top_k_recall(
        train_df=df1,
        test_df=df2,
        method='test_method',
        distance_metric='euclidean',
        normalize=True,
        max_k=2,
        average='macro',
    )
    assert result['model'] == 'test_method'
    assert len(result['accuracies']) == 2  # k=1 and k=2
    assert result['accuracies'][0] == 1.0  # k=1 should be perfect recall
    assert result['accuracies'][1] == 1.0  # k=2 should also be perfect recall

    result = evaluate_top_k_recall(
        train_df=df1,
        test_df=df2,
        method='test_method',
        distance_metric='euclidean',
        normalize=True,
        max_k=2,
        average='micro',
    )
    assert result['model'] == 'test_method'
    assert len(result['accuracies']) == 2  # k=1 and k=2
    assert result['accuracies'][0] == 1.0  # k=1 should be perfect recall
    assert result['accuracies'][1] == 1.0  # k=2 should also be perfect recall

def test_evaluate_top_k_recall_2():
    df1 = pd.DataFrame({
        'embedding': [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
        'true_label': ['A', 'B', 'A'],
        'strain_name': ['strain1', 'strain2', 'strain3'],
    })
    df2 = pd.DataFrame({
        'embedding': [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
        'true_label': ['A', 'B', 'B'],
        'strain_name': ['strain4', 'strain5', 'strain6'],
    })
    result = evaluate_top_k_recall(
        train_df=df1,
        test_df=df2,
        method='test_method',
        distance_metric='euclidean',
        normalize=True,
        max_k=2,
        average='macro',
    )
    assert result['model'] == 'test_method'
    assert len(result['accuracies']) == 2  # k=1 and k=2
    assert np.isclose(result['accuracies'][1], 0.750)
    assert np.isclose(result['accuracies'][1], 0.750)
    result = evaluate_top_k_recall(
        train_df=df1,
        test_df=df2,
        method='test_method',
        distance_metric='euclidean',
        normalize=True,
        max_k=2,
        average='micro',
    )
    assert result['model'] == 'test_method'
    assert len(result['accuracies']) == 2  # k=1 and k=2
    assert np.isclose(result['accuracies'][0], 0.6666, atol=0.0001), f"Expected 0.667, got {result['accuracies'][0]}"
    assert np.isclose(result['accuracies'][1], 0.6666, atol=0.0001), f"Expected 0.667, got {result['accuracies'][1]}"

def test_evaluate_top_k_recall_3():
    df1 = pd.DataFrame({
        'embedding': [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
        'true_label': ['A', 'B', 'A'],
        'strain_name': ['strain1', 'strain2', 'strain3'],
    })
    df2 = pd.DataFrame({
        'embedding': [np.array([1, 0]), np.array([0, 1]), np.array([1, 0])],
        'true_label': ['A', 'B', 'B'],
        'strain_name': ['strain4', 'strain5', 'strain6'],
    })
    result = evaluate_top_k_recall(
        train_df=df1,
        test_df=df2,
        method='test_method',
        distance_metric='euclidean',
        normalize=True,
        max_k=1,
        average='none',
    )
    print("result", result)
    assert 'A' in result
    assert 'B' in result
    assert np.isclose(result['A']['mean_accuracy'], 1.0, atol=0.0001), f"Expected 1.0 for A, got {result['A']['mean_accuracy']}"
    assert np.isclose(result['B']['mean_accuracy'], 0.5, atol=0.0001), f"Expected 0.5 for B, got {result['B']['mean_accuracy']}"

# test_evaluate_top_k_recall_1()
# test_evaluate_top_k_recall_2()
# test_evaluate_top_k_recall_3()


def top_k_recall_plot(dataset, target, split_type, n_jobs=-1, within_test=False, macro=False, cross_species=False, db_dataset=None):
    embeddings = gather_embeddings(dataset, target, split_type)

    # Add intensity-agnostic embeddings
    print("Adding intensity-agnostic embeddings...")
    embeddings['cosine_intensity_agnostic'] = {'train': [None]*len(embeddings['cosine']['train']),
                                               'test': [None]*len(embeddings['cosine']['test'])}
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x>0.02).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x>0.02).astype(int))

    if db_dataset is not None:
        if within_test:
            raise ValueError("Cannot use within_test with db_dataset")
        # We're going to replace the training set using embeddings from the db_dataset
        db_embeddings = gather_embeddings(db_dataset, target, split_type)
        db_embeddings['cosine_intensity_agnostic'] = {'train': [None]*len(db_embeddings['cosine']['train']),
                                                     'test': [None]*len(db_embeddings['cosine']['test'])}
        for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
            db_embeddings['cosine_intensity_agnostic']['train'][i] = db_embeddings['cosine']['train'][i].copy(deep=True)
            db_embeddings['cosine_intensity_agnostic']['test'][i] = db_embeddings['cosine']['test'][i].copy(deep=True)
            db_embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = db_embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x>0.02).astype(int))
            db_embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = db_embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x>0.02).astype(int))

        # Replace training sets
        for method in embeddings.keys():
            if method == 'metadata':
                continue
            if method in db_embeddings:
                embeddings[method]['train'] = db_embeddings[method]['train']
            else:
                print(f"Warning: {method} not found in db_embeddings, skipping.")
                del embeddings[method]

    # DEBUG Keep only relevant embeddings
    # for key in list(embeddings.keys()):
    #     if key not in {'clip_transformer', 'cosine_intensity_agnostic'}:
    #         del embeddings[key]

    # Prepare job arguments
    job_args = []
    for key in embeddings.keys():
        if key in {'metadata', 'multinomial_classifier'}:
            continue
        distance_metric = 'cosine' if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic'} else 'euclidean'

        for i in range(len(embeddings[key]['train'])):
            train_df = embeddings[key]['train'][i]
            test_df = embeddings[key]['test'][i]
            job_args.append((key, train_df, test_df, distance_metric, i))

    average = 'macro' if macro else 'micro'

    # Run evaluation in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_top_k_recall)(
            train_df=train_df,
            test_df=test_df,
            method=key,
            distance_metric=distance_metric,
            normalize=True,
            max_k=10,
            average=average,
            within_test=within_test,
            metadata={'cv_fold': cvf},
            cross_species=cross_species
        )
        for key, train_df, test_df, distance_metric, cvf in tqdm(job_args)
    )


    # return results
    # print(results)

    # Unpack into final format
    final_results = []
    for res in results:
        for k, acc in enumerate(res['accuracies'], start=1):
            d = {
                'model': res['model'],
                'k': k,
                'accuracy': acc,
                'label_to_counts_k': res.get('label_to_counts_k', None),
            }
            if 'genera_counts' in res:
                d['genera_counts'] = res['genera_counts']
            if 'cv_fold' in res:
                d['cv_fold'] = res['cv_fold']
            final_results.append(d)


    # return pd.DataFrame(final_results)

    if 'multinomial_classifier' in embeddings:
        print("Adding static recall calculation for multinomial classifier...")
        for i in range(len(embeddings['multinomial_classifier']['test'])):
            test_df = embeddings['multinomial_classifier']['test'][i]
            if macro:
                recall = recall_score(test_df['true_label'], test_df['pred_class'], average='macro')
            else:
                recall = recall_score(test_df['true_label'], test_df['pred_class'], average='micro')
            for k in range(1, 11):
                final_results.append({
                    'model': 'multinomial_classifier',
                    'k': k,
                    'accuracy': recall,
                    'genera_counts': len(test_df['true_label'].unique()),
                })

    accuracies_df = pd.DataFrame(final_results)
    accuracies_df['k'] = accuracies_df['k'].astype(int)
    accuracies_df['accuracy'] = accuracies_df['accuracy'].astype(float)
    accuracies_df = accuracies_df.sort_values(by=['model', 'k'])

    print(accuracies_df.head())

    # Propagate the theoretical max out
    for res in results:
        if 'theoretical_max' in res:
            max_df = [{
                'model': 'Theoretical Max',
                'accuracy': res['theoretical_max'][0],
                'genera_counts': res['genera_counts'],
                'k': k
            } for k in range(1, 11)]
            max_df = pd.DataFrame(max_df)
            print('max_df')
            print(max_df.head())
            accuracies_df = pd.concat([accuracies_df, max_df], ignore_index=True)

            print('accuracies_df')
            print(accuracies_df.head)
            continue

    # Map accuracies_df models to names
    accuracies_df['model_name'] = accuracies_df['model'].apply(lambda x: NAME_MAPPINGS.get(x, x))
    accuracies_df['color'] = accuracies_df['model'].apply(lambda x: colors.get(x, "#000000"))

    # Print average accuracies across CV for k=1
    _acc_df = accuracies_df[accuracies_df['k'] == 1]
    print("Average accuracies for k=1:")
    for model in _acc_df['model'].unique():
        avg_acc = _acc_df[_acc_df['model'] == model]['accuracy'].mean()
        print(f"{model}: {avg_acc:.4f}")

    print("***********Using weighted averages.")
    print(accuracies_df)
    accuracies_df['genera_counts'] = accuracies_df['genera_counts'].apply(lambda x: x if isinstance(x, int) else len(x) if isinstance(x, list) else np.nan)

    fig=None
    try:
        # Line plot
        fig = plt.figure(figsize=(12, 8))
        sns.lineplot(data=accuracies_df, 
                    x='k', 
                    y='accuracy',
                    hue='model_name',
                    style='model_name',
                    weights='genera_counts' if average == 'macro' else None,
                    markers=True,
                    dashes=False,
                    errorbar="ci",
                    palette=dict(zip(accuracies_df['model_name'], accuracies_df['color'])))
        plt.title(f"Train-Test Recall for {dataset} - {target} - {split_type}")
        plt.xlabel('Number of Neighbors Considered (k)')
        plt.ylabel('Macro Recall')
        plt.legend(title='Model')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.show()
    except Exception as e:
        # Print the traceback using traceback
        print(f"An error occurred while plotting: {e}")
        traceback.print_exc()
        pass

    return fig, accuracies_df



# %%
def top_k_barplot_per_method_by_label(dataset, target, split_type, k=1, n_jobs=-1, within_test=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    embeddings = gather_embeddings(dataset, target, split_type)

    # Optional intensity-agnostic transformation (copied from existing function)
    print("Adding intensity-agnostic embeddings...")
    embeddings['cosine_intensity_agnostic'] = {'train': [None for _ in range(len(embeddings['cosine']['train']))], 
                                               'test': [None for _ in range(len(embeddings['cosine']['test']))]}
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))

    job_args = []
    for key in embeddings:
        if key in {'metadata', 'multinomial_classifier'}:
            continue
        distance_metric = 'cosine' if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic'} else 'euclidean'
        for i in range(len(embeddings[key]['train'])):
            train_df = embeddings[key]['train'][i]
            test_df = embeddings[key]['test'][i]
            if within_test:
                train_df = test_df
            job_args.append((key, train_df, test_df, distance_metric))


    def compute_macro_label_scores(key, train_df, test_df, metric):
        if within_test:
            train_df = test_df

        train_embeddings = np.vstack(train_df['embedding'].values)
        test_embeddings = np.vstack(test_df['embedding'].values)

        # Normalize
        train_embeddings = np.nan_to_num(train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True))
        test_embeddings = np.nan_to_num(test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True))

        if metric == 'cosine':
            scores = cosine_similarity(test_embeddings, train_embeddings)
        elif metric == 'euclidean':
            scores = -euclidean_distances(test_embeddings, train_embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

        sorted_indices = np.argsort(-scores, axis=1)[:, :k]
        predicted_k_labels = train_df['true_label'].values[sorted_indices]
        test_labels = test_df['true_label'].values

        label_to_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
        for i, true_label in enumerate(test_labels):
            label_to_counts[true_label]['total'] += 1
            if true_label in predicted_k_labels[i, :k]:
                label_to_counts[true_label]['correct'] += 1

        return [
            {'model': key, 'true_label': label, 'recall': v['correct'] / v['total']}
            for label, v in label_to_counts.items() if v['total'] > 0
        ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_macro_label_scores)(key, train_df, test_df, metric)
        for key, train_df, test_df, metric in tqdm(job_args)
    )

    # Flatten and create dataframe
    flattened = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened)

    fig = plt.figure(figsize=(18, 8))
    sns.barplot(
        data=df,
        x='true_label',
        y='recall',
        hue='model',
        errorbar=('sd'),
        capsize=0.05,     # Shorter error bar caps
        errwidth=1.0,     # Thinner error bars
    )
    plt.title(f"Per-Label Top-{k} Recall (Bar Plot) — {dataset} | {target} | {split_type}")
    plt.xlabel('Label')
    plt.ylabel(f'Mean Recall@{k}')
    plt.xticks(rotation=90)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


    label_sizes = defaultdict(int)
    for _, _, test_df, _ in job_args:
        for label in test_df['true_label']:
            label_sizes[label] += 1

    # Convert to DataFrame
    size_df = pd.DataFrame(list(label_sizes.items()), columns=['true_label', 'genus_size'])

    # Merge with recall results
    merged_df = pd.merge(df, size_df, on='true_label')

    # Plot: Genus size vs Recall
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged_df,
        x='genus_size',
        y='recall',
        hue='model',
        alpha=0.7,
        s=60,
    )
    plt.xscale('log')
    plt.xlabel('Genus Size (log scale)')
    plt.ylabel(f'Recall@{k}')
    plt.title(f"Genus Size vs Recall — {dataset} | {target} | {split_type}")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return fig, df

# %%
def nn_accuracy( 
                test_df:pd.DataFrame,
                random_seed:int=42,
                k:int=5,
                distance_metric:str='euclidean',
                normalize:bool=True, 
                dedicated_db:pd.DataFrame=False,
                average:str='micro', # 'macro' or 'micro'
                require_cross_species:bool=False,
                require_same_species:bool=False,
                )-> float:
    """
    Computes the accuracy of the nearest neighbor classifier using a faux or dedicated database.
    Subsamples classes in the database to max(k , n_c//2) samples per taxa (c).

    Parameters
    ----------
    test_df : pd.DataFrame
        DataFrame containing the test set with columns 'embedding', 'strain_name', and 'true_label'.
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    k : int, optional
        Number of nearest neighbors to consider, by default 5
    distance_metric : str, optional
        Distance metric to use, either 'euclidean' or 'cosine', by default 'euclidean'
    normalize : bool, optional
        Whether to normalize the embeddings, by default True
    dedicated_db : pd.DataFrame, optional
        DataFrame containing the dedicated database with columns 'embedding', 'strain_name', and 'true_label', by default None
    average : str, optional
        Type of averaging to use for accuracy calculation, either 'micro', 'macro', or 'none', by default 'micro'
    
    Returns
    -------
    float
        Accuracy of the nearest neighbor classifier
    """
    if require_cross_species and require_same_species:
        raise ValueError("Cannot require both cross and same species. Choose one or the other.")

    df = test_df.copy(deep=True)
    np.random.seed(random_seed)

    if dedicated_db is None: # Generate a faux db using the test set
        label_counts = df['true_label'].value_counts()
        singletons = label_counts[label_counts == 1].index.tolist()

        df = df[~df['true_label'].isin(singletons)]
        singleton_df = df[df['true_label'].isin(singletons)]

        
        if normalize:
            # L2 Norm
            df['embedding'] = df['embedding'].apply(lambda x: x / np.linalg.norm(x))
            # Check if any are nan, set to 0
            df['embedding'] = df['embedding'].apply(lambda x: np.nan_to_num(x))


        taxa_indices = {}
        for idx, row in df.iterrows():
            if row['true_label'] not in taxa_indices:
                taxa_indices[row['true_label']] = []
            taxa_indices[row['true_label']].append(idx)

        # For each taxa, sample k spectra (up to 50% of test set)
        sampled_indices_set = set()
        for taxa, indices in taxa_indices.items():
            _k = min(k, len(indices)//2)
            sampled_indices = np.random.choice(indices, size=_k, replace=False)
            for i in sampled_indices:
                sampled_indices_set.add(i)

        # Add singletons into the faux_db THIS IS NEW 6/26
        # Ensure singleton's aren't already in that set 
        assert sampled_indices_set.isdisjoint(singletons), "Singletons should not be in the sampled indices set."

        faux_db = pd.concat([df.loc[list(sampled_indices_set), :], singleton_df]).values
    else:
        # Use the provided faux_db
        dedicated_db = dedicated_db.copy(deep=True)
        # Subsample the faux_db to k samples per taxa
        grouped_db = dedicated_db.groupby('true_label')
        sampled_indices_set = set()
        sampled_indices_dedicated_db = set()    # Different set because we don't want to remove these indices
        for taxa, group in grouped_db:
            _k = min(k, len(group)//2)
            sampled_indices = np.random.choice(group.index, size=_k, replace=False)
            for i in sampled_indices:
                sampled_indices_dedicated_db.add(i)

        faux_db = dedicated_db.loc[list(sampled_indices_dedicated_db), :].values

    predictions = []
    labels = []
    failure_cases = []

    faux_db_labels = np.unique(faux_db[:, 4])
          
    for i, row in df.iterrows():
        if i in sampled_indices_set:
            continue  # Skip sampled indices used in faux_db

        embedding = row['embedding']
        strain_name = row['strain_name']
        true_label = row['true_label']
        species = row['species']

        if true_label not in faux_db_labels:# and \
            # (require_cross_species or require_same_species):
            # If the true label is not in the faux_db, we skip this row
            # print(f"Skipping {strain_name} with true label {true_label} as it is not in the faux_db.")
            continue

        similarities = []
        correction_factor = 0
        for faux_row in faux_db:
            faux_embedding = faux_row[2]
            faux_true_label = faux_row[4]
            faux_species = faux_row[6]
            if require_cross_species and faux_species == species:
                continue
            if require_same_species and faux_species != species:
                continue
            if distance_metric == 'cosine':
                similarity = np.dot(embedding, faux_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(faux_embedding)
                )
            elif distance_metric == 'euclidean':
                similarity = -1 * np.linalg.norm(embedding - faux_embedding)
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")

            similarities.append((faux_true_label, similarity))

        # Scale to 100% Accuracy
        if true_label not in [x[0] for x in similarities]:
            continue

        predicted_label = max(similarities, key=lambda x: x[1])[0]
        similarity_count = len(similarities)

        predictions.append(predicted_label)
        labels.append(true_label)

        if predicted_label != true_label:
            failure_cases.append({
                'strain_name': strain_name,
                'true_label': true_label,
                'predicted_genus': predicted_label,
                'total_comparisons': similarity_count,
            })

    if average == 'micro':
        correct = sum([p == l for p, l in zip(predictions, labels)])
        total = len(predictions)
        acc = correct / total
    elif average == 'macro' or average == 'none':
        from collections import defaultdict

        class_correct = defaultdict(int)
        test_class_total = defaultdict(int)

        for p, l in zip(predictions, labels):
            test_class_total[l] += 1
            if p == l:
                class_correct[l] += 1

        if average == 'none':
            # Return per-class accuracy, and total number in each class label, along with labels
            per_class_accuracies = {
                label: class_correct[label] / test_class_total[label] for label in test_class_total
            }

            return per_class_accuracies, test_class_total,
        else:
            per_class_accuracies = [
                class_correct[label] / test_class_total[label] for label in test_class_total
            ]
            acc = np.mean(per_class_accuracies)

    else:
        raise ValueError(f"Invalid average type: {average}")

    return acc, failure_cases


# %%
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def within_test_prototype_accuracy_plot(dataset, target, split_type, n_jobs=-1):
    embeddings = gather_embeddings(dataset, target, split_type)

    # Ensure test sets are same size
    test_lengths = None
    test_length_error_str = ""
    for key, train_test_dict_lst in embeddings.items():
        if key == 'metadata':
            continue
        if test_lengths is None:
            test_length_error_str = f"Inital length was from {key} {len(train_test_dict_lst['test'][0])}"
            test_lengths = len(train_test_dict_lst['test'][0])
        these_lengths = [len(train_test_dict) for train_test_dict in train_test_dict_lst['test']]
        if not len(set(these_lengths)) == 1:
            for i, train_test_dict in enumerate(train_test_dict_lst):
                print(f"{key} {i}: {len(train_test_dict['test'])}")
            raise ValueError(f"Test sets within {key} are not the same length. Got {these_lengths}. Expected: {test_lengths}")
        if not these_lengths[0] == test_lengths:
            for i, train_test_dict in enumerate(train_test_dict_lst):
                print(f"{key} {i}: {len(train_test_dict['test'])}")
            raise ValueError(f"Test sets are not the same length. {test_length_error_str}")

    # Define a small helper function for a single evaluation
    def compute_prototype_accuracy(key, test_embeddings, distance_metric, k, random_seed):
        acc, _ = get_within_test_prototype_accuracy(
            test_embeddings,
            k=k,
            random_seed=random_seed,
            distance_metric=distance_metric,
            normalize=True
        )
        return {
            'model': key,
            'k': k,
            'accuracy': acc,
            'random_seed': random_seed,
        }

    tasks = []
    for key, train_test_dict in embeddings.items():
        if key == 'metadata':
            continue
        for i in range(len(train_test_dict['train'])):
            distance_metric = 'cosine' if key == 'cosine' else 'euclidean'
            test_embeddings = train_test_dict['test'][i]

            for random_seed in np.arange(42, 42+10):
                for k in np.arange(1, 11):
                    tasks.append((key, test_embeddings, distance_metric, k, random_seed))

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_prototype_accuracy)(key, test_embeddings, distance_metric, k, random_seed)
        for key, test_embeddings, distance_metric, k, random_seed in tqdm(tasks)
    )

    accuracies_df = pd.DataFrame(results)

    # Line plot
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=accuracies_df, x='k', y='accuracy', hue='model', style='model', markers=True, dashes=False)

    # Rename legend entries based on NAME_MAPPINGS
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [NAME_MAPPINGS.get(label, label) for label in labels]
    plt.legend(handles=handles, labels=new_labels, title='Model')

    plt.title(f"Within Test Prototype Accuracy for {dataset} - {target} - {split_type}")
    plt.xlabel('Number of Prototypes (k)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    return fig, accuracies_df


# %%
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

def nn_accuracy_plot(dataset, target, split_type, n_jobs=-1, within_test=False, num_samples_per_k=1,
                     average='macro', require_cross_species=False, require_same_species=False):
    
    if require_cross_species and require_same_species:
        raise ValueError("Cannot require both cross-species and same-species accuracy at the same time.")

    embeddings = gather_embeddings(dataset, target, split_type)

    if require_cross_species and target != 'genera':
        raise ValueError("Cross-species accuracy is only applicable for genus-level predictions")
    if require_same_species and target != 'genera':
        raise ValueError("Same-species accuracy is only applicable for genus-level predictions")

    # Add a "Cosine (Intensity Agnostic)" method
    embeddings['cosine_intensity_agnostic'] = {'train': [None for _ in range(len(embeddings['cosine']['train']))],
                                               'test': [None for _ in range(len(embeddings['cosine']['test']))]}
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))

    if require_cross_species or require_same_species:
        # Remove multinomial classifier, as the denominator is incongruent
        if 'multinomial_classifier' in embeddings:
            print("Removing multinomial classifier for same species accuracy")
            assert 'multinomial_classifier' in embeddings.keys(), "Multinomial classifier should be in embeddings"
            del embeddings['multinomial_classifier']

    # Quick sanity check, remove everything but contrastive transformer TODO DEBUG
    # for key in list(embeddings.keys()):
    #     if key not in {'clip_transformer', 'cosine'}:
    #         print(f"Removing {key} from embeddings")
    #         del embeddings[key]

    # Check that all dataframes have a 'target' value
    for key, train_test_dict in embeddings.items():
        if key == 'metadata':
            continue
        for i in range(len(train_test_dict['train'])):
            train_df = train_test_dict['train'][i]
            test_df = train_test_dict['test'][i]
            if not hasattr(train_df, 'target') or not hasattr(test_df, 'target'):
                raise ValueError(f"DataFrame for {key} at index {i} does not have a 'target' attribute.")
            # Ensure it's in _metadata
            if 'target' not in train_df._metadata or 'target' not in test_df._metadata:
                raise ValueError(f"DataFrame for {key} at index {i} does not have 'target' in _metadata.")

    def compute_accuracy(key, test_embeddings, distance_metric, k, random_seed, retrieval_db):
        acc, _ = nn_accuracy(
            test_embeddings, 
            k=k,
            random_seed=random_seed,
            distance_metric=distance_metric,
            normalize=True,
            dedicated_db=retrieval_db,
            average=average,
            require_cross_species=require_cross_species,
            require_same_species=require_same_species,
        )
        return {
            'model': key,
            'k': k,
            'accuracy': acc,
            'random_seed': random_seed,
        }

    tasks = []
    for key, train_test_dict in embeddings.items():
        if key == 'metadata':
            continue
        if key == 'multinomial_classifier':
            continue

        for i in range(len(train_test_dict['train'])):
            retrieval_db = None
            if not within_test:
                retrieval_db = embeddings[key]['train'][i]
            if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic'}:
                print(f"Using cosine distance for {key}")
                distance_metric = 'cosine'
            else:
                print(f"Using euclidean distance for {key}")
                distance_metric = 'euclidean'

            test_embeddings = train_test_dict['test'][i]

            for random_seed in np.arange(42, 42+num_samples_per_k):
                for k in np.arange(1, 11):
                    tasks.append((key, test_embeddings, distance_metric, k, random_seed, retrieval_db))

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_accuracy)(key, test_embeddings, distance_metric, k, random_seed, retrieval_db)
        for key, test_embeddings, distance_metric, k, random_seed, retrieval_db in tqdm(tasks)
    )

    if 'multinomial_classifier' in embeddings:
        # Calculate accuracy for multinomial classifier
        print("Adding static accuracy calculation for multinomial classifier...")
        for i in range(len(embeddings['multinomial_classifier']['train'])):
            test_df = embeddings['multinomial_classifier']['test'][i]
            acc = recall_score(test_df['true_label'], test_df['pred_class'], average=average)
            for k in np.arange(1, 11):
                results.append({
                    'model': 'multinomial_classifier',
                    'k': k,
                    'accuracy': acc,
                    'random_seed': 42,
                })

    accuracies_df = pd.DataFrame(results)
   
    # Line plot
    fig = plt.figure(figsize=(8, 5))
    sns.lineplot(data=accuracies_df, x='k', y='accuracy', hue='model', style='model', markers=True, dashes=False, errorbar='sd')

    # Rename legend entries based on NAME_MAPPINGS
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [NAME_MAPPINGS.get(label, label) for label in labels]
    plt.legend(handles=handles, labels=new_labels, title='Model')

    if within_test:
        plt.title(f"Within Test Nearest-Neighbor Accuracy for {dataset} - {target} - {split_type}")
    else:
        plt.title(f"Train-Test Nearest-Neighbor Accuracy for {dataset} - {target} - {split_type}")
    plt.xlabel(f'Number of Strains Per {target.capitalize()} (k)')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

    return fig, accuracies_df


# %%
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

def train_test_curves_nn_plot(dataset, target, split_type, n_jobs=-1):
    """
    Computes and plots ROC, Precision-Recall, and FDR curves for annotation via 1-NN similarity,
    showing mean and standard deviation across random seeds.
    Returns a dictionary of matplotlib figures.

    Args:
        dataset (str): Dataset name.
        target (str): Label column (e.g. 'species').
        split_type (str): Split type (e.g. 'species_even').
        n_jobs (int): Number of parallel jobs.

    Returns:
        dict: Dictionary of figures with keys: 'roc', 'precision_recall', 'fdr'.
    """
    embeddings = gather_embeddings(dataset, target, split_type)

    # Ensure test sets are same size
    test_lengths = None
    test_length_error_str = ""
    for key, train_test_dict_lst in embeddings.items():
        if key == 'metadata':
            continue
        if test_lengths is None:
            test_length_error_str = f"Inital length was from {key} {len(train_test_dict_lst['test'][0])}"
            test_lengths = len(train_test_dict_lst['test'][0])
        these_lengths = [len(train_test_dict) for train_test_dict in train_test_dict_lst['test']]
        if not len(set(these_lengths)) == 1:
            for i, train_test_dict in enumerate(train_test_dict_lst):
                print(f"{key} {i}: {len(train_test_dict['test'])}")
            raise ValueError(f"Test sets within {key} are not the same length. Got {these_lengths}. Expected: {test_lengths}")
        if not these_lengths[0] == test_lengths:
            for i, train_test_dict in enumerate(train_test_dict_lst):
                print(f"{key} {i}: {len(train_test_dict['test'])}")
            raise ValueError(f"Test sets are not the same length. {test_length_error_str}")

    tasks = []
    for key, dict_lst in embeddings.items():
        if key == 'metadata':
            continue
        distance_metric = 'cosine' if key == 'cosine' else 'euclidean'
        for i in range(len(dict_lst['train'])):
            train_df = dict_lst['train'][i].copy()
            test_df = dict_lst['test'][i].copy()
            tasks.append((key, train_df, test_df, distance_metric))

    def compute_curves(model_key, train_df, test_df, distance_metric):
        # Normalize embeddings
        train_df['embedding'] = train_df['embedding'].apply(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x)
        test_df['embedding'] = test_df['embedding'].apply(lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x)
        train_df['embedding'] = train_df['embedding'].apply(np.nan_to_num)
        test_df['embedding'] = test_df['embedding'].apply(np.nan_to_num)

        train_embeddings = np.vstack(train_df['embedding'])
        test_embeddings = np.vstack(test_df['embedding'])

        if distance_metric == 'cosine':
            similarities = cosine_similarity(test_embeddings, train_embeddings)
        elif distance_metric == 'euclidean':
            similarities = -euclidean_distances(test_embeddings, train_embeddings)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        train_labels = train_df['true_label'].values
        test_labels = test_df['true_label'].values

        y_true = []
        y_score = []

        for i in range(len(test_embeddings)):
            top_idx = np.argmax(similarities[i])
            pred_label = train_labels[top_idx]
            sim_score = similarities[i, top_idx]

            y_score.append(sim_score)
            y_true.append(int(pred_label == test_labels[i]))

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        fdr = 1 - precision
        fdr[np.isnan(fdr)] = 1.0

        return {
            'model': model_key,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': auc(fpr, tpr),
            'precision': precision,
            'recall': recall,
            'pr_auc': auc(recall, precision),
            'fdr': fdr,
            'fdr_auc': auc(recall, fdr)
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_curves)(key, train_df, test_df, distance_metric)
        for key, train_df, test_df, distance_metric in tqdm(tasks)
    )

    # === Plotting with Mean and Standard Deviation ===
    interp_points = np.linspace(0, 1, 100)
    all_roc_curves = {}
    all_pr_curves = {}
    all_fdr_curves = {}

    for res in results:
        model_name = res['model']
        if model_name not in all_roc_curves:
            all_roc_curves[model_name] = []
            all_pr_curves[model_name] = []
            all_fdr_curves[model_name] = []

        # Interpolate curves
        try:
            interp_tpr = np.interp(interp_points, res['fpr'], res['tpr'])
            interp_precision = np.interp(interp_points, res['recall'][::-1], res['precision'][::-1])
            interp_fdr = np.interp(interp_points, res['recall'][::-1], res['fdr'][::-1])

            all_roc_curves[model_name].append(interp_tpr)
            all_pr_curves[model_name].append(interp_precision)
            all_fdr_curves[model_name].append(interp_fdr)
        except Exception as e:
            print(f"Interpolation failed for {model_name}: {e}")

    figs = {}

    # Plot ROC curves with mean and std
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for model_name, curves in all_roc_curves.items():
        mean_tpr = np.mean(curves, axis=0)
        std_tpr = np.std(curves, axis=0)
        mean_auc = auc(interp_points, mean_tpr)
        ax_roc.plot(interp_points, mean_tpr, label=f"{model_name} (AUC = {mean_auc:.3f})")
        ax_roc.fill_between(interp_points, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_title("Train-Test Nearest Neighbor ROC Curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend()
    ax_roc.grid(False)
    ax_roc.set_xlim([0, 1])  # Set x-axis limits from 0 to 1
    ax_roc.set_ylim([0, 1])  # Set y-axis limits from 0 to 1
    figs['roc'] = fig_roc

    # Plot Precision-Recall curves with mean and std
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for model_name, curves in all_pr_curves.items():
        mean_precision = np.mean(curves, axis=0)
        std_precision = np.std(curves, axis=0)
        mean_auc_pr = auc(interp_points, mean_precision)
        ax_pr.plot(interp_points, mean_precision, label=f"{model_name} (AUC = {mean_auc_pr:.3f})")
        ax_pr.fill_between(interp_points, mean_precision - std_precision, mean_precision + std_precision, alpha=0.2)
    ax_pr.set_title("Train-Test Nearest Neighbor Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    ax_pr.grid(False)
    ax_pr.set_xlim([0, 1])  # Set x-axis limits from 0 to 1
    ax_pr.set_ylim([0, 1])  # Set y-axis limits from 0 to 1
    figs['precision_recall'] = fig_pr

    # Plot FDR curves with mean and std
    fig_fdr, ax_fdr = plt.subplots(figsize=(8, 6))
    for model_name, curves in all_fdr_curves.items():
        mean_fdr = np.mean(curves, axis=0)
        std_fdr = np.std(curves, axis=0)
        mean_auc_fdr = auc(interp_points, mean_fdr)
        ax_fdr.plot(interp_points, mean_fdr, label=f"{model_name} (AUC = {mean_auc_fdr:.3f})")
        ax_fdr.fill_between(interp_points, mean_fdr - std_fdr, mean_fdr + std_fdr, alpha=0.2)
    ax_fdr.set_title("Train-Test Nearest Neighbor False Discovery Rate vs Recall")
    ax_fdr.set_xlabel("Recall")
    ax_fdr.set_ylabel("FDR (1 - Precision)")
    ax_fdr.legend()
    ax_fdr.grid(False)
    ax_fdr.set_xlim([0, 1])  # Set x-axis limits from 0 to 1
    ax_fdr.set_ylim([0, 1])  # Set y-axis limits from 0 to 1
    figs['fdr'] = fig_fdr

    return figs

# %%
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import traceback

def _binary_similarity_curves(df1, df2, distance_metric='cosine', max_pairs=None, between_species=False,):
    """
    Computes binary labels and similarity scores for all pairs between df1 and df2,
    then returns ROC and PR curve components.

    Returns:
        fpr, tpr, roc_auc, precision, recall, pr_auc
    """
    X = np.stack(df1['embedding'].values).astype(np.float32)
    Y = np.stack(df2['embedding'].values).astype(np.float32)
    # L2 Normalize per embedding
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    # Check for NaN values
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    labels_X = np.array(df1['true_label'])
    labels_Y = np.array(df2['true_label'])

    n, m = len(X), len(Y)
    if n*m < max_pairs:
        pair_indices = [(i, j) for i in range(n) for j in range(m)]
        if between_species:
            species_X = np.array(df1['species'])
            species_Y = np.array(df2['species'])
            pair_indices = [(i, j) for i, j in pair_indices if species_X[i] != species_Y[j]]
            
    else:
        # Generate them randomly
        rng = np.random.default_rng(42)
        _max_pairs = max_pairs
        if between_species: # Sample some extra to give us buffer
            _max_pairs = min(m*n, int(max_pairs * 10))
        flat_indices = rng.choice(n * m, size=_max_pairs, replace=False)
        pair_indices = [(i // m, i % m) for i in flat_indices]
        if between_species:
            # Require that the pairs are from different species
            species_X = np.array(df1['species'])
            species_Y = np.array(df2['species'])
            pair_indices = [(i, j) for i, j in pair_indices if species_X[i] != species_Y[j]]
            if len(pair_indices) < 0.9 * max_pairs:
                print(f"Warning: Only {len(pair_indices)} pairs selected with different species.")
            if len(pair_indices) > max_pairs:
                # Subsample
                pair_indices = rng.choice(pair_indices, size=max_pairs, replace=False).tolist()

    i1, i2 = zip(*pair_indices)
    X1 = X[list(i1)]
    X2 = Y[list(i2)]
    l1 = labels_X[list(i1)]
    l2 = labels_Y[list(i2)]
    y_true = (l1 == l2).astype(int)

    if distance_metric == 'cosine':
        sim = np.sum(X1 * X2, axis=1) / (np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1))
    elif distance_metric == 'euclidean':
        sim = -np.linalg.norm(X1 - X2, axis=1)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    # Set nan similarity scores to 0
    sim[np.isnan(sim)] = 0.0

    if np.isnan(sim).any():
        raise ValueError("NaN values found in similarity scores.")

    acc = (np.round(y_true.flatten(), 0) == np.round(sim.flatten(), 0)).sum()/len(y_true)
    print(f"DEBUG ** Computing PR ACC={acc}")

    fpr, tpr, _ = roc_curve(y_true.flatten(), sim.flatten(), drop_intermediate=False)
    precision, recall, _ = precision_recall_curve(y_true.flatten(), sim.flatten())
    return fpr, tpr, auc(fpr, tpr), precision, recall, auc(recall, precision)

def _compute_interpolated_curves(df1, df2, method, data_idx, interp_points, metric, max_pairs, between_species=False):
    print(f"_compute_interpolated_curves; {method} : {metric}")
    try:
        if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
            print(f"Method {method} is using intensity agnostic embeddings")
            df1['embedding'] = df1['embedding'].apply(lambda x: (x > 0.02).astype(int))
            df2['embedding'] = df2['embedding'].apply(lambda x: (x > 0.02).astype(int))
        
        if method == 'cosine_intensity_agnostic_between_species':
            print("Using cosine intensity agnostic embeddings for between species")
            df1['embedding'] = df1['embedding'].apply(lambda x: (x > 0.02).astype(int))
            df2['embedding'] = df2['embedding'].apply(lambda x: (x > 0.02).astype(int))
            between_species = True

        fpr, tpr, roc_auc_val, prec, rec, pr_auc_val = _binary_similarity_curves(df1, df2, metric, max_pairs, between_species=between_species)
        fdr = 1 - prec

        print(f"Raw Precision: {prec}")
        print(f"Raw Recall: {rec}")

        print(f"ROC AUC before interpolation for {method} seed {data_idx}: {roc_auc_val:.3f}")
        print(f"PR AUC before interpolation for {method} seed {data_idx}: {pr_auc_val:.3f}")
        

        return (
            np.interp(interp_points, fpr, tpr),
            np.interp(interp_points, rec[::-1], prec[::-1]),
            np.interp(interp_points, rec[::-1], fdr[::-1])
        )
    except Exception as e:
        print(f"Interpolation failed for {method} seed {data_idx}: {e}")
        traceback.print_exc()
        # raise e
        return None

def static_precision_recall_roc(df1, df2, target, split_type, method, max_pairs=None, between_species=False):
    """
    Computes static precision and recall based on predicted class equality
    between embeddings in df1 and df2. Intended for classifier predictions
    where sweeping a threshold is not applicable.

    Returns:
        precision (float), recall (float)
    """
    pred1 = np.array(df1['pred_class'])
    pred2 = np.array(df2['pred_class'])
    true1 = np.array(df1['true_label'])
    true2 = np.array(df2['true_label'])

    n, m = len(df1), len(df2)
    if n*m < max_pairs:
        pair_indices = [(i, j) for i in range(n) for j in range(m)]
        if between_species:
            species1 = np.array(df1['species'])
            species2 = np.array(df2['species'])
            # Require that the pairs are from different species
            pair_indices = [(i, j) for i, j in pair_indices if species1[i] != species2[j]]
    else:
        # Generate them randomly
        rng = np.random.default_rng(42)
        flat_indices = rng.choice(n * m, size=max_pairs, replace=False)
        pair_indices = [(i // m, i % m) for i in flat_indices]
        if between_species:
            # Require that the pairs are from different species
            species1 = np.array(df1['species'])
            species2 = np.array(df2['species'])
            pair_indices = [(i, j) for i, j in pair_indices if species1[i] != species2[j]]
            if len(pair_indices) < 0.9 * max_pairs:
                print(f"Warning: Only {len(pair_indices)} pairs selected with different species.")

    i1, i2 = zip(*pair_indices)
    pred_match = (pred1[list(i1)] == pred2[list(i2)])
    true_match = (true1[list(i1)] == true2[list(i2)])

    # Compute binary prediction and ground truth
    y_pred = pred_match.astype(int)
    y_true = true_match.astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return precision, recall, fpr

import re
def binary_curves_plot(dataset, target, split_type, test_only=False, max_pairs=None,
                       cosine_ablation=False, n_jobs=-1, between_species=False):
    if cosine_ablation:
        embeddings = gather_embeddings_cosine_only(dataset, target, split_type)
    else:
        embeddings = gather_embeddings(dataset, target, split_type)

    if between_species:
        assert target == 'genera', "Between species accuracy is only applicable for genus-level predictions"

    # Add a "Cosine (Intensity Agnostic)" method
    if not cosine_ablation:
        embeddings['cosine_intensity_agnostic'] = {
            'train': [None for _ in range(len(embeddings['cosine']['train']))],
            'test': [None for _ in range(len(embeddings['cosine']['test']))]
        }
        for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
            embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
            embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
            embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
            embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
    else:
        if True:
            # For each cosine_[0-9]+ method, add a cosine_intensity_agnostic version
            for key in list(embeddings.keys()):
                if re.match(r'cosine_[0-9]+', key):
                    print(f"Adding intensity agnostic version for {key}")
                    embeddings[f"{key}_intensity_agnostic"] = {
                        'train': [None for _ in range(len(embeddings[key]['train']))],
                        'test': [None for _ in range(len(embeddings[key]['test']))]
                    }
                    for i in range(len(embeddings[f"{key}_intensity_agnostic"]['test'])):
                        embeddings[f"{key}_intensity_agnostic"]['train'][i] = embeddings[key]['train'][i].copy(deep=True)
                        embeddings[f"{key}_intensity_agnostic"]['test'][i] = embeddings[key]['test'][i].copy(deep=True)
                        embeddings[f"{key}_intensity_agnostic"]['train'][i]['embedding'] = embeddings[f"{key}_intensity_agnostic"]['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
                        embeddings[f"{key}_intensity_agnostic"]['test'][i]['embedding'] = embeddings[f"{key}_intensity_agnostic"]['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
                # Debug, remove the old one
                del embeddings[key]

    # TEMP DUPLICATE INTENSITY AGNOSTIC, MAKE BETWEEN SPECIES
    # if not cosine_ablation:
    #     embeddings['cosine_intensity_agnostic_between_species'] = {
    #         'train': [None for _ in range(len(embeddings['cosine_intensity_agnostic']['train']))],
    #         'test': [None for _ in range(len(embeddings['cosine_intensity_agnostic']['test']))]
    #     }
    #     for i in range(len(embeddings['cosine_intensity_agnostic_between_species']['test'])):
    #         embeddings['cosine_intensity_agnostic_between_species']['train'][i] = embeddings['cosine_intensity_agnostic']['train'][i].copy(deep=True)
    #         embeddings['cosine_intensity_agnostic_between_species']['test'][i] = embeddings['cosine_intensity_agnostic']['test'][i].copy(deep=True)
    #         embeddings['cosine_intensity_agnostic_between_species']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic_between_species']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
    #         embeddings['cosine_intensity_agnostic_between_species']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic_between_species']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))

    # Add a "Euclidean" method
    # embeddings['Euclidean'] = {
    #     'train': [None for _ in range(len(embeddings['cosine']['train']))],
    #     'test': [None for _ in range(len(embeddings['cosine']['test']))]
    # }
    # for i in range(len(embeddings['Euclidean']['test'])):
    #     embeddings['Euclidean']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
    #     embeddings['Euclidean']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)


    # DEBUG, MALDI_TRANSFORMER_TS and COSINE ONLY
    # for key in list(embeddings.keys()):
    #     if key not in {'maldi_transformer_ts', 'cosine_10_intensity_agnostic'}:
    #         print(f"Removing {key} from embeddings")
    #         del embeddings[key]
       
    # Print prior probability of equal and unequal taxa
    a_key = [x for x in list(embeddings.keys()) if x != 'metadata'][0]
    df = embeddings[a_key]['test'][0]

    total_pos = 0
    total_count = 0

    for df in embeddings[a_key]['test']:
        labels = df['true_label'].values
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_count += len(labels) ** 2
        for label, count in zip(unique_labels, counts):
            total_pos += count ** 2
    
    print(f"% of pairs with equal taxa: {total_pos / total_count:.2f}")

    roc_fig, roc_ax = plt.subplots(figsize=(7, 5), dpi=200)
    pr_fig, pr_ax = plt.subplots(figsize=(7, 5), dpi=200)
    fdr_fig, fdr_ax = plt.subplots(figsize=(7, 5), dpi=200)

    interp_points = np.linspace(0, 1, 100)

    print("Found methods:", list(embeddings.keys()))

    for method, data_list in embeddings.items():
        print("running method:", method)
        if method == 'metadata':
            continue

        if method == 'cosine' or method == 'cosine_intensity_agnostic' \
            or method == 'clip_transformer' or method == 'clip_transformer_genus_genus' \
            or re.match(r'cosine_[0-9]+', method) or method == 'cosine_intensity_agnostic_between_species' \
            or method == 'maldi_transformer_ts':
             print(f"Method using {method} cosine distance metric")
             metric = 'cosine'
        else:
            print(f"Method using {method} euclidean distance metric")
            metric = 'euclidean'

        if method == 'multinomial_classifier':
            # We don't have embeddings, calculate precision/recall as a single point using
            # paired classification

            # Require that the target level is below the split level
            if level_heirarchy[target] <= level_heirarchy[split_type]:
                continue

            precisions = []
            recalls = []
            fprs = []
        
            for data_idx in range(len(data_list['test'])):
                df1 = data_list['test'][data_idx].copy()
                df2 = df1.copy() if test_only else data_list['train'][data_idx].copy()

                precision, recall, fpr = static_precision_recall_roc(
                    df1,
                    df2,
                    target=target,
                    split_type=split_type,
                    method=method,
                    max_pairs=max_pairs,
                    between_species=between_species
                )

                if np.isnan(precision) or np.isnan(recall) or np.isnan(fpr):
                    print(f"[{method}] Skipping fold {data_idx} due to NaN: P={precision}, R={recall} FDR={fpr}")
                    continue

                precisions.append(precision)
                recalls.append(recall)
                fprs.append(fpr)

            if not precisions:
                print(f"[{method}] No valid folds for plotting.")
                continue

            precisions = np.array(precisions)
            recalls = np.array(recalls)
            fprs = np.array(fprs)

            precision_mean = precisions.mean()
            precision_std = precisions.std()
            recall_mean = recalls.mean()
            recall_std = recalls.std()
            fpr_mean = fprs.mean()
            fpr_std = fprs.std()

            name = NAME_MAPPINGS.get(method, method)

            print(f"[{method}] Plotting mean FPR={fpr_mean:.2f}, R={recall_mean:.2f}")
            roc_ax.errorbar([fpr_mean], [recall_mean], xerr=[fpr_std], yerr=[recall_std],
                            fmt='o', capsize=4, label=f"{name}", #  (FPR={fpr_mean:.2f})
                            color=colors.get(method))

            print(f"[{method}] Plotting mean P={precision_mean:.2f}, R={recall_mean:.2f}")
            pr_ax.errorbar([recall_mean], [precision_mean], xerr=[recall_std], yerr=[precision_std],
                        fmt='o', capsize=4, label=f"{name}",    #  (P={precision_mean:.2f}, R={recall_mean:.2f})
                        color=colors.get(method))

            fdr_ax.errorbar([recall_mean], [1 - precision_mean], xerr=[recall_std], yerr=[precision_std],
                            fmt='o', capsize=4, label=f"{name}",    #  (FDR={1 - precision_mean:.2f})
                            color=colors.get(method))

            # Add a fill between to keep the colors consistent
            roc_ax.fill_between([], [], alpha=0.2)
            pr_ax.fill_between([], [], alpha=0.2)
            fdr_ax.fill_between([], [], alpha=0.2)

        else:
                
            tasks = []
            for data_idx in range(len(data_list['test'])):
                df1 = data_list['test'][data_idx].copy()
                df2 = df1.copy() if test_only else data_list['train'][data_idx].copy()

                tasks.append((df1, df2, method, data_idx))

            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_interpolated_curves)(df1, df2, method, data_idx, interp_points, metric, max_pairs, between_species)
                for df1, df2, method, data_idx in tasks
            )

            # Filter out failed results
            results = [res for res in results if res is not None]
            if not results:
                continue

            roc_curves, pr_curves, fdr_curves = map(np.array, zip(*results))

            roc_mean, roc_std = roc_curves.mean(axis=0), roc_curves.std(axis=0)
            pr_mean, pr_std = pr_curves.mean(axis=0), pr_curves.std(axis=0)
            fdr_mean, fdr_std = fdr_curves.mean(axis=0), fdr_curves.std(axis=0)

            print(pr_mean)

            roc_auc_val = auc(interp_points, roc_mean)
            pr_auc_val = auc(interp_points, pr_mean)
            fdr_auc_val = auc(interp_points, fdr_mean)

            roc_ax.plot(interp_points, roc_mean, label=f"{NAME_MAPPINGS[method]}", color=colors.get(method))  #  (auc={roc_auc_val:.2f})
            roc_ax.fill_between(interp_points, roc_mean - roc_std, roc_mean + roc_std, alpha=0.2, color=colors.get(method))

            pr_ax.plot(interp_points, pr_mean, label=f"{NAME_MAPPINGS[method]}", color=colors.get(method))    #  (auc={pr_auc_val:.2f})
            pr_ax.fill_between(interp_points, pr_mean - pr_std, pr_mean + pr_std, alpha=0.2, color=colors.get(method))

            fdr_ax.plot(interp_points, fdr_mean, label=f"{NAME_MAPPINGS[method]}", color=colors.get(method)) #  (auc={fdr_auc_val:.2f})
            fdr_ax.fill_between(interp_points, fdr_mean - fdr_std, fdr_mean + fdr_std, alpha=0.2, color=colors.get(method))

            # Print the Method, and Standard Deviation at 5 points
            print(f"[{method}] ROC AUC: {roc_auc_val:.3f}, PR AUC: {pr_auc_val:.3f}, FDR AUC: {fdr_auc_val:.3f}")
            for i in [0, 25, 50, 74, 79, 84, 89, 99]:
                print(f"[{method}] ROC at {interp_points[i]:.2f}: {roc_mean[i]:.3f} ± {roc_std[i]:.3f}")
                print(f"[{method}] PR at {interp_points[i]:.2f}: {pr_mean[i]:.3f} ± {pr_std[i]:.3f}")
                print(f"[{method}] FDR at {interp_points[i]:.2f}: {fdr_mean[i]:.3f} ± {fdr_std[i]:.3f}")

    # Finalize ROC
    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate') # , title='ROC Curve'
    roc_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    roc_ax.set_xlim(-0.05, 1.05)
    roc_ax.set_ylim(-0.05, 1.05)

    # Finalize PR
    pr_ax.set(xlabel='Recall', ylabel='Precision') # , title='Precision-Recall Curve'
    pr_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    pr_ax.set_xlim(-0.05, 1.05)
    pr_ax.set_ylim(-0.05, 1.05)

    # Finalize FDR
    fdr_ax.set(xlabel='Recall', ylabel='False Discovery Rate') # , title='FDR vs Recall'
    fdr_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fdr_ax.set_xlim(-0.05, 1.05)
    fdr_ax.set_ylim(-0.05, 1.05)

    return {
        'roc': roc_fig,
        'precision_recall': pr_fig,
        'fdr': fdr_fig
    }



# %%
def compute_theoretical_max_precision(train_df, test_df, k: int) -> float:
    """
    Computes the theoretical maximum precision@k, assuming perfect retrieval.
    This is the best possible global precision@k given the available labels in training.
    """
    train_label_counts = train_df['true_label'].value_counts().to_dict()
    test_labels = test_df['true_label'].values

    total_possible_hits = sum(min(k, train_label_counts.get(label, 0)) for label in test_labels)
    total_predictions = k * len(test_labels)

    return total_possible_hits / total_predictions

def compute_macro_averaged_theoretical_max_precision(train_df, test_df, k: int) -> float:
    """
    Computes the macro-averaged theoretical maximum precision@k.
    For each class, assumes perfect retrieval and calculates per-class precision@k,
    then averages across all classes present in the test set.
    """
    train_label_counts = train_df['true_label'].value_counts().to_dict()
    test_label_counts = test_df['true_label'].value_counts().to_dict()

    precisions = []
    for label, count in test_label_counts.items():
        max_hits = min(k, train_label_counts.get(label, 0))
        precisions.append(min(max_hits / k, 1.0))  # perfect precision@k capped at 1.0

    return np.mean(precisions)
def evaluate_top_k_precision(
    train_df,
    test_df,
    k: int,
    method: str = 'clip_transformer',
    distance_metric: str = 'euclidean',
    normalize: bool = True,
    average: str = "micro",  # "micro" or "macro"
    return_failure_cases: bool = False
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Computes precision@k using either micro or macro averaging.

    average:
        - "micro": total correct / total predictions (default)
        - "macro": average of per-label mean precision@k
    """

    # if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
    #     print(f"Method {method} is using intensity agnostic embeddings")
    #     train_df['embedding'] = train_df['embedding'].apply(lambda x: (x > 0.02).astype(int))
    #     test_df['embedding'] = test_df['embedding'].apply(lambda x: (x > 0.02).astype(int))

    if normalize:
        def _normalize_fn(x):
            norm = np.linalg.norm(x)
            return x / norm if norm > 0 else x
        train_df['embedding'] = train_df['embedding'].apply(_normalize_fn)
        test_df['embedding'] = test_df['embedding'].apply(_normalize_fn)

    train_embeddings = np.vstack(train_df['embedding'])
    test_embeddings = np.vstack(test_df['embedding'])

    if distance_metric == 'cosine':
        similarities = cosine_similarity(test_embeddings, train_embeddings)
    elif distance_metric == 'euclidean':
        similarities = -euclidean_distances(test_embeddings, train_embeddings)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

    test_labels = test_df['true_label'].values
    train_labels = train_df['true_label'].values

    failure_cases = []
    strain_names = test_df['strain_name'].values

    if average == "micro":
        total_correct = 0
        total_predictions = 0

        for i in range(test_embeddings.shape[0]):
            top_k_indices = np.argsort(similarities[i])[::-1][:k]
            predicted_labels = train_labels[top_k_indices]

            num_correct = np.sum(predicted_labels == test_labels[i])
            total_correct += num_correct
            total_predictions += k

            if num_correct == 0:
                failure_cases.append({
                    'strain_name': strain_names[i],
                    'true_label': test_labels[i],
                    'predicted_labels': predicted_labels.tolist(),
                })

        precision = total_correct / total_predictions

    elif average == "macro":
        per_label_precisions = {}

        for i in range(test_embeddings.shape[0]):
            true_label = test_labels[i]
            top_k_indices = np.argsort(similarities[i])[::-1][:k]
            predicted_labels = train_labels[top_k_indices]
            correct = np.sum(predicted_labels == true_label) / k

            per_label_precisions.setdefault(true_label, []).append(correct)

            if correct == 0.0:
                failure_cases.append({
                    'strain_name': strain_names[i],
                    'true_label': true_label,
                    'predicted_labels': predicted_labels.tolist(),
                })

        label_means = [np.mean(p_list) for p_list in per_label_precisions.values()]
        precision = np.mean(label_means)

    else:
        raise ValueError(f"Invalid average type: {average}. Use 'micro' or 'macro'.")

    if return_failure_cases:
        return {'method': method, 'precision': precision, 'k':k}, failure_cases
    else:
        return {'method': method, 'precision': precision, 'k': k}

def top_k_precision_plot(dataset, target, split_type, n_jobs=-1, within_test=False, average='macro'):
    embeddings = gather_embeddings(dataset, target, split_type)


    # Add intensity-agnostic embeddings
    print("Adding intensity-agnostic embeddings...")
    embeddings['cosine_intensity_agnostic'] = {'train': [None for _ in range(len(embeddings['cosine']['train']))], 
                                               'test': [None for _ in range(len(embeddings['cosine']['test']))]}
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))

    tasks = []
    for key in embeddings.keys():
        if key == "metadata":
            continue
        if key == "multinomial_classifier":
            continue
        if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic'}:
            distance_metric = 'cosine'
        else:
            distance_metric = 'euclidean'
        for i in range(len(embeddings[key]['train'])):
            train_df = embeddings[key]['train'][i]
            test_df = embeddings[key]['test'][i]
            if within_test:
                train_df = test_df
            for k in range(1, 11):  # k from 1 to 10
                tasks.append((key, train_df, test_df, k, distance_metric))

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_top_k_precision)(
            train_df, test_df, k, method=key, distance_metric=distance_metric, average=average
        ) for key, train_df, test_df, k, distance_metric in tqdm(tasks)
    )

    # Unpack
    final_results = []
    for res in results:
        final_results.append({
            'model': res['method'],
            'k': res['k'],
            'precision': res['precision']
        })

    # Add multinomial classifier
    if 'multinomial_classifier' in embeddings:
        print("Adding static precision calculation for multinomial classifier...")
        for i in range(len(embeddings['multinomial_classifier']['test'])):
            test_df = embeddings['multinomial_classifier']['test'][i]
            if average == 'macro':
                precision = recall_score(test_df['true_label'], test_df['pred_class'], average='macro') # In multiclass classification, recall for a class is the same as precision for that class
            else:
                precision = recall_score(test_df['true_label'], test_df['pred_class'], average='micro')
            for k in range(1, 11):
                final_results.append({
                    'model': 'multinomial_classifier',
                    'k': k,
                    'precision': precision,
                })

    df = pd.DataFrame(final_results)
    df['k'] = df['k'].astype(int)
    df['precision'] = df['precision'].astype(float)
    df = df.sort_values(by=['model', 'k'])

    print(df[df['model'] == 'multinomial_classifier'])

    # Print average precision for k=1
    _prec_df = df[df['k'] == 1]
    print("Average precision for k=1:")
    for model in _prec_df['model'].unique():
        avg_prec = _prec_df[_prec_df['model'] == model]['precision'].mean()
        print(f"{model}: {avg_prec:.4f}")

    # Plot
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, 
                 x='k', 
                 y='precision',
                 hue='model',
                 style='model',
                 markers=True,
                 dashes=False,
                 errorbar="sd")
    plt.title(f"Train-Test Precision for {dataset} - {target} - {split_type}")
    plt.xlabel('Number of Neighbors Considered (k)')
    plt.ylabel(f"{average.capitalize()} Precision")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    return fig, df


# %%
from time import time
def plot_nn_accuracy_vs_train_taxa_size(dataset, target, split_type, n_jobs=-1, within_test=False, n_bins=4):
    """
    Plots the accuracy of 1-nearest neighbor classification against the number taxa in the training (or test) set.
    
    Args:
        dataset (str): Dataset name.
        target (str): Label column (e.g. 'species').
        split_type (str): Split type (e.g. 'species_even').
        n_jobs (int): Number of parallel jobs.
        within_test (bool): If True, use the test set for both training and testing.

    Returns:
        fig: Matplotlib figure object.
        accuracies_df: DataFrame containing accuracies for each model and k value.
    """
    embeddings = gather_embeddings(dataset, target, split_type)

    # Add a "Cosine (Intensity Agnostic)" method
    embeddings['cosine_intensity_agnostic'] = {
        'train': [None for _ in range(len(embeddings['cosine']['train']))],
        'test': [None for _ in range(len(embeddings['cosine']['test']))]
    }
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))

    # Only do cosine_intensity_agnostic and clip_transformer methods
    for key in list(embeddings.keys()):
        if key not in {'cosine_intensity_agnostic', 'clip_transformer'}:
            del embeddings[key]

    tasks = []
    for key, d in embeddings.items():
        if key == 'metadata':
            continue
        if key in {'cosine', 'cosine_intensity_agnostic', 'clip_transformer', 'clip_transformer_genus_genus'}:
            print(f"Method using {key} cosine distance metric")
            distance_metric = 'cosine'
        else:
            distance_metric = 'euclidean'

        # for k in range(1, 11):
        k = 1
        print(f"Found a total of {len(d['train'])} CV folds.")
        for j in range(len(d['train'])):
            tasks.append((key, k, d['train'][j], d['test'][j], distance_metric, j))


    def _compute_accuracy(key, k, train_df, test_df, distance_metric, cv_fold):
        if key == 'multinomial_classifier':
            # Plot the accuracy as a horizontal line
            # Require that the target level is below the split level
            if level_heirarchy[target] <= level_heirarchy[split_type]:
                return None
            # Calculate accuracy directly on test set per class (no difference for train-test inference)
            g = test_df.groupby('true_label')
            per_class_acc_dict = g.apply(lambda x: np.sum(x['pred_class'] == x['true_label']) / len(x)).to_dict()
            test_total_count_dict = g.size().to_dict()
            train_total_count_dict = train_df['true_label'].value_counts().to_dict()

        else:
            per_class_acc_dict = evaluate_top_k_recall(
                train_df, test_df, key, distance_metric, normalize=True, max_k=1, average="none"
            )
            # This is bugged, but seemed to perform better?
            # per_class_acc_dict = evaluate_top_k_recall(
            #     train_df, test_df, distance_metric, normalize=True, max_k=1, average="none"
            # )
            train_total_count_dict = train_df['true_label'].value_counts().to_dict()
            test_total_count_dict = test_df['true_label'].value_counts().to_dict()

        return {
            'model': key,
            'k': k,
            'per_class_acc_dict': per_class_acc_dict,
            'train_total_count_dict': train_total_count_dict,
            'test_total_count_dict': test_total_count_dict,
            'cv_fold': cv_fold
        }
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_accuracy)(key, k, tr, te, dm, cvf)
        for key, k, tr, te, dm, cvf in tqdm(tasks)
    )

    # Unpack results
    # results = [res for res in results if res is not None]  # Filter out None results
    
    start_time = time()
    accuracies = []
    for res in results:
        model = res['model']
        k = res['k']
        per_class_acc_dict = res['per_class_acc_dict']
        train_total_count_dict = res['train_total_count_dict']
        test_total_count_dict = res['test_total_count_dict']
        cv_fold = res['cv_fold']

        for label, acc in per_class_acc_dict.items():
            # print(f"{model}: Adding for class {label}: {acc:.3f}, train size: {train_total_count_dict.get(label)}")
            train_size = train_total_count_dict.get(label)
            test_size = test_total_count_dict.get(label)
            accuracies.append({
                'model': model,
                'k': k,
                'train_size': train_size,
                'test_size': test_size,
                'accuracy': acc['mean_accuracy'],
                'label': label,
                'cv_fold': cv_fold
            })
    print(f"Time taken to compute accuracies: {time() - start_time:.2f} seconds")

    accuracies_df = pd.DataFrame(accuracies)
    assert accuracies_df['k'].nunique() == 1, "There should be only one k value per model in the results."

    print("CV Size Weighted K=1 Recall:")
    print(accuracies_df.groupby(['model'])['accuracy'].mean())

    print("Independent of CV Size K=1 Recall:")
    print(accuracies_df.groupby(['model', 'cv_fold'])['accuracy'].mean().reset_index().groupby('model')['accuracy'].mean())


    # Plot 
    # x: train class size
    # y: accuracy
    # color: model

    # print(accuracies_df)

    # Plot histogram of train_size 
    start_time = time()
    plt.figure(figsize=(12, 6))
    sns.histplot(accuracies_df['train_size'], bins=30, kde=False)
    plt.title(f"Distribution of Training Class Sizes\n{dataset} - {target} -{split_type}")
    plt.xlabel('Training Class Size')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print(f"Time taken to plot histogram: {time() - start_time:.2f} seconds")


    # fig = plt.figure(figsize=(12, 8))
    # sns.scatterplot(data=accuracies_df, x='train_size', y='accuracy', hue='model', style='model', markers=True) 
    # plt.title(f"1-NN Accuracy vs Training Class Size\n{dataset} - {target} - {split_type}")
    # plt.xlabel('Training Class Size')
    # plt.ylabel('1-NN Accuracy')

    # plt.grid(True)
    # plt.legend(title='Model')
    # plt.tight_layout()
    # plt.show()

    # Get the the range of the middle three quartiles
    accuracies_df['size_bin'] = pd.qcut(accuracies_df['train_size'], n_bins, duplicates='raise')
    accuracies_df['test_size_bin'] = pd.qcut(accuracies_df['test_size'], n_bins, duplicates='raise')

    plt.figure(figsize=(16, 8))
    start_time = time()
    sns.barplot(
        data=accuracies_df,
        x='size_bin',
        y='accuracy',
        hue='model',
        estimator='mean',
        errorbar=None,
        alpha=0.5,
        dodge=True
    )
    plt.title(f"1-NN Accuracy vs Training Class Size (Binned & Filtered)\n{dataset} - {target} - {split_type}")
    plt.xlabel('Training Class Size Bin')
    plt.ylabel('1-NN Accuracy')
    plt.grid(True)
    plt.legend(title='Model')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print(f"Time taken to plot binned accuracies: {time() - start_time:.2f} seconds")

    plt.figure(figsize=(16, 8))
    sns.violinplot(
        data=accuracies_df,
        x='size_bin',
        y='accuracy',
        hue='model',
        split=False,
        inner="quartile",
        width=0.7
    )
    plt.title(f"Distribution of 1-NN Accuracy by Training Class Size Bin\n{dataset} - {target} - {split_type}")
    plt.xlabel('Training Class Size Bin')
    plt.ylabel('1-NN Accuracy')
    plt.grid(True)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    # 2D heatmap of test_size and train_size
    

# %%
def similarity_histogram(dataset, target, split_type, model, mode='train-test', max_pairs=None, distance_metric='euclidean',
                         density=True):
    """
    Plots a histogram of pairwise similarities or distances between embeddings:
    - Within the same label (true_label)
    - Between different labels

    Args:
        dataset (str): Dataset name.
        target (str): Column with class labels.
        split_type (str): Split strategy (e.g., 'species_even').
        model (str): The embedding model to use.
        mode (str): If True, compare only test-test pairs. If False, compare train-test.
        max_pairs (int): Max number of pairs to consider for speed.
        distance_metric (str): 'cosine' or 'euclidean'.

    Returns:
        matplotlib.figure.Figure: The histogram figure.
    """
    embeddings = gather_embeddings(dataset, target, split_type)

     # Add cosine intensity agnostic if not present
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
    print("Warning: Taking first model only")

    if mode == 'test-test':
        df1 = data['test'][0].copy()
        df2 = df1.copy()
    if mode == 'train-test':
        df1 = data['train'][0].copy()
        df2 = data['test'][0].copy()
    elif mode == 'train-train':
        df1 = data['train'][0].copy()
        df2 = df1.copy()

    X = np.stack(df1['embedding'].values)
    Y = np.stack(df2['embedding'].values)
    labels_X = np.array(df1['true_label'])
    labels_Y = np.array(df2['true_label'])


    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)

    n, m = len(X), len(Y)
    pair_indices = [(i, j) for i in range(n) for j in range(m)]

    if max_pairs and max_pairs < len(pair_indices):
        rng = np.random.default_rng(42)
        pair_indices = rng.choice(pair_indices, size=max_pairs, replace=False)

    i1, i2 = zip(*pair_indices)
    X1 = X[list(i1)]
    X2 = Y[list(i2)]
    l1 = labels_X[list(i1)]
    l2 = labels_Y[list(i2)]

    if distance_metric == 'cosine':
        scores = np.sum(X1 * X2, axis=1)  # Higher = more similar
        xlabel = 'Cosine Similarity'
    elif distance_metric == 'euclidean':
        scores = - np.linalg.norm(X1 - X2, axis=1)  # Lower = more similar
        xlabel = 'Euclidean Distance'
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    within = scores[np.array(l1) == np.array(l2)]
    between = scores[np.array(l1) != np.array(l2)]

    bins = np.linspace(min(scores), max(scores), 51)
    bin_width = bins[1] - bins[0]
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    within_hist, _ = np.histogram(within, bins=bins, density=density)
    between_hist, _ = np.histogram(between, bins=bins, density=density)
    if density:
        y_label = 'Density'
    else:
        y_label = 'Count'

    desired_precision = 0.95
    # Compute Precision Threshold directly from scores
    # For each possible threshold, compute precision = TP / (TP + FP)
    # TP: within pairs above threshold, FP: between pairs above threshold (for cosine; reverse for euclidean)
    if distance_metric == 'cosine':
        thresholds = np.linspace(min(scores), max(scores), 500)
        tp = np.array([(within >= t).sum() for t in thresholds])
        fp = np.array([(between >= t).sum() for t in thresholds])
    elif distance_metric == 'euclidean':
        thresholds = np.linspace(min(scores), max(scores), 500)
        tp = np.array([(within <= t).sum() for t in thresholds])
        fp = np.array([(between <= t).sum() for t in thresholds])
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    valid = np.where(precision >= desired_precision)[0]
    if len(valid) > 0:
        precision_threshold = thresholds[valid[0]]
        print(f"{desired_precision} Precision Threshold: {precision_threshold:.3f} ({xlabel})")
    else:
        precision_threshold = None
        print(f"No threshold achieves {desired_precision} precision.")

    # Area estimates
    shared_area = np.sum(np.minimum(within_hist, between_hist)) * bin_width
    unique_within = np.sum(np.maximum(0, within_hist - between_hist)) * bin_width
    unique_between = np.sum(np.maximum(0, between_hist - within_hist)) * bin_width

    print(f"Shared Area: {shared_area:.3f}")
    print(f"Unique Within Area: {unique_within:.3f}")
    print(f"Unique Between Area: {unique_between:.3f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(bin_centers, within_hist, width=bin_width, alpha=0.5, label='Within label', color='blue', align='center')
    ax.bar(bin_centers, between_hist, width=bin_width, alpha=0.5, label='Between labels', color='orange', align='center')
    ax.set_title(f'Pairwise {"Similarity" if distance_metric == "cosine" else "Distance"} for Model: {model}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_label)
    ax.legend()

    return fig  

# %%
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

# %%
_ = top_k_boxplot_per_method_by_label(
    dataset='driams-a',
    target='genera',
    split_type='genera',
    within_test=False,
    n_jobs=-1,
)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=6, macro=False)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'genera', cross_species=True, within_test=True, n_jobs=12, macro=True)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=12, macro=True)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=12, macro=False, cross_species=False)

# %%
tdf = _[1]
tdf[(tdf['model'] == 'multinomial_classifier') & (tdf['k'] == 3)].accuracy.mean()

# %%
tdf['model'].unique()

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=True, n_jobs=12, macro=False, cross_species=True)

# %%
#Old metric
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=True, n_jobs=12, macro=False, cross_species=True)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'genera', within_test=True, n_jobs=12, macro=True, cross_species=True)

# %% [markdown]
# ## DRIAMS Plots Genera/Genera

# %%
_ = binary_curves_plot('driams-a', 'genera', 'genera', test_only=True, max_pairs=1_000_000, n_jobs=4)

# %%
_ = binary_curves_plot('driams-a', 'genera', 'genera', test_only=True, max_pairs=1_000_000, n_jobs=12, between_species=True)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=16, average='micro', num_samples_per_k=5)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=16, average='macro', num_samples_per_k=5)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=16, average='macro', num_samples_per_k=5, require_cross_species=True)

# %%

_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=16, average='micro', num_samples_per_k=5, require_cross_species=True)

# %% [markdown]
# ## DRIAMS Plots Genera/Species

# %%
_ = binary_curves_plot('driams-a', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=12)

# %%
gather_embeddings('driams-a', 'genera', 'species')['multinomial_classifier']

# %%
# DRIAMS Plots
_ = binary_curves_plot('driams-a', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=12)

# %%
_ = binary_curves_plot('driams-a', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=12, between_species=True)

# %%
_ = binary_curves_plot('driams-a', 'genera', 'genera', test_only=True, max_pairs=1_000_000, n_jobs=4, between_species=True)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=False, n_jobs=12, num_samples_per_k=5, average='macro')
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=12, num_samples_per_k=5, average='macro')

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=12, num_samples_per_k=5, average='macro')
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=False, n_jobs=12, num_samples_per_k=5, average='macro')

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=12, num_samples_per_k=5, average='macro', require_cross_species=True)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=12, num_samples_per_k=5, average='macro', require_cross_species=False)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=12, num_samples_per_k=5, average='macro', require_cross_species=True)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=12, num_samples_per_k=5, average='macro', require_cross_species=True)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=12, num_samples_per_k=2, average='macro', require_cross_species=True)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=16, num_samples_per_k=5, average='macro', require_same_species=True)

# %%
_[1].head(50)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=6)
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=False, n_jobs=6)

# %%
s = gather_embeddings('driams-a', 'genera', 'species')['clip_transformer']['train'][0]['species']

# Plot histogram of species counts
plt.figure(figsize=(12, 6))
sns.histplot(s.value_counts(), bins=30, kde=False)
plt.title('Distribution of Species Counts in DRIAMS-A')
plt.xlabel('Number of Species')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False
)

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='driams-a',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False,
    n_bins=10
)

# %%
_temp_df = plot_nn_accuracy_vs_train_taxa_size(
    dataset='driams-a',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False
)

# %%
old_fun_df = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=6, macro=True)

# %%
old_fun_df2 = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=6, macro=True)
_old_fun_df2 = old_fun_df2.copy(deep=True)

# %%
_old_fun_df = old_fun_df.copy()


# %%
_old_fun_df2

# %%
def manual_average(lst):
    averages = []
    d = lst[0]
    print(d)
    for v in d.values():
        if v['total'] > 0:
            averages.append(v['correct']/v['total'])
    return sum(averages) / len(averages) if averages else 0
_old_fun_df2['manual_avg'] = _old_fun_df2.label_to_counts_k.apply(lambda x: manual_average(x))

# %%
_old_fun_df2

# %%
_old_fun_df2.groupby(['model', 'k'])['accuracy'].mean()

# %%
# Expand to one row per model and entry in label_to_counts
def expand_top_k_recall_df(d:dict):
    """
    Expands the top_k_recall_df to have one row per model and entry in label_to_counts.
    """
    expanded_rows = []
    for entry in d:
        model = entry['model']
        accuracies = entry['accuracies']
        genera_counts = entry['genera_counts']
        label_to_counts = entry['label_to_counts_k'][0]

        for i, (label, counts) in enumerate(label_to_counts.items()):
            expanded_rows.append({
                'model': model,
                'k': i + 1,
                'calculated_macro_accuracy': accuracies[0],
                'label': label,
                'correct': counts['correct'],
                'total': counts['total'],
                'accuracy': counts['correct'] / counts['total'] if counts['total'] > 0 else 0,
                'cv_fold': entry.get('cv_fold', None),
                # 'mask_method_acc': entry['accuracies_per_class'].get(label)['mean_accuracy'],
                # 'mask_method_correct': entry['accuracies_per_class'].get(label)['correct'],
                # 'mask_method_total': entry['accuracies_per_class'].get(label)['total'],
                # 'mask_incorrect': entry['accuracies_per_class'].get(label)['incorrect'],
            })

    return pd.DataFrame(expanded_rows)
    

_old_fun_df = expand_top_k_recall_df(_old_fun_df)

# %%
_old_fun_df2.sort_values(by=['model', 'cv_fold'])

# %%
_temp_df[(_temp_df['cv_fold'] == 0) & (_temp_df['model'] == 'cosine_intensity_agnostic')].sort_values(by=['label', 'model'])

# %%
_temp_df.groupby(['model', 'cv_fold'])['accuracy'].mean()#.reset_index().groupby('model')['accuracy'].mean().reset_index()
_temp_df.groupby(['model', 'cv_fold'])['accuracy'].mean().reset_index().groupby('model')['accuracy'].mean().reset_index(), \
_temp_df.groupby(['model'])['accuracy'].mean()

# %%
_temp_df.groupby(['model', 'cv_fold'])['accuracy'].mean()

# %%
_temp_df.groupby(['model', 'cv_fold'])['accuracy'].mean().reset_index().groupby('model')['accuracy'].mean().reset_index(), \
_temp_df.groupby(['model'])['accuracy'].mean()

# %%
_old_fun_df.groupby(['model'])['accuracy'].mean()

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='driams-a',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False
)

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='driams-a',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False,
    n_bins=10
)

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False,
    n_bins=4
)

# %%
species_counts = gather_embeddings('driams-a', 'genera', 'species')['clip_transformer']['train'][0].species.value_counts()

# %%
species_counts.min()

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    n_jobs=6,
    within_test=False
)

# %% [markdown]
# ## DRIAMS Plots Species/Species

# %%
_ = binary_curves_plot('driams-a', 'species', 'species', test_only=True, max_pairs=1_000_000, n_jobs=4)

# %%
_ = nn_accuracy_plot('driams-a', 'species', 'species', within_test=True, n_jobs=12, num_samples_per_k=5)

# %%
_ = nn_accuracy_plot('driams-a', 'species', 'species', within_test=True, n_jobs=16, num_samples_per_k=5, average='micro')

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species_even', within_test=False, n_jobs=16, num_samples_per_k=5, average='macro')

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species_even', within_test=True, n_jobs=16, num_samples_per_k=5, average='macro')

# %%
temp = gather_embeddings('driams-a', 'genera', 'species_even')['cosine']['test'][1]

# %%
temp

# Quick and dirty, compute average cosine similarity per class
grouped_temp = temp.groupby('true_label')['embedding'].apply(lambda x: np.min(cosine_similarity(np.vstack(x))))
grouped_temp = grouped_temp.reset_index()
print(grouped_temp)
# Get min
print(grouped_temp['embedding'].min())

# For each entry, get max similarity to any other entry of a different class
max_sims = []
for i, row in tqdm(temp.iterrows(), total=len(temp)):
    other_embeddings = temp[temp['true_label'] != row['true_label']]['embedding'].values
    if len(other_embeddings) > 0:
        max_sim = np.max(cosine_similarity([row['embedding']], np.vstack(other_embeddings)))
    else:
        max_sim = 0.0
    max_sims.append(max_sim)

temp['max_similar_other'] = max_sims
# Plot histogram of max similarities
plt.figure(figsize=(10, 6))
plt.hist(temp['max_similar_other'], bins=50, alpha=0.7, color='blue')
plt.title('Histogram of Max Similarity to Other Classes')
plt.xlabel('Max Similarity to Other Classes')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# %%
temp

# Manual precision recall curve on temp

ground_truth = []
similarity = []

for i, row in tqdm(temp.iterrows(), total=len(temp)):
    sims = cosine_similarity([row['embedding']], np.vstack(temp['embedding'].values))
    ground_truth.extend(row['true_label'] == temp['true_label'].values)
    similarity.extend(sims[0])

# Create a DataFrame for the predictions
pred_df = pd.DataFrame({
    'true_label': ground_truth,
    'similarity': similarity
})

# %%
pred_df

# %%
# Histogram of similarities per true_label value

plt.figure(figsize=(10, 6))

for label in pred_df['true_label'].unique():
    subset = pred_df[pred_df['true_label'] == label]
    sns.histplot(
        data=subset,
        x='similarity',
        bins=50,
        stat='density',  # Density within each label group
        element='step',
        fill=True,
        alpha=0.5,
        label=str(label)
    )

plt.title('Histogram of Similarities by True Label (Normalized per Label)')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.grid(True)
plt.legend(title='True Label')
plt.show()

# %%
split_root_path = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/data/driams/processed_data/species_even/')
test_folds = [f for f in split_root_path.glob('test_fold_*.pt') if f.is_file()]
train_folds= [f for f in split_root_path.glob('train_fold_*.pt') if f.is_file()]
# Sort the folds to ensure consistent order
test_folds.sort()
train_folds.sort()

print(f"Found {len(test_folds)} test folds and {len(train_folds)} train folds.")
import torch
test_folds = [torch.load(f, weights_only=False) for f in test_folds]
train_folds = [torch.load(f, weights_only=False) for f in train_folds]

# Check for overlap between test_folds
for i in range(len(test_folds)):
    print(f"Total size of fold {i}: {len(test_folds[i])}")
    for j in range(i + 1, len(test_folds)):
        overlap = set(test_folds[i]).intersection(set(test_folds[j]))
        if overlap:
            print(f"Overlap between test fold {i} and {j}: {len(overlap)}")

# Check for overlap between test and train folds
for i in range(len(test_folds)):
    overlap = set(test_folds[i]).intersection(set(train_folds[i]))
    if overlap:
        print(f"Overlap between test fold {i} and train fold: {len(overlap)}")



# %%
test_folds[0]

# %%
# Compute precision and recall at various thresholds
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
    pred_df['true_label'], pred_df['similarity'], pos_label=1
)
# Plot the precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.show()


# %%
precision, recall, thresholds

# %%
pd.read_feather('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score/cosine_10/genera/species_even/k=0/test_inference.feather')

# %%
pd.read_feather('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score/cosine_10/genera/species_even/k=1/test_inference.feather')

# %%
gather_embeddings('driams-a', 'genera', 'species_even')['cosine']['test'][0].head(10)

# %%
_ = binary_curves_plot('driams-a', 'genera', 'species_even', test_only=True, max_pairs=1_000_000, n_jobs=12)

# %% [markdown]
# ## IDBac Plots

# %%
plot_nn_accuracy_vs_train_taxa_size(
    dataset='IDBac-kb',
    target='genera',
    split_type='species',
    n_jobs=12,
    within_test=False
)

# %%
_ = top_k_recall_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, macro=False)
_ = top_k_recall_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, macro=True)

# %%
_ = top_k_recall_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, macro=False)

# %%
_ = nn_accuracy_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, num_samples_per_k=5, average='micro')
_ = nn_accuracy_plot('IDBac-kb', 'genera', 'species', within_test=True, n_jobs=6, num_samples_per_k=5, average='micro')

# %%
_ = nn_accuracy_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, num_samples_per_k=5, average='macro')
_ = nn_accuracy_plot('IDBac-kb', 'genera', 'species', within_test=True, n_jobs=6, num_samples_per_k=5, average='macro')

# %%
_ = nn_accuracy_plot('IDBac-kb', 'genera', 'species', within_test=True, n_jobs=1, num_samples_per_k=5, average='macro', require_cross_species=True)

# %%
_ = binary_curves_plot('IDBac-kb', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=4)

# %%
_ = binary_curves_plot('driams-a', 'genera', 'species', test_only=True, max_pairs=500_000, n_jobs=2, cosine_ablation=True)

# %%
_ = binary_curves_plot('IDBac-kb', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=4, cosine_ablation=True)

# %%
_ = binary_curves_plot('IDBac-kb', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=4, between_species=True)

# %%
_ = binary_curves_plot('IDBac-kb', 'genera', 'genera', test_only=True, max_pairs=1_000_000, n_jobs=4)

# %%


# %%
_ = top_k_precision_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, average='micro')
_ = top_k_precision_plot('IDBac-kb', 'genera', 'species', within_test=False, n_jobs=6, average='macro')

# %%
binary_curves_plot('idbac-kb', 'genera', 'species_even', test_only=True, max_pairs=1_000_000, n_jobs=4)

# %%
_ = dendrogram_from_embeddings(
    dataset='IDBac-kb',
    target='genera',
    split_type='species',
    model='cosine_intensity_agnostic',
    )

# %%
_ = dendrogram_from_embeddings(
    dataset='IDBac-kb',
    target='genera',
    split_type='species',
    model='clip_transformer',
    )

# %%
_ = similarity_histogram(
    dataset='IDBac-kb',
    target='genera',
    split_type='species',
    model='cosine_intensity_agnostic',
    mode='train-test',
    max_pairs=1_000_000,
    distance_metric='cosine',
    density=False
)

# %%
# Histogram of pairwise similarities for IDBac-kb
_ = similarity_histogram(
    dataset='IDBac-kb',
    target='genera',
    split_type='species',
    model='clip_transformer',
    mode='train-test',
    max_pairs=1_000_000,
    distance_metric='cosine',
    density=False
)

# %% [markdown]
# ## End official IDBac Plots

# %%
_ = train_test_precision_plot('idbac-kb', 'genera', 'species')
_ = train_test_recall_plot('idbac-kb', 'g enera', 'species')

# %%
top_k_precision_plot('idbac-kb', 'genera', 'species')
top_k_precision_plot('idbac-kb', 'genera', 'species', within_test=True)

# %%
train_test_curves_nn_plot('driams-a', 'genera', 'species')

# %%
_ = binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=1_000_000)

# %%
_ = binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=1_000_000, between_species=True)

# %%
_ = binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=500_000)

# %% [markdown]
# ## DRIAMS-B Plots

# %%
# Binary 
_ = binary_curves_plot('DRIAMS-B', 'genera', 'species', test_only=True, max_pairs=1_000_000)

# %%
_ = binary_curves_plot('DRIAMS-B', 'genera', 'species', test_only=True, max_pairs=1_000_000, between_species=True)

# %% [markdown]
# ## DRIAMS-C Plots
# 

# %%
_ = binary_curves_plot('DRIAMS-C', 'genera', 'species', test_only=True, max_pairs=1_000_000, between_species=False)

# %%
_ = binary_curves_plot('DRIAMS-C', 'genera', 'species', test_only=True, max_pairs=1_000_000, between_species=True)

# %% [markdown]
# ## DRIAMS-D Plots

# %%
_ = binary_curves_plot('DRIAMS-D', 'genera', 'species', test_only=True, max_pairs=1_000_000, between_species=False)

# %%
_ = binary_curves_plot('DRIAMS-D', 'genera', 'species', test_only=True, max_pairs=1_000_000, between_species=True)

# %% [markdown]
# ## RKI Plots

# %%
_ = binary_curves_plot('RKI', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=1)

# %%
_ = binary_curves_plot('RKI', 'genera', 'species', test_only=True, max_pairs=1_000_000, between_species=True)

# %% [markdown]
# ## Retrieval Across Datasets

# %%
top_k_recall_plot('DRIAMS-B', 'genera', 'species', within_test=False, n_jobs=6, macro=False, db_dataset='DRIAMS-D', cross_species=False)

# %%
top_k_recall_plot('DRIAMS-B', 'genera', 'species', within_test=False, n_jobs=6, macro=True, db_dataset='DRIAMS-D', cross_species=True)

# %%
top_k_recall_plot('DRIAMS-C', 'genera', 'species', within_test=False, n_jobs=6, macro=False, db_dataset='DRIAMS-D', cross_species=False)

# %%
top_k_recall_plot('DRIAMS-C', 'genera', 'species', within_test=False, n_jobs=6, macro=True, db_dataset='DRIAMS-D', cross_species=True)

# %%
top_k_recall_plot('DRIAMS-D', 'genera', 'species', within_test=True, n_jobs=6, macro=False, cross_species=True)

# %%
top_k_recall_plot('RKI', 'genera', 'species', within_test=True, n_jobs=6, macro=False, cross_species=True, db_dataset='DRIAMS-D')

# %%
nn_accuracy_plot('DRIAMS-C', 'genera', 'species', within_test=True, n_jobs=6, num_samples_per_k=5, average='macro', require_cross_species=True)

# %%
_ = similarity_histogram(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=None,
)

_ = similarity_histogram(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    model='clip_transformer_classifier',
    distance_metric='euclidean',
    test_only=True,
    max_pairs=None,
)

_ = similarity_histogram(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    model='clip_transformer',
    distance_metric='euclidean',
    test_only=True,
    max_pairs=None,
)

# %%
_ = similarity_histogram(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=None,
)

_ = similarity_histogram(
    dataset='IDBac-KB',
    target='genera',
    split_type='species',
    model='clip_transformer_classifier',
    distance_metric='euclidean',
    test_only=True,
    max_pairs=None,
)

# %%
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=1_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='clip_transformer_classifier',
    distance_metric='euclidean',
    test_only=True,
    max_pairs=1_000_000 # ML Methods don't reqiure subsampling, but it's added for congruence
)

# %%
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=1_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)

# %%
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=3_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='species',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=3_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)

# %%
# Get a DRIAMS genus / genus test set
driams_a_genera_test = gather_embeddings('driams-a', 'genera', 'genera')['cosine']['test'][0]
driams_a_species_test = gather_embeddings('driams-a', 'genera', 'species')['cosine']['test'][0]
driams_a_genera_train = gather_embeddings('driams-a', 'genera', 'genera')['cosine']['train'][0]
driams_a_species_train = gather_embeddings('driams-a', 'genera', 'species')['cosine']['train'][0]

# %%
# Annotate with a species label
m_data = pd.read_csv('../data/driams/preprocessing/merged_metadata.csv')
species_mapping = m_data.set_index('code')['species'].to_dict()
genus_mapping = m_data.set_index('code')['genus'].to_dict()
driams_a_genera_test['species'] = driams_a_genera_test['strain_name'].map(species_mapping)
driams_a_genera_test['genus'] = driams_a_genera_test['strain_name'].map(genus_mapping)
driams_a_species_test['species'] = driams_a_species_test['strain_name'].map(species_mapping)
driams_a_species_test['genus'] = driams_a_species_test['strain_name'].map(genus_mapping)

driams_a_genera_train['species'] = driams_a_genera_train['strain_name'].map(species_mapping)
driams_a_genera_train['genus'] = driams_a_genera_train['strain_name'].map(genus_mapping)
driams_a_species_train['species'] = driams_a_species_train['strain_name'].map(species_mapping)
driams_a_species_train['genus'] = driams_a_species_train['strain_name'].map(genus_mapping)

# %%
m_data.genus.value_counts()

# %%
m_data[m_data.genus == 'Corynebacterium'].species.value_counts()

# %%
m_data[m_data.genus == 'Pseudomonas'].shape

# %%
# Plot histogram of within genus cosine similarity vs within species cosine similarity for species and genera disjoint sets

def _get_scores_and_samples(df, chunk_size=1000, k=10, threshold=0.2, max_genera=None):
    if max_genera is not None:
        # Randomly sample a subset of genera
        unique_genera = df['genus'].unique()
        rng = np.random.default_rng(42)
        sampled_genera = rng.choice(unique_genera, size=min(max_genera, len(unique_genera)), replace=False)
        df = df[df['genus'].isin(sampled_genera)]

    # Calculate pairwise cosine similarities in chunks to avoid memory issues
    X = np.stack(df['embedding'].values)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = np.nan_to_num(X)
    n = len(X)

    print("Number of unique genera:", df['genus'].nunique())

    within_genus = []
    within_species = []
    within_genus_genus_names = []
    within_species_genus_names = []
    genus_pairs = []
    species_pairs = []

    for start in tqdm(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        X_chunk = X[start:end]
        l1_genus_chunk = df['genus'].values[start:end]
        l1_species_chunk = df['species'].values[start:end]

        # Compute cosine similarity between the chunk and the entire dataset
        scores_chunk = np.dot(X_chunk, X.T)

        for i, scores_row in enumerate(scores_chunk):
            l1_genus = l1_genus_chunk[i]
            l1_species = l1_species_chunk[i]

            # Mask for within genus
            within_genus_mask = (df['genus'].values == l1_genus)
            within_genus.extend(scores_row[within_genus_mask])

            # Collect pairs for genus with similarity below threshold
            genus_indices = np.where(within_genus_mask)[0]
            genus_pairs.extend([(start + i, j) for j in genus_indices if scores_row[j] < threshold])

            # Mask for within species
            within_species_mask = (df['species'].values == l1_species)
            within_species.extend(scores_row[within_species_mask])

            # Collect pairs for species with similarity below threshold
            species_indices = np.where(within_species_mask)[0]
            species_pairs.extend([(start + i, j) for j in species_indices if scores_row[j] < threshold])

            # Store genus names for within genus and species
            within_genus_genus_names.extend([l1_genus] * len(scores_row[within_genus_mask]))
            within_species_genus_names.extend([l1_genus] * len(scores_row[within_species_mask]))

    # Randomly sample k pairs for genus and species
    rng = np.random.default_rng(42)
    sampled_genus_pairs = rng.choice(genus_pairs, size=min(k, len(genus_pairs)), replace=False).tolist()
    sampled_species_pairs = rng.choice(species_pairs, size=min(k, len(species_pairs)), replace=False).tolist()

    # Convert sampled pairs to dict with code, embedding, cosine similarity, and genus/species
    genus_pairs_dict = [
        {
            "code_1": df.iloc[pair[0]]['strain_name'],
            "embedding_1": df.iloc[pair[0]]['embedding'],
            "code_2": df.iloc[pair[1]]['strain_name'],
            "embedding_2": df.iloc[pair[1]]['embedding'],
            "cosine": np.dot(df.iloc[pair[0]]['embedding'], df.iloc[pair[1]]['embedding']),
            "genus": df.iloc[pair[0]]['genus'],
        }
        for pair in sampled_genus_pairs
    ]
    species_pairs_dict = [
        {
            "code_1": df.iloc[pair[0]]['strain_name'],
            "embedding_1": df.iloc[pair[0]]['embedding'],
            "code_2": df.iloc[pair[1]]['strain_name'],
            "embedding_2": df.iloc[pair[1]]['embedding'],
            "cosine": np.dot(df.iloc[pair[0]]['embedding'], df.iloc[pair[1]]['embedding']),
            "species": df.iloc[pair[0]]['species'],
        }
        for pair in sampled_species_pairs
    ]

    return (
        np.array(within_genus),
        np.array(within_species),
        genus_pairs_dict,
        species_pairs_dict,s
        within_genus_genus_names,
        within_species_genus_names,
    )

genera_within_genus, genera_within_species, genera_genus_pairs, genera_species_pairs, genera_genera_pair_labels, genera_species_pair_labels = _get_scores_and_samples(
    driams_a_genera_test
)

species_within_genus, species_within_species, species_genus_pairs, species_species_pairs, species_genera_pair_labels, species_species_pair_labels = _get_scores_and_samples(
    driams_a_species_test
)

# %%
len(genera_within_genus), len(genera_genera_pair_labels)

# %%
bplot_within_genus_df

# %%
# Box plot of within genus similarity by genus in each pair

plt.figure(figsize=(12, 6))
genera_bplot_within_genus_df = pd.DataFrame({
    'pair_similarity': genera_within_genus,
    'genus': genera_genera_pair_labels
})

# Sort the DataFrame by genus alphabetically
genera_bplot_within_genus_df = genera_bplot_within_genus_df.sort_values(by='genus')

sns.boxplot(data=genera_bplot_within_genus_df, x='genus', y='pair_similarity')

# Rotate x-axis labels
plt.xticks(rotation=90)
plt.title('Within Genus Similarity by Genus (Genus Disjoint)')
plt.xlabel('Genus')
plt.ylabel('Pair Similarity')
plt.tight_layout()
plt.show()

# %%
from matplotlib.patches import Patch
# Box plot of within genus similarity by genus in each pair

plt.figure(figsize=(12, 6))
species_bplot_within_genus_df = pd.DataFrame({
    'pair_similarity': species_within_genus,
    'genus': species_genera_pair_labels
})

# Sort the DataFrame by genus alphabetically
species_bplot_within_genus_df = species_bplot_within_genus_df.sort_values(by='genus')

sns.boxplot(data=species_bplot_within_genus_df, x='genus', y='pair_similarity')

# Rotate x-axis labels
plt.xticks(rotation=90)
plt.title('Within Genus Similarity by Genus (Species Disjoint)')
plt.xlabel('Genus')
plt.ylabel('Pair Similarity')
plt.tight_layout()
plt.show()

# Get common genus names between species and genera splits
common_genera = set(genera_genera_pair_labels).intersection(set(species_genera_pair_labels))
print(f"Common genera: {len(common_genera)}")

# Do a boxplot for just the common genera with both splits
common_genera = list(common_genera)

# Add an identifier column
species_bplot_within_genus_df['split'] = 'species'
genera_bplot_within_genus_df['split'] = 'genera'

bplot_with_labels = pd.concat([species_bplot_within_genus_df, genera_bplot_within_genus_df])
bplot_with_labels_common = bplot_with_labels[bplot_with_labels['genus'].isin(common_genera)]
bplot_with_labels_remainder = bplot_with_labels[~bplot_with_labels['genus'].isin(common_genera)]

plt.figure(figsize=(12, 6))
sns.boxplot(data=bplot_with_labels_common, x='genus', y='pair_similarity', hue='split')
plt.title('Within Genus Similarity by Genus (Common Genera)')
plt.xlabel('Genus')
plt.ylabel('Pair Similarity')
plt.legend(title='Split')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Histogram of common vs remainder genera (in each set)
sns.histplot(data=bplot_with_labels_common, x='pair_similarity', bins=50, 
             color='violet', label='Shared', alpha=0.5)

# Plot split histograms with hue
ax = sns.histplot(data=bplot_with_labels_remainder, x='pair_similarity', hue='split', color=['blue', 'orange'],
                  bins=50, alpha=0.5)

# Create custom legend
handles = [Patch(color='violet', alpha=0.5, label='Shared')]
handles += [Patch(color='blue', alpha=0.5, label='Species'),
            Patch(color='orange', alpha=0.5, label='Genera')]
plt.title('Within Genus Similarity Distribution (Common Genera)')
plt.xlabel('Pair Similarity')
plt.ylabel('Density')
plt.legend(handles=handles, title='Split')
plt.tight_layout()
plt.show()


# %%
# Plot mean cosine similarity vs number of species in genus
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True)

# First subplot: genera_bplot_within_genus_df
genera_bplot_within_genus_df = pd.DataFrame({
    'pair_similarity': genera_within_genus,
    'genus': genera_genera_pair_labels
})

# Calculate mean pair similarity for each genus
genera_bplot_within_genus_df = genera_bplot_within_genus_df.groupby('genus').agg(
    pair_similarity=('pair_similarity', 'mean'),
).reset_index()
genera_bplot_within_genus_df['num_species'] = genera_bplot_within_genus_df['genus'].map(
    lambda x: len(driams_a_genera_test[driams_a_genera_test['genus'] == x]['species'].unique())
)

sns.scatterplot(
    data=genera_bplot_within_genus_df,
    x='num_species',
    y='pair_similarity',
    alpha=0.8,
    ax=axes[0]
)
axes[0].set_title('Genera Disjoint')
axes[0].set_xlabel('Number of Species')
axes[0].set_ylabel('Pair Similarity')

# Second subplot: species_bplot_within_genus_df
species_bplot_within_genus_df = pd.DataFrame({
    'pair_similarity': species_within_genus,
    'genus': species_genera_pair_labels
})

# Calculate mean pair similarity for each genus
species_bplot_within_genus_df = species_bplot_within_genus_df.groupby('genus').agg(
    pair_similarity=('pair_similarity', 'mean'),
).reset_index()
species_bplot_within_genus_df['num_species'] = species_bplot_within_genus_df['genus'].map(
    lambda x: len(driams_a_species_test[driams_a_species_test['genus'] == x]['species'].unique())
)

sns.scatterplot(
    data=species_bplot_within_genus_df,
    x='num_species',
    y='pair_similarity',
    alpha=0.8,
    ax=axes[1]
)
axes[1].set_title('Species Disjoint')
axes[1].set_xlabel('Number of Species')

# Add grids
axes[0].grid(True)
axes[1].grid(True)
axes[0].set_xlim(0, 13)
axes[1].set_xlim(0, 13)
axes[0].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# %%
import pandas as pd
all_idbac_data = pd.read_feather('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score/cosine_10/all/all_inference.feather')
idbac_metadata = pd.read_csv('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/data/idbac_db/raw/ammended_db.csv')

idbac_name_to_genus = idbac_metadata.set_index('Strain name')['genus'].to_dict()
all_idbac_data['strain_name'] = all_idbac_data['strain_name'].apply(lambda x: x[0])
all_idbac_data['true_label'] = all_idbac_data['strain_name'].map(idbac_name_to_genus)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

def plot_cosine_similarity_histogram(df, title, bins=50, binarize=False):
    # Compute all pairs similarity within label
    X = np.stack(df['embedding'].values)
    if binarize:
        # Binarize the embeddings
        X = (X > 0.02).astype(float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = np.nan_to_num(X)
    n = len(X)

    scores = np.dot(X, X.T)
    within_label = []
    between_label = []
    labels = df['true_label'].values
    # Create masks for within-label and between-label
    within_label_mask = labels[:, None] == labels[None, :]
    between_label_mask = ~within_label_mask

    # Extract scores using the masks
    within_label = scores[within_label_mask]
    between_label = scores[between_label_mask]

    print(f"Within label: {len(within_label)}")
    print(f"Between label: {len(between_label)}")

    # Plot histogram
    # plt.figure(figsize=(12, 6))
    # plt.hist(within_label.flatten(), bins=bins, alpha=0.5, label='Within label', color='blue', density=True)
    # plt.hist(between_label.flatten(), bins=bins, alpha=0.5, label='Between labels', color='orange', density=True)
    # plt.title(title)
    # plt.xlabel("Cosine Similarity")
    # plt.ylabel("Density")
    # plt.legend()

    # Cleanup what we can to save memory
    del X
    del within_label
    del between_label
    del between_label_mask
    del labels
    del bins
    
    within_label_mask = within_label_mask.flatten()
    scores = scores.flatten()

    num_samples = 10_000_000
    # Random subsammple if greater than num_samples
    if len(scores) > num_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(scores), size=num_samples, replace=False)
        scores = scores[indices]
        within_label_mask = within_label_mask[indices]

    # Plot precision/recall curve
    precision, recall, _ = precision_recall_curve(within_label_mask, scores)
    plt.figure(figsize=(12, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # Sort recall and precision in ascending order of recall
    sorted_indices = np.argsort(recall)
    recall = recall[sorted_indices]
    precision = precision[sorted_indices]
    auc = np.trapz(precision, recall)
    plt.title(f'Precision-Recall Curve (auc={auc:.2f})')
    plt.legend()
    plt.show()

plot_cosine_similarity_histogram(
    all_idbac_data,
    title="Cosine Similarity Histogram for All IDBac Data",
    bins=50
)
plot_cosine_similarity_histogram(
    all_idbac_data,
    title="Cosine Similarity Histogram for All IDBac Data",
    bins=50, 
    binarize=True
)

# %%
all_dirams_data = pd.read_feather('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score/cosine_10/all/all_inference.feather')
driams_metadata = pd.read_csv('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/data/driams/preprocessing/merged_metadata.csv')
driams_name_to_genus = driams_metadata.set_index('code')['genus'].to_dict()
all_dirams_data['strain_name'] = all_dirams_data['strain_name'].apply(lambda x: x[0])
all_dirams_data['true_label'] = all_dirams_data['strain_name'].map(driams_name_to_genus)

# %%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set up the figure with gridspec
fig = plt.figure(figsize=(10, 4), constrained_layout=True)
gs = gridspec.GridSpec(
    2, 4,
    height_ratios=[1, 4],
    width_ratios=[4, 0.8, 4, 0.8],
    hspace=0.05,
    wspace=0.15  # Reduce if overlap persists
)

# Genera subplot
ax_scatter_1 = fig.add_subplot(gs[1, 0])
ax_histx_1 = fig.add_subplot(gs[0, 0], sharex=ax_scatter_1)
ax_histy_1 = fig.add_subplot(gs[1, 1], sharey=ax_scatter_1)

# Species subplot
ax_scatter_2 = fig.add_subplot(gs[1, 2])
ax_histx_2 = fig.add_subplot(gs[0, 2], sharex=ax_scatter_2)
ax_histy_2 = fig.add_subplot(gs[1, 3], sharey=ax_scatter_2)

# First plot: Genera disjoint
sns.scatterplot(
    data=genera_bplot_within_genus_df,
    x='num_species',
    y='pair_similarity',
    ax=ax_scatter_1,
    alpha=0.8
)
clean_df = genera_bplot_within_genus_df.dropna(subset=['num_species', 'pair_similarity'])

sns.histplot(
    data=clean_df,
    x='num_species',
    ax=ax_histx_1,
    kde=True,
    bins=20
)

sns.histplot(
    data=clean_df,
    y='pair_similarity',
    ax=ax_histy_1,
    kde=True,
    bins=20
)

ax_scatter_1.set_title('Genera Disjoint', loc='right')
ax_scatter_1.set_xlabel('Number of Species')
ax_scatter_1.set_ylabel('Pair Similarity')
ax_histx_1.axis('off')
ax_histy_1.axis('off')
ax_scatter_1.grid(True)

# Second plot: Species disjoint
sns.scatterplot(
    data=species_bplot_within_genus_df,
    x='num_species',
    y='pair_similarity',
    ax=ax_scatter_2,
    alpha=0.8
)

clean_df = species_bplot_within_genus_df.dropna(subset=['num_species', 'pair_similarity'])
sns.histplot(
    data=clean_df,
    x='num_species',
    ax=ax_histx_2,
    kde=True,
    bins=20
)
sns.histplot(
    data=clean_df,
    y='pair_similarity',
    ax=ax_histy_2,
    kde=True,
    bins=20
)

ax_scatter_2.set_title('Species Disjoint', loc='right')
ax_scatter_2.set_xlabel('Number of Species')
ax_scatter_2.set_ylabel('')
ax_histx_2.axis('off')
ax_histy_2.axis('off')
ax_scatter_2.grid(True)

# Set y and x lims
ax_scatter_1.set_ylim(0, 1)
ax_scatter_2.set_ylim(0, 1)
ax_scatter_1.set_xlim(0, 13)
ax_scatter_2.set_xlim(0, 13)

plt.tight_layout()
plt.show()


# %%
# Plot histogram for within genus for both sets
plt.figure()
plt.hist(genera_within_genus, bins=50, alpha=0.5, label='Within Genus (Genus Disjoint Test Set)', color='blue', density=True)
plt.hist(species_within_genus, bins=50, alpha=0.5, label='Within Genus (Species Disjoint Test Set)', color='orange', density=True)

plt.title('Within Genus Cosine Similarity')

plt.xlabel('Cosine Similarity')
plt.ylabel('Density')

plt.legend()
plt.show()

# %%
# Plot histogram for within species for both sets
plt.figure()
plt.hist(genera_within_species, bins=50, alpha=0.5, label='Within Species (Genus Disjoint Test Set)', color='blue', density=True)
plt.hist(species_within_species, bins=50, alpha=0.5, label='Within Species (Species Disjoint Test Set)', color='orange', density=True)
plt.title('Within Species Cosine Similarity')
plt.xlabel('Cosine Similarity')

plt.ylabel('Density')
plt.legend()
plt.show()

# %%
import numpy as np

import matplotlib.pyplot as plt

def plot_mirror_embeddings(pair_dicts, idx=0, title="Mirror Plot of Embeddings"):
    """
    Creates a mirror plot of embedding_1 and embedding_2 from the given dictionary list.

    Args:
        pair_dicts (list): List of dictionaries containing 'embedding_1' and 'embedding_2'.
        title (str): Title of the plot.
    """
    pair = pair_dicts[idx]
    # for i, pair in enumerate(pair_dicts):
    embedding_1 = pair['embedding_1']
    embedding_2 = pair['embedding_2']
    print(pair['cosine'])
    print(f"Num peaks in embedding 1: {np.sum(embedding_1 > 0.02)}")
    print(f"Num peaks in embedding 2: {np.sum(embedding_2 > 0.02)}")

    plt.figure(figsize=(10, 4))
    print(embedding_1)
    plt.stem(range(len(embedding_1)), embedding_1, label='Embedding 1', basefmt=" ", markerfmt=" ", )
    plt.stem(range(len(embedding_2)), -embedding_2, label='Embedding 2', basefmt=" ", markerfmt=" ", )
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title(f"{title} - Pair {idx+1}")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage with genera_genus_pairs
plot_mirror_embeddings(genera_genus_pairs, idx=4, title="Mirror Plot of Genera Genus Embeddings")

# %%
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='species',
    split_type='genera',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=3_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='species',
    split_type='species',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=3_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)

# %%
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=1_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='clip_transformer_classifier',
    distance_metric='euclidean',
    test_only=True,
    max_pairs=1_000_000 # ML Methods don't reqiure subsampling, but it's added for congruence
)


