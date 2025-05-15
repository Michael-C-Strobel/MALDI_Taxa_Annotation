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

# %%
# Set font size to large
plt.rcParams.update({'font.size': 20})
# Set the style of seaborn
sns.set(style="white")
sns.set_palette("Set2")

# %%
os.getcwd()

# %%
NAME_MAPPINGS = {
    'clip_transformer': "Contrastive Transformer",
    'clip_transformer_intensity_agnostic': "Contrastive Transformer (Int. Agn.)",
    'clip_transformer_classifier': "Classifier Embeddings",
    'cosine': "Cosine Similarity",
    'prototypical_transformer': "Prototypical Transformer",
    'cosine_intensity_agnostic': "Cosine Similarity (Int. Agn.)",
    'multinomial_classifier': "Multinomial Classifier",
    'clip_transformer_genus_genus': 'Contrastive Transformer (Task-Specific HParams)'
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
    elif dataset.lower() == 'idbac-kb':
        base_dir = base_dir / 'lightning_logs'
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

    # inference/{args.target}/{args.split_type}/"
    if target == 'genera':
        if split_type == 'genera':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / 'genera'/ 'genera' / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type
                    for i in range(0,6) # Omit fold 7 where val is used for test
                ]
                clip_transformer_genus_genus_path = [
                    _base_dir / 'genera'/ 'genera' / f'k={i}' / 'CLIP_Transformer' / 'version_1' / 'inference' / target / split_type
                    for i in range(0,6)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(1,6)]
                prototypical_transformer_path = None
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None
                clip_transformer_intensity_agnostic_path = None
            
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / 'genera'/ 'species' / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type
                    for i in range(0,3)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(1,6)]
                prototypical_transformer_path = None
                multinomial_classifier_path = [
                    _base_dir / 'genera'/ 'species' / f'k={i}' / 'Multinomial_Logistic_Classifier' / 'version_0' / 'inference' / target / split_type
                    for i in range(0,3)
                ]
            elif dataset.lower() == 'idbac-kb':
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')

                clip_transformer_path = [
                    _base_dir / 'genera'/ 'species' / f'k={i}' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type
                    for i in range(0,3)
                ]
                clip_transformer_classifer_path = None
                cosine_path = [_base_dir / 'cosine_10' / target / split_type / f'k={i}' for i in range(1,6)]
                prototypical_transformer_path = None


        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata_code_accessions.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
                prototypical_transformer_path = None
                           


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
                clip_transformer_path = None
                clip_transformer_classifer_path = None
                cosine_path = None
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

    output_dict['metadata'] = metadata_table
    
    return output_dict
        

# %%
def gather_embeddings_cosine_only(dataset:str, target:str, split_type:str):
    base_dir = Path('../bin/ml/')

    if dataset.lower() == 'driams-a':
        base_dir = base_dir / 'lightning_logs_DRIAMS_A'
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
                raise NotImplementedError("This configuration is not implemented")
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
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import recall_score
from collections import defaultdict

def evaluate_top_k_recall(
    train_df,
    test_df,
    distance_metric='euclidean',
    normalize=True,
    max_k=10,
    macro=False,
    within_test=False,
) -> Tuple[List[float], List[Dict[str, str]]]:
    if within_test:
        train_df = test_df

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

    sorted_indices = np.argsort(-scores, axis=1)[:, :max_k]

    train_labels = train_df['true_label'].values
    test_labels = test_df['true_label'].values
    strain_names = test_df['strain_name'].values
    predicted_k_labels = train_labels[sorted_indices]

    # Accuracy per k
    accuracies = []
    failure_indices_per_k = []

    if macro:
        label_to_counts_k = [defaultdict(lambda: {'correct': 0, 'total': 0}) for _ in range(max_k)]
        for i, true_label in enumerate(test_labels):
            for k in range(1, max_k + 1):
                label_to_counts_k[k - 1][true_label]['total'] += 1
                if true_label in predicted_k_labels[i, :k]:
                    label_to_counts_k[k - 1][true_label]['correct'] += 1

        for label_to_counts in label_to_counts_k:
            recalls = [v['correct'] / v['total'] for v in label_to_counts.values() if v['total'] > 0]
            accuracies.append(np.mean(recalls))
    else:
        match_matrix = predicted_k_labels == test_labels[:, None]
        for k in range(1, max_k + 1):
            hits = match_matrix[:, :k].any(axis=1)
            accuracies.append(hits.mean())
            failure_indices_per_k.append(np.where(~hits)[0])

    if len(failure_indices_per_k) == 0:
        failure_cases = [np.array([]) for _ in range(max_k)]
    else:
        failure_cases = [
            {
                'strain_name': strain_names[i],
                'true_label': test_labels[i],
                'predicted_labels': predicted_k_labels[i].tolist(),
            }
            for i in failure_indices_per_k[-1]  # only keep failures at max_k
        ]

    return accuracies, failure_cases


def top_k_recall_plot(dataset, target, split_type, n_jobs=-1, within_test=False, macro=False):
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

    # Step 1: Schedule all compute_sorted_neighbors jobs
    job_args = []
    for key in embeddings.keys():
        if key == "metadata":
            continue
        if key == 'multinomial_classifier':
            continue
        if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic' }:
            distance_metric = 'cosine'
        else:
            distance_metric = 'euclidean'
        for i in range(len(embeddings[key]['train'])):
            train_df = embeddings[key]['train'][i]
            test_df = embeddings[key]['test'][i]
            if within_test:
                train_df = test_df
            job_args.append((key, train_df, test_df, distance_metric))

    results =  Parallel(n_jobs=n_jobs)(
        delayed(lambda key, train_df, test_df, metric: {
            'model': key,
            'accuracies': evaluate_top_k_recall(train_df, test_df, distance_metric=metric, max_k=10, macro=macro, within_test=within_test)[0]
        })(key, train_df, test_df, distance_metric)
        for key, train_df, test_df, distance_metric in tqdm(job_args)
    )

    # Unpack into final format
    final_results = []
    for res in results:
        for k, acc in enumerate(res['accuracies'], start=1):
            final_results.append({
                'model': res['model'],
                'k': k,
                'accuracy': acc,
            })

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
                })

    accuracies_df = pd.DataFrame(final_results)
    accuracies_df['k'] = accuracies_df['k'].astype(int)
    accuracies_df['accuracy'] = accuracies_df['accuracy'].astype(float)
    accuracies_df = accuracies_df.sort_values(by=['model', 'k'])

    # Line plot
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=accuracies_df, 
                 x='k', 
                y='accuracy',
                hue='model',
                style='model',
                markers=True,
                dashes=False,
                errorbar="sd",)
    plt.title(f"Train-Test Recall for {dataset} - {target} - {split_type}")
    plt.xlabel('Number of Neighbors Considered (k)')
    plt.ylabel('% of Queries with Correct Taxa in Top k')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()

    return fig, accuracies_df

# def get_recall_in_top_k(
#     train_df,
#     test_df,
#     k: int,
#     distance_metric: str = 'euclidean',
#     normalize: bool = True,
#     macro: bool = False,
# ) -> Tuple[float, List[Dict[str, str]]]:
#     """Computes recall@k: either micro (overall) or macro (averaged across classes)."""
    
#     train_embeddings = np.vstack(train_df['embedding'].values)
#     test_embeddings = np.vstack(test_df['embedding'].values)

#     if normalize:
#         train_embeddings = np.nan_to_num(train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True))
#         test_embeddings = np.nan_to_num(test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True))

#     if distance_metric == 'cosine':
#         scores = cosine_similarity(test_embeddings, train_embeddings)
#     elif distance_metric == 'euclidean':
#         scores = -euclidean_distances(test_embeddings, train_embeddings)
#     else:
#         raise ValueError(f"Unknown distance metric: {distance_metric}")

#     top_k_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
#     row_indices = np.arange(scores.shape[0])[:, None]
#     top_k_sorted = np.argsort(scores[row_indices, top_k_idx], axis=1)[:, ::-1]
#     top_k_final = top_k_idx[row_indices, top_k_sorted]

#     test_labels = test_df['true_label'].values
#     train_labels = train_df['true_label'].values
#     strain_names = test_df['strain_name'].values
#     predicted_k_labels = train_labels[top_k_final]

#     if macro:
#         label_to_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
#         for i, true_label in enumerate(test_labels):
#             label_to_counts[true_label]['total'] += 1
#             if true_label in predicted_k_labels[i]:
#                 label_to_counts[true_label]['correct'] += 1
#         recalls = [
#             v['correct'] / v['total'] for v in label_to_counts.values() if v['total'] > 0
#         ]
#         acc = np.mean(recalls)
#         failure_indices = [i for i, true_label in enumerate(test_labels) if true_label not in predicted_k_labels[i]]
#     else:
#         match_matrix = predicted_k_labels == test_labels[:, None]
#         hits = match_matrix.any(axis=1)
#         acc = hits.mean()
#         failure_indices = np.where(~hits)[0]

#     failure_cases = [
#         {
#             'strain_name': strain_names[i],
#             'true_label': test_labels[i],
#             'predicted_labels': predicted_k_labels[i].tolist(),
#         }
#         for i in failure_indices
#     ]

#     return acc, failure_cases

# def top_k_recall_plot(dataset, target, split_type, n_jobs=-1, within_test=False, macro=False):
#     embeddings = gather_embeddings(dataset, target, split_type)

#     # Add a "Cosine (Intensity Agnostic)" method
#     embeddings['cosine_intensity_agnostic'] = embeddings['cosine'].copy()
    
#     for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
#         embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine_intensity_agnostic']['train'][i].copy(deep=True)
#         embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine_intensity_agnostic']['test'][i].copy(deep=True)
#         embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
#         embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
    
#     tasks = []
#     for key, train_test_dict in embeddings.items():
#         if key == 'metadata':
#             continue
#         for i in range(len(train_test_dict['train'])):
#             distance_metric = 'euclidean'
#             if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic'}:
#                 print(f"Using cosine distance for {key}")
#                 distance_metric = 'cosine'
#             train_embeddings = train_test_dict['train'][i]
#             test_embeddings = train_test_dict['test'][i]
#             if within_test:
#                 train_embeddings = test_embeddings

#             for k in np.arange(1, 11):
#                 tasks.append((key, k, train_embeddings, test_embeddings, distance_metric))

#     def compute_accuracy(key, k, train_embeddings, test_embeddings, distance_metric):
#         acc, _ = get_recall_in_top_k(
#             train_embeddings,
#             test_embeddings,
#             k=k,
#             distance_metric=distance_metric,
#             normalize=True,
#             macro=macro,
#         )
#         return {
#             'model': key,
#             'k': k,
#             'accuracy': acc,
#         }

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(compute_accuracy)(key, k, train_embeddings, test_embeddings, distance_metric)
#         for key, k, train_embeddings, test_embeddings, distance_metric in tqdm(tasks)
#     )

#     accuracies_df = pd.DataFrame(results)
#     accuracies_df['k'] = accuracies_df['k'].astype(int)
#     accuracies_df['accuracy'] = accuracies_df['accuracy'].astype(float)
#     accuracies_df = accuracies_df.sort_values(by=['model', 'k'])

#     # Line plot
#     fig = plt.figure(figsize=(12, 8))
#     sns.lineplot(data=accuracies_df, x='k', y='accuracy', hue='model', style='model', markers=True, dashes=False)
#     plt.title(f"Train-Test Recall for {dataset} - {target} - {split_type}")
#     plt.xlabel('Number of Neighbors Considered (k)')
#     plt.ylabel('% of Queries with Correct Taxa in Top k')
#     plt.legend(title='Model')
#     plt.grid(True)
#     plt.show()
#     return fig, accuracies_df
        

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
                )-> float:
    df = test_df.copy(deep=True)
    np.random.seed(random_seed)


    if dedicated_db is None: # Generate a faux db using the test set
        label_counts = df['true_label'].value_counts()
        singletons = label_counts[label_counts == 1].index.tolist()

        df = df[~df['true_label'].isin(singletons)]

        
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

        faux_db = df.loc[list(sampled_indices_set), :].values
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
        for j, faux_row in enumerate(faux_db):
            faux_embedding = faux_row[2]
            faux_strain_name = faux_row[1]
            faux_true_label = faux_row[4]    # I HAD TO CHANGE TO 4
            if distance_metric == 'cosine':
                    similarity = np.dot(embedding, faux_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(faux_embedding))
            elif distance_metric == 'euclidean':
                    similarity = -1 * np.linalg.norm(embedding - faux_embedding)
            else:
                    raise ValueError("Unknown distance metric: {}".format(distance_metric))
            similarities.append((faux_true_label, similarity))

        # Always take largest
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

def nn_accuracy_plot(dataset, target, split_type, n_jobs=-1, within_test=False):
    embeddings = gather_embeddings(dataset, target, split_type)

    # Add a "Cosine (Intensity Agnostic)" method
    embeddings['cosine_intensity_agnostic'] = {'train': [None for _ in range(len(embeddings['cosine']['train']))],
                                               'test': [None for _ in range(len(embeddings['cosine']['test']))]}
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.02).astype(int))

    def compute_accuracy(key, test_embeddings, distance_metric, k, random_seed, retrieval_db):
        acc, _ = nn_accuracy(
            test_embeddings, 
            k=k,
            random_seed=random_seed,
            distance_metric=distance_metric,
            normalize=True,
            dedicated_db=retrieval_db,
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

            for random_seed in [42]: #np.arange(42, 42+10):
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
            acc = recall_score(test_df['true_label'], test_df['pred_class'], average='micro')    # Micro is equivlent to accuracy
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
    plt.xlabel('Number of Neighbors Considered (k)')
    plt.ylabel('Accuracy')
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

def _binary_similarity_curves(df1, df2, distance_metric='cosine', max_pairs=None):
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
    else:
        # Generate them randomly
        rng = np.random.default_rng(42)
        flat_indices = rng.choice(n * m, size=max_pairs, replace=False)
        pair_indices = [(i // m, i % m) for i in flat_indices]

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


    fpr, tpr, _ = roc_curve(y_true.flatten(), sim.flatten())
    precision, recall, _ = precision_recall_curve(y_true.flatten(), sim.flatten())
    return fpr, tpr, auc(fpr, tpr), precision, recall, auc(recall, precision)

def _compute_interpolated_curves(df1, df2, method, data_idx, interp_points, metric, max_pairs, ignore_intensity=False):
    try:
        if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
            print(f"Method {method} is using intensity agnostic embeddings")
            df1['embedding'] = df1['embedding'].apply(lambda x: (x > 0.02).astype(int))
            df2['embedding'] = df2['embedding'].apply(lambda x: (x > 0.02).astype(int))

        fpr, tpr, roc_auc_val, prec, rec, pr_auc_val = _binary_similarity_curves(df1, df2, metric, max_pairs)
        fdr = 1 - prec

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

def static_precision_recall_roc(df1, df2, target, split_type, method, max_pairs=None):
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
    else:
        # Generate them randomly
        rng = np.random.default_rng(42)
        flat_indices = rng.choice(n * m, size=max_pairs, replace=False)
        pair_indices = [(i // m, i % m) for i in flat_indices]

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


def binary_curves_plot(dataset, target, split_type, test_only=False, max_pairs=None,
                       cosine_ablation=False, n_jobs=-1):
    if cosine_ablation:
        embeddings = gather_embeddings_cosine_only(dataset, target, split_type)
    else:
        embeddings = gather_embeddings(dataset, target, split_type)

    # Add a "Cosine (Intensity Agnostic)" method
    embeddings['cosine_intensity_agnostic'] = embeddings['cosine'].copy()

    # Print prior probability of equal and unequal taxa
    a_key = [x for x in list(embeddings.keys()) if x != 'metadata'][0]
    df = embeddings[a_key]['test'][0]
    labels = df['true_label'].values
    # print(f"Labels", df['true_label'].value_counts())
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_pos = 0
    total = len(labels) ** 2
    for label, count in zip(unique_labels, counts):
        total_pos += count **2
    
    print(f"% of pairs with equal taxa: {total_pos / total:.2f}")

    roc_fig, roc_ax = plt.subplots()
    pr_fig, pr_ax = plt.subplots()
    fdr_fig, fdr_ax = plt.subplots()

    interp_points = np.linspace(0, 1, 100)

    print("Found methods:", list(embeddings.keys()))

    for method, data_list in embeddings.items():
        if method == 'metadata':
            continue

        if method == 'cosine' or method == 'cosine_intensity_agnostic' \
            or method == 'clip_transformer' or method == 'clip_transformer_genus_genus':
             print(f"Method using {method} cosine distance metric")
             metric = 'cosine'
        else:
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

            roc_ax.errorbar([fpr_mean], [recall_mean], xerr=[fpr_std], yerr=[recall_std],
                            fmt='o', capsize=4, label=f"{name} (FPR={fpr_mean:.2f})")

            print(f"[{method}] Plotting mean P={precision_mean:.2f}, R={recall_mean:.2f}")
            pr_ax.errorbar([recall_mean], [precision_mean], xerr=[recall_std], yerr=[precision_std],
                        fmt='o', capsize=4, label=f"{name} (P={precision_mean:.2f}, R={recall_mean:.2f})")

            fdr_ax.errorbar([recall_mean], [1 - precision_mean], xerr=[recall_std], yerr=[precision_std],
                            fmt='o', capsize=4, label=f"{name} (FDR={1 - precision_mean:.2f})")

        else:
                
            tasks = []
            for data_idx in range(len(data_list['test'])):
                df1 = data_list['test'][data_idx].copy()
                df2 = df1.copy() if test_only else data_list['train'][data_idx].copy()

                tasks.append((df1, df2, method, data_idx))

            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_interpolated_curves)(df1, df2, method, data_idx, interp_points, metric, max_pairs)
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

            roc_auc_val = auc(interp_points, roc_mean)
            pr_auc_val = auc(interp_points, pr_mean)
            fdr_auc_val = auc(interp_points, fdr_mean)

            roc_ax.plot(interp_points, roc_mean, label=f"{NAME_MAPPINGS[method]} (auc={roc_auc_val:.2f})")
            roc_ax.fill_between(interp_points, roc_mean - roc_std, roc_mean + roc_std, alpha=0.2)

            pr_ax.plot(interp_points, pr_mean, label=f"{NAME_MAPPINGS[method]} (auc={pr_auc_val:.2f})")
            pr_ax.fill_between(interp_points, pr_mean - pr_std, pr_mean + pr_std, alpha=0.2)

            fdr_ax.plot(interp_points, fdr_mean, label=f"{NAME_MAPPINGS[method]} (auc={fdr_auc_val:.2f})")
            fdr_ax.fill_between(interp_points, fdr_mean - fdr_std, fdr_mean + fdr_std, alpha=0.2)

    # Finalize ROC
    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curve')
    roc_ax.legend()

    # Finalize PR
    pr_ax.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curve')
    pr_ax.legend(loc='lower left')

    # Finalize FDR
    fdr_ax.set(xlabel='Recall', ylabel='False Discovery Rate', title='FDR Curve')
    fdr_ax.legend()

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
def get_precision_in_top_k(
    train_df,
    test_df,
    k: int,
    method: str = 'clip_transformer',
    distance_metric: str = 'euclidean',
    normalize: bool = True,
    average: str = "micro",  # "micro" or "macro"
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Computes precision@k using either micro or macro averaging.

    average:
        - "micro": total correct / total predictions (default)
        - "macro": average of per-label mean precision@k
    """
    if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
        print(f"Method {method} is using intensity agnostic embeddings")
        train_df['embedding'] = train_df['embedding'].apply(lambda x: (x > 0.02).astype(int))
        test_df['embedding'] = test_df['embedding'].apply(lambda x: (x > 0.02).astype(int))

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

    return precision, failure_cases

def top_k_precision_plot(dataset, target, split_type, n_jobs=-1, within_test=False, average="micro"):
    embeddings = gather_embeddings(dataset, target, split_type)

    # Add a "Cosine (Intensity Agnostic)" method
    embeddings['cosine_intensity_agnostic'] = embeddings['cosine'].copy()
    
    for train_test_df in embeddings['cosine_intensity_agnostic']:
        train_test_df['train'] = train_test_df['train'].copy()
        train_test_df['test'] = train_test_df['test'].copy()
        train_test_df['train']['embedding'] = train_test_df['train']['embedding'].apply(lambda x: (x > 0.02).astype(int))
        train_test_df['test']['embedding'] = train_test_df['test']['embedding'].apply(lambda x: (x > 0.02).astype(int))

    base_train_df = list(embeddings.values())[0]['train'][0]
    base_test_df = list(embeddings.values())[0]['test'][0]
    if within_test:
        base_train_df = base_test_df

    # Precompute theoretical max precision@k once
    if average == "micro":
        theoretical_max_per_k = {
            k: compute_theoretical_max_precision(base_train_df, base_test_df, k)
            for k in range(1, 11)
        }
    else:
        theoretical_max_per_k = {
            k: compute_macro_averaged_theoretical_max_precision(base_train_df, base_test_df, k)
            for k in range(1, 11)
        }

    tasks = []
    for key, d in embeddings.items():
        if key == 'metadata':
            continue
        if key in {'cosine', 'cosine_intensity_agnostic', 'clip_transformer', 'clip_transformer_genus_genus'}:
            print(f"Method using {key} cosine distance metric")
            distance_metric = 'cosine'
        else:
            distance_metric = 'euclidean'

        for k in range(1, 11):
            for j in range(len(d['train'])):
                tasks.append((key, k, d['train'][j], d['test'][j], distance_metric))

    def compute_precision(key, k, train_df, test_df, distance_metric):
        if key == 'multinomial_classifier':
            # Plot the precision as a horizontal line
            # Require that the target level is below the split level
            if level_heirarchy[target] <= level_heirarchy[split_type]:
                return None
            # Calculate precision directly on test set (no difference for train-test inference)
            tp = np.sum(test_df['pred_class'] == test_df['true_label'])
            fp = np.sum(test_df['pred_class'] != test_df['true_label'])
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            precision, _ = get_precision_in_top_k(
                train_df, test_df, k, key, distance_metric, normalize=True, average=average
            )
        return {
            'model': key,
            'k': k,
            'precision': precision,
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_precision)(key, k, tr, te, dm)
        for key, k, tr, te, dm in tqdm(tasks)
    )
    results = [res for res in results if res is not None]  # Filter out None results

    df = pd.DataFrame(results)
    df['k'] = df['k'].astype(int)
    df = df.sort_values(by=['model', 'k'])

    # --- Plot ---
    fig = plt.figure(figsize=(12, 8))

    # Plot each model
    sns.lineplot(data=df, x='k', y='precision', hue='model', style='model', markers=True, dashes=False)

    # Plot the theoretical max separately
    k_vals = list(theoretical_max_per_k.keys())
    theoretical_vals = list(theoretical_max_per_k.values())
    plt.plot(k_vals, theoretical_vals, label='Theoretical Max', linestyle='dashed', color='black', linewidth=2)

    plt.title(f"Precision@k ({average.capitalize()} Average) with Theoretical Max\n{dataset} - {target} - {split_type}")
    plt.xlabel('k (Top-k Nearest Neighbors)')
    plt.ylabel('% of Retrievals in Correct Taxa')
    plt.grid(True)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    return fig, df

# def get_precision_in_top_k(
#     train_df,
#     test_df,
#     k: int,
#     method: str = 'clip_transformer',
#     distance_metric: str = 'euclidean',
#     normalize: bool = True,
# ) -> Tuple[float, List[Dict[str, str]]]:

#     if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
#         print(f"Method {method} is using intensity agnostic embeddings")
#         train_df['embedding'] = train_df['embedding'].apply(lambda x: (x > 0.02).astype(int))
#         test_df['embedding'] = test_df['embedding'].apply(lambda x: (x > 0.02).astype(int))

#     if normalize:
#         def _normalize_fn(x):
#             norm = np.linalg.norm(x)
#             return x / norm if norm > 0 else x
#         train_df['embedding'] = train_df['embedding'].apply(_normalize_fn)
#         test_df['embedding'] = test_df['embedding'].apply(_normalize_fn)

#     train_embeddings = np.vstack(train_df['embedding'])
#     test_embeddings = np.vstack(test_df['embedding'])

#     if distance_metric == 'cosine':
#         similarities = cosine_similarity(test_embeddings, train_embeddings)
#     elif distance_metric == 'euclidean':
#         similarities = -euclidean_distances(test_embeddings, train_embeddings)
#     else:
#         raise ValueError(f"Unknown distance metric: {distance_metric}")

#     test_labels = (test_df['true_label'].values).astype(np.float32)
#     train_labels = (train_df['true_label'].values).astype(np.float32)

#     total_correct = 0
#     total_predictions = 0
#     failure_cases = []
#     strain_names = test_df['strain_name'].values
#     true_labels = test_df['true_label'].values

#     for i in range(test_embeddings.shape[0]):
#         top_k_indices = np.argsort(similarities[i])[::-1][:k]
#         predicted_labels = train_labels[top_k_indices]

#         num_correct = np.sum(predicted_labels == test_labels[i])
#         total_correct += num_correct  # Sum up total correct across all samples
#         total_predictions += len(predicted_labels)

#         if num_correct == 0:
#             failure_cases.append({
#                 'strain_name': strain_names[i],
#                 'true_label': true_labels[i],
#                 'predicted_labels': predicted_labels.tolist(),
#             })

#     precision = total_correct / total_predictions  # Global precision
#     return precision, failure_cases


# def top_k_precision_plot(dataset, target, split_type, n_jobs=-1, within_test=False):
#     embeddings = gather_embeddings(dataset, target, split_type)

#     # Add a "Cosine (Intensity Agnostic)" method
#     embeddings['cosine_intensity_agnostic'] = embeddings['cosine'].copy()

#     base_train_df = list(embeddings.values())[0]['train'][0]
#     base_test_df = list(embeddings.values())[0]['test'][0]
#     if within_test:
#         base_train_df = base_test_df

#     # Precompute theoretical max precision@k once
#     theoretical_max_per_k = {
#         k: compute_theoretical_max_precision(base_train_df, base_test_df, k)
#         for k in range(1, 11)
#     }

#     tasks = []
#     for key, d in embeddings.items():
#         if key == 'metadata':
#             continue
#         if key == 'cosine' or key == 'cosine_intensity_agnostic' \
#             or key == 'clip_transformer':
#             print(f"Method using {key} cosine distance metric")
#             distance_metric = 'cosine'
#         else:
#             distance_metric = 'euclidean'

#         for k in range(1, 11):
#             for j in range(len(d['train'])):
#                 tasks.append((key, k, d['train'][j], d['test'][j], distance_metric))

#     def compute_precision(key, k, train_df, test_df, distance_metric):
#         if key == 'multinomial_classifier':
#             # Plot the precision as a horizontal line
#             # Require that the target level is below the split level
#             if level_heirarchy[target] <= level_heirarchy[split_type]:
#                 return None
#             # Calculate precision directly on test set (no difference for train-test inference)
#             tp = np.sum(test_df['pred_class'] == test_df['true_label'])
#             fp = np.sum(test_df['pred_class'] != test_df['true_label'])
#             precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         else:
#             precision, _ = get_precision_in_top_k(
#                 train_df, test_df, k, key, distance_metric, normalize=True
#             )
#         return {
#             'model': key,
#             'k': k,
#             'precision': precision,
#         }

#     results = Parallel(n_jobs=n_jobs)(
#         delayed(compute_precision)(key, k, tr, te, dm)
#         for key, k, tr, te, dm in tqdm(tasks)
#     )
#     results = [res for res in results if res is not None]  # Filter out None results

#     df = pd.DataFrame(results)
#     df['k'] = df['k'].astype(int)
#     df = df.sort_values(by=['model', 'k'])

#     # --- Plot ---
#     fig = plt.figure(figsize=(12, 8))

#     # Plot each model
#     sns.lineplot(data=df, x='k', y='precision', hue='model', style='model', markers=True, dashes=False)

#     # Plot the theoretical max separately
#     k_vals = list(theoretical_max_per_k.keys())
#     theoretical_vals = list(theoretical_max_per_k.values())
#     plt.plot(k_vals, theoretical_vals, label='Theoretical Max', linestyle='dashed', color='black', linewidth=2)

#     plt.title(f"Precision@k with Theoretical Max\n{dataset} - {target} - {split_type}")
#     plt.xlabel('k (Top-k Nearest Neighbors)')
#     plt.ylabel('% of Retrievals in Correct Taxa')
#     plt.grid(True)
#     plt.legend(title='Model')
#     plt.tight_layout()
#     plt.show()

#     return fig, df

# %%
def similarity_histogram(dataset, target, split_type, model, test_only=False, max_pairs=None, distance_metric='euclidean'):
    """
    Plots a histogram of pairwise similarities or distances between embeddings:
    - Within the same label (true_label)
    - Between different labels

    Args:
        dataset (str): Dataset name.
        target (str): Column with class labels.
        split_type (str): Split strategy (e.g., 'species_even').
        model (str): The embedding model to use.
        test_only (bool): If True, compare only test-test pairs. If False, compare train-test.
        max_pairs (int): Max number of pairs to consider for speed.
        distance_metric (str): 'cosine' or 'euclidean'.

    Returns:
        matplotlib.figure.Figure: The histogram figure.
    """
    embeddings = gather_embeddings(dataset, target, split_type)
    if model not in embeddings:
        raise ValueError(f"Model '{model}' not found in embeddings.")

    data = embeddings[model]
    print("Warning: Taking first model only")

    df1 = data['test'][0].copy()
    df2 = df1.copy() if test_only else data['train'][0].copy()

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

    within_hist, _ = np.histogram(within, bins=bins, density=True)
    between_hist, _ = np.histogram(between, bins=bins, density=True)

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
    ax.set_ylabel('Density')
    ax.legend()
    return fig

# %%
debug_embeddings = gather_embeddings('driams-a', 'genera', 'species')
debug_embeddings_train = debug_embeddings['cosine']['test']
debug_embeddings_test = debug_embeddings['cosine']['test']

# %%
debug_acc_lst = []
for k in range(1, 21, 5):
  debug_acc = get_accuracy_in_top_k(
                              train_df=debug_embeddings_train,
                              test_df=debug_embeddings_test,
                              k=k,
                              distance_metric='euclidean',
                              normalize=True,
                            )
  debug_acc_lst.append(debug_acc[0])

debug_acc_lst

# %%
binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=1_000_000)

# %%
nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=-1)

# %%
binary_curves_plot('DRIAMS-A', 'species', 'species', test_only=True, max_pairs=1_000_000)

# %%
binary_curves_plot('DRIAMS-A', 'genera', 'species', test_only=True, max_pairs=1_000_000)

# %%
binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=1_000_000)

# %%
_ = top_k_boxplot_per_method_by_label(
    dataset='driams-a',
    target='genera',
    split_type='species',
    within_test=False,
    n_jobs=-1,
)

# %%
_ = top_k_barplot_per_method_by_label(
    dataset='driams-a',
    target='genera',
    split_type='species',
    within_test=False,
    n_jobs=10,
)

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
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=5, macro=True)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=5, macro=False)

# %%
_ = top_k_recall_plot('DRIAMS-A', 'genera', 'species', within_test=False, n_jobs=5, macro=True)

# %%
# DRIAMS Plots
_ = binary_curves_plot('driams-a', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=4)


# %%
# DRIAMS Plots
_ = binary_curves_plot('driams-a', 'genera', 'genera', test_only=True, max_pairs=1_000_000, n_jobs=4)


# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'species', within_test=True, n_jobs=4)

# %%
_ = nn_accuracy_plot('driams-a', 'genera', 'genera', within_test=True, n_jobs=6)

# %% [markdown]
# ## IDBac Plots

# %%
# Get embeddings for the idbac dataset with cosine
t = gather_embeddings('IDBac-kb', 'genera', 'species')['cosine']

# %%
for i in range(5):
    print(t['test'][i].true_label.value_counts())

# %%
_ = nn_accuracy_plot('IDBac-kb', 'genera', 'species', within_test=True, n_jobs=6)

# %%
_ = binary_curves_plot('IDBac-kb', 'genera', 'species', test_only=True, max_pairs=1_000_000, n_jobs=4)
# _ = binary_curves_plot('IDBac-kb', 'genera', 'genera', test_only=True, max_pairs=1_000_000, n_jobs=4) # TODO

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
_ = binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=500_000)

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
plot_cosine_similarity_histogram(
    all_dirams_data,
    title="Cosine Similarity Histogram for All DRIAMS Data",
    bins=50
)

# %%
plot_cosine_similarity_histogram(
    all_dirams_data,
    title="Cosine Similarity Histogram for All DRIAMS Data",
    bins=50,
    binarize=True
)

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


