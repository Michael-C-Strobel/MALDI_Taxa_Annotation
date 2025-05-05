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
    'clip_transformer_classifier': "Classifier Embeddings",
    'cosine': "Cosine Similarity",
    'prototypical_transformer': "Prototypical Transformer",
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
    clip_transformer_path = None
    clip_transformer_classifer_path = None
    cosine_path = None
    prototypical_transformer_path = None

    # inference/{args.target}/{args.split_type}/"
    if target == 'genera':
        if split_type == 'genera':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                # clip_transformer_path = [base_dir / 'CLIP_Transformer' / 'genera' / 'version_3' / 'inference' / target / split_type]
                # # clip_transformer_classifer_path = [base_dir / 'CLIP_Transformer_Classifier' / 'version_9' / 'inference' / target / split_type]
                # clip_transformer_classifer_path = [base_dir / 'CLIP_Transformer_Classifier' / 'genera' / 'version_1' / 'inference' / target / split_type]
                # cosine_path = [base_dir / 'cosine' / target / split_type]
                # prototypical_transformer_path = [base_dir / 'Prototyical_Transformer' / 'genera' / 'version_10'  / 'inference' / target / split_type]
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / 'genera'/ 'genera' / 'CLIP_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                clip_transformer_classifer_path = [
                    _base_dir / 'genera'/ 'genera' / 'CLIP_Transformer_Classifier' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                cosine_path = [_base_dir / 'cosine_10' / target / split_type]
                prototypical_transformer_path = [
                    _base_dir / 'genera'/ 'genera' / 'Prototyical_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = [
                    _base_dir / 'genera'/ 'genera' / 'CLIP_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                clip_transformer_classifer_path = [
                    _base_dir / 'genera'/ 'genera' / 'CLIP_Transformer_Classifier' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                cosine_path = [_base_dir / 'cosine_10' / target / split_type]
                prototypical_transformer_path = [
                    _base_dir / 'genera'/ 'genera' / 'Prototyical_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
            
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                                            _base_dir / 'genera'/ 'species' / 'CLIP_Transformer' / 'version_0' / 'inference' / target / split_type,
                                            _base_dir / 'genera'/ 'species' / 'CLIP_Transformer' / 'version_1' / 'inference' / target / split_type,
                                            _base_dir / 'genera'/ 'species' / 'CLIP_Transformer' / 'version_2' / 'inference' / target / split_type,
                                            _base_dir / 'genera'/ 'species' / 'CLIP_Transformer' / 'version_3' / 'inference' / target / split_type,
                                            _base_dir / 'genera'/ 'species' / 'CLIP_Transformer' / 'version_4' / 'inference' / target / split_type,
                                         ]
                clip_transformer_classifer_path = [
                                                    _base_dir / 'genera'/ 'species' / 'CLIP_Transformer_Classifier' / 'version_0' / 'inference' / target / split_type,
                                                    _base_dir / 'genera'/ 'species' / 'CLIP_Transformer_Classifier' / 'version_1' / 'inference' / target / split_type,
                                                    _base_dir / 'genera'/ 'species' / 'CLIP_Transformer_Classifier' / 'version_2' / 'inference' / target / split_type,
                                                    _base_dir / 'genera'/ 'species' / 'CLIP_Transformer_Classifier' / 'version_3' / 'inference' / target / split_type,
                                                    _base_dir / 'genera'/ 'species' / 'CLIP_Transformer_Classifier' / 'version_4' / 'inference' / target / split_type,
                                                ]
                
                cosine_path = [base_dir / 'cosine' / target / split_type]
                prototypical_transformer_path = [
                    _base_dir / 'genera'/ 'species' / 'Prototyical_Transformer' / 'version_0' / 'inference' / target / split_type,
                    _base_dir / 'genera'/ 'species' / 'Prototyical_Transformer' / 'version_1' / 'inference' / target / split_type,
                    _base_dir / 'genera'/ 'species' / 'Prototyical_Transformer' / 'version_2' / 'inference' / target / split_type,
                    _base_dir / 'genera'/ 'species' / 'Prototyical_Transformer' / 'version_3' / 'inference' / target / split_type,
                    _base_dir / 'genera'/ 'species' / 'Prototyical_Transformer' / 'version_4' / 'inference' / target / split_type,
                ]
            elif dataset.lower() == 'idbac-kb':
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')

                clip_transformer_path = [
                        _base_dir / 'genera'/ 'species' / 'CLIP_Transformer' / f'version_{i}' / 'inference' / target / split_type
                        for i in range(5)
                    ]
                clip_transformer_classifer_path = [
                                                    _base_dir / 'genera'/ 'species' / 'CLIP_Transformer_Classifier' / f'version_{i}' / 'inference' / target / split_type
                                                    for i in range(5)
                                                ]
                cosine_path = [_base_dir / 'cosine_10' / target / split_type]
                prototypical_transformer_path = [
                    _base_dir / 'genera'/ 'species' / 'Prototyical_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]


        elif split_type == 'species_even':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata_code_accessions.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_DRIAMS_A_for_score')
                clip_transformer_path = [
                    _base_dir / 'genera'/ 'species_even' / 'CLIP_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                clip_transformer_classifer_path = [
                    _base_dir / 'genera'/ 'species_even' / 'CLIP_Transformer_Classifier' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                cosine_path = [base_dir / 'cosine_10' / target / split_type]
                prototypical_transformer_path = [
                    _base_dir / 'genera'/ 'species_even' / 'Prototyical_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
                _base_dir = Path('/data/nas-gpu/wang/mstro016/SourceCode/16s_sim_pred/bin/ml/lightning_logs_idbac_for_score')
                clip_transformer_path = [
                    _base_dir / 'genera'/ 'species_even' / 'CLIP_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                clip_transformer_classifer_path = [
                    _base_dir / 'genera'/ 'species_even' / 'CLIP_Transformer_Classifier' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                cosine_path = [_base_dir / 'cosine_10' / target / split_type]
                prototypical_transformer_path = [
                    _base_dir / 'genera'/ 'species_even' / 'Prototyical_Transformer' / f'version_{i}' / 'inference' / target / split_type
                    for i in range(5)
                ]
                           


    elif target == 'species':
        if split_type == 'genera':
            # I guess we could do this, but it feels like a bit of a stretch
            raise ValueError(f"Unknown split_type: {split_type} for target: {target}")
    
        elif split_type == 'species':
            if dataset.lower() == 'driams-a':
                metadata_path = Path('../data/driams/preprocessing/merged_metadata.csv')
                clip_transformer_path = [base_dir / 'CLIP_Transformer' / 'species' / 'version_1' / 'inference' / target / split_type]
                clip_transformer_classifer_path = [base_dir / 'CLIP_Transformer_Classifier' / 'version_7' / 'inference' / target / split_type]
                cosine_path = [base_dir / 'cosine' / target / split_type]
                prototypical_transformer_path = [base_dir / 'Prototyical_Transformer' / 'species' / 'version_9' / 'inference' / target / split_type]
            elif dataset.lower() == 'idbac-kb':
                metadata_path = Path('../data/idbac_db/raw/ammended_db.csv')
            

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
    if cosine_path:
        output_dict['cosine'] = {}
        output_dict['cosine']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in cosine_path]
        output_dict['cosine']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in cosine_path] 
    if prototypical_transformer_path:
        output_dict['prototypical_transformer'] = {}
        output_dict['prototypical_transformer']['train'] = [pd.read_feather(x / 'train_inference.feather') for x in prototypical_transformer_path]
        output_dict['prototypical_transformer']['test'] = [pd.read_feather(x / 'test_inference.feather') for x in prototypical_transformer_path]

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
def get_recall_in_top_k(
                            train_df,
                            test_df,
                            k:int,
                            distance_metric:str='euclidean',
                            normalize:bool=True,
                        )-> Tuple[float, List[Dict[str, str]]]:
    """Annotates test data points with the top k most similar training data points. 
    Considered correct if the true label of the test data point is in the top k most 
    similar training data points.
    Returns overall accuracy @ k.

    Args:
        train_df (pd.DataFrame): DataFrame containing training data with columns 'embedding' and 'true_label'.
        test_df (pd.DataFrame): DataFrame containing test data with columns 'embedding' and 'true_label'.
        k (int): Number of nearest neighbors to consider.
        distance_metric (str): Distance metric to use. Can be 'euclidean' or 'cosine'.
        normalize (bool): Whether to normalize the embeddings.
    Returns:
        float: Overall accuracy @ k.
        List[Dict[str, str]]: List of failure cases with strain name, true label, and predicted labels.
    """
    train_df = train_df.copy(deep=True)
    test_df = test_df.copy(deep=True)

    # Normalize embeddings if required
    if normalize:
        train_df['embedding'] = train_df['embedding'].apply(lambda x: x / np.linalg.norm(x))
        test_df['embedding'] = test_df['embedding'].apply(lambda x: x / np.linalg.norm(x))
        # Check if any are nan, set to 0
        train_df['embedding'] = train_df['embedding'].apply(lambda x: np.nan_to_num(x))
        test_df['embedding'] = test_df['embedding'].apply(lambda x: np.nan_to_num(x))
    
    # Extract the embeddings as numpy arrays
    train_embeddings = np.vstack(train_df['embedding'])
    test_embeddings = np.vstack(test_df['embedding'])
    
    # Compute the similarity/distance matrix in a vectorized manner
    if distance_metric == 'cosine':
        similarities = cosine_similarity(test_embeddings, train_embeddings)
    elif distance_metric == 'euclidean':
        similarities = -euclidean_distances(test_embeddings, train_embeddings)
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Extract true labels
    test_labels = test_df['true_label'].values
    train_labels = train_df['true_label'].values
    
    correct = 0
    total = 0
    failure_cases = []

    # Iterate over the test set and compute accuracy
    for i in range(test_embeddings.shape[0]):
        true_label = test_labels[i]
        
        # Get the indices of the top k similar training samples
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        predicted_labels = train_labels[top_k_indices]
        
        # Check if true label is in top k predictions
        if true_label in predicted_labels:
            correct += 1
        else:
            failure_cases.append({
                'strain_name': test_df.iloc[i]['strain_name'],
                'true_label': true_label,
                'predicted_labels': predicted_labels.tolist(),
            })
        
        total += 1
    
    acc = correct / total
    return acc, failure_cases


def top_k_recall_plot(dataset, target, split_type, n_jobs=-1, within_test=False):
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
    for key, train_test_dict in embeddings.items():
        if key == 'metadata':
            continue
        for i in range(len(train_test_dict['train'])):
            distance_metric = 'cosine' if key == 'cosine' else 'euclidean'
            train_embeddings = train_test_dict['train'][i]
            test_embeddings = train_test_dict['test'][i]
            if within_test:
                train_embeddings = test_embeddings

            for k in np.arange(1, 11):
                tasks.append((key, k, train_embeddings, test_embeddings, distance_metric))

    def compute_accuracy(key, k, train_embeddings, test_embeddings, distance_metric):
        acc, _ = get_recall_in_top_k(
            train_embeddings,
            test_embeddings,
            k=k,
            distance_metric=distance_metric,
            normalize=True
        )
        return {
            'model': key,
            'k': k,
            'accuracy': acc,
        }

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_accuracy)(key, k, train_embeddings, test_embeddings, distance_metric)
        for key, k, train_embeddings, test_embeddings, distance_metric in tqdm(tasks)
    )

    accuracies_df = pd.DataFrame(results)
    accuracies_df['k'] = accuracies_df['k'].astype(int)
    accuracies_df['accuracy'] = accuracies_df['accuracy'].astype(float)
    accuracies_df = accuracies_df.sort_values(by=['model', 'k'])

    # Line plot
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=accuracies_df, x='k', y='accuracy', hue='model', style='model', markers=True, dashes=False)
    plt.title(f"Train-Test Recall for {dataset} - {target} - {split_type}")
    plt.xlabel('Number of Neighbors Considered (k)')
    plt.ylabel('% of Queries with Correct Taxa in Top k')
    plt.legend(title='Model')
    plt.grid(True)
    plt.show()
    return fig, accuracies_df
        

# %%
def get_within_test_nn_accuracy( 
                            test_df:pd.DataFrame,
                            random_seed:int=42,
                            k:int=5,
                            distance_metric:str='euclidean',
                            normalize:bool=True, 
                          )-> float:
    df = test_df.copy(deep=True)

    label_counts = df['true_label'].value_counts()
    singletons = label_counts[label_counts == 1].index.tolist()

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
    sampled_indices_set = set()
    for taxa, indices in taxa_indices.items():
        _k = min(k, len(indices)//2)
        sampled_indices = np.random.choice(indices, size=_k, replace=False)
        for i in sampled_indices:
            sampled_indices_set.add(i)

    faux_db = df.loc[list(sampled_indices_set), :].values

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
            faux_true_label = faux_row[3]
            if distance_metric == 'cosine':
                    similarity = np.dot(embedding, faux_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(faux_embedding))
            elif distance_metric == 'euclidean':
                    similarity = -1 * np.linalg.norm(embedding - faux_embedding)
            else:
                    raise ValueError("Unknown distance metric: {}".format(distance_metric))
            similarities.append((faux_true_label, similarity))

        # similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:k] # Pretty sure this doesn't do anything?
        predicted_genus = max(similarities, key=lambda x: x[1])[0]
      
        if predicted_genus == true_label:
            correct += 1
        else:
            failure_cases.append({
                'strain_name': strain_name,
                'true_label': true_label, 
                'predicted_genus': predicted_genus,
                })
        # print(f"True label: {true_label}, Predicted label: {predicted_genus}")
        
        total += 1

    acc = correct / total
    return acc, failure_cases


# %%
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

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
from tqdm import tqdm

def within_test_nn_accuracy_plot(dataset, target, split_type, n_jobs=-1):
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

    def compute_accuracy(key, test_embeddings, distance_metric, k, random_seed):
        acc, _ = get_within_test_nn_accuracy(
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
        delayed(compute_accuracy)(key, test_embeddings, distance_metric, k, random_seed)
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

    plt.title(f"Within Test Nearest-Neighbor Accuracy for {dataset} - {target} - {split_type}")
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

def _binary_similarity_curves(df1, df2, distance_metric='cosine', max_pairs=None):
    """
    Computes binary labels and similarity scores for all pairs between df1 and df2,
    then returns ROC and PR curve components.

    Returns:
        fpr, tpr, roc_auc, precision, recall, pr_auc
    """
    X = np.stack(df1['embedding'].values)
    Y = np.stack(df2['embedding'].values)
    # L2 Normalize per embedding
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    # Check for NaN values
    X = np.nan_to_num(X)
    Y = np.nan_to_num(Y)
    labels_X = np.array(df1['true_label'])
    labels_Y = np.array(df2['true_label'])

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
    y_true = (l1 == l2).astype(int)

    if distance_metric == 'cosine':
        sim = np.sum(X1 * X2, axis=1) / (np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1))
    elif distance_metric == 'euclidean':
        sim = -np.linalg.norm(X1 - X2, axis=1)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # ROC + PR curves
    fpr, tpr, _ = roc_curve(y_true, sim)
    precision, recall, _ = precision_recall_curve(y_true, sim)
    return fpr, tpr, auc(fpr, tpr), precision, recall, auc(recall, precision)

def _compute_interpolated_curves(df1, df2, method, data_idx, interp_points, metric, max_pairs):
    try:
        if method.split('_')[0] == 'cosine':
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
        return None


def binary_curves_plot(dataset, target, split_type, test_only=False, max_pairs=None,
                       cosine_ablation=False, ignore_intensity=False):
    if cosine_ablation:
        embeddings = gather_embeddings_cosine_only(dataset, target, split_type)
    else:
        embeddings = gather_embeddings(dataset, target, split_type)

    roc_fig, roc_ax = plt.subplots()
    pr_fig, pr_ax = plt.subplots()
    fdr_fig, fdr_ax = plt.subplots()

    interp_points = np.linspace(0, 1, 100)

    for method, data_list in embeddings.items():
        if method == 'metadata':
            continue

        metric = 'cosine' if method == 'cosine' else 'euclidean'

        tasks = []
        for data_idx in range(len(data_list['train'])):
            df1 = data_list['test'][data_idx].copy()
            df2 = df1.copy() if test_only else data_list['train'][data_idx].copy()
            tasks.append((df1, df2, method, data_idx))

        results = Parallel(n_jobs=2)(
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



def get_precision_in_top_k(
    train_df,
    test_df,
    k: int,
    distance_metric: str = 'euclidean',
    normalize: bool = True,
) -> Tuple[float, List[Dict[str, str]]]:

    train_df = train_df.copy()
    test_df = test_df.copy()

    if normalize:
        normalize_fn = lambda x: np.nan_to_num(x / np.linalg.norm(x))
        train_df['embedding'] = train_df['embedding'].apply(normalize_fn)
        test_df['embedding'] = test_df['embedding'].apply(normalize_fn)

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

    total_correct = 0
    total_predictions = 0
    failure_cases = []

    for i in range(test_embeddings.shape[0]):
        top_k_indices = np.argsort(similarities[i])[::-1][:k]
        predicted_labels = train_labels[top_k_indices]

        num_correct = np.sum(predicted_labels == test_labels[i])
        total_correct += num_correct  # Sum up total correct across all samples
        total_predictions += len(predicted_labels)

        if num_correct == 0:
            failure_cases.append({
                'strain_name': test_df.iloc[i]['strain_name'],
                'true_label': test_labels[i],
                'predicted_labels': predicted_labels.tolist(),
            })

    precision = total_correct / total_predictions  # Global precision
    return precision, failure_cases


def top_k_precision_plot(dataset, target, split_type, n_jobs=-1, within_test=False):
    embeddings = gather_embeddings(dataset, target, split_type)

    base_train_df = list(embeddings.values())[0]['train'][0]
    base_test_df = list(embeddings.values())[0]['test'][0]
    if within_test:
        base_train_df = base_test_df

    # Precompute theoretical max precision@k once
    theoretical_max_per_k = {
        k: compute_theoretical_max_precision(base_train_df, base_test_df, k)
        for k in range(1, 11)
    }

    tasks = []
    for key, d in embeddings.items():
        if key == 'metadata':
            continue
        distance_metric = 'cosine' if key == 'cosine' else 'euclidean'
        for k in range(1, 11):
            for j in range(len(d['train'])):
                tasks.append((key, k, d['train'][j], d['test'][j], distance_metric))

    def compute_precision(key, k, train_df, test_df, distance_metric):
        precision, _ = get_precision_in_top_k(
            train_df, test_df, k, distance_metric, normalize=True
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

    plt.title(f"Precision@k with Theoretical Max\n{dataset} - {target} - {split_type}")
    plt.xlabel('k (Top-k Nearest Neighbors)')
    plt.ylabel('% of Retrievals in Correct Taxa')
    plt.grid(True)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.show()

    return fig, df

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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(within, bins=50, alpha=0.5, label='Within label', color='blue', density=True)
    ax.hist(between, bins=50, alpha=0.5, label='Between labels', color='orange', density=True)
    ax.set_title(f'Pairwise {"Similarity" if distance_metric == "cosine" else "Distance"} for Model: {model}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend()
    return fig

# %%
_ = train_test_recall_plot('driams-a', 'genera', 'species')
_ = train_test_precision_plot('driams-a', 'genera', 'species')

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
binary_curves_plot('driams-a', 'genera', 'genera', test_only=True, max_pairs=1_000_000)

# %%
within_test_nn_accuracy_plot('driams-a', 'genera', 'genera')

# %%
binary_curves_plot('DRIAMS-A', 'genera', 'genera', test_only=True, max_pairs=1_000_000)

# %%
binary_curves_plot('IDBac-KB', 'genera', 'species', test_only=True)

# %%
binary_curves_plot('IDBac-KB', 'genera', 'genera', test_only=True)

# %%
_ = top_k_recall_plot('IDBac-KB', 'genera', 'species', within_test=True)

# %%
_ = top_k_precision_plot('IDBac-KB', 'genera', 'species', within_test=True)

# %%
within_test_nn_accuracy_plot('DRIAMS-A', 'genera', 'species')

# %%
within_test_nn_accuracy_plot('IDBac-KB', 'genera', 'species')

# %%
_ = top_k_precision_plot('DRIAMS-A', 'genera', 'species')

# %%
_ = top_k_precision_plot('DRIAMS-A', 'genera', 'species', within_test=True)

# %%
train_test_curves_nn_plot('DRIAMS-A', 'genera', 'species')

# %%
# DRIAMS Plots
_ = binary_curves_plot('driams-a', 'genera', 'species_even', test_only=True, max_pairs=1_000_000)


# %% [markdown]
# ## IDBac Plots

# %%
_ = within_test_prototype_accuracy_plot('idbac-kb', 'genera', 'species_even')
_ = within_test_nn_accuracy_plot('idbac-kb', 'genera', 'species_even')

# %%
_ = train_test_precision_plot('idbac-kb', 'genera', 'species')
_ = train_test_recall_plot('idbac-kb', 'g enera', 'species')

# %%
top_k_precision_plot('idbac-kb', 'genera', 'species')
top_k_precision_plot('idbac-kb', 'genera', 'species', within_test=True)

# %%
train_test_curves_nn_plot('driams-a', 'genera', 'species')

# %%
_ = binary_curves_plot('IDBac-KB', 'genera', 'species_even', test_only=True)

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
    split_type='species',
    model='cosine',
    distance_metric='cosine',
    test_only=True,
    max_pairs=1_000_000 # Cosine requires subsampling because the vector is high dimensional    (5 M ~= 1-- GB)
)
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='species',
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
_ = similarity_histogram(
    dataset='DRIAMS-A',
    target='genera',
    split_type='genera',
    model='clip_transformer_classifier',
    distance_metric='euclidean',
    test_only=True,
    max_pairs=1_000_000 # ML Methods don't reqiure subsampling, but it's added for congruence
)


