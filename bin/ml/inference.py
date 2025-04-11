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

from datamodule import Spectrum_DataModule
from utils import mirror_plot, shannon_entropy, estimate_convexity

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, cophenet
from sklearn.metrics import fowlkes_mallows_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

# def compute_binary 

def compute_taxa_clustering_scores(prediction_table:pd.DataFrame, output_path:Path, metadata:pd.DataFrame, tax_level:str='genus', plot=False)->dict:
    """ Computes the average precision ('purity') and recall ('completeness') for each taxon at the specified taxonomic level. 
    Metrics are returned with both macro and micro averages.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        The DataFrame containing the predictions and metadata
    output_path : Path
        The path to save the output to
    metadata : pd.DataFrame
        The metadata associated with the spectra
    tax_level : str, optional
        The taxonomic level to compute clustering scores for, by default 'genus'    
    """
    table = prediction_table.copy()
    # Make the prediction_table 'square' by swapping accessions and concatenating
    reversed_table = table.rename(columns={
        'accession_a': 'accession_b',
        'accession_b': 'accession_a',
        'taxa_a': 'taxa_b',
        'taxa_b': 'taxa_a',
    })
    table = pd.concat([table, reversed_table], ignore_index=True)
    
    # Get the taxa for each accession
    metadata['Genbank accession'] = metadata['Genbank accession'].str.strip().str.split('.').str[0]
    metadata.dropna(subset=[tax_level], inplace=True)
    accession_taxa_mapping = metadata.set_index('Genbank accession')[tax_level].to_dict()

    assert 'JQ691549' in accession_taxa_mapping, "JQ691549 not in accession_taxa_mapping"
    assert 'AB184413' in accession_taxa_mapping, "AB184413 not in accession_taxa_mapping"

    table['taxa_a'] = table['accession_a'].map(accession_taxa_mapping)
    table['taxa_b'] = table['accession_b'].map(accession_taxa_mapping)
    table['equal_taxa'] = table['taxa_a'] == table['taxa_b']

    # Remove any taxa with only one member
    table = table[table['taxa_a'].map(table['taxa_a'].value_counts()) > 1]

    table.to_csv(f'./debug/pr_table_{tax_level}.csv')

    # Group by taxa
    grouped = table.groupby('taxa_a')

    # Compute precision and recall for each taxon across thresholds
    thresholds = list(np.arange(-1.0, 1.1, 0.05))

    micro_precision = []
    micro_recall = []

    num_positives = table.loc[table['equal_taxa'], 'equal_taxa'].sum()
    for threshold in thresholds:
        precision = table.loc[table['predicted_similarity'] >= threshold, 'equal_taxa'].mean()
        tp = table.loc[(table['predicted_similarity'] >= threshold) & (table['equal_taxa']), 'equal_taxa'].sum()
        recall = tp / num_positives

        micro_precision.append(precision)
        micro_recall.append(recall)

    per_taxa_precision = []
    per_taxa_recall = []

    per_group_precision = {}
    per_group_recall = {}
    for taxa, group in grouped:
        num_positives = group['equal_taxa'].sum()
        group_precision = []
        group_recall = []
        for threshold in thresholds:
            precision = group.loc[group['predicted_similarity'] >= threshold, 'equal_taxa'].mean() if len(group.loc[group['predicted_similarity'] >= threshold]) > 0 else 0
            tp = group.loc[(group['predicted_similarity'] >= threshold) & (group['equal_taxa']), 'equal_taxa'].sum()
            recall = tp / num_positives if num_positives > 0 else 0

            group_precision.append(precision)
            group_recall.append(recall)

        per_group_precision[taxa] = group_precision
        per_group_recall[taxa] = group_recall

    per_taxa_precision = [np.nanmean([v[i] for v in per_group_precision.values()]) for i in range(len(thresholds))]
    per_taxa_recall = [np.nanmean([v[i] for v in per_group_recall.values()]) for i in range(len(thresholds))]

    clustering_scores = {
        "params": {"tax_level": tax_level},
        "thresholds": thresholds,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "per_taxa_precision": per_taxa_precision,
        "per_taxa_recall": per_taxa_recall,
        "per_group_precision": per_group_precision,
        "per_group_recall": per_group_recall,
    }

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustering_scores, f, indent=4)

    if plot:
        # Plot the precision recall curve for averaged scores
        fig = plt.figure()

        sorted_indices = np.argsort(micro_recall)
        plt.plot(np.array(micro_recall)[sorted_indices], np.array(micro_precision)[sorted_indices], label="Micro Average")

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.title(f"Precision-Recall Curve for Taxonomic Level: {tax_level}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(output_path.parent / f"precision_recall_{tax_level}_micro.png")

        # Macro average
        fig = plt.figure()
        plt.plot(per_taxa_recall, per_taxa_precision, label="Macro Average")

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.title(f"Precision-Recall Curve for Taxonomic Level: {tax_level}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.savefig(output_path.parent / f"precision_recall_{tax_level}_macro.png")

def compute_top_in_top_k(prediction_table:pd.DataFrame, output_path:Path, k_range:List[int]=[1, 3, 5, 7, 10])->List[float]:
    """Computes the highest ground truth rank from rank 0-k. 
    
    Parameters
    ----------
    prediction_table : pd.DataFrame
        The DataFrame containing the predictions and metadata
    output_path : Path
        The path to save the output to
    k_range : List[int], optional
        The range of k values to compute top in top k for, by default [1, 3, 5, 7, 10]
    
    Returns
    -------
    List[float]
        The top in top k scores
    """
    table = prediction_table.copy()
    # Make the prediction_table 'square' by swapping accessions and concatenating
    reversed_table = table.rename(columns={
        'accession_a': 'accession_b',
        'accession_b': 'accession_a'
    })
    table = pd.concat([table, reversed_table], ignore_index=True)

    grouped = table.groupby('accession_a')
    # Rank predicted and true similarities
    method='dense'
    table['true_similarity'] = np.round(table['true_similarity'], 2)
    table['predicted_similarity'] = np.round(table['predicted_similarity'], 2)
    table['predicted_rank'] = grouped['predicted_similarity'].rank(method=method).reset_index(drop=True)
    table['true_rank'] = grouped['true_similarity'].rank(method=method).reset_index(drop=True)
    # Calculate the minimum rank for predicted rank 1...k
    top_in_top_k = []
    avg_in_top_k = []
    for k in k_range:
        per_accession = grouped.apply(lambda x: x.loc[x['predicted_rank'] <= k, 'true_rank'].min())
        top_in_top_k.append(per_accession.mean())
        per_accession = grouped.apply(lambda x: x.loc[x['predicted_rank'] <= k, 'true_rank'].mean())
        avg_in_top_k.append(per_accession.mean())

    top_in_top_k = {
        'params': {'k_range': k_range, 'method': method},
        'top_in_top_k': top_in_top_k,
        'avg_in_top_k': avg_in_top_k,
    }

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(top_in_top_k, f, indent=4)

    return top_in_top_k

def compute_information_imbalance(prediction_table:pd.DataFrame, output_path:Path, k:int=1, num_permutations:int=30, method='dense')->float:
    """ Computes the information imbalance as described in https://doi.org/10.1093/pnasnexus/pgac039.
    Briefly, the metric is descibed by two values Delta(B → A), Delta(A → B). Where
    Delta(A → B) ≈ 2〈r^B | r^A = 1〉/ N where r is rank for similarity measures A and B, 
    and brackets denote an expected value (which we compute explicity).

    Note: This metric is calculated per accession, not per spectrum.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        The DataFrame containing the predictions and metadata
    output_path : Path
        The path to save the output to

    Returns
    ----------
    float
        The information imbalance measurement
    """
    if method == 'dense':
        num_permutations = 1
    table = prediction_table.copy()
    # Make the prediction_table 'square' by swapping accessions and concatenating
    reversed_table = table.rename(columns={
        'accession_a': 'accession_b',
        'accession_b': 'accession_a'
    })
    table = pd.concat([table, reversed_table], ignore_index=True)
    # Quantize table for fair comparison
    table['true_similarity'] = np.round(table['true_similarity'], 2)
    table['predicted_similarity'] = np.round(table['predicted_similarity'], 2)
    def _compute_rank_metrics(k):
        grouped = table.groupby('accession_a')
        # Rank predicted and true similarities
        table['predicted_rank'] = grouped['predicted_similarity'].rank(method=method).reset_index(drop=True)
        table['true_rank'] = grouped['true_similarity'].rank(method=method).reset_index(drop=True)
        # Calculate expected ranks
        expected_rank_true_given_pred = []
        expected_rank_pred_given_true = []
        for accession, group in grouped:
            true_rank_sel = group.loc[group['predicted_rank'] == k, 'true_rank']
            pred_rank_sel = group.loc[group['true_rank'] == k, 'predicted_rank']
            if len(true_rank_sel) == 0:
                print(f"Warning: No values for group {accession} at predicted rank {k}")
            expected_rank_true_given_pred.append(true_rank_sel.mean())
            if len(pred_rank_sel) == 0:
                print(f"Warning: No values for group {accession} at true rank {k}")
            expected_rank_pred_given_true.append(pred_rank_sel.mean())
        # Compute deltas
        delta_pred_given_true = 2 * np.nanmean(expected_rank_true_given_pred) / max(table['true_rank'].max(), table['predicted_rank'].max()) 
        delta_true_given_pred = 2 * np.nanmean(expected_rank_pred_given_true) / max(table['true_rank'].max(), table['predicted_rank'].max()) 
        
        raw_ranks_pred_given_true = np.nanmean(expected_rank_true_given_pred)
        raw_ranks_true_given_pred = np.nanmean(expected_rank_pred_given_true)
        
        return delta_true_given_pred, delta_pred_given_true, raw_ranks_pred_given_true, raw_ranks_true_given_pred
    deltas = []
    for _ in range(num_permutations):
        delta_true_given_pred, delta_pred_given_true, raw_ranks_pred_given_true, raw_ranks_true_given_pred = _compute_rank_metrics(k)
        deltas.append((delta_true_given_pred, delta_pred_given_true, raw_ranks_pred_given_true, raw_ranks_true_given_pred))
    avg_delta_true_given_pred = np.mean([d[0] for d in deltas])
    avg_delta_pred_given_true = np.mean([d[1] for d in deltas])
    avg_rank_true_given_pred = np.mean([d[2] for d in deltas])
    avg_rank_pred_given_true = np.mean([d[3] for d in deltas])
    information_imbalance = {
        'params': {'k': k, 'num_permutations': num_permutations, 'method': method},
        'delta_true_given_pred': avg_delta_true_given_pred,
        'delta_pred_given_true': avg_delta_pred_given_true,
        'avg_rank_true_given_pred': avg_rank_true_given_pred,
        'avg_rank_pred_given_true': avg_rank_pred_given_true,
    }
    print(information_imbalance)

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(information_imbalance, f, indent=4)

    return information_imbalance


def compute_clustering_scores_per_genera(prediction_table,
                                         metadata:pd.DataFrame,
                                         output_path:Path=None,
                                         figure_path:Path=None,
                                         method:str='average'):
    """ Compute the fowlkes_mallows_score per genera. 
    
    Parameters
    ----------
    y_true : np.ndarray
        The true similarity matrix as a condensed similarity matrix
    y_pred : np.ndarray
        The predicted similarity matrix as a condensed similarity matrix
    metadata : pd.DataFrame
        The metadata associated with the spectra
    output_path : Path, optional
        The path to save the clustering scores, by default None
    figure_path : Path, optional
        The path to save the dendrogram comparison plot, by default None
        
    Returns
    -------
    dict
        Dictionary containing lists of clustering scores: 'fowlkes_mallows', 'rand_index', 'nmi', 'ami'.
    """

    def _helper(genus_group):
        print('genus_group', genus_group.shape)
        print(genus_group)

        # Convert to square distance matrix
        true_square_df = genus_group.pivot(index='strain_a', columns='strain_b', values='true_similarity')
        true_square = true_square_df.values
        true_square = (true_square + true_square.T) / 2
        pred_square_df = genus_group.pivot(index='strain_a', columns='strain_b', values='predicted_similarity')
        pred_square = pred_square_df.values
        pred_square = (pred_square + pred_square.T) / 2

        # Save to "./debug"
        os.makedirs('./debug', exist_ok=True)
        genus_group.to_csv('./debug/genus_group.csv')
        true_square_df.to_csv('./debug/true_square_df.csv')
        

        print('true_square_df', true_square_df.shape)
        print(true_square_df)
        
        print('true_square', true_square.shape)
        print('pred_square', pred_square.shape)

        # Print indices with nan values in true_square_df
        nan_indices = true_square_df.isna().sum().sum()
        if nan_indices > 0:
            print(f"Warning: {nan_indices} nan values in true_square_df")
            # Print all indices where this happens
            print(true_square_df[true_square_df.isna().any(axis=1)].index)

        # Convert to condensed distance matrix
        true_square = true_square
        pred_square = pred_square

        # To scipy squareform vector
        true_square = squareform(true_square, force='tovector', checks=False)
        pred_square = squareform(pred_square, force='tovector', checks=False)

        # Compute Scores
        scores = compute_clustering_scores(true_square, pred_square, method=method, max_k=None)

        true_label_mapping = {i: label for i, label in enumerate(true_square_df.index)}
        pred_label_mapping = {i: label for i, label in enumerate(true_square_df.index)}

        return scores, true_square, pred_square, true_label_mapping, pred_label_mapping

    tax_level = 'genus'
    table = prediction_table.copy()
    # Make the prediction_table 'square' by swapping accessions and concatenating
    reversed_table = table.rename(columns={
        'accession_a': 'accession_b',
        'accession_b': 'accession_a',
        'strain_a': 'strain_b',
        'strain_b': 'strain_a',
    })
    table = pd.concat([table, reversed_table], ignore_index=True)
    
    # Get the taxa for each accession
    metadata['Genbank accession'] = metadata['Genbank accession'].str.strip().str.split('.').str[0]
    accession_taxa_mapping = metadata.set_index('Genbank accession')[tax_level].to_dict()

    table['taxa_a'] = table['accession_a'].map(accession_taxa_mapping)
    table['taxa_b'] = table['accession_b'].map(accession_taxa_mapping)
    table['equal_taxa'] = table['taxa_a'] == table['taxa_b']

    table['names'] = table['strain_a'] + ';' + table['strain_b']

    
    table.to_csv('./debug/table.csv')

    # Remove duplicate pairs (this is coming from poor input data todo: deprecate)
    table = table.drop_duplicates(subset=['names'])

    # We only want things with equal taxa
    table = table[table['equal_taxa']]

    # Remove any taxa with less than x members (need 3 to get any non-trivial clustering metric), but x is arbitrary
    print(table['taxa_a'].value_counts())
    counts = table['taxa_a'].value_counts()
    table = table[table['taxa_a'].map(counts) >= 3]
    logging.info(f"Computing clustering scores for {len(table['taxa_a'].unique())} genera")


    # Group by taxa
    grouped = table.groupby('taxa_a')

    output_dict = {}
    true_matrix_dict = {}
    pred_matrix_dict = {}
    # Compute clustering scores for each genus
    for genus, group in grouped:
        print(f"Running helper for {genus}")
        out = _helper(group)
        output_dict[genus] = out[0]
        true_matrix_dict[genus] = {'data':out[1], 'mapping':out[3]}
        pred_matrix_dict[genus] = {'data':out[2], 'mapping':out[4]}

    json.dump(output_dict, open('./temp_test_output.json', "w", encoding="utf-8"), indent=4)

    if output_path:
        output_path = Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4)

    if figure_path:
        species_mapping = metadata.set_index('Strain name')['species'].to_dict()

        for genus in list(output_dict.keys()):

            # Get species with more than 5 members
            _strain_names = metadata.loc[metadata['genus'] == genus, 'Strain name'].unique()
            species_counts = metadata.loc[metadata['genus'] == genus, 'species'].value_counts()
            species_counts = species_counts[species_counts >=2]
            # if len(species_counts) > 20:
            #     raise NotImplementedError("Too many species to plot")
            species_color_mapping = {species: sns.color_palette('tab20')[i] for i, species in enumerate(species_counts.index.tolist()[:20])}
            strain_name_color_mapping = {strain: species_color_mapping.get(species_mapping[strain], 'black') for strain in _strain_names}   # If we have a genus but no species, will be black

            genus_path = figure_path / f"{genus}" / "clustering_score.png"
            if not genus_path.parent.exists():
                genus_path.parent.mkdir(parents=True, exist_ok=True)

            # Plot the average Fowlkes-Mallows score
            fig = plt.figure()
            scores = output_dict[genus]
            plt.plot(np.arange(2, len(scores['fowlkes_mallows'])+2), scores['fowlkes_mallows'], label="Fowlkes Mallows")

            plt.ylabel("Clustering Score")
            plt.xlabel("Number of Clusters")
            plt.title("Clustering Scores per Genus")

            plt.savefig(genus_path, dpi=300)
            plt.close(fig)  # Close to free memory

            #### Plot the true dendrogram ####
            fig, ax = plt.subplots(figsize=(0.08 * len(true_matrix_dict[genus]['mapping']), 10))
            
            true_linkage_as_dist = 1 - true_matrix_dict[genus]['data']
            true_linkage_as_dist = np.clip(true_linkage_as_dist, 0, 1)  # Clamp values

            true_linkage_as_dist = linkage(true_linkage_as_dist, method=method)
            dendrogram(true_linkage_as_dist, color_threshold=999)

            plt.title(f"True Linkage for {genus}")

            # Set x-axis labels with colors
            current_x_locs = ax.get_xticks()
            current_x_ticks = ax.get_xticklabels()
            new_x_ticks = [true_matrix_dict[genus]['mapping'][int(tick.get_text())] for tick in current_x_ticks]

            strain_name_colors = [strain_name_color_mapping.get(strain_name, "black") for strain_name in new_x_ticks]

            ax.set_xticks(current_x_locs)
            ax.set_xticklabels(new_x_ticks, rotation=90, fontsize=10, fontweight='bold')

            for tick, color in zip(ax.get_xticklabels(), strain_name_colors):
                tick.set_color(color)

            plt.savefig(figure_path / f"{genus}" / "true_linkage.png", dpi=300)
            plt.close(fig)  # Free memory

            #### Plot the predicted dendrogram ####
            fig, ax = plt.subplots(figsize=(0.08 * len(pred_matrix_dict[genus]['mapping']), 10))

            pred_linkage_as_dist = 1 - pred_matrix_dict[genus]['data']
            pred_linkage_as_dist = np.clip(pred_linkage_as_dist, 0, 1)  # Clamp values

            pred_linkage_as_dist = linkage(pred_linkage_as_dist, method=method)
            dendrogram(pred_linkage_as_dist, color_threshold=999)

            plt.title(f"Predicted Linkage for {genus}")

            # Set x-axis labels with colors
            current_x_locs = ax.get_xticks()
            current_x_ticks = ax.get_xticklabels()
            new_x_ticks = [pred_matrix_dict[genus]['mapping'][int(tick.get_text())] for tick in current_x_ticks]

            strain_name_colors = [strain_name_color_mapping.get(strain_name, "black") for strain_name in new_x_ticks]

            ax.set_xticks(current_x_locs)
            ax.set_xticklabels(new_x_ticks, rotation=90, fontsize=10, fontweight='bold')

            for tick, color in zip(ax.get_xticklabels(), strain_name_colors):
                tick.set_color(color)

            plt.savefig(figure_path / f"{genus}" / "predicted_linkage.png", dpi=300)
            plt.close(fig)  # Free memory


def compute_clustering_scores(y_true, y_pred, figure_path:Path=None, method:str='average',
                              max_k:int=100)->dict:
    """ Compute the fowlkes_mallows_score.

    Parameters
    ----------
    y_true : np.ndarray
        The true similarity matrix as a condensed similarity matrix
    y_pred : np.ndarray
        The predicted similarity matrix as a condensed similarity matrix
    figure_path : Path, optional
        The path to save the dendrogram comparison plot, by default None
    method : str, optional
        The method to use for clustering, by default 'complete'
    max_k : int, optional
        The maximum number of clusters to evaluate, by default 100. None will use the maximum number of clusters.

    Returns
    -------
    dict
        Dictionary containing lists of clustering scores: 'fowlkes_mallows', 'rand_index', 'nmi', 'ami'.

    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nan_trues = np.isnan(y_true)
    if sum(nan_trues) > 0:
        print("Warning: NaNs in true similarity matrix, dropping nans")
        # return {
        #     'fowlkes_mallows': None,
        #     'rand_index': None,
        #     'nmi': None,
        #     'ami': None,
        # }
        # Let's remove any cells that have more than the mode number of nans
        # Note that this is a heuristic approahch, and can't be guarenteed to work in all cases (it assumes a few sequences cause _all_ problems)
        # Convert back to square matrix
        y_true = squareform(y_true, force='tomatrix')
        # Calculate the mode of the number of nans
        # mode = mode_fn(np.sum(np.isnan(y_true).astype(int), axis=0))
        # Remove rows and columns with more than the mode number of nans
        mode = int(mode[0])
        print("Mode is ", mode)
        row_wise = np.sum(np.isnan(y_true).astype(int), axis=0) <= mode
        col_wise = np.sum(np.isnan(y_true).astype(int), axis=1) <= mode
        good_indices = row_wise & col_wise
        print("A total of ", len(good_indices), " indices are good")
        print(f"Nan Total: {np.sum(np.isnan(y_true))}")
        print("Original shape", y_true.shape)
        y_true = y_true[np.ix_(good_indices, good_indices)]
        print("New shape", y_true.shape)
        print(y_true)
        # Assert still square
        assert y_true.shape[0] == y_true.shape[1]
        # Convert back to condensed form
        y_true = squareform(y_true, force='tovector')

        # Do the same for the predicted matrix
        y_pred = squareform(y_pred, force='tomatrix')
        y_pred = y_pred[np.ix_(good_indices, good_indices)]
        assert y_pred.shape[0] == y_pred.shape[1]
        y_pred = squareform(y_pred, force='tovector')

    # Cluster
    y_true = 1 - np.array(y_true)
    y_pred = 1- np.array(y_pred)
    true_linkage = linkage(y_true, method=method)
    pred_linkage = linkage(y_pred, method=method)

    if max_k is not None:
        max_k = min(len(pred_linkage), 100) # Limit number of clusters to 100
    else:
        max_k = len(pred_linkage)

    # Fowlkes Mallows Score
    fm_scores = []
    for k in tqdm(list(range(2, max_k))):
        tc = np.squeeze(cut_tree(true_linkage, k))
        pc = np.squeeze(cut_tree(pred_linkage, k))
        fm_scores.append(fowlkes_mallows_score(tc, pc))

    # Rand Index
    rand_scores = []
    for k in tqdm(list(range(2, max_k))):
        tc = np.squeeze(cut_tree(true_linkage, k))
        pc = np.squeeze(cut_tree(pred_linkage, k))
        rand_scores.append(adjusted_rand_score(tc, pc))
        

    # Normalized Mutual Information
    nmi_scores = []
    for k in tqdm(list(range(2, max_k))):
        tc = np.squeeze(cut_tree(true_linkage, k))
        pc = np.squeeze(cut_tree(pred_linkage, k))
        nmi_scores.append(normalized_mutual_info_score(tc, pc))

    # Adjusted Mutual Information
    ami_scores = []
    for k in tqdm(list(range(2, max_k))):
        tc = np.squeeze(cut_tree(true_linkage, k))
        pc = np.squeeze(cut_tree(pred_linkage, k))
        ami_scores.append(adjusted_mutual_info_score(tc, pc))
    
    # Calculate cophenetic correlation
    true_cophenetic_corr = cophenet(true_linkage, y_true)[0]
    pred_cophenetic_corr = cophenet(pred_linkage, y_pred)[0]


    if figure_path:
        figure_path = Path(figure_path)
        if not figure_path.parent.exists():
            figure_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(1, 3, figsize=(20, 7))

        # Plot the dendrograms
        dendrogram(true_linkage, ax=axs[0], color_threshold=0.7 * max(true_linkage[:, 2]))
        # axs[0].set_title(f"True Linkage\nCophenetic Correlation: {true_cophenetic_corr:.2f}")
        axs[0].set_xlabel("Sample index")
        axs[0].set_ylabel("Distance")
        axs[0].set_title(f"True Linkage\nCophenetic Correlation: {true_cophenetic_corr:.2f}")

        dendrogram(pred_linkage, ax=axs[1], color_threshold=0.7 * max(pred_linkage[:, 2]))

        axs[1].set_xlabel("Sample index")
        axs[1].set_ylabel("Distance")
        axs[1].set_title(f"Predicted Linkage\nCophenetic Correlation: {pred_cophenetic_corr:.2f}")

        # Plot the Fowlkes Mallows Score
        auc = np.trapezoid(fm_scores, range(2, max_k)) / (max_k - 2)
        axs[2].plot(range(2, max_k), fm_scores, label=f"FM Score AUC={auc:.2f}")
        
        auc = np.trapezoid(rand_scores, range(2, max_k)) / (max_k - 2)
        axs[2].plot(range(2, max_k), rand_scores, label=f"Rand Score AUC={auc:.2f}")

        auc = np.trapezoid(nmi_scores, range(2, max_k)) / (max_k - 2)
        axs[2].plot(range(2, max_k), nmi_scores, label=f"NMI Score AUC={auc:.2f}")

        auc = np.trapezoid(ami_scores, range(2, max_k)) / (max_k - 2)
        axs[2].plot(range(2, max_k), ami_scores, label=f"AMI Score AUC={auc:.2f}")

        axs[2].legend()
        
        axs[2].set_xlabel("Number of Clusters")
        axs[2].set_ylabel("Clustering Score")
        axs[2].set_title(f"Clustering Evaluation")
        axs[2].set_ylim(-0.1, 1.1)

        plt.suptitle(f"Fowlkeys Mallows Score (Method={method})")
        plt.tight_layout()

        # Add some left, right margin
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        print("********************************")
        print("Saving Tree Metrics to ", figure_path)
        plt.savefig(figure_path)
        plt.close(fig)

    return {
        'fowlkes_mallows': fm_scores,
        'rand_index': rand_scores,
        'nmi': nmi_scores,
        'ami': ami_scores,
    }

def create_report(predictions:List[float], true_similarity:List[float], metadata:List[dict], output_path:Path, k:int=5):
    """Output the k worst predictions, with associated metadata to a folder.
    Spectra are saved as images and numpy arrays, metadata is saved as a json file.

    Parameters
    ----------
    predictions : List[float]
        The predicted similarities
    true_similarity : List[float]
        The true similarities
    metadata : List[dict]
        The metadata associated with the spectra
    output_path : Path
        The path to save the output to

    Returns
    ----------
    None
    """
    # Sort the predictions
    worst_predictions = np.argsort(np.abs(np.array(predictions) - np.array(true_similarity)))[::-1][:k]

    print("len(metadata)", len(metadata))
    print("len(predictions)", len(predictions))
    print("len(true_similarity)", len(true_similarity))

    for i, idx in enumerate(worst_predictions):
        subdir_path = output_path / f"worst_{i}"
        if not subdir_path.exists():
            subdir_path.mkdir(parents=True, exist_ok=True)

        # Save the spectra
        spectrum_a = torch.tensor(metadata[idx]['spectrum_a'])
        spectrum_b = torch.tensor(metadata[idx]['spectrum_b'])
        accession_a = metadata[idx]['accession_a']
        accession_b = metadata[idx]['accession_b']
        np.save(subdir_path / f"spectrum_a.npy", spectrum_a)
        np.save(subdir_path / f"spectrum_b.npy", spectrum_b)

        mirror_plot(spectrum_a, spectrum_b, subdir_path / f"mirror.png", 
                    title=f"True Similarity: {true_similarity[idx]:.2f}\nPredicted Similarity: {predictions[idx]:.2f}",
                    x_label="Mass Bin",
                    y_label="Intensity",
                    top_label=accession_a,
                    bottom_label=accession_b)
 
        # Save the metadata
        output_metadata = {} #metadata[idx].copy()
        with open(subdir_path / f"metadata.json", "w", encoding="utf-8") as f:
            output_metadata['accession_a'] = accession_a
            output_metadata['accession_b'] = accession_b
            output_metadata['true_similarity'] = true_similarity[idx].item()
            output_metadata['predicted_similarity'] = predictions[idx].item()
            output_metadata['index'] = idx.item()
            output_metadata['error'] = np.abs(predictions[idx].item() - true_similarity[idx].item())
            output_metadata['num_peaks_in_a'] = metadata[idx]['num_peaks_in_a']#.item()
            output_metadata['num_peaks_in_b'] = metadata[idx]['num_peaks_in_b']#.item()
            # if len(spectrum_a.shape) == 2:
            output_metadata['shannon_entropy_a'] = shannon_entropy(spectrum_a)
            output_metadata['shannon_entropy_b'] = shannon_entropy(spectrum_b)
            if len(spectrum_a.shape) > 1:
                print("spectrum_a.shape", spectrum_a.shape)
                output_metadata['convexity_a'] = estimate_convexity(spectrum_a, rescale=True)
                output_metadata['convexity_b'] = estimate_convexity(spectrum_b, rescale=True)
            # else:
            #     output_metadata['shannon_entropy_a'] = None
            #     output_metadata['shannon_entropy_b'] = None
            json.dump(output_metadata, f, indent=4)

    # if len(spectrum_a.shape) == 2:
    # TODO: This is super inefficent, but it's fine for now
    shannon_entropy_a = [shannon_entropy(meta['spectrum_a']) for meta in metadata]
    shannon_entropy_b = [shannon_entropy(meta['spectrum_b']) for meta in metadata]
    if len(spectrum_a.shape) > 1:
        convexity_a = [estimate_convexity(meta['spectrum_a'], rescale=True) for meta in metadata]
        convexity_b = [estimate_convexity(meta['spectrum_b'], rescale=True) for meta in metadata]
    else:
        convexity_a = [None] * len(metadata)
        convexity_b = [None] * len(metadata)

    # else:
    #     shannon_entropy_a = [None] * len(metadata)
    #     shannon_entropy_b = [None] * len(metadata)

    # Generate a dataframe of all predictions, true values, and metadata
    summary_df = pd.DataFrame({
        'predicted_similarity': predictions,
        'true_similarity': true_similarity,
        'error': np.abs(np.array(predictions) - np.array(true_similarity)),
        'strain_a': [meta['strain_a'] for meta in metadata],
        'strain_b': [meta['strain_b'] for meta in metadata],
        'accession_a': [meta['accession_a'] for meta in metadata],
        'accession_b': [meta['accession_b'] for meta in metadata],
        'num_peaks_in_a': [meta['num_peaks_in_a'] for meta in metadata],
        'num_peaks_in_b': [meta['num_peaks_in_b'] for meta in metadata],
        'shannon_entropy_a': shannon_entropy_a,
        'shannon_entropy_b': shannon_entropy_b,
        'convexity_a': convexity_a,
        'convexity_b': convexity_b,
    })

    # Sort by error
    summary_df = summary_df.sort_values(by='error', ascending=False)

    # Save the summary DataFrame as a CSV file.
    summary_df.to_csv(output_path / "summary.csv", index=False)

    return summary_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--metric_path", type=str, default="metrics")
    parser.add_argument("--inference_set", type=str, default="test", choices=["test", "val", "train", "all"])
    args = parser.parse_args()

    if str(args.model_name).lower() == 'cosine':
        model = RawCosine()
        trans = transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

    else:
        model_dir = Path('lightning_logs') / str(args.model_name) / 'checkpoints'
        model_path = list(model_dir.glob('*.ckpt'))
        # hparams = Path('lightning_logs') / str(args.model_name) / 'hparams.yaml'
        # hparams = torch.load(hparams)
        if len(model_path) == 0:
            raise FileNotFoundError(f"No model found in {model_dir}")
        if len(model_path) > 1:
            raise FileNotFoundError(f"Multiple models found in {model_dir}")
        model_path = model_path[0]
        print(f"Loding model from {model_path}")


        trans=None
        # Load the model
        if 'mlp_model' in args.model_name.lower():
            print("Performing inference on MLP")
            model = MLP.load_from_checkpoint(model_path)
            trans =  transforms.Compose([BinSpectrum(10, 2_000, 20_000), SquareRootTransform(), NormalizeIntensity()])
        elif 'autoencoder' in args.model_name.lower():
            print("Performing inference on Autoencoder")
            hyperparameters = { # Temporary fix until I figure out how to deal with this
                    'input_dim': 1800,
                    'output_dim': 1800,
                    'hidden_dim': 600,
                    'bottleneck_dim': 300,
                    'hidden_layers': 3,
                    'weight_decay': 1e-5,
                    'dropout': 0.2,
                }
            trans = transforms.Compose([BinSpectrum(10, 2_000, 20_000), SquareRootTransform(), NormalizeIntensity()])
            model = Autoencoder(hyperparameters)
            model.load_converted_from_checkpoint(model_path)
        elif 'transformer_embedding_prediction_model' == args.model_name.lower().split('.')[0]:
            print("Performing inference on TransformerPredictionHead")
            hyperparameters = {
                'hidden_dim': 300,
                'latent_dim': 300,
                'hidden_layers': 3,
                'dropout': 0.2,
                'lr': 5e-4,
                'weight_decay': 1e-5,
                'freeze_encoder': False,
                'encoder_path': './lightning_logs/maldi_transformer_model/version_175/checkpoints/epoch=2193-step=43880.ckpt'
            }
            trans = transforms.Compose([SelectMassRange(2_000, 20_000), 
                                    NormalizeIntensity(),
                                    SelectTopKPeaks(150),
                                    PadToLength(150),])
            # model = TransformerPredictionHead(hyperparameters)
            model = TransformerPredictionHead.load_from_checkpoint(model_path)
        elif 'Sentence_MALDI' in args.model_name:
            print("Performing inference on Sentence_MALDI")
            model = Sentence_MALDI.load_from_checkpoint(model_path)

            trans =  transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])
        elif 'MLPBinaryClassifier' in args.model_name:
            print("Performing inference on MLPBinaryClassifier")
            model = MLPBinaryClassifier.load_from_checkpoint(model_path)

            trans = transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])

        elif 'binary_transformer_embedding' in args.model_name:
            print("Performing inference on BinaryTransformerPredictionHead")
            model = BinaryTransformerPredictionHead.load_from_checkpoint(model_path)

            trans = transforms.Compose([SelectMassRange(2_000, 20_000),
                                    NormalizeIntensity(),
                                    SelectTopKPeaks(150),
                                    PadToLength(150),])
        elif args.model_name.split('/',1)[0] == "CLIP_MLP":
            print("Performing inference on CLIP_MLP")
            model = CLIP_MALDI.load_from_checkpoint(model_path)

            trans = transforms.Compose([BinSpectrum(10, 3_000, 20_000), SquareRootTransform(), NormalizeIntensity()])
        elif args.model_name.split('/',1)[0] == "CLIP_Transformer":
            print("Performing inference on CLIP_Transformer")
            model = CLIP_MALDI.load_from_checkpoint(model_path)

            trans =  transforms.Compose([
                                    SquareRootTransform(),
                                    NormalizeIntensity(),
                                    SelectTopKPeaks(150),
                                    PadToLength(150, padding_value=-1.0),
                                    ])
        else:
            raise ValueError(f"Unknown model name {args.model_name}")
        
    datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                '../../data/idbac_db/raw/ammended_db.csv',
                                '../../data/idbac_db/processed_data',
                                num_workers=7, 
                                wipe_test_sets=False,
                                inference_set_to_use=args.inference_set,
                                transforms=trans,
                                batch_size=1,
                                split_method='species')
                                # split_method='genera')
    
    metadata_table = '../../data/idbac_db/raw/ammended_db.csv'
    metadata_table = pd.read_csv(metadata_table)

    # Perform inference on all data
    datamodule.setup('test')

    model.eval()
    logger = TensorBoardLogger('lightning_logs', name=str(args.model_name)+'/prediction')

    # Get the predictions
    trainer = L.Trainer(logger=logger, devices=[0], inference_mode=False)   # Inference mode false to allow gradients

    # Use the Trainer to run predictions
    outputs = trainer.predict(model, datamodule=datamodule, return_predictions=True)

    # Extract predictions and ground truth values
    predictions = []
    true_similarity = []
    metadata = []

    for output in outputs:
        predictions.extend(output['predictions'].detach().cpu().numpy())  # Assuming you want numpy arrays
        true_similarity.extend(output['similarity'].detach().cpu().numpy())
        # output['metadata'] is a dict of lists
        # want to convert it to a list of dicts
        if 'metadata' in output:
            for i in range(len(output['metadata']['accession_a'])):
                metadata.append({
                    k: v[i] for k, v in output['metadata'].items()
                })

    print("true_similarity", true_similarity)

    # Save the predictions
    predictions = np.array(predictions)
    true_similarity = np.array(true_similarity)
    if np.max(predictions) > 1 + 1e-6 or np.min(predictions) < 0 - 1e-6:
        raise ValueError("Predictions are not in the range [0, 1]")
    # Clamp (to handle numerical errors)
    predictions = np.clip(predictions, 0, 1)


    metric_path = Path('lightning_logs') / str(args.model_name) / str(args.metric_path)
    if not metric_path.exists():
        metric_path.mkdir(parents=True, exist_ok=True)
    print("Saving metrics to", metric_path)

    # print(metadata)
    # print(len(metadata))

    np.save(metric_path / "preds.npy", predictions)
    np.save(metric_path / "true.npy", true_similarity)
    # Save metadata (list of dicts)
    with open(metric_path / "prediction_metadata.json", "w", encoding="utf-8") as f:
        # Change all tensors to lists for json serialization
        for i in tqdm(range(len(metadata)), desc="Processing metadata"):
            for k, v in metadata[i].items():
                if isinstance(v, np.ndarray):
                    metadata[i][k] = v.tolist()
                if isinstance(v, torch.Tensor):
                    metadata[i][k] = v.tolist()

                # Check if object is serializable, if not print it
                try:
                    json.dumps(metadata[i][k])
                except:
                    print(f"Could not serialize {metadata[i][k]}")
        json.dump(metadata, f, indent=4)

    # Create a report of the worst predictions
    if len(metadata) > 0:
        prediction_table = create_report(predictions, true_similarity, metadata, metric_path / "worst_predictions", k=5)
        # for k in [1,3,5,7,10]:
        #     compute_information_imbalance(prediction_table, metric_path / "information_imbalance"/ f"information_imbalance_{k}_dense.txt", k=k, method='dense')
        #     compute_information_imbalance(prediction_table, metric_path / "information_imbalance"/ f"information_imbalance_{k}_min.txt", k=k, method='min')

        # compute_top_in_top_k(prediction_table, metric_path / "top_in_top_k.json", k_range=list(range(1, 15)))

        # Compute taxa-dependent clustering scores
        for tax_level in ["genus", "species"]:
            print(f"Computing clustering scores for {tax_level}")
            compute_taxa_clustering_scores(prediction_table, metric_path / f"clustering_scores_{tax_level}.json", metadata_table, tax_level=tax_level, plot=True)

        # Computer per-genus clustering scores
        # compute_clustering_scores_per_genera(prediction_table, metadata_table, metric_path / "clustering_scores_per_genera.json", metric_path / "clustering_scores_per_genera/")

    f = open(metric_path / "metrics.txt", 'w', encoding='utf-8')

    # Fowlkes Mallows Score
    # print("Computing Fowlkes Mallows Score")
    # scores = compute_clustering_scores(true_similarity, predictions, metric_path / "dendrograms.png")

    # if not (metric_path / 'clustering_scores').exists():
    #     (metric_path / 'clustering_scores').mkdir(parents=True, exist_ok=True)
    # np.save(metric_path / 'clustering_scores' / "fowlkes_mallows_scores.npy", scores['fowlkes_mallows'])
    # np.save(metric_path / 'clustering_scores' / "rand_index_scores.npy", scores['rand_index'])
    # np.save(metric_path / 'clustering_scores' / "nmi_scores.npy", scores['nmi'])
    # np.save(metric_path / 'clustering_scores' / "ami_scores.npy", scores['ami'])

    # Correlation of predictions with true values
    correlation = np.corrcoef(predictions, true_similarity)[0, 1]
    mae = np.mean(np.abs(predictions - true_similarity))
    rmse = np.sqrt(np.mean((predictions - true_similarity) ** 2))
    print("Correlation of predictions with true values:", correlation)
    # Save to txt
    f.write(f"Correlation of predictions with true values: R2={correlation:.2f} \n")
    f.write(f"Mean Absolute Error: {mae:.2f} \n")
    f.write(f"Root Mean Squared Error: {rmse:.2f} \n")
    # Plot
    fig = plt.figure()
    sns.scatterplot(x=true_similarity, y=predictions, alpha=0.5)
    x_min = np.nanmin([np.nanmin(true_similarity), np.nanmin(predictions)])
    x_max = np.nanmax([np.nanmax(true_similarity), np.nanmax(predictions)])
    y_min = np.nanmin([np.nanmin(true_similarity), np.nanmin(predictions)])
    y_max = np.nanmax([np.nanmax(true_similarity), np.nanmax(predictions)])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("True Sequence Similarity Similarity")
    plt.ylabel("Predicted Sequence Similarity")
    plt.title(f"Predicted vs True similarity (R2={correlation:.2f})")
    plt.savefig(metric_path / "scatter.png")

    # Hex Density Plot
    hexplot = sns.jointplot(x=true_similarity, y=predictions, kind='hex', xlim=(x_min, x_max), ylim=(y_min, y_max))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = hexplot.figure.add_axes([.85, .25, .05, .4])  # x, y, width, height
    hexplot.set_axis_labels("True Sequence Similarity Similarity", "Predicted Sequence Similarity")
    plt.colorbar(cax=cbar_ax)
    plt.suptitle(f"Predicted vs True similarity (R2={correlation:.2f})")
    # plt.tight_layout()
    plt.savefig(metric_path / "hex.png")

    f.close()


if __name__ == "__main__":
    main()