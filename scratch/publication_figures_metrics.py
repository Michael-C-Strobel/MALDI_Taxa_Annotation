from publication_figures_helpers import gather_embeddings, NAME_MAPPINGS, colors, PAIRED_MODELS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from pathlib import Path
import re
from joblib import Parallel, delayed

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import recall_score
from collections import defaultdict
from typing import Optional, Tuple, List, Dict

from scipy import stats

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
        valid_mask = (~np.isnan(scores[i]) & ~np.isinf(scores[i]))
        # invalid_mask_sum = sum(np.isnan(scores[i]))
        valid_sorted = [idx for idx in sorted_indices[i] if valid_mask[idx]]
        # Pad with a sentinel (e.g., None) if fewer than k valid predictions
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
        valid_scores = ~(np.isnan(scores[i]) | np.isinf(scores[i]))
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
                'strain_name': strain_names[1][i],
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

    train_species = train_df['species'].astype(str).values
    test_species = test_df['species'].astype(str).values

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

    return scores, train_df['true_label'].values, test_df['true_label'].values, (train_df['strain_name'].values, test_df['strain_name'].values)


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
        
        if False:
            print("************* REMOVING SINGLETON GENERA (ONE SPECIES GENERA)")
            singleton_genera = train_df['true_label'].value_counts()
            singleton_genera = singleton_genera[singleton_genera == 1].index.tolist()
            train_df = train_df[~train_df['true_label'].isin(singleton_genera)]
            
            singleton_genera = test_df['true_label'].value_counts()
            singleton_genera = singleton_genera[singleton_genera == 1].index.tolist()
            test_df = test_df[~test_df['true_label'].isin(singleton_genera)]

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


def top_k_recall_plot(
        dataset,
        target,
        split_type,
        n_jobs=-1,
        within_test=False,
        macro=False,
        cross_species=False,
        db_dataset=None, 
        legend=True,
        title=False,
    ):
    embeddings = gather_embeddings(dataset, target, split_type)
    print(embeddings.keys())

    # Add intensity-agnostic embeddings
    print("Adding intensity-agnostic embeddings...")
    embeddings['cosine_intensity_agnostic'] = {'train': [None]*len(embeddings['cosine']['train']),
                                               'test': [None]*len(embeddings['cosine']['test'])}
    for i in range(len(embeddings['cosine_intensity_agnostic']['test'])):
        embeddings['cosine_intensity_agnostic']['train'][i] = embeddings['cosine']['train'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['test'][i] = embeddings['cosine']['test'][i].copy(deep=True)
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x>0.00).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x>0.00).astype(int))

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
            db_embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = db_embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x>0.00).astype(int))
            db_embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = db_embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x>0.00).astype(int))

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
    # print("Removing all embeddings except clip_transformer, cosine_intensity_agnostic, cross_encoder...")
    # for key in list(embeddings.keys()):
    #     if key not in {'clip_transformer', 'cosine_intensity_agnostic', 'cross_encoder'}:
    #         del embeddings[key]
    # print("Remaining embeddings:", embeddings.keys())

    # Prepare job arguments
    job_args = []
    for key in embeddings.keys():
        if key in {'metadata', 'multinomial_classifier', 'cross_encoder'}:
            continue
        distance_metric = 'cosine' if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic'} else 'euclidean'

        for i in range(len(embeddings[key]['train'])):
            train_df = embeddings[key]['train'][i]
            test_df = embeddings[key]['test'][i]

            # Subsample train_df and test_df to 5 spectra per genera
            # if target == 'genera':
            #     train_df = train_df.groupby('true_label').apply(lambda x: x.sample(n=min(5, len(x)), random_state=42)).reset_index(drop=True)
            #     test_df = test_df.groupby('true_label').apply(lambda x: x.sample(n=min(5, len(x)), random_state=42)).reset_index(drop=True)

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

    # Get top 50 hits with clip_transformer, then use cross_encoder to rerank
    if 'clip_transformer' in embeddings and 'cross_encoder' in embeddings and False:
        print("Calculating super score...")

        print(len(embeddings['cross_encoder']['train']))
        print(len(embeddings['clip_transformer']['test']))

        job_args = []
        for i in range(len(embeddings['cross_encoder']['test'])):
            train_df = embeddings['cross_encoder']['train'][i]
            test_df = embeddings['cross_encoder']['test'][i]

            # Order the embeddings by strain_name for consistency
            # train_df = train_df.sort_values(by='strain_name')
            # test_df = test_df.sort_values(by='strain_name')
            
            if dataset.lower() == 'driams-a':
                clip_embeddings_train = embeddings['clip_transformer']['train'][i].sort_values(by='strain_name')
                clip_embeddings_test = embeddings['clip_transformer']['test'][i].sort_values(by='strain_name')
            else:
                clip_embeddings_train = embeddings['clip_transformer']['train'][0].sort_values(by='strain_name')
                clip_embeddings_test = embeddings['clip_transformer']['test'][0].sort_values(by='strain_name')

            clip_scores, train_labels, test_labels, strain_names = compute_scores(
                clip_embeddings_train,
                clip_embeddings_test,
                distance_metric='cosine',
                normalize=True,
                within_test=within_test,
                cross_species=cross_species,
            )
            print('strain_names', strain_names[0][:5])
            print('strain_names.shape', strain_names[0].shape)

            print('clip_scores.shape', clip_scores.shape)
            print('clip_scores:', clip_scores[:5, :5])

            print('train_labels', train_labels[:5])
            print('test_labels', test_labels[:5])

            print('"cross_encoder" test_df', test_df.head())

            # Print where this is happening
            if any(isinstance(x, np.ndarray) for x in test_df['strain_name_a']):
                print('"cross_encoder" test_df strain_name_a contains arrays:', test_df['strain_name_a'][test_df['strain_name_a'].apply(lambda x: isinstance(x, np.ndarray))])
            if any(isinstance(x, np.ndarray) for x in test_df['strain_name_b']):
                print('"cross_encoder" test_df strain_name_b contains arrays:', test_df['strain_name_b'][test_df['strain_name_b'].apply(lambda x: isinstance(x, np.ndarray))])

            # test_df is the cross_encoder test set, we need to reverse all cols with _a, _b and reconcat to make it square
            test_df = test_df.pivot(index='strain_name_a', columns='strain_name_b', values='binary_pred')
            test_df = test_df.combine_first(test_df.T)  # Make it symmetric
            np.fill_diagonal(test_df.values, -np.inf)

            # Count how many rows/cols are common with strain_names[0], strain_names[1]
            common_strains_a = test_df.index.intersection(strain_names[0])
            common_strains_b = test_df.columns.intersection(strain_names[1])
            print(f"Test set dimensions: {len(strain_names[0])}, {len(strain_names[1])}")
            print(f"Common strains in cross_encoder test set: {len(common_strains_a)} rows, {len(common_strains_b)} columns")
            test_df = test_df.loc[common_strains_a, common_strains_b]

            # Top 50 indices
            sorted_idx = np.argsort(-clip_scores, axis=1)   # Highest similarity first
            # Get lowest similarity indices (everything out of top-50)
            other_idx = sorted_idx[:, 50:]

            super_scores = clip_scores.copy()
            print('super_scores.shape', super_scores.shape)
            print("super_scores", super_scores[:5, :10])
            print('test_df.values.shape', test_df.values.shape)
            print('sorted_idx.shape', sorted_idx.shape)
            print('sorted_idx', sorted_idx[:5, :10])
            rows = np.arange(super_scores.shape[0])[:, None]
            super_scores[rows, sorted_idx] = test_df.values[rows, sorted_idx]
            super_scores[rows, other_idx] = -np.inf  # Mask out all but top 50
            super_scores[~np.isfinite(clip_scores)] = -np.inf  # Reapply original mask

            job_args.append((super_scores, train_labels, test_labels, strain_names, i))

        super_results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_top_k_recall)(
                scores=scores,
                train_labels=train_labels,
                test_labels=test_labels,
                strain_names=strain_names,
                method='Super Score (Clip + Cosine)',
                max_k=10,
                average=average,
                within_test=within_test,
                metadata={'cv_fold': cvf},
                cross_species=cross_species
            )
            for scores, train_labels, test_labels, strain_names, cvf in tqdm(job_args)
        )
        results.extend(super_results)

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
    accuracies_df['genera_counts'] = accuracies_df['genera_counts'].apply(lambda x: x if isinstance(x, int) else len(x) if isinstance(x, list) else np.nan)

    #  Print the accuracies that the plot will use (weighted mean)
    if average == 'macro':
        weighted_means = (
            accuracies_df
            .groupby(['model_name', 'k'])
            .apply(lambda g: np.average(g['accuracy'], weights=g['genera_counts']))
            .reset_index(name='weighted_accuracy')
        )
        print(weighted_means)
    else:
        simple_means = (
            accuracies_df
            .groupby(['model_name', 'k'])
            .agg(mean_accuracy=('accuracy', 'mean'))
            .reset_index()
        )
        print(simple_means)
    

    fig=None
    fig_leg = None

    try:
        # Line plot
        fig, ax = plt.subplots(figsize=(6, 4))
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
        if title:
            plt.title(f"{dataset}-{target}-{split_type}", pad=20)
        plt.xlabel('# Neighbors Considered (k)')
        if average == 'macro':
            plt.ylabel('Macro Recall')
        else:
            plt.ylabel('Micro Recall')
        plt.ylim(0, 1)
        plt.grid(False)

        # Show every x-tick
        x_vals = sorted(accuracies_df['k'].unique())
        # Skip evens
        # x_vals = [x for x in x_vals if x % 2 == 1]
        plt.xticks(x_vals, labels=[str(int(x)) for x in x_vals])

        # Remove top and right spines
        sns.despine()
        # Spine width 0.8
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)


        ax.legend_.remove()
        if legend:
            # 2. Create the Legend Canvas
            if legend:
                # Grab handles/labels from the main axis
                handles, labels = ax.get_legend_handles_labels()
                
                # Create a new figure for the legend
                fig_leg = plt.figure(figsize=(3, 4)) 
                fig_leg.legend(handles, labels, loc='center', ncols=5)
                plt.axis('off') # Hide the empty plot lines/ticks
                fig_leg.show()
                
        

        fig.show()
    except Exception as e:
        # Print the traceback using traceback
        print(f"An error occurred while plotting: {e}")
        traceback.print_exc()
        pass

    return fig, accuracies_df, fig_leg


def nn_accuracy( 
                test_df:pd.DataFrame,
                random_seed:int=42,
                k:int=5,
                distance_metric:str='euclidean',
                normalize:bool=True, 
                dedicated_db:pd.DataFrame=None,
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

    sampling_mode = 'species'

    # ----------------- Spectrum Sampling Mode ----------------- #
    if sampling_mode == 'spectrum':
        if dedicated_db is None: # Generate a faux db using the test set
            label_counts = df['true_label'].value_counts()
            singletons = label_counts[label_counts == 1].index.tolist()

            singleton_df = df[df['true_label'].isin(singletons)]
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

            # Add singletons into the faux_db THIS IS NEW 6/26
            # Ensure singleton's aren't already in that set 
            assert sampled_indices_set.isdisjoint(singletons), "Singletons should not be in the sampled indices set."

            faux_db = pd.concat([df.loc[list(sampled_indices_set), :], singleton_df]).values
        else:
            # Use the provided faux_db
            dedicated_db = dedicated_db.copy(deep=True)
            # Subsample the faux_db to k samples per taxa
            grouped_db = dedicated_db.groupby('true_label')
            sampled_indices_set = set()             # Empty set because we don't want to remove anything
            sampled_indices_dedicated_db = set()    # Different set because we don't want to remove these indices
            for taxa, group in grouped_db:
                _k = min(k, len(group)//2)
                sampled_indices = np.random.choice(group.index, size=_k, replace=False)
                for i in sampled_indices:
                    sampled_indices_dedicated_db.add(i)

            faux_db = dedicated_db.loc[list(sampled_indices_dedicated_db), :].values
    # ----------------- Species Sampling Mode ----------------- #
    elif sampling_mode == 'species':
        if dedicated_db is None: # Generate a faux db using the test set
            single_species_genera = []
            grouped = df.groupby('genus', observed=True)
            for genus, group in grouped:
                if group.species.nunique() == 1:
                    single_species_genera.extend(group.index.tolist())

            single_species_df = df.loc[single_species_genera, :]
            df = df.drop(index=single_species_genera)

            if normalize:
                # L2 Norm
                df['embedding'] = df['embedding'].apply(lambda x: x / np.linalg.norm(x))
                # Check if any are nan, set to 0
                df['embedding'] = df['embedding'].apply(lambda x: np.nan_to_num(x))

            # Generate a dict[genus][species] -> indices
            genus_species_indices = {}
            for l, group in df.groupby('genus', observed=True):
                if l not in genus_species_indices:
                    genus_species_indices[l] = {}
                for s, species_group in group.groupby('species', observed=True):
                    genus_species_indices[l][s] = species_group.index.tolist()

            sampled_indices_set = set()
            # For each genus, sample k species, and take all spectra from those species
            for genus, species_dict in genus_species_indices.items():
                _k = min(k, len(species_dict)-1) # Ensure at least one species is left out
                sampled_species = np.random.choice(list(species_dict.keys()), size=_k, replace=False)
                
                for species in sampled_species:
                    species_indices = species_dict[species]
                    for i in species_indices:
                        sampled_indices_set.add(i)

            assert sampled_indices_set.isdisjoint(single_species_genera), "Single-species genera should not be in the sampled indices set."

            faux_db = pd.concat([df.loc[list(sampled_indices_set), :], single_species_df])

            # Show the faux_db
            # print(faux_db.head())
            # print('faux_db.shape', faux_db.shape)

            faux_db = faux_db.values

        else:
            # ----------------- Species-Provided DB Sampling Mode ----------------- #
            # Use the provided faux_db
            dedicated_db = dedicated_db.copy(deep=True)
            # Subsample the faux_db to k samples per taxa
            grouped_db = dedicated_db.groupby('true_label')
            sampled_indices_set = set()
            sampled_indices_dedicated_db = set()    # Different set because we don't want to remove these indices
            for taxa, group in grouped_db:
                species_grouped = group.groupby('species')
                _k = min(k, len(species_grouped)-1) # Ensure at least one species is left out
                sampled_species = np.random.choice(list(species_grouped.groups.keys()), size=_k, replace=False)
                for species in sampled_species:
                    species_group = species_grouped.get_group(species)
                    for i in species_group.index:
                        sampled_indices_dedicated_db.add(i)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    predictions = []
    labels = []
    failure_cases = []

    faux_db_labels = np.unique(faux_db[:, 4])

    # Print the number of rows that aren't in the sampled-index set
    # print(f"Evaluating on {len(df) - len(sampled_indices_set)} test samples.")
          
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

        predicted_label = max(similarities, key=lambda x: x[1])[0]  # Always take the top-1 prediction
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

def nn_accuracy_plot(dataset, target, split_type, n_jobs=-1, within_test=False, num_samples_per_k=1,
                     average='macro', require_cross_species=False, require_same_species=False,
                     dedicated_db:str=None):
    
    if require_cross_species and require_same_species:
        raise ValueError("Cannot require both cross-species and same-species accuracy at the same time.")

    if dedicated_db and within_test:
        raise ValueError("Either a 'dedicated_db' or 'within_test' can be specified, but not both.")

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
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))

    if require_cross_species or require_same_species:
        # Remove multinomial classifier, as the denominator is incongruent
        if 'multinomial_classifier' in embeddings:
            print("Removing multinomial classifier for same species accuracy")
            assert 'multinomial_classifier' in embeddings.keys(), "Multinomial classifier should be in embeddings"
            del embeddings['multinomial_classifier']

    if dedicated_db:
        print(f"Using dedicated retrieval database: {dedicated_db}")
        dedicated_embeddings = gather_embeddings(dedicated_db, target, split_type)

        # Generate an intensity_agnostic cosine for dedicated db
        dedicated_embeddings['cosine_intensity_agnostic'] = {'train': [None for _ in range(len(dedicated_embeddings['cosine']['train']))],
                                                            'test': [None for _ in range(len(dedicated_embeddings['cosine']['test']))]}

        for i in range(len(dedicated_embeddings['cosine_intensity_agnostic']['test'])):
            dedicated_embeddings['cosine_intensity_agnostic']['train'][i] = dedicated_embeddings['cosine']['train'][i].copy(deep=True)
            dedicated_embeddings['cosine_intensity_agnostic']['test'][i] = dedicated_embeddings['cosine']['test'][i].copy(deep=True)
            dedicated_embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = dedicated_embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
            dedicated_embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = dedicated_embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))

    # Quick sanity check, remove everything but contrastive transformer TODO DEBUG
    # for key in list(embeddings.keys()):
    #     if key not in {'clip_transformer', 'cosine', 'cosine_intensity_agnostic'}:
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
            'color': colors.get(key, "#000000"),
        }

    tasks = []

    k_range = np.arange(1, 11, 1)

    for key, train_test_dict in embeddings.items():
        if key == 'metadata':
            continue
        if key == 'multinomial_classifier':
            continue

        for i in range(len(train_test_dict['train'])):
            retrieval_db = None
            if not within_test:
                retrieval_db = embeddings[key]['train'][i]
            if dedicated_db:
                retrieval_db = dedicated_embeddings[key]['test'][0]
            if key in {'clip_transformer', 'clip_transformer_genus_genus', 'cosine', 'cosine_intensity_agnostic', 'maldi_transformer_ts'}:
                print(f"Using cosine distance for {key}")
                distance_metric = 'cosine'
            else:
                print(f"Using euclidean distance for {key}")
                distance_metric = 'euclidean'

            test_embeddings = train_test_dict['test'][i]

            for random_seed in np.arange(42, 42+num_samples_per_k):
                for k in k_range:
                    tasks.append((key, test_embeddings, distance_metric, k, random_seed, retrieval_db))

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_accuracy)(key, test_embeddings, distance_metric, k, random_seed, retrieval_db)
        for key, test_embeddings, distance_metric, k, random_seed, retrieval_db in tqdm(tasks, desc="Computing nearest neighbor accuracies")
    )

    if 'multinomial_classifier' in embeddings:
        # Calculate accuracy for multinomial classifier
        print("Adding static accuracy calculation for multinomial classifier...")
        for i in range(len(embeddings['multinomial_classifier']['train'])):
            test_df = embeddings['multinomial_classifier']['test'][i]
            acc = recall_score(test_df['true_label'], test_df['pred_class'], average=average)
            for k in k_range:
                results.append({
                    'model': 'multinomial_classifier',
                    'k': k,
                    'accuracy': acc,
                    'random_seed': 42,
                    'color': colors.get('multinomial_classifier', "#000000"),
                })

    accuracies_df = pd.DataFrame(results)
   
    # Line plot
    fig = plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=accuracies_df,
        x='k',
        y='accuracy',
        hue='model',
        style='model',
        markers=True,
        dashes=False,
        errorbar='ci',
        palette=dict(zip(accuracies_df['model'], accuracies_df['color']))
        )

    # Rename legend entries based on NAME_MAPPINGS
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = [NAME_MAPPINGS.get(label, label) for label in labels]
    # plt.legend(handles=handles, labels=new_labels)

    if False:
        if within_test:
            plt.title(f"Within Test Nearest-Neighbor Accuracy for {dataset} - {target} - {split_type}")
        else:
            plt.title(f"Train-Test Nearest-Neighbor Accuracy for {dataset} - {target} - {split_type}")
    plt.xlabel(f'Number of Species Per {target.capitalize()} (k)')
    if average == 'macro':
        plt.ylabel('Macro Accuracy')
    elif average == 'micro':
        plt.ylabel('Micro Accuracy')
    else:
        plt.ylabel('Accuracy')
    plt.xticks(k_range)
    plt.ylim(0, 1)
    plt.grid(False)
    # Completely disable the legend, even if already plotted
    plt.legend([],[], frameon=False)

    # Remove top and right splines
    ax = plt.gca() # Get current axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Spine width to 0.8
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    plt.show()

    return fig, accuracies_df


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
    interp_points = np.linspace(0, 1, 300)
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
            print(traceback.format_exception(e))
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

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import traceback
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def compute_sim_numba(X, Y, i1, i2, metric_code):
    """
    metric_code: 0 for cosine, 1 for euclidean
    """
    n_pairs = i1.shape[0]
    dim = X.shape[1]
    sim = np.empty(n_pairs, dtype=np.float32)
    
    for k in prange(n_pairs):
        idx1 = i1[k]
        idx2 = i2[k]
        
        if metric_code == 0:  # Cosine
            dot = 0.0
            for d in range(dim):
                dot += X[idx1, d] * Y[idx2, d]
            sim[k] = (dot + 1.0) / 2.0  # Scaled dot
            
        else:  # Euclidean
            dist_sq = 0.0
            for d in range(dim):
                diff = X[idx1, d] - Y[idx2, d]
                dist_sq += diff * diff
            sim[k] = -np.sqrt(dist_sq)
            
    return sim

def _binary_similarity_curves(df1, df2, distance_metric='cosine', max_pairs=None, between_species=False, balance_pairs=False, remove_singleton_genera=False):
    """
    Computes binary labels and similarity scores for all pairs between df1 and df2,
    then returns ROC and PR curve components.

    Returns:
        fpr, tpr, roc_auc, precision, recall, pr_auc
    """
    if remove_singleton_genera:
        print("******************* REMOVING SINGLETON GENERA")

        non_singleton_genera = df1.groupby('genus').apply(lambda g: g.species.nunique() > 1)
        non_singleton_genera = non_singleton_genera[non_singleton_genera].index.tolist()
        df1 = df1[df1['genus'].isin(non_singleton_genera)]


        non_singleton_genera = df2.groupby('genus').apply(lambda g: g.species.nunique() > 1)
        non_singleton_genera = non_singleton_genera[non_singleton_genera].index.tolist()
        df2 = df2[df2['genus'].isin(non_singleton_genera)]

    if df2.attrs.get('paired'):
        assert df2.test == True
        y_true = df2['true_label'].values
        sim    = df2['binary_pred'].values

        print('y_true', y_true)
        print('sim', sim)
    else:
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
        genera_X = np.array(df1['genus'])
        genera_Y = np.array(df2['genus'])

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

        i1 = np.array([p[0] for p in pair_indices])
        i2 = np.array([p[1] for p in pair_indices])

        # 2. Pre-normalize for cosine, or if your metric requires it
        if distance_metric == 'cosine':
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            m_code = 0
        else:
            m_code = 1

        i1_arr = np.array([p[0] for p in pair_indices], dtype=np.int32)
        i2_arr = np.array([p[1] for p in pair_indices], dtype=np.int32)

        # 3. Call Numba (This replaces the chunking loop entirely)
        # It will use all CPU cores and almost zero extra RAM.
        sim = compute_sim_numba(X, Y, i1_arr, i2_arr, m_code)

        l1 = labels_X[i1]
        l2 = labels_Y[i2]
        y_true = (l1 == l2).astype(int)

    balance_labels = []

    for idx1, idx2 in pair_indices:
        genus_pair = tuple(sorted((genera_X[idx1], genera_Y[idx2])))
        balance_labels.append(genus_pair) 
        
    # Print # of nan similarity scores
    if np.isnan(sim).any():
        print(f"Warning: {np.isnan(sim).sum()} NaN similarity scores found out of {len(sim)}")
        # Set nan similarity scores to 0
        sim[np.isnan(sim)] = 0.0

    acc = (np.round(y_true.flatten(), 0) == np.round(sim.flatten(), 0)).sum()/len(y_true)

    sample_weights = None
    if balance_pairs:
        sample_weights = compute_sample_weight(class_weight='balanced', y=balance_labels)

    # Assert all positive
    assert np.all(sim.flatten() >= 0), "sim.flatten() contain negative values."

    fpr, tpr, roc_thresholds = roc_curve(y_true.flatten(), sim.flatten(), drop_intermediate=False, sample_weight=sample_weights)
    precision, recall, pr_thresholds = precision_recall_curve(y_true.flatten(), sim.flatten(), sample_weight=sample_weights, drop_intermediate=False)
    return (fpr, tpr, roc_thresholds, auc(fpr, tpr)), (precision, recall, pr_thresholds, auc(recall, precision))

def _compute_interpolated_curves(
        df1,
        df2,
        method,
        data_idx,
        interp_points,
        metric,
        max_pairs,
        between_species=False,
        balance_pairs=False,
        remove_singleton_genera=False):
    print(f"_compute_interpolated_curves; {method} : {metric}")
    try:
        if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
            print(f"Method {method} is using intensity agnostic embeddings")
            df1['embedding'] = df1['embedding'].apply(lambda x: (x > 0.00).astype(int))
            df2['embedding'] = df2['embedding'].apply(lambda x: (x > 0.00).astype(int))
        
        if method == 'cosine_intensity_agnostic_between_species':
            print("Using cosine intensity agnostic embeddings for between species")
            df1['embedding'] = df1['embedding'].apply(lambda x: (x > 0.00).astype(int))
            df2['embedding'] = df2['embedding'].apply(lambda x: (x > 0.00).astype(int))
            between_species = True

        (fpr, tpr, roc_thresholds, roc_auc_val),(prec, rec, pr_thresholds, pr_auc_val) = _binary_similarity_curves(df1, df2, metric, max_pairs, between_species=between_species, balance_pairs=balance_pairs, remove_singleton_genera=remove_singleton_genera)
        fdr = 1 - prec

        # --- DEBUG PRINT BLOCK ---
        # print(f"\n[DEBUG {method} Seed {data_idx}] Raw PR Curve Samples:")
        # thresh_array is n-1, precision is n. We zip them to see the relationship.
        # We look at the tail end where thresholds are highest (most strict)
        # print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10}")
        # print("-" * 40)
        # Show ~10 points across the distribution
        # step = max(1, len(pr_thresholds) // 10)
        # for i in range(0, len(pr_thresholds), step):
        #     print(f"{pr_thresholds[i]:<12.4f} | {prec[i]:<10.4f} | {rec[i]:<10.4f}")
        # Always check the very last thresholded point
        # print(f"{pr_thresholds[-1]:<12.4f} | {prec[-2]:<10.4f} | {rec[-2]:<10.4f}")
        # print("-" * 40 + "\n")
        # -------------------------

        # print(f"Raw Precision: {prec}")
        # print(f"Raw Recall: {rec}")

        # print(f"ROC AUC before interpolation for {method} seed {data_idx}: {roc_auc_val:.3f}")
        # print(f"PR AUC before interpolation for {method} seed {data_idx}: {pr_auc_val:.3f}")


        target_precision = 0.95
        meeting_indices = np.where(prec >= target_precision)[0]
        
        informed_point = None
        if len(meeting_indices) > 0:
            # In sklearn, the last precision value is 1.0 (at rec 0), 
            # so we look for the index that gives us the best recall
            best_idx = meeting_indices[0] # Usually the one with highest recall
            
            # Guard against the n vs n-1 length of pr_thresholds
            t_val = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else pr_thresholds[-1]
            r_val = rec[best_idx]
            p_val = prec[best_idx]
            informed_point = (t_val, p_val, r_val)
        

        return (
            np.interp(interp_points, fpr, tpr),
            np.interp(interp_points, rec[::-1], prec[::-1]),
            np.interp(interp_points, rec[::-1], fdr[::-1])
        ), informed_point
    
    except Exception as e:
        print(f"Interpolation failed for {method} seed {data_idx}: {e}")
        traceback.print_exc()
        # raise e
        return None

def static_precision_recall_roc(df1, df2, target, split_type, method, max_pairs=None, between_species=False,
    balance_pairs=False
    ):
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
    genera_X = np.array(df1['genus'])
    genera_Y = np.array(df2['genus'])

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

    # Use 'genus' column to make combined labels, must be sorted order
    balance_labels = []
    for idx1, idx2 in pair_indices:
        genus_pair = tuple(sorted((genera_X[idx1], genera_Y[idx2])))
        balance_labels.append(genus_pair)

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    sample_weights = None
    if balance_pairs:
        sample_weights = compute_sample_weight(class_weight='balanced', y=balance_labels)

    def wsum(mask):
        if sample_weights is None:
            return np.sum(mask)
        return np.sum(sample_weights[mask])

    tp = wsum(tp_mask)
    fp = wsum(fp_mask)
    fn = wsum(fn_mask)
    tn = wsum(tn_mask)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    return precision, recall, fpr

def binary_curves_plot(dataset, target, split_type, test_only=False, max_pairs=None,
                       cosine_ablation=False, n_jobs=-1, between_species=False, balance_pairs=False, rki_disjoint=None, remove_singleton_genera=False):
    if cosine_ablation:
        embeddings = gather_embeddings_cosine_only(dataset, target, split_type)
    else:
        embeddings = gather_embeddings(dataset, target, split_type, rki_disjoint=rki_disjoint)

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
            embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
            embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
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
                        embeddings[f"{key}_intensity_agnostic"]['train'][i]['embedding'] = embeddings[f"{key}_intensity_agnostic"]['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
                        embeddings[f"{key}_intensity_agnostic"]['test'][i]['embedding'] = embeddings[f"{key}_intensity_agnostic"]['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
                # Debug, remove the old one
                del embeddings[key]

    for key in list(embeddings.keys()):
        if key in PAIRED_MODELS and not test_only:
            print(f"Removing paired model {key} from embeddings")
            del embeddings[key]


    # Print prior probability of equal and unequal taxa
    a_key = [x for x in list(embeddings.keys()) if x != 'metadata' and x not in PAIRED_MODELS][0]
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

    output_table = []

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
            if remove_singleton_genera:
                raise ValueError("Removing singleton genera is not compatible with multinomial_classifier method.")
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
                    between_species=between_species,
                    balance_pairs=balance_pairs
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

            output_table.append({
                'method': method,
                'precision_mean': precision_mean,
                'precision_95_ci_upper': None,
                'precision_95_ci_lower': None,
                'recall_mean': recall_mean,
                'recall_95_ci_upper': None,
                'recall_95_ci_lower': None,
                'fpr_mean': fpr_mean,
                'fpr_95_ci_upper': None,
                'fpr_95_ci_lower': None
            })

        else:
                
            tasks = []
            for data_idx in range(len(data_list['test'])):
                df1 = data_list['test'][data_idx].copy()
                df2 = df1.copy() if test_only else data_list['train'][data_idx].copy()

                tasks.append((df1, df2, method, data_idx))

            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_interpolated_curves)(df1, df2, method, data_idx, interp_points, metric, max_pairs, between_species, balance_pairs, remove_singleton_genera=remove_singleton_genera)
                for df1, df2, method, data_idx in tasks
            )

            # Filter out failed results
            results = [res for res in results if res is not None]
            if not results:
                continue

            roc_curves = []
            pr_curves = []
            fdr_curves = []
            informed_points = []
            # Unpack the results
            for curves, informed_point in results:
                roc_curve, pr_curve, fdr_curve = curves
                roc_curves.append(roc_curve)
                pr_curves.append(pr_curve)
                fdr_curves.append(fdr_curve)
                informed_points.append(informed_point)

            if informed_points:
                opt_thresholds = [pt[0] for pt in informed_points if pt is not None]
                avg_threshold = np.mean(opt_thresholds)
                std_threshold = np.std(opt_thresholds)
                print(f"[{method}] Average lowest SIMILARITY threshold: {avg_threshold:.3f} ± {std_threshold:.3f}")
                print(f"[{method}] Average lowest DISTANCE threshold: {1-avg_threshold:.3f} ± {std_threshold:.3f}")
            roc_curves = np.array(roc_curves)
            pr_curves = np.array(pr_curves)
            fdr_curves = np.array(fdr_curves)

            # Logit transform the curves for better normality, but only if values are strictly between 0 and 1
            if not np.all((roc_curves >= 0) & (roc_curves <= 1)):
                # Show the erring value
                err_idx = np.where((roc_curves <= 0) | (roc_curves >= 1))
                print(f"ROC curve values outside (0, 1) at indices: {err_idx}, values: {roc_curves[err_idx]}")
                raise ValueError(f"ROC curves contain values outside (0, 1), cannot logit transform.")
        
            def logit(p, eps=1e-6):
                p = np.clip(p, eps, 1 - eps)
                return np.log(p / (1 - p))

            def inv_logit(x):
                return 1 / (1 + np.exp(-x))

            confidence = 0.95
            n = pr_curves.shape[0]  # Number of folds (k)
            dof = n - 1             # Degrees of freedom
            t_crit = stats.t.ppf((1 + confidence) / 2., dof)

            confidence = 0.95
            n = pr_curves.shape[0]
            dof = n - 1
            t_crit = stats.t.ppf((1 + confidence) / 2., dof)

            roc_logit = logit(roc_curves)
            pr_logit = logit(pr_curves)
            fdr_logit = logit(fdr_curves)
            

            # Mean and SEM in logit space
            roc_mean_logit = roc_logit.mean(axis=0)
            roc_sem_logit = stats.sem(roc_logit, axis=0)
            pr_mean_logit = pr_logit.mean(axis=0)
            pr_sem_logit = stats.sem(pr_logit, axis=0)
            fdr_mean_logit = fdr_logit.mean(axis=0)
            fdr_sem_logit = stats.sem(fdr_logit, axis=0)

            # CI in logit space
            roc_lower_logit = roc_mean_logit - t_crit * roc_sem_logit
            roc_upper_logit = roc_mean_logit + t_crit * roc_sem_logit
            pr_lower_logit = pr_mean_logit - t_crit * pr_sem_logit
            pr_upper_logit = pr_mean_logit + t_crit * pr_sem_logit
            fdr_lower_logit = fdr_mean_logit - t_crit * fdr_sem_logit
            fdr_upper_logit = fdr_mean_logit + t_crit * fdr_sem_logit

            # Transform EVERYTHING back
            roc_mean = inv_logit(roc_mean_logit)
            roc_upper = inv_logit(roc_upper_logit)
            roc_lower = inv_logit(roc_lower_logit)
            pr_mean = inv_logit(pr_mean_logit)
            pr_lower = inv_logit(pr_lower_logit)
            pr_upper = inv_logit(pr_upper_logit)
            fdr_mean = inv_logit(fdr_mean_logit)
            fdr_upper = inv_logit(fdr_upper_logit)
            fdr_lower = inv_logit(fdr_lower_logit)


            roc_auc_val = auc(interp_points, roc_mean)
            pr_auc_val = auc(interp_points, pr_mean)
            fdr_auc_val = auc(interp_points, fdr_mean)

            roc_ax.plot(interp_points, roc_mean, label=f"{NAME_MAPPINGS[method]}", color=colors.get(method))  #  (auc={roc_auc_val:.2f})
            roc_ax.fill_between(interp_points, roc_lower, roc_upper, alpha=0.15, color=colors.get(method))
            # Plot individual runs with alpha
            # for curve in roc_curves:
            #     roc_ax.plot(interp_points, curve, color=colors.get(method), alpha=0.3, linewidth=0.5)

            pr_ax.plot(interp_points, pr_mean, label=f"{NAME_MAPPINGS[method]}", color=colors.get(method))    #  (auc={pr_auc_val:.2f})
            pr_ax.fill_between(interp_points, pr_lower, pr_upper, alpha=0.15, color=colors.get(method))
            # Plot individual runs with alpha
            # for curve in pr_curves:
            #     pr_ax.plot(interp_points, curve, color=colors.get(method), alpha=0.3, linewidth=0.5)

            fdr_ax.plot(interp_points, fdr_mean, label=f"{NAME_MAPPINGS[method]}", color=colors.get(method)) #  (auc={fdr_auc_val:.2f})
            fdr_ax.fill_between(interp_points, fdr_lower, fdr_upper, alpha=0.15, color=colors.get(method))
            # Plot individual runs with alpha
            # for curve in fdr_curves:
            #     fdr_ax.plot(interp_points, curve, color=colors.get(method), alpha=0.3, linewidth=0.5)

            # Print the Method, and Standard Deviation at 5 points
            print(f"[{method}] ROC AUC: {roc_auc_val:.3f}, PR AUC: {pr_auc_val:.3f}, FDR AUC: {fdr_auc_val:.3f}")
            for i in [0, 25, 50, 74, 79, 84, 89, 99]:
                print(f"[{method}] X- Axis is {interp_points[i]:.2f}")
                print(f"[{method}] ROC at {interp_points[i]:.2f}")
                print(f"[{method}] PR at {interp_points[i]:.2f}")
                print(f"[{method}] FDR at {interp_points[i]:.2f}")

            output_table.append({
                'method': method,
                'roc_auc': roc_auc_val,
                'roc_95_ci_lower': auc(interp_points, pr_lower),
                'roc_95_ci_upper': auc(interp_points, pr_upper),
                'pr_auc': pr_auc_val,
                'pr_95_ci_lower': auc(interp_points, pr_lower),
                'pr_95_ci_upper': auc(interp_points, pr_upper),
                'fdr_auc': fdr_auc_val,
                'fdr_95_ci_lower': auc(interp_points, fdr_lower),
                'fdr_95_ci_upper': auc(interp_points, fdr_upper)
            })
    # Finalize ROC
    roc_ax.plot([0, 1], [0, 1], 'k--')
    roc_ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate') # , title='ROC Curve'
    # roc_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    roc_ax.set_xlim(-0.05, 1.05)
    roc_ax.set_ylim(-0.05, 1.05)

    # Finalize PR
    ylabel = "Macro Precision" if balance_pairs == True else "Micro Precision"
    pr_ax.set(xlabel='Recall', ylabel=ylabel) # , title='Precision-Recall Curve'
    # pr_ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    pr_ax.set_xlim(-0.05, 1.05)
    pr_ax.set_ylim(-0.05, 1.05)

    # Finalize FDR
    fdr_ax.set(xlabel='Recall', ylabel='False Discovery Rate') # , title='FDR vs Recall'
    fdr_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    fdr_ax.set_xlim(-0.05, 1.05)
    fdr_ax.set_ylim(-0.05, 1.05)
    
    for ax in [roc_ax, pr_ax, fdr_ax]:
        # Remove top and right spines for all plots
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Spine width to 0.8
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        ax.tick_params(direction='in', length=3, width=0.8)

    return {
        'roc': roc_fig,
        'precision_recall': pr_fig,
        'fdr': fdr_fig,
        'metrics_table': pd.DataFrame(output_table)
    }


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
    return_failure_cases: bool = False,
    within_test: bool = True,
) -> Tuple[float, List[Dict[str, str]]]:
    """
    Computes precision@k using either micro or macro averaging.

    average:
        - "micro": total correct / total predictions (default)
        - "macro": average of per-label mean precision@k
    """

    # if 'intensity_agnostic' in method and method != 'clip_transformer_intensity_agnostic':
    #     print(f"Method {method} is using intensity agnostic embeddings")
    #     train_df['embedding'] = train_df['embedding'].apply(lambda x: (x > 0.00).astype(int))
    #     test_df['embedding'] = test_df['embedding'].apply(lambda x: (x > 0.00).astype(int))

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

    if within_test:
        assert train_df.shape[0] == test_df.shape[0], "Train and test must be the same size for within_test"
        np.fill_diagonal(similarities, -np.inf)  # Exclude self in nearest neighbors

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
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))

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
            train_df, test_df, k, method=key, distance_metric=distance_metric, average=average, within_test=within_test
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
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))

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
        embeddings['cosine_intensity_agnostic']['train'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['train'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))
        embeddings['cosine_intensity_agnostic']['test'][i]['embedding'] = embeddings['cosine_intensity_agnostic']['test'][i]['embedding'].apply(lambda x: (x > 0.00).astype(int))


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