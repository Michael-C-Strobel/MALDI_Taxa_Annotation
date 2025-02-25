from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from preprocess import postprocess_files, convert_spectra_to_tensor
import os 
import sys
import numpy as np
import torch
from torch import default_generator
from torch.utils.data import Subset, Sampler, IterableDataset
import scipy
import pytest
import copy

import time

from typing import (
    cast,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

def get_most_recent_modified_time()->float:
    most_recent_time = None
    
    # Iterate over all imported modules
    for module_name, module in sys.modules.items():
        # Check if the module has a file associated with it
        if hasattr(module, '__file__') and module.__file__:
            file_path = module.__file__
            try:
                # Get the last modified time
                mod_time = os.path.getmtime(file_path)
                
                # Update the most recent time if applicable
                if most_recent_time is None or mod_time > most_recent_time:
                    most_recent_time = mod_time
            except FileNotFoundError:
                # Ignore if the module file is not found
                pass
    
    return most_recent_time

class single_MALDI_TOF_DS(Dataset):
    def __init__(self, preprocessing_dir:str,
                metadata_table:str,
                root_dir:str,
                process:bool=True,
                transform:callable=None,
                balance:str='accession',
                require_genus:bool=False):
        self.root_dir = root_dir
        self.preprocessing_dir = preprocessing_dir
        self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))

        balance = str(balance).lower()
        if balance not in ['accession', 'genus', 'species']:
            raise ValueError(f"Invalid balance method. Expected one of ['accession', 'class'], got '{balance}'")
        if balance == 'class':
            raise NotImplementedError("Class balancing not yet implemented")
        self.balance = balance

        all_spectra_names = [x.stem for x in self.all_spectra]

        metadata_table = pd.read_csv(metadata_table)
        if 'accession' not in metadata_table.columns:
            metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
        else:
            metadata_table['accession'] = metadata_table['accession'].str.strip()
        metadata_table = metadata_table.loc[metadata_table['Strain name'].isin(all_spectra_names)]
        metadata_table = metadata_table.drop_duplicates(subset='Strain name')   # Some strains occur twice due to multuple csv files

        self.metadata_table = metadata_table
        if require_genus:
            self.metadata_table = self.metadata_table.loc[self.metadata_table['genus'].notna()]

        self.transform = transform

        self.num_turns = 2

        similarities = Path(self.root_dir) / 'similarities.feather'
        self.similarities = None
        if similarities.exists():
            temp_similarities = pd.read_feather(similarities)
            post_filtration_accessions = self.metadata_table.accession.unique()
            temp_similarities = temp_similarities.loc[temp_similarities['query_genbank'].isin(post_filtration_accessions) & \
                                                  temp_similarities['subject_genbank'].isin(post_filtration_accessions)]
            square_similarities = temp_similarities.pivot_table(index='query_genbank', columns='subject_genbank', values='pident')
            # Set Diag to 100
            np.fill_diagonal(square_similarities.values, 100)
            
            # Ensure actually square
            if square_similarities.shape[0] != square_similarities.shape[1]:
                raise ValueError(f"Similarities matrix is not square: {square_similarities.shape}")
            self.similarities = square_similarities


        if not process:
            preprocessing_dir_stat = Path(self.preprocessing_dir).stat()
            most_recent_m_time = get_most_recent_modified_time()

            if Path(self.root_dir).exists():
                root_dir_stat = Path(self.root_dir).stat()
            else:
                root_dir_stat = None

            if preprocessing_dir_stat.st_mtime > root_dir_stat.st_mtime or \
                most_recent_m_time > root_dir_stat.st_mtime or \
                not Path(self.root_dir).exists():
                    self.preprocess()
        else:
            self.preprocess()

        self.all_accessions = self.metadata_table.loc[self.metadata_table['accession'].notna(), 'accession'].unique().astype(str)
        
        if self.similarities is not None:
            self.all_accessions = np.intersect1d(self.all_accessions, self.similarities.index.values.astype(str))
            self.metadata_table = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)]

        # DEBUG, TEMPORARY
        # Remove any accessions whose spectra were removed
        removed = ['strain_B032', 'nan']
        self.all_accessions = np.array([x for x in self.all_accessions if not (str(x) in removed)])

        self.n_genus = len(self.metadata_table['genus'].unique())

        labels, uniques = pd.factorize(self.metadata_table['genus'].sort_values().unique())
        # Create the class indices dictionary
        one_hot_encoder = dict(zip(uniques, range(1, len(uniques) + 1)))  # Starting from 1
        # Add 'nan' as class 0
        one_hot_encoder['nan'] = 0
        self.one_hot_encoder = one_hot_encoder

    def sample_strain_from_accession(self, accession):
        choices = self.metadata_table[self.metadata_table['accession'] == accession]
        if len(choices) == 0:
            raise ValueError(f"No strain found for accession '{accession}'")
        return choices.sample(1).to_dict(orient='records')[0]
    
    def __len__(self):
        return len(self.all_accessions) * self.num_turns

    def __getitem__(self, idx):
        # For DRIAMS, use the species name as the accession
        # strain_name will be the hash
        accession = self.all_accessions[idx % len(self.all_accessions)]
        sampled_row = self.sample_strain_from_accession(accession)
        strain_name = sampled_row['Strain name']
        database_id = sampled_row['database_id']
        
        spectrum = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name}.pt', weights_only=True).to(torch.float32)

        if self.transform:
            try:
                spectrum = self.transform(spectrum)
            except Exception as e:
                raise RuntimeError(f"Error transforming spectrum for strain {strain_name} with accession {accession}") from e

        metadata = {
            'accession': accession,
            'Strain name': strain_name,
            'class': str(self.metadata_table[self.metadata_table['database_id'] == database_id]['genus'].values[0]),
            'database_id': database_id,
        }

        return spectrum, metadata
    
    def get_one_hot_encoded_classes(self):
        """
        
        """
        def one_hot_encode(classes:List):
            return torch.Tensor([self.one_hot_encoder[str(x)] for x in classes]).to(torch.long)  # str() important to cover nan
        return one_hot_encode
         
    def _genus_triplets(self, metadata:dict):
        """ Generates positive and negative triplets based on genus.

        The positive triplet is from the same genus, while the negative triplet is from a different genus.

        Args:
            metadata (dict): The metadata dictionary.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The positive and negative triplets.
            Tuple[dict, dict]: The positive and negative metadata.
        """

        anchor_strain_name = metadata['Strain name']
        anchor_accession = metadata['accession']
        anchor_class = metadata['class']

        # Positive triplet
        positive_mask = (self.metadata_table['genus'] == anchor_class)

        # Valid distance
        relevant_dists = self.similarities.loc[anchor_accession, :].notna()
        # print("na distances found", (len(relevant_dists) - sum(relevant_dists)))
        relevant_dists = relevant_dists.loc[relevant_dists].index.values.tolist()
        r_dists =relevant_dists
        relevant_dists = self.metadata_table['accession'].astype(str).isin(relevant_dists)
        # print("relevant_dists", relevant_dists.sum(), '/', len(relevant_dists))
        # print("relevant_dists", relevant_dists.sum())
        # print("positive_mask", positive_mask.sum())
        positive_mask = positive_mask & relevant_dists

        assert positive_mask.index.equals(self.metadata_table.index)

        all_possible_accessions = self.metadata_table.loc[positive_mask, 'accession'].values.tolist()
        # print('len(self.metadata_table)', len(self.metadata_table))
        # print('len(all_possible_accessions)', len(all_possible_accessions))
        # print('sum(positive_mask)', sum(positive_mask))
        # print('len(pos_mask)', len(positive_mask))


        if self.similarities.loc[anchor_accession, all_possible_accessions].isna().any():
            print("self.similarities.loc[anchor_accession, all_possible_accessions].isna().any()")
            print(self.similarities.loc[anchor_accession, all_possible_accessions].isna().any(), flush=True)


            # Get location where this happens
            print("anchor_accession", anchor_accession)
            missing_accessions = []
            for accession in all_possible_accessions:
                if np.isnan(self.similarities.at[anchor_accession, accession]):
                    print(f"Missing similarity for {accession}")
                    print('In relevant_dists', accession in r_dists)
                    idx_in_metadata = (self.metadata_table[self.metadata_table['accession'] == accession]).index
                    print('In positive_mask', positive_mask[idx_in_metadata])
                    print('value in relevant_dists', relevant_dists[idx_in_metadata])   
                    missing_accessions.append(accession)

            raise ValueError(f"Positive mask contains NaN values for {anchor_accession}")

        # new_positive_mask = positive_mask & \
        #                 (self.metadata_table['Strain name'] != anchor_strain_name)
        # if sum(new_positive_mask) > 0:   # Avoid self comparison if possible
        #     positive_mask = new_positive_mask
        
        positive_row = self.metadata_table[positive_mask].sample(1).to_dict(orient='records')[0]
        positive_metadata = {
            'accession': positive_row['accession'],
            'Strain name': positive_row['Strain name'],
            'class': positive_row['genus'],
            'database_id': positive_row['database_id'],
        }
        pos_sim = self.similarities.loc[anchor_accession, positive_metadata['accession']]
        positive_spectrum = torch.load(Path(self.root_dir) / 'spectra' / f"{positive_metadata['Strain name']}.pt", weights_only=True).to(torch.float32)

        if self.transform:
            try:
                positive_spectrum = torch.tensor(self.transform(positive_spectrum))
            except Exception as e:
                raise RuntimeError(f"Error transforming positive spectrum for strain {positive_metadata['Strain name']} with accession {positive_metadata['accession']}") from e

        # Negative triplet
        negative_row = self.metadata_table[self.metadata_table['genus'] != anchor_class].sample(1).to_dict(orient='records')[0]
        negative_metadata = {
            'accession': negative_row['accession'],
            'Strain name': negative_row['Strain name'],
            'class': negative_row['genus'],
            'database_id': negative_row['database_id'],
        }
        neg_sim = self.similarities.loc[metadata['accession'], negative_metadata['accession']]
        negative_spectrum = torch.load(Path(self.root_dir) / 'spectra' / f"{negative_metadata['Strain name']}.pt", weights_only=True).to(torch.float32)

        if self.transform:
            try:
                negative_spectrum = torch.tensor(self.transform(negative_spectrum))
            except Exception as e:
                raise RuntimeError(f"Error transforming negative spectrum for strain {negative_metadata['Strain name']} with accession {negative_metadata['accession']}") from e

        pos_sim = pos_sim/100
        assert 0.0 <= pos_sim <= 1.0, f"pos_sim is {pos_sim} for {metadata['accession']} and {positive_metadata['accession']}"
        # We don't actually care, since we're not backpropogating on it:
        # neg_sim = neg_sim/100
        # assert 0.0 <= neg_sim <= 1.0, f"neg_sim is {neg_sim}"


        return (positive_spectrum, negative_spectrum), (positive_metadata, negative_metadata), (pos_sim, neg_sim)

    def generate_triplets(self, spectrum:torch.Tensor, metadata:dict, strategy:str='genus'):
        """ Generates the positive and negative triplets for a given metadata input and strategy.

        Args:
            metadata (dict): The metadata dictionary.
            strategy (str, optional): The strategy to use for generating triplets. Defaults to 'genus'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The positive and negative triplets.
            Tuple[dict, dict]: The positive and negative metadata.
        """

        if strategy == 'genus':
            (positive_spectrum, negative_spectrum), (positive_metadata, negative_metadata), (pos_sim, neg_sim) = self._genus_triplets(metadata)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return (torch.tensor(spectrum), positive_spectrum, negative_spectrum), (metadata, positive_metadata, negative_metadata), (None, pos_sim, neg_sim)
    
    def subset(self, indices:List):
        #  Make a copy, in this way the sliced similarities can be used to identify the subset
        subset_dataset = copy.deepcopy(self)
        # Match indices to accessions
        mapped_indices = [idx % len(self.all_accessions) for idx in indices]    # TODO: This is untennable. Need to switch to unique IDS
        accessions = np.unique(self.all_accessions[mapped_indices])

        # Remove any accessions whose spectra were removed
        subset_dataset.similarities = subset_dataset.similarities.loc[accessions, accessions]

        return Subset(subset_dataset, indices)
        

    def preprocess(self,):
        return
        if not (Path(self.root_dir) / 'spectra/').exists():
            (Path(self.root_dir) / 'spectra/').mkdir(parents=True, exist_ok=True)
        print("Preprocessing files...")
        for strain_name, spectrum_as_tensor in convert_spectra_to_tensor(Path(self.preprocessing_dir) / 'baseline_corrected.json'):
            torch.save(spectrum_as_tensor, Path(self.root_dir) / f'spectra/{strain_name}.pt')

class Paired_MALDI_TOF_DS(Dataset):
    def __init__(self, preprocessing_dir:str,
                 metadata_table:str,
                 root_dir: str,
                 process:bool=True,
                 transform:callable=None):
        self.root_dir = root_dir
        self.preprocessing_dir = preprocessing_dir
        self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))
        all_spectra_names = [x.stem for x in self.all_spectra]

        metadata_table = pd.read_csv(metadata_table)
        metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
        metadata_table = metadata_table.loc[metadata_table['Strain name'].isin(all_spectra_names)]

        self.metadata_table = metadata_table
        self.transform = transform

        self.num_turns = 2
        
        if not process:
            preprocessing_dir_stat = Path(self.preprocessing_dir).stat()
            most_recent_m_time = get_most_recent_modified_time()

            if Path(self.root_dir).exists():
                root_dir_stat = Path(self.root_dir).stat()
            else:
                root_dir_stat = None

            if preprocessing_dir_stat.st_mtime > root_dir_stat.st_mtime or \
                most_recent_m_time > root_dir_stat.st_mtime or \
                not Path(self.root_dir).exists():
                    self.preprocess()
        else:
            self.preprocess()

        similarities = Path(self.root_dir) / 'similarities.feather'
        temp_similarities = pd.read_feather(similarities)
        post_filtration_accessions = self.metadata_table.accession.unique()

        assert 'strain_B016' in temp_similarities['query_genbank'].values


        # Remove any accessiosn whose spectra were removed
        temp_similarities = temp_similarities.loc[temp_similarities['query_genbank'].isin(post_filtration_accessions) & \
                                                  temp_similarities['subject_genbank'].isin(post_filtration_accessions)]

        # assert 'strain_B016' in temp_similarities['query_genbank'].values

        # Remove all accessions with poor BLASTN results
        square_similarities = temp_similarities.pivot_table(index='query_genbank', columns='subject_genbank', values='pident')
        # Ensure actually square
        if square_similarities.shape[0] != square_similarities.shape[1]:
            raise ValueError(f"Similarities matrix is not square: {square_similarities.shape}")

        # This implicitly assumes, you have more good than bad results, which is risky
        is_na = square_similarities.isna().sum(axis=1)
        # na_mode = is_na.mode().item()
        na_mode = is_na.median().item()
        not_na_accessions   = np.unique(is_na.loc[is_na <= na_mode].index.values)
        na_accessions       = np.unique(is_na.loc[is_na > na_mode].index.values)
        print(f"Found {len(na_accessions)} accessions with limited number of BLASTN results. Removing them.")
        print(f"Found {len(not_na_accessions)} accessions with sufficient BLASTN results.")
        # DEBUG
        # temp_similarities = temp_similarities.loc[temp_similarities['query_genbank'].isin(not_na_accessions) & \
        #                                             temp_similarities['subject_genbank'].isin(not_na_accessions)]
        print(f"Left with {len(temp_similarities)} pairs.")
        # Recalculate the square similarities
        # DEBUG
        # square_similarities = square_similarities.loc[not_na_accessions, not_na_accessions]

        # Must be sorted to maintain train/test set consistency 
        # self.all_accessions = np.sort(np.unique(np.concatenate((temp_similarities['query_genbank'].values, temp_similarities['subject_genbank'].values))))

        # assert 'strain_B016' in self.all_accessions

        # self.metadata_table = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)]

        self.all_accessions = self.metadata_table.loc[self.metadata_table['accession'].notna(), 'accession'].unique().astype(str)

        self.sim_bins = np.linspace(temp_similarities['pident'].min(), temp_similarities['pident'].max(), 21)   # Data leakage in the _absolute_ strictest sense
        self.similarities = square_similarities
        self.sliced_similarities = self._preslice_similarities(temp_similarities)

        # Initialize the linkage and clustered accessions
        self.linkage = None
        self.train_test_sim = None
        self.clustered_accessions = None

    def __len__(self):
        return len(self.all_accessions) * self.num_turns
    
    def __getitem__(self, idx):
        accession_a = self.all_accessions[idx % len(self.all_accessions)]
        strain_name_a = self.sample_strain_from_accession(accession_a)

        # DEBUG TO DELETE STUFF
        rand_int = np.random.randint(0, self.sim_bins.shape[0] - 1)
        accession_b = None
        # while accession_b is None or accession_b in ['EF178692', 'AB184357', 'AB122711', 'AB184476', 'strain_B017', 'AB122711']:  # SS preprocessing
        while accession_b is None or accession_b in ['strain_B032',]:            
            strain_name_b, accession_b, similarity = self.find_match_in_range(accession_a, strain_name_a, rand_int)
        spectrum_a = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name_a}.pt', weights_only=True).to(torch.float32)
        spectrum_b = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name_b}.pt', weights_only=True).to(torch.float32)

        if self.transform:
            try:
                spectrum_a = self.transform(spectrum_a)
            except Exception as e:
                raise RuntimeError(f"Error transforming spectrum_a for strain {strain_name_a} with accession {accession_a}") from e
            try:
                spectrum_b = self.transform(spectrum_b)
            except Exception as e:
                raise RuntimeError(f"Error transforming spectrum_b for strain {strain_name_b} with accession {accession_b}") from e

        num_peaks_in_a = None
        num_peaks_in_b = None
        if len(spectrum_a.shape) == 2:
            num_peaks_in_a = spectrum_a.shape[0]
            num_peaks_in_b = spectrum_b.shape[0]
        elif len(spectrum_a.shape) == 1:
            num_peaks_in_a = (spectrum_a > 0).sum().item()
            num_peaks_in_b = (spectrum_b > 0).sum().item()

        metadata = {
            'accession_a': accession_a,
            'accession_b': accession_b,
            'strain_a': strain_name_a,
            'strain_b': strain_name_b,
            'num_peaks_in_a': num_peaks_in_a,
            'num_peaks_in_b': num_peaks_in_b,
            'spectrum_a': spectrum_a,
            'spectrum_b': spectrum_b,
        }

        ##### DEBUG PLEASE PLEASE PLEASE REMOVE 
        # print("***DEBUG***")
        # time.sleep(3)
        # print("***DEBUG***")

        # Original target:torch.tensor(similarity/100, dtype=torch.float32)

        target = similarity/100
        target = torch.tensor(target, dtype=torch.float32)
        assert 0.0 <= target <= 1.0, f"Target is {target}"


        return spectrum_a, spectrum_b, target, metadata
    
    def get_by_strain_name(self, strain_name):
        accession = self.metadata_table[self.metadata_table['Strain name'] == strain_name]['accession'].values[0]
        spectrum = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name}.pt', weights_only=True).to(torch.float32)
        
        metadata = {}

        num_peaks = None
        if len(spectrum.shape) == 2:
            num_peaks = spectrum.shape[0]
        elif len(spectrum.shape) == 1:
            num_peaks = (spectrum > 0).sum().item()

        metadata.update({
            'accession': accession,
            'strain_name': strain_name,
            'num_peaks': num_peaks,
        })

        if self.transform:
            try:
                spectrum = self.transform(spectrum)
            except Exception as e:
                raise RuntimeError(f"Error transforming spectrum for strain {strain_name} with accession {accession}") from e

        return spectrum, metadata

    def strain_to_accession(self, strain_name):
        return self.metadata_table[self.metadata_table['Strain name'] == strain_name]['accession'].values[0]
    
    def sample_strain_from_accession(self, accession, strain_name=None):
        try:
            if strain_name:
                choices = self.metadata_table[(self.metadata_table['accession'] == accession) & (self.metadata_table['Strain name'] != strain_name)]
            else:
                choices = self.metadata_table[self.metadata_table['accession'] == accession]
            if len(choices) == 0:
                return None
            return choices['Strain name'].sample(1).values[0]
        except ValueError as ve:
            raise ValueError(f'No strain found for accession "{accession}"') from ve

    def find_match_in_range(self, accession, strain_name, bin_index):
        """ Find a strain name that has a similarity in the specified bin range. Excludes identical strains.

        Args:
            accession (str): The Genbank accession of the query strain.
            strain_name (str): The strain name of the query strain.
            bin_index (int): The index of the similarity bin.

        Returns:
            Tuple[str, str, float]: The strain name, accession, and similarity of the matched strain.
        """
        relevant_df = self.sliced_similarities[accession]

        lb = self.sim_bins[bin_index]
        ub = self.sim_bins[bin_index + 1]

        result_strain_name  = None
        result_accession    = None
        result_similarity   = None
        tried_candidates = set()

        in_range = relevant_df.loc[lb:ub,]
        while result_strain_name is None:
            in_range = relevant_df.loc[lb:ub,]
            lb -= 0.05
            ub += 0.05
            # Iterate over the possible accessions, try each until we find a strain name
            if len(in_range) > 0:
                in_range = in_range.sample(1)
                candidates = in_range['subject_genbank'].values
                candidates = list(set(candidates) - tried_candidates)
                if len(candidates) == 0:
                    continue
                for _result_accession in np.random.choice(candidates, replace=False, size=len(in_range)):
                    _result_similarity = in_range.index.values[0]
                    _result_strain_name = self.sample_strain_from_accession(_result_accession, strain_name)
                    if _result_strain_name is not None:
                        result_accession = _result_accession
                        result_strain_name = _result_strain_name
                        result_similarity = _result_similarity
                        break
                    else:
                        tried_candidates.add(_result_accession)

        return result_strain_name, result_accession, result_similarity

    def _preslice_similarities(self, sims:pd.DataFrame):
        out_dict = {}

        grouped = sims.groupby('query_genbank')
        for key, item in grouped:
            out_dict[key] = item.set_index('pident').sort_index()   # Now we can slice by pident (e.g., df.loc[78:98])

        return out_dict
    
    def preprocess(self,):
        if not Path(self.root_dir).exists():
            Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        print("Preprocessing files...")
        postprocess_files(Path(self.preprocessing_dir), Path(self.root_dir))

    def _split_dendrogam(self, return_indices):
        # Split based on the ground-truth dendrogram
        sims = self.similarities
        # Convert to distance matrix
        if sims.max().max() < 99.9:
            raise ValueError('Similarity matrix is not in percentage format. Also, ensure diagonal is 100%')
        distance_matrix = 100 - sims
        np.fill_diagonal(distance_matrix.values, 0)
        # Force symmetric matrix
        distance_matrix.iloc[:,:] = (distance_matrix.values + distance_matrix.values.T) / 2

        # Fill nans with zeros (no ideal, but we're working with what we've got here)
        # DEBUG
        distance_matrix.fillna(100.0, inplace=True)

        # Convert to condensed distance matrix
        condensed = scipy.spatial.distance.squareform(distance_matrix.to_numpy(), force='tovector', checks=True)
        # Perform hierarchical clustering
        linkage = scipy.cluster.hierarchy.linkage(condensed, method='complete')
        # Cut the dendrogram
        clusters = scipy.cluster.hierarchy.cut_tree(linkage, n_clusters=9.0).flatten() # Start with 9 clusters, merge into 3 to try to get even sets# Start with 9 clusters, merge into 3 to try to get even sets
        clustered_accessions = pd.DataFrame({'accession': distance_matrix.index, 'cluster': clusters})
        clustered_accessions.groupby('cluster').apply(lambda x: distance_matrix.loc[x.accession, x.accession].mean().mean(), include_groups=False)

        def _merge_smallest_clusters(clustered_accessions):
            cluster_sizes = clustered_accessions.cluster.value_counts()
            smallest_cluster_size = cluster_sizes.nsmallest(1).item()
            largest_cluster_size  = cluster_sizes.nlargest(1).item()
            if smallest_cluster_size > 0.15 * largest_cluster_size:
                # Add to largest cluster
                smallest_cluster = cluster_sizes.nsmallest(1).index
                smallest_cluster_accessions = clustered_accessions.loc[clustered_accessions.cluster.isin(smallest_cluster)]
                new_cluster = smallest_cluster_accessions.cluster.min()
                largest_cluster  = cluster_sizes.nlargest(1).index
                # largest_cluster_accessions = clustered_accessions.loc[clustered_accessions.cluster.isin(largest_cluster)]
                to_merge = smallest_cluster.append(largest_cluster)
                clustered_accessions.loc[clustered_accessions.cluster.isin(to_merge), 'cluster'] = new_cluster

            else:
                # Merge two smallest clusters
                smallest_clusters = cluster_sizes.nsmallest(2).index
                smallest_cluster_accessions = clustered_accessions.loc[clustered_accessions.cluster.isin(smallest_clusters)]
                new_cluster = smallest_cluster_accessions.cluster.min()
                clustered_accessions.loc[clustered_accessions.cluster.isin(smallest_clusters), 'cluster'] = new_cluster
            return clustered_accessions

        while len(clustered_accessions.cluster.unique()) > 3:
            clustered_accessions = _merge_smallest_clusters(clustered_accessions)

        cluster_counts = clustered_accessions.cluster.value_counts()
        train_cluster_id = cluster_counts.idxmax().item()
        val_cluster_id = cluster_counts.idxmin().item()
        test_cluster_id = cluster_counts[cluster_counts.index != train_cluster_id].idxmax().item()
        # print("train_cluster_id", train_cluster_id)
        # print("val_cluster_id", val_cluster_id)
        # print("test_cluster_id", test_cluster_id)

        # print("clustered_accessions", clustered_accessions)

        train_accessions = clustered_accessions.loc[clustered_accessions.cluster == train_cluster_id, 'accession'].values
        val_accessions = clustered_accessions.loc[clustered_accessions.cluster == val_cluster_id, 'accession'].values
        test_accessions = clustered_accessions.loc[clustered_accessions.cluster == test_cluster_id, 'accession'].values

        # print('train_accessions', train_accessions)
        # print('val_accessions', val_accessions)
        # print('test_accessions', test_accessions)

        # Convert accessions to indices
        train_indices = [np.where(self.all_accessions == x)[0][0] for x in train_accessions]
        val_indices = [np.where(self.all_accessions == x)[0][0] for x in val_accessions]
        test_indices = [np.where(self.all_accessions == x)[0][0] for x in test_accessions]
        
        self.clustered_accessions = (train_accessions, val_accessions, test_accessions)
        self.linkage = linkage

        # Calculate minimum train+val/test dendrogram distance
        train_val_accessions = np.concatenate((train_accessions, val_accessions))
        self.train_test_sim = distance_matrix.loc[train_val_accessions, test_accessions].mean().mean()

        if return_indices:
            return train_indices, val_indices, test_indices
        
        # Need to repeat train_indices num_turns times
        train_indices = np.repeat(train_indices, self.num_turns)
        val_indices = np.repeat(val_indices, self.num_turns)

        return (self.subset(train_indices), self.subset(val_indices), self.subset(test_indices)), \
                (train_indices, val_indices, test_indices)

    def _split_genera(self, return_indices):
        """ Create a train/val/test split based on genera. We'll take the largest genera first 
        60% of genera with > 5 accessions will be train, 20% val and test.

        The remaining will be added to train.

        Args:
            return_indices (bool): Whether to return the indices or the actual datasets.

        Returns:
            Tuple: The train, val, and test datasets or the train, val, and test indices.
        """

        # Get the genera with more than 5 accessions
        genera = self.metadata_table['genus'].value_counts()
        large_genera = genera[genera > 5].index
        small_genera = genera[genera <= 5].index

        # Get the accessions for each genus
        train_genera = large_genera[:int(0.4 * len(large_genera))].tolist()
        test_genera  = large_genera[int(0.4 * len(large_genera)):int(0.8 * len(large_genera))].tolist()
        val_genera   = large_genera[int(0.8 * len(large_genera)):].tolist()
        print("train_genera", train_genera)
        print("val_genera", val_genera)
        print("test_genera", test_genera)


        train_accessions = self.metadata_table[self.metadata_table['genus'].isin(train_genera)]['accession'].values
        val_accessions = self.metadata_table[self.metadata_table['genus'].isin(val_genera)]['accession'].values
        test_accessions = self.metadata_table[self.metadata_table['genus'].isin(test_genera)]['accession'].values

        # Assert no overlap
        assert len(set(train_accessions) & set(val_accessions)) == 0
        assert len(set(train_accessions) & set(test_accessions)) == 0
        assert len(set(val_accessions) & set(test_accessions)) == 0

        # Get the accessions for the small genera
        small_genera_accessions = self.metadata_table[self.metadata_table['genus'].isin(small_genera)]['accession'].values
        train_accessions = np.concatenate((train_accessions, small_genera_accessions[:len(small_genera_accessions) // 2]))

        # Assert no overlap
        assert len(set(train_accessions) & set(val_accessions)) == 0
        assert len(set(train_accessions) & set(test_accessions)) == 0
        assert len(set(val_accessions) & set(test_accessions)) == 0

        # print("train_accessions", train_accessions)
        # print("val_accessions", val_accessions)
        # print("test_accessions", test_accessions)

        # Convert accessions to indices
        train_indices = [np.where(self.all_accessions == x)[0][0] for x in train_accessions]        # WHY
        val_indices = [np.where(self.all_accessions == x)[0][0] for x in val_accessions]
        test_indices = [np.where(self.all_accessions == x)[0][0] for x in test_accessions]

        if return_indices:
            return train_indices, val_indices, test_indices

        # Need to repeat train_indices num_turns times
        train_indices = np.repeat(train_indices, self.num_turns)
        val_indices = np.repeat(val_indices, self.num_turns)

        return (self.subset(train_indices), self.subset(val_indices), self.subset(test_indices)), \
                (train_indices, val_indices, test_indices)

    def split_train_val_test(self, split_method:str='dendrogram',
                             return_indices: bool = False,
                             ) -> List:
        split_method = str(split_method).lower()

        if not split_method in ['dendrogram', 'genera']:
            raise ValueError(f"Invalid split method: {split_method}")
        
        if split_method == 'dendrogram':
            return self._split_dendrogam(return_indices=return_indices)
        
        if split_method == 'genera':
            return self._split_genera(return_indices=return_indices)

        
    def subset(self, indices:List):
        # Make a copy, in this way the sliced similarities can be used to identify the subset
        subset_dataset = copy.deepcopy(self)
        # Match indices to accessions
        mapped_indices = [idx % len(self.all_accessions) for idx in indices]    # TODO: This is untennable. Need to switch to unique IDS
        accessions = self.all_accessions[mapped_indices]

        # Remove any accessions whose spectra were removed
        subset_dataset.sliced_similarities = {k: v.loc[v['subject_genbank'].isin(accessions)] for k, v in self.sliced_similarities.items() if str(k) in accessions}       # OVERLAPS FOR EACH SUBSET

        # The same thing but long-winded and good for debugging:
        # updated_sliced_similarities = {}
        # for k, v in self.sliced_similarities.items():
        #     print(f"Checking key: {k}")
        #     if k in accessions:
        #         print(f"Key {k} is in accessions")
        #         filtered_values = v.loc[v['subject_genbank'].isin(accessions)]
        #         print(f"Filtered DataFrame for key {k} from {v.shape} to {filtered_values.shape}")
        #         updated_sliced_similarities[k] = filtered_values
        #     else:
        #         print(f"Key {k} is NOT in accessions and will be removed")

        # # Update self.sliced_similarities
        # self.sliced_similarities = updated_sliced_similarities

        return Subset(subset_dataset, indices)

    def plot_split(self,output_path:str=None):
        if self.linkage is None or self.clustered_accessions is None:
            raise ValueError('Split the dataset first')
        import matplotlib.pyplot as plt
        

        # Add train/test/val labels
        stock_labels = self.all_accessions
        new_labels = []
        for label in stock_labels:
            if label in self.clustered_accessions[0]:
                new_labels.append(f'(train) {label}')
            elif label in self.clustered_accessions[1]:
                new_labels.append(f'(val) {label}')
            elif label in self.clustered_accessions[2]:
                new_labels.append(f'(test) {label}')
            else:
                new_labels.append(f'(unused) {label}')

        Path(output_path).mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 0.09 * len(new_labels)))
        plt.title("Train/Val/Test Split")
        scipy.cluster.hierarchy.dendrogram(self.linkage, labels=new_labels, orientation='right', color_threshold=self.train_test_sim)
        
        # Dashed vertical line 
        plt.axvline(x=self.train_test_sim, color='black', linestyle='--')
        
        plt.show()
        plt.savefig(Path(output_path) / 'dendrogram.svg')
        plt.savefig(Path(output_path) / 'dendrogram.png', dpi=300)

class ExhaustiveMALDI_TOF_DS(IterableDataset):
    def __init__(self, ds, indices):
        self.sampler = ExhaustiveSampler(ds, indices)
        # self.len = len(self.sampler)
    
    def __iter__(self):
        return iter(self.sampler)

    # def __len__(self):
    #     return self.len

    def __getitem__(self, idx):
        spectrum_a, spectrum_b, similarity, metadata = next(self.sampler)
        assert 0.0 <= similarity <= 1.0
        return spectrum_a, spectrum_b, similarity, metadata

class ExhaustiveSampler():
    def __init__(self, data: Paired_MALDI_TOF_DS, indices: torch.Tensor):
        self.data = data

        worker_total_num = torch.utils.data.get_worker_info()
        if worker_total_num is not None:
            worker_total_num = worker_total_num.num_workers
            if worker_total_num > 1:
                raise ValueError("ExhaustiveSampler does not support multi-processing")
        
        # Get unique
        indices = np.unique(indices)

        self.accessions = self.data.all_accessions[indices]
        self.metadata = self.data.metadata_table.loc[self.data.metadata_table['accession'].isin(self.accessions)]
        print("Found a total of", len(self.metadata), "strains.")
        print("Found a total of", len(self.accessions), "accessions.")
        self.all_strains = self.metadata['Strain name'].values
        self.num_strains = len(self.all_strains)
        self._iterator = None

    def _iter_strains(self):
        for i in range(self.num_strains):
            for j in range(i+1, self.num_strains):
                try:
                    strain_a, meta_a = self.data.get_by_strain_name(self.all_strains[i])
                    strain_b, meta_b = self.data.get_by_strain_name(self.all_strains[j])
                except Exception as e:
                    continue
                
                try:
                    sim = self.data.similarities.loc[meta_a['accession'], meta_b['accession']]
                except Exception:
                    sim = np.nan
                # if np.isnan(sim):
                #     continue

                metadata = {
                    'accession_a': meta_a['accession'],
                    'accession_b': meta_b['accession'],
                    'strain_a': self.all_strains[i],
                    'strain_b': self.all_strains[j],
                    'spectrum_a': np.array(strain_a),
                    'spectrum_b': np.array(strain_b),
                    'num_peaks_in_a': meta_a['num_peaks'],
                    'num_peaks_in_b': meta_b['num_peaks'],
                }

                target = sim/100
                target = torch.tensor(target, dtype=torch.float32)

                yield strain_a, strain_b, target, metadata
                      
    def __iter__(self):
        """Iterates overall all unique combinations of spectra.
        
        Returns:
            Iterable: An iterator over all possible combinations of spectra.
        """
        self._iterator = self._iter_strains()
        return self

    def __next__(self):
        if self._iterator is None:
            self.__iter__()
        return next(self._iterator)
    
    # def __len__(self):
    #     x =  self.num_strains
    #     return x * (x + 1) // 2
        
    

@pytest.fixture
def ds():
    # Setup code: create the MALDI_TOF_DS instance
    dataset = Paired_MALDI_TOF_DS('../../data/idbac_db/preprocessing', 
                            '../../data/idbac_db/raw/db.csv', 
                            '../../data/idbac_db/preprocessed',
                            process=True)
    yield dataset  # This allows the test to use the dataset
    # Teardown code: delete the dataset instance
    del dataset

def test_dataset_initialization(ds):
    print(ds)

def test_dataset_find_match_in_range(ds):
    accession = pd.read_csv('../../data/idbac_db/raw/db.csv', nrows=5)['Genbank accession'].str.split('.').str[0].values[0]
    strain_name = ''
    print(ds.find_match_in_range(accession, strain_name, 2))

def test_getitem(ds):
    print(ds[0])

def test_train_test_split(ds):
    (train, val, test), (_, _, _) = ds.split_train_val_test()
    print(len(train), len(val), len(test))

def test_plot_split(ds):
    (train, val, test), (_, _, _) = ds.split_train_val_test()
    ds.plot_split('./')

    
@pytest.fixture
def single_ds():
    dataset = single_MALDI_TOF_DS('../../data/idbac_db/preprocessing',
                            '../../data/idbac_db/raw/db.csv',
                            '../../data/idbac_db/preprocessed',
                            process=True)
    yield dataset
    del dataset

def test_single_ds_initialization(single_ds):
    print(single_ds)

def test_single_ds_getitem(single_ds):
    print(single_ds[0])

def test_single_ds_sample_strain_from_accession(single_ds):
    accession = pd.read_csv('../../data/idbac_db/raw/db.csv', nrows=5)['Genbank accession'].str.split('.').str[0].values[0]
    print(single_ds.sample_strain_from_accession(accession))

def test_single_ds_preprocess(single_ds):
    single_ds.preprocess()

def test_single_ds_metadata(single_ds):
    metadata_df = pd.read_csv('../../data/idbac_db/raw/db.csv', nrows=5)
    accession = metadata_df['Genbank accession'].str.split('.').str[0].values[0]
    metadata = single_ds.sample_strain_from_accession(accession)
    assert 'accession' in metadata
    assert 'Strain name' in metadata
    assert 'genus' in metadata

    assert metadata['accession'] == accession
    assert metadata['Strain name'] == metadata_df['Strain name'].values[0]
    assert metadata['genus'] == metadata_df['genus'].values[0]



@pytest.fixture
def driams_ds():
    dataset = single_MALDI_TOF_DS('../../data/driams/preprocessing/',
                                    '../../data/driams/preprocessing/merged_metadata.csv',
                                    '../../data/driams/preprocessed',
                                    process=True)
    yield dataset
    del dataset

def test_driams_ds_initialization(driams_ds):
    print(driams_ds)