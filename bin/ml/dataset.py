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
                transform:callable=None):
        self.root_dir = root_dir
        self.preprocessing_dir = preprocessing_dir
        self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))
        all_spectra_names = [x.stem for x in self.all_spectra]

        metadata_table = pd.read_csv(metadata_table)
        if 'accession' not in metadata_table.columns:
            metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
        else:
            metadata_table['accession'] = metadata_table['accession'].str.strip()
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

        self.all_accessions = self.metadata_table['accession'].unique()

    def sample_strain_from_accession(self, accession):
        choices = self.metadata_table[self.metadata_table['accession'] == accession]
        if len(choices) == 0:
            raise ValueError(f"No strain found for accession '{accession}'")
        return choices['Strain name'].sample(1).values[0]
    
    def __len__(self):
        return len(self.all_accessions) * self.num_turns

    def __getitem__(self, idx):
        # For DRIAMS, use the species name as the accession
        # strain_name will be the hash
        accession = self.all_accessions[idx % len(self.all_accessions)]
        strain_name = self.sample_strain_from_accession(accession)
        
        spectrum = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name}.pt', weights_only=True).to(torch.float32)

        if self.transform:
            spectrum = self.transform(spectrum)

        return spectrum, strain_name
    
    def preprocess(self,):
        if not Path(self.root_dir).exists():
            Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        print("Preprocessing files...")
        for strain_name, spectrum_as_tensor in convert_spectra_to_tensor(Path(self.preprocessing_dir) / 'baseline_corrected.json'):
            torch.save(spectrum_as_tensor, Path(self.root_dir) / f'{strain_name}.pt')

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
                                                    # temp_similarities['subject_genbank'].isin(not_na_accessions)]
        print(f"Left with {len(temp_similarities)} pairs.")
        # Recalculate the square similarities
        # DEBUG
        # square_similarities = square_similarities.loc[not_na_accessions, not_na_accessions]

        self.all_accessions = np.sort(np.unique(np.concatenate((temp_similarities['query_genbank'].values, temp_similarities['subject_genbank'].values))))

        # assert 'strain_B016' in self.all_accessions

        self.metadata_table = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)]

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

        strain_name_b, accession_b, similarity = self.find_match_in_range(accession_a, strain_name_a, np.random.randint(0, self.sim_bins.shape[0] - 1))
        spectrum_a = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name_a}.pt', weights_only=True).to(torch.float32)
        spectrum_b = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name_b}.pt', weights_only=True).to(torch.float32)

        if self.transform:
            spectrum_a = self.transform(spectrum_a)
            spectrum_b = self.transform(spectrum_b)

        return spectrum_a, spectrum_b, torch.tensor(similarity/100, dtype=torch.float32)
    
    def get_by_strain_name(self, strain_name):
        accession = self.metadata_table[self.metadata_table['Strain name'] == strain_name]['accession'].values[0]
        spectrum = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name}.pt', weights_only=True).to(torch.float32)
        
        if self.transform:
            spectrum = self.transform(spectrum)

        return spectrum, accession

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

    def split_train_val_test(self,
                             return_indices: bool = False,
                             ) -> List:

        # Split based on the ground-truth dendrogram
        sims = self.similarities
        # Covnert to distance matrix
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

        train_accessions = clustered_accessions.loc[clustered_accessions.cluster == train_cluster_id, 'accession'].values
        val_accessions = clustered_accessions.loc[clustered_accessions.cluster == val_cluster_id, 'accession'].values
        test_accessions = clustered_accessions.loc[clustered_accessions.cluster == test_cluster_id, 'accession'].values

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

        return (Subset(self, train_indices), Subset(self, val_indices), Subset(self, test_indices)), \
                (train_indices, val_indices, test_indices)


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
        spectrum_a, spectrum_b, similarity = next(self.sampler)
        assert 0.0 <= similarity <= 1.0
        return spectrum_a, spectrum_b, similarity/100

class ExhaustiveSampler():
    def __init__(self, data: Paired_MALDI_TOF_DS, indices: torch.Tensor):
        self.data = data
        
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
                strain_i, accession_i = self.data.get_by_strain_name(self.all_strains[i])
                strain_j, accession_j = self.data.get_by_strain_name(self.all_strains[j])

                sim = self.data.similarities.loc[accession_i, accession_j]
                # if np.isnan(sim):
                #     continue

                yield strain_i, strain_j, sim/100
                      
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