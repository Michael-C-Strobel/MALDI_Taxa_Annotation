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
from joblib import Parallel, delayed
from tqdm import tqdm

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
                sampling_mode:str=None,
                cast_to_classification:bool=False,
                num_turns:int=1,
                targets:str='genera',
                in_memory:bool=True,
                n_workers:int=-1,
                prefer_hard:bool=False,
                singletons_as_anchors:bool=True,):

        self.root_dir = root_dir
        self.preprocessing_dir = preprocessing_dir

        root_dir = Path(root_dir)
        if not root_dir.exists():
            # Process it
            print(f"Root dir {root_dir} does not exist, processing...")
            self.preprocess()

        self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))
        if len(self.all_spectra) == 0:
            print(f"No spectra found in {root_dir}/spectra, processing...")
            self.preprocess()
            self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))

        self.cast_to_classification = cast_to_classification
        self.prefer_hard = prefer_hard
        self.singletons_as_anchors = singletons_as_anchors
    

        balance = str(balance).lower()
        if balance not in ['accession', 'strain']:
            raise ValueError(f"Invalid balance method. Expected one of ['accession', 'class'], got '{balance}'")
        if balance == 'class':
            raise NotImplementedError("Class balancing not yet implemented")
        self.balance = balance

        if targets not in ['genera', 'species']:
            raise ValueError(f"Expected targets to be 'genera' or 'species', but got {targets}")
        
        if targets == 'species':
            self.target_col = 'species'
        else:
            self.target_col = 'genus'


        all_spectra_names = [x.stem for x in self.all_spectra]

        metadata_table = pd.read_csv(metadata_table)
        metadata_table.dropna(subset=[self.target_col], inplace=True)

        ### DEBUG REMOVE ANY SPECIES THAT OCCURS LESS THAN 5 TIMES
        # print("************************* DEBUG: Removing species that occur less than 5 times *************************")
        # metadata_table = metadata_table[metadata_table['species'].map(metadata_table['species'].value_counts()) >= 5]

        if self.cast_to_classification:
            # Sort unique values in 'genus' column, annotate with number
            self.class_to_int = {genus: i for i, genus in enumerate(sorted(metadata_table[self.target_col].unique()))}
            self.int_to_class = {i: genus for genus, i in self.class_to_int.items()}
            print(f"Got maximum class number {max(self.class_to_int.values())}")

        if 'accession' not in metadata_table.columns:
            print("Column 'accession' not found in metadata table, trying 'Genbank accession'")
            metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
        else:
            metadata_table['accession'] = metadata_table['accession'].astype(str).str.strip()
        metadata_table['accession'] = metadata_table['accession'].astype(str)
        initial_len = len(metadata_table)
        metadata_table = metadata_table.loc[metadata_table['Strain name'].isin(all_spectra_names)]
        filtered_len = len(metadata_table)
        print("Filtered metadata table from", initial_len, "to", filtered_len, "based on available spectra")
        metadata_table = metadata_table.drop_duplicates(subset='Strain name')   # Some strains occur twice due to multuple csv files

        self.metadata_table = metadata_table

        self.transform = transform

        self.num_turns = num_turns
        self.triplets = False
        self.nce = False
        if sampling_mode is not None:
            sampling_mode = str(sampling_mode).lower().strip()
            if sampling_mode == 'triplets':
                print("Using triplet sampling")
                self.triplets = True
            elif sampling_mode == 'nce':
                print("Using NCE sampling")
                self.nce = True
            else:
                raise ValueError(f"Invalid sampling mode. Expected one of ['triplets', 'nce'], got '{sampling_mode}'")

        if (cast_to_classification and self.triplets):
            raise ValueError("Cannot use triplet  sampling with cast_to_classification")

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
            # Drop duplicate rows/cols (keeping first)
            print("Removing duplicates from similarities")
            print("Original Shape:", self.similarities.shape)
            self.similarities = self.similarities.loc[~self.similarities.index.duplicated(keep='first')]
            self.similarities = self.similarities.loc[:, ~self.similarities.columns.duplicated(keep='first')]
            print("New Shape:", self.similarities.shape)
            # Check for dupes one last time
            assert not self.similarities.index.duplicated().any()
            assert not self.similarities.columns.duplicated().any()

        if not process:
            pass
            # preprocessing_dir_stat = Path(self.preprocessing_dir).stat()
            # most_recent_m_time = get_most_recent_modified_time()

            # if Path(self.root_dir).exists():
            #     root_dir_stat = Path(self.root_dir).stat()
            # else:
            #     root_dir_stat = None

            # if not Path(self.root_dir).exists() or \
            #     preprocessing_dir_stat.st_mtime > root_dir_stat.st_mtime or \
            #     most_recent_m_time > root_dir_stat.st_mtime:
            #         self.preprocess()
        else:
            self.preprocess()

        self.all_accessions = self.metadata_table.loc[self.metadata_table['accession'].str.lower() != 'nan', 'accession'].unique().astype(str)
        self.all_strain_names = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions), 'Strain name'].unique().astype(str)
        print(f"Found a total of {len(self.all_accessions)} accessions in the metadata table")
        if (not self.singletons_as_anchors) and self.nce:
            # Remove accessions with only one strain
            self.all_accessions = [x for x in self.all_accessions if self.metadata_table[self.metadata_table['accession'] == x]['Strain name'].nunique() > 1]
            self.all_strain_names = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions), 'Strain name'].unique().astype(str)
            print("Removing singletons from accessions")
            print("New number of accessions", len(self.all_accessions))
        
        if self.similarities is not None:
            print("Filtering accessions based on similarities")
            # self.all_accessions = np.intersect1d(self.all_accessions, self.similarities.index.values.astype(str))
            # self.metadata_table = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)]

        # DEBUG, TEMPORARY
        # Remove any accessions whose spectra were removed
        # removed = ['strain_B032', 'nan']
        # self.all_accessions = np.array([x for x in self.all_accessions if not (str(x) in removed)])

        self.n_classes = len(self.metadata_table[self.target_col].unique())

        print("Prefetching accession to genus")
        self.accession_to_genus = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)].set_index('accession')['genus'].to_dict()

        print("Prefetching accession to species")
        self.accession_to_species = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)].set_index('accession')['species'].to_dict()

        print("Prefetching accession to class")
        if self.target_col == 'genus':
            self.accession_to_class = self.accession_to_genus
        elif self.target_col == 'species':
            self.accession_to_class = self.accession_to_species
        else:
            raise ValueError(f"Invalid target column: {self.target_col}")

        print("Prefetching accession to strain")
        # self.accession_to_strain_names = {}
        # for accession in tqdm(self.metadata_table['accession'].unique()):
        #     # Get the class for this accession
        #     rows = self.metadata_table[self.metadata_table['accession'] == accession]['Strain name'].values
        #     self.accession_to_strain_names[accession] = rows
        self.accession_to_strain_names = (
            self.metadata_table
            .groupby('accession')['Strain name']
            .apply(lambda x: np.array(x.values))
            .to_dict()
        )

        print("Prefetching strain name to metadata")
        assert self.metadata_table['Strain name'].is_unique, "Strain names are not unique"
        _metadata_table = self.metadata_table.copy()
        _metadata_table['_strain_name'] = _metadata_table['Strain name'].astype(str)
        self.strain_name_to_metadata = (
            _metadata_table
            .groupby('_strain_name', sort=False)
            .first()
            .to_dict(orient='index')
        )
        print("Done")

        # This happens before the subset, so it will iterate over the entire dataset
        self.in_memory = in_memory
        self.n_workers = n_workers
        if in_memory:
            self.spectra = self.preload()
            
    def preload(self):
        """Joblib-parallelized loading of spectra into memory"""
        def _fetch(strain_names_chunk):
            chunk_results = []
            for strain_name in strain_names_chunk:
                spectrum = torch.load(Path(self.root_dir) / 'spectra' / f"{strain_name}.pt", weights_only=True).to(torch.float32)
                if self.transform:
                    spectrum = self.transform(spectrum)
                chunk_results.append((strain_name, spectrum))
            return chunk_results

        strain_names = self.metadata_table['Strain name'].values
        n_workers = self.n_workers
        if n_workers == -1:
            n_workers = os.cpu_count()
        if n_workers is None or n_workers < 1:
            _n_workers = 1
        else:
            _n_workers = n_workers
        chunk_size = max(1, len(strain_names) // (_n_workers * 2))
        strain_name_chunks = [strain_names[i:i + chunk_size] for i in range(0, len(strain_names), chunk_size)]

        results = Parallel(n_jobs=_n_workers)(
            delayed(_fetch)(chunk) for chunk in tqdm(strain_name_chunks, desc=f"Loading Spectra into Memory with {_n_workers} cpus")
        )
        results = [item for sublist in results for item in sublist]

        return {strain_name: spectrum for strain_name, spectrum in results}
        

    def sample_strain_from_accession(self, accession):
        # choices = self.metadata_table[self.metadata_table['accession'] == accession]
        # choices = self.accession_to_strains[accession]
        # if len(choices) == 0:
        #     raise ValueError(f"No strain found for accession '{accession}'")
        # return choices.sample(1).to_dict(orient='records')[0]
    
        choices = self.accession_to_strain_names[accession]
        if len(choices) == 0:
            raise ValueError(f"No strain found for accession '{accession}'")
        # Randomly sample 1
        sampled_row = np.random.choice(choices)
        # Get the metadata for this strain 
        return self.strain_name_to_metadata[sampled_row]
    
    def __len__(self):
        if self.balance == 'accession':
            return len(self.all_accessions) * self.num_turns
        elif self.balance == 'strain':
            # Count unique strains
            return len(self.all_strain_names) * self.num_turns
        else:
            raise ValueError(f"Invalid balance method: {self.balance}. Expected one of ['accession', 'strain']")

    def __getitem__(self, idx):
        # For DRIAMS, use the species name as the accession
        # strain_name will be the hash
        if self.balance == 'accession':
            accession = self.all_accessions[idx % len(self.all_accessions)]
            sampled_row = self.sample_strain_from_accession(accession)
            strain_name = sampled_row['Strain name']
            database_id = sampled_row['database_id']
        elif self.balance == 'strain':
            strain_name = self.all_strain_names[idx % len(self.all_strain_names)]
            mdata = self.strain_name_to_metadata[strain_name]
            accession = mdata['accession']
            database_id = mdata['database_id']
            
        if self.in_memory:
            # They're already transformed
            spectrum = self.spectra[strain_name]
        else:
            spectrum = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name}.pt', weights_only=True).to(torch.float32)
            
            if self.transform:
                try:
                    spectrum = self.transform(spectrum)
                except Exception as e:
                    raise RuntimeError(f"Error transforming spectrum for strain {strain_name} with accession {accession}") from e

        metadata = {
            'accession': accession,
            'Strain name': strain_name,
            'class': str(self.accession_to_class[accession]),
            'database_id': database_id,
        }

        if self.triplets:
            triplets = self.generate_triplets(spectrum, metadata)
            return triplets 
        elif self.nce:
            pair = self.generate_pair(spectrum, metadata)
            return pair
        elif self.cast_to_classification:
            return spectrum, self.class_to_int[metadata['class']]
        else:
            return spectrum, metadata
            
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

    def get_one_hot_encoded_classes(self):
        """
        
        """
        raise NotImplementedError("One hot encoding has been deprecated at the dataset level.")
        def one_hot_encode(classes:List):
            return torch.Tensor([self.one_hot_encoder[str(x)] for x in classes]).to(torch.long)  # str() important to cover nan
        return one_hot_encode
    
    def _genus_pair(self, metadata:dict):
        """ Generates a positive pair based on genus.
        
        Args:
            metadata (dict): The metadata dictionary.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The positive pair.
            Tuple[dict, dict]: The positive metadata.
        """

        anchor_strain_name = metadata['Strain name']
        anchor_accession = metadata['accession']
        anchor_class = metadata['class']

        # Positive pair
        positive_mask = (self.metadata_table[self.target_col] == anchor_class)
        
        # Try to get a non-identity pair
        strain_mask = (self.metadata_table['Strain name'] != anchor_strain_name)
        if sum(positive_mask & strain_mask) > 0:
            if self.prefer_hard and self.target_col == 'genus':
                # Prefer hard positives (if target == genus: different species, if target == species: N/A)
                anchor_species = self.accession_to_species[anchor_accession]
                different_species_mask = (self.metadata_table['species'] != anchor_species)
                if sum(positive_mask & strain_mask & different_species_mask) > 0:
                    positive_mask = positive_mask & strain_mask & different_species_mask
                else:
                    positive_mask = positive_mask & strain_mask

        # TODO: filter for distances, if we ever want to require them

        if len(self.metadata_table[positive_mask]) < 1:
            raise ValueError(f"Could not find a match for {anchor_accession}, this shuould never happen")

        positive_row = self.metadata_table[positive_mask].sample(1).to_dict(orient='records')[0]
        positive_metadata = {
            'accession': positive_row['accession'],
            'Strain name': positive_row['Strain name'],
            'class': positive_row[self.target_col],
            'database_id': positive_row['database_id'],
        }
        try:
            if self.similarities is not None: 
                pos_sim = self.similarities.loc[anchor_accession, positive_metadata['accession']]
            else:
                pos_sim = np.nan
        except KeyError as ke:
            pos_sim = np.nan
        positive_spectrum = torch.load(Path(self.root_dir) / 'spectra' / f"{positive_metadata['Strain name']}.pt", weights_only=True).to(torch.float32)

        if self.transform:
            try:
                positive_spectrum = self.transform(positive_spectrum)
            except Exception as e:
                raise RuntimeError(f"Error transforming positive spectrum for strain {positive_metadata['Strain name']} with accession {positive_metadata['accession']}") from e
            
        return positive_spectrum, positive_metadata, pos_sim

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
        try:
            neg_sim = self.similarities.loc[metadata['accession'], negative_metadata['accession']]
        except:
            neg_sim = np.nan
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
    
    def generate_pair(self, spectrum:torch.Tensor, metadata:dict):
        """ Generates a positive pair for a CLIP-like batch for a given metadata input.

        Args:
            spectrum (torch.Tensor): The spectrum tensor.
            metadata (dict): The metadata dictionary.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The positive pair.
            Tuple[dict, dict]: The positive metadata.
            None: Similarity TODO
        """
        positive_spectrum, positive_metadata, similarity = self._genus_pair(metadata)

        # Cast to classification if needed
        if self.cast_to_classification:
            # Cast main spectrum to classification
            metadata['class_as_int'] = self.class_to_int[metadata['class']]
        
        return (spectrum, positive_spectrum), (metadata, positive_metadata), (similarity,)
    
    def taxa_triplets(self, metadata:dict):
        """ Generates a positive and negative triplet for a given metadata input.

        Args:
            metadata (dict): The metadata dictionary.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The positive and negative triplet.
            Tuple[dict, dict]: The positive and negative metadata.
        """
        # Unpack metadata
        anchor_strain_name = metadata['Strain name']
        anchor_accession = metadata['accession']
        anchor_class = metadata['class']

        # Positive triplet
        positive_options = self.metadata_table[self.metadata_table[self.target_col] == anchor_class]['Strain name'].values
        if len(positive_options) > 1:
            # Try to pick a different one
            positive_options = positive_options[positive_options != anchor_strain_name]

        random_positive_strain_name = np.random.choice(list(positive_options))

        # Negative triplet
        neg_classes = self.metadata_table[self.metadata_table[self.target_col] != anchor_class][self.target_col].unique()

        random_class_name = np.random.choice(list(neg_classes))
        negative_options = self.metadata_table[self.metadata_table[self.target_col] == random_class_name]['Strain name'].values
        random_negative_strain_name = np.random.choice(list(negative_options))

        # Get all data 
        positive_metadata = self.metadata_table[self.metadata_table['Strain name'] == random_positive_strain_name].to_dict(orient='records')[0]
        negative_metadata = self.metadata_table[self.metadata_table['Strain name'] == random_negative_strain_name].to_dict(orient='records')[0]
        positive_metadata = {
            'accession': positive_metadata['accession'],
            'Strain name': positive_metadata['Strain name'],
            'class': positive_metadata[self.target_col],
            'database_id': positive_metadata['database_id'],
        }
        negative_metadata = {
            'accession': negative_metadata['accession'],
            'Strain name': negative_metadata['Strain name'],
            'class': negative_metadata[self.target_col],
            'database_id': negative_metadata['database_id'],
        }
        # Get the spectra
        if self.in_memory:
            positive_spectrum = self.spectra[random_positive_strain_name]
            negative_spectrum = self.spectra[random_negative_strain_name]
        else:
            positive_spectrum = torch.load(Path(self.root_dir) / 'spectra' / f"{random_positive_strain_name}.pt", weights_only=True).to(torch.float32)
            negative_spectrum = torch.load(Path(self.root_dir) / 'spectra' / f"{random_negative_strain_name}.pt", weights_only=True).to(torch.float32)
            if self.transform:
                try:
                    positive_spectrum = self.transform(positive_spectrum)
                    negative_spectrum = self.transform(negative_spectrum)
                except Exception as e:
                    raise RuntimeError(f"Error transforming spectrum for strain {random_positive_strain_name} with accession {random_negative_strain_name}") from e
                
        return (positive_spectrum, negative_spectrum), (positive_metadata, negative_metadata), (np.nan, np.nan)




    def generate_triplets(self, spectrum:torch.Tensor, metadata:dict):
        """ Generates the positive and negative triplets for a given metadata input and strategy.

        Args:
            metadata (dict): The metadata dictionary.
            strategy (str, optional): The strategy to use for generating triplets. Defaults to 'genus'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The positive and negative triplets.
            Tuple[dict, dict]: The positive and negative metadata.
        """

        if self.target_col == 'genus':
            # (positive_spectrum, negative_spectrum), (positive_metadata, negative_metadata), (pos_sim, neg_sim) = self._genus_triplets(metadata)
            (positive_spectrum, negative_spectrum), (positive_metadata, negative_metadata), (pos_sim, neg_sim) = self.taxa_triplets(metadata)
        else:
            raise NotImplementedError(f"Triplet generation has not been implemented for {self.target_col} yet")

        if False:
            # Print everything
            print("Anchor metadata", metadata)
            print("Positive metadata", positive_metadata)
            print("Negative metadata", negative_metadata)

            print("Anchor spectrum", spectrum)
            print("Positive spectrum", positive_spectrum)
            print("Negative spectrum", negative_spectrum)

            

        return (spectrum, positive_spectrum, negative_spectrum), (metadata, positive_metadata, negative_metadata), (pos_sim, neg_sim)
    
    def subset(self, accessions:List):
        #  Make a copy, in this way the sliced similarities can be used to identify the subset
        if self.balance == 'accession':
            indices = [i for i, x in enumerate(self.all_accessions) if x in accessions]
        else:
            strain_name_to_accession = self.metadata_table.set_index('Strain name')['accession'].to_dict()
            indices = [i for i, x in enumerate(self.all_strain_names) if strain_name_to_accession.get(x) in accessions]
        subset_dataset = copy.deepcopy(self)

        relevant_accessions = [x for x in self.all_accessions if x in accessions]

        print("Debug: only incluiding accessions in subset that actually exist")
        print("Original accessions", len(np.unique(accessions)))
        print("Accessions in splits but not found in training data", len(set(accessions) - set(self.all_accessions)))
        accessions = np.intersect1d(accessions, self.all_accessions)
        print("New accessions", len(np.unique(accessions)))

        # Remove any accessions whose spectra were removed
        # subset_dataset.similarities = subset_dataset.similarities.loc[accessions, accessions]
        # assert subset_dataset.similarities.shape[0] < self.similarities.shape[0], "Expected subset similarities to be smaller"
        # assert subset_dataset.similarities.shape[1] < self.similarities.shape[1], "Expected subset similarities to be smaller"
        subset_dataset.metadata_table = subset_dataset.metadata_table.loc[subset_dataset.metadata_table['accession'].isin(accessions)]
        subset_dataset.relevant_accessions = relevant_accessions
        # assert subset_dataset.metadata_table.shape[0] < self.metadata_table.shape[0], "Expected subset metadata to be smaller"

        indices = np.repeat(indices, self.num_turns)

        return Subset(subset_dataset, indices)
        

    def preprocess(self,):
        # return
        if not (Path(self.root_dir) / 'spectra/').exists():
            (Path(self.root_dir) / 'spectra/').mkdir(parents=True, exist_ok=True)
        print("Preprocessing files...")
        for strain_name, spectrum_as_tensor in tqdm(convert_spectra_to_tensor(Path(self.preprocessing_dir) / 'baseline_corrected.json')):
            torch.save(spectrum_as_tensor, Path(self.root_dir) / f'spectra/{strain_name}.pt')

    def calculate_transformed_stats(self, train_accessions)->Dict[str, float]:
        """ Use Welford's online algorithm to compute the mean and variance of the transformed spectra,
        in a memory efficient way.

        Returns:
            Dict[str, float]: The mean and variance of the transformed spectra.
        """

        # Collect all strain_names from the accessions
        strain_names = self.metadata_table.loc[self.metadata_table['accession'].isin(train_accessions), 'Strain name'].values

        mean = None
        M2 = None
        count = 0
        length = None

        relevant_paths = [Path(self.root_dir) / 'spectra' / f"{strain_name}.pt" for strain_name in strain_names]

        for f in relevant_paths:
            spectrum = torch.load(f)
            transformed_spectrum = torch.tensor(self.transform(spectrum))

            if length is not None:
                assert length == transformed_spectrum.shape[0], f"Length mismatch: {length} vs {transformed_spectrum.shape[0]}"
            else:
                length = transformed_spectrum.shape[0]

            # Flatten to [num_samples, feature_dim]
            batch = transformed_spectrum.view(-1, transformed_spectrum.shape[-1])
            batch_n = batch.shape[0]
            batch_mean = batch.mean(dim=0)
            batch_var = batch.var(dim=0, unbiased=False)

            if mean is None:
                mean = batch_mean
                M2 = batch_var * batch_n
            else:
                delta = batch_mean - mean
                total = count + batch_n
                mean += delta * batch_n / total
                M2 += batch_var * batch_n + delta**2 * count * batch_n / total

            count += batch_n

        variance = M2 / count
        std = torch.sqrt(variance)
        return {'mean': mean, 'std': std,}

class Paired_MALDI_TOF_DS(Dataset):
    def __init__(self, preprocessing_dir:str,
                 metadata_table:str,
                 root_dir: str,
                 process:bool=True,
                 transform:callable=None,
                 targets:str='genera',):
        self.root_dir = root_dir
        self.preprocessing_dir = preprocessing_dir
        self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))
        if len(self.all_spectra) == 0:
            print("No spectra found, running initial setup.")
            self.preprocess()
            self.all_spectra = list(Path(self.root_dir).glob('spectra/*.pt'))

        if targets not in ['genera', 'species']:
            raise ValueError(f"Expected targets to be 'genera' or 'species', but got {targets}")
        
        if targets == 'species':
            self.target_col = 'species'
        else:
            self.target_col = 'genus'

        all_spectra_names = [x.stem for x in self.all_spectra]

        metadata_table = pd.read_csv(metadata_table)
        print(f"DEBUG 767: metadata table contains {len(metadata_table)} rows")

        if 'Genbank accession' in metadata_table.columns and \
            'accession' not in metadata_table.columns:
            metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0].str.strip()
        metadata_table = metadata_table.loc[metadata_table['Strain name'].isin(all_spectra_names)]

        print(f"DEBUG 773: metadata table contains {len(metadata_table)} rows")

        self.metadata_table = metadata_table
        self.transform = transform

        self.num_turns = 1 # Not implemented in subset so locked at 1

        if not process:
            pass
            # preprocessing_dir_stat = Path(self.preprocessing_dir).stat()
            # most_recent_m_time = get_most_recent_modified_time()

            # if Path(self.root_dir).exists():
            #     root_dir_stat = Path(self.root_dir).stat()
            # else:
            #     root_dir_stat = None

            # if preprocessing_dir_stat.st_mtime > root_dir_stat.st_mtime or \
            #     most_recent_m_time > root_dir_stat.st_mtime or \
            #     not Path(self.root_dir).exists():
            #         self.preprocess()
        else:
            self.preprocess()

        similarities = Path(self.root_dir) / 'similarities.feather'
        if not similarities.exists():
            temp_similarities = pd.DataFrame()
            self.similarities = None
            self.sim_bins = None
            self.sliced_similarities = None
        else:
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
            self.similarities = square_similarities
            self.sim_bins = np.linspace(temp_similarities['pident'].min(), temp_similarities['pident'].max(), 21)   # Data leakage in the _absolute_ strictest sense
            self.sliced_similarities = self._preslice_similarities(temp_similarities)

        print(f"DEBUG 5: metadata table contains {len(self.metadata_table)} rows")
        self.all_accessions = self.metadata_table.loc[self.metadata_table['accession'].notna(), 'accession'].unique().astype(str)
        print(f"Found {len(self.all_accessions)} accessions in the metadata table.")

        if self.similarities is not None:
            print("Filtering accessions based on similarities")
            # self.all_accessions = np.intersect1d(self.all_accessions, self.similarities.index.values.astype(str))
            # self.metadata_table = self.metadata_table.loc[self.metadata_table['accession'].isin(self.all_accessions)]

        # Initialize the linkage and clustered accessions
        self.linkage = None
        self.train_test_sim = None
        self.clustered_accessions = None

    def __len__(self):
        # Will always be the same, independent of whether we've subset
        return len(self.all_accessions) * self.num_turns
    
    def __getitem__(self, idx):
        accession_a = self.all_accessions[idx % len(self.all_accessions)]
        strain_name_a = self.sample_strain_from_accession(accession_a)

        # DEBUG TO DELETE STUFF
        rand_int = np.random.randint(0, self.sim_bins.shape[0] - 1)
        accession_b = None
        # while accession_b is None or accession_b in ['EF178692', 'AB184357', 'AB122711', 'AB184476', 'strain_B017', 'AB122711']:  # SS preprocessing
        while accession_b is None: #or accession_b in ['strain_B032',]:            
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

        if hasattr(self, 'relevant_accessions'):
            if accession_a not in self.relevant_accessions or accession_b not in self.relevant_accessions:
                raise ValueError(f"accession_a {accession_a} or accession_b {accession_b} not in relevant accessions")

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
        print('find_match_in_range')
        print('accession', accession)
        relevant_df = self.sliced_similarities[str(accession)]

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
            out_dict[str(key)] = item.set_index('pident').sort_index()   # Now we can slice by pident (e.g., df.loc[78:98])

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
        # distance_matrix.fillna(100.0, inplace=True)

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

        train_accessions = clustered_accessions.loc[clustered_accessions.cluster == train_cluster_id, 'accession'].astype(str).values
        val_accessions = clustered_accessions.loc[clustered_accessions.cluster == val_cluster_id, 'accession'].astype(str).values
        test_accessions = clustered_accessions.loc[clustered_accessions.cluster == test_cluster_id, 'accession'].astype(str).values

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
        # train_indices = np.repeat(train_indices, self.num_turns)
        # val_indices = np.repeat(val_indices, self.num_turns)
        train_accessions = np.repeat(train_accessions, self.num_turns)
        val_accessions = np.repeat(val_accessions, self.num_turns)

        return (self.subset(train_accessions), self.subset(val_accessions), self.subset(test_accessions)), \
                (train_accessions, val_accessions, test_accessions)

    def _split_taxa(self, return_indices, level='genus'):
        """ Create a train/val/test split based on genera. We'll take the largest genera first 
        60% of genera with > 5 accessions will be train, 20% val and test.

        The remaining will be added to train.

        Args:
            return_indices (bool): Whether to return the indices or the actual datasets.

        Returns:
            Tuple: The train, val, and test datasets or the train, val, and test indices.
        """
        assert level in ['genus', 'species'], f"Invalid level: {level}"

        _metadata_table = self.metadata_table.copy()
        # Remove nan accessions
        _metadata_table = _metadata_table.loc[_metadata_table['accession'].notna()]

        # Get the genera with more than 5 accessions
        genera = _metadata_table[level].value_counts()
        large_genera = genera[genera > 5].index
        small_genera = genera[genera <= 5].index

        # Get the accessions for each taxa
        train_genera = large_genera[:int(0.4 * len(large_genera))].tolist()
        test_genera  = large_genera[int(0.4 * len(large_genera)):int(0.8 * len(large_genera))].tolist()
        val_genera   = large_genera[int(0.8 * len(large_genera)):].tolist()
        print("train_genera", train_genera)
        print("val_genera", val_genera)
        print("test_genera", test_genera)


        train_accessions = _metadata_table[_metadata_table[level].isin(train_genera)]['accession'].astype(str).values
        val_accessions = _metadata_table[_metadata_table[level].isin(val_genera)]['accession'].astype(str).values
        test_accessions = _metadata_table[_metadata_table[level].isin(test_genera)]['accession'].astype(str).values

        # Assert no overlap
        assert len(set(train_accessions) & set(val_accessions)) == 0, f"Expected no overlap but found {set(train_accessions) & set(val_accessions)} in common between train and validation"
        assert len(set(train_accessions) & set(test_accessions)) == 0, f"Expected no overlap but found {set(train_accessions) & set(test_accessions)} in common between train and test"
        assert len(set(val_accessions) & set(test_accessions)) == 0, f"Expected no overlap but found {set(val_accessions) & set(test_accessions)} in common between validation and test"

        # Get the accessions for the small genera
        small_genera_accessions = _metadata_table[_metadata_table[level].isin(small_genera)]['accession'].values
        train_accessions = np.concatenate((train_accessions, small_genera_accessions[:len(small_genera_accessions) // 2]))  # Wot

        # Assert no overlap
        assert len(set(train_accessions) & set(val_accessions)) == 0
        assert len(set(train_accessions) & set(test_accessions)) == 0
        assert len(set(val_accessions) & set(test_accessions)) == 0

        if return_indices:
            raise NotImplementedError("Returning indices is not implemented for genera splits")

        # Need to repeat train_indices num_turns times
        train_accessions = np.repeat(train_accessions, self.num_turns).tolist()
        val_accessions = np.repeat(val_accessions, self.num_turns).tolist()

        print("train_accessions", train_accessions)

        return (self.subset(train_accessions), self.subset(val_accessions), self.subset(test_accessions)), \
                (train_accessions, val_accessions, test_accessions)
    
    def _even_taxa_split(self, return_indices, level='genus'):
        """For each species, add 80% to the training set, 10% to the validation set, and 10% to the test set.
        Singular species will be added to the training set.
        """
        if return_indices:
            raise NotImplementedError("Returning indices is not implemented for genera splits")
        
        train_accessions = list()
        val_accessions = list()
        test_accessions = list()

        for taxa in self.metadata_table[level].unique():
            species_accessions = self.metadata_table[self.metadata_table[level] == taxa]['accession'].unique().tolist()
            if len(species_accessions) <= 3:
                train_accessions += species_accessions  # Add all to train
            else:
                train_end = int(np.floor(0.7 * len(species_accessions)))
                val_end = int(np.floor(0.85 * len(species_accessions)))
                print(train_end, val_end)
                if len(set(species_accessions[:train_end]) & set(species_accessions[train_end:val_end])) > 0:
                    print(set(species_accessions[:train_end]) & set(species_accessions[train_end:val_end]))
                    raise ValueError("Overlap between train and val")
                train_accessions += species_accessions[:train_end]
                val_accessions += species_accessions[train_end:val_end]
                test_accessions += species_accessions[val_end:]

        # Assert no overlap
        assert len(set(train_accessions) & set(val_accessions)) == 0, f"Expected no overlap but got {set(train_accessions) & set(val_accessions)}"
        assert len(set(train_accessions) & set(test_accessions)) == 0, f"Expected no overlap but got {set(train_accessions) & set(test_accessions)}"
        assert len(set(val_accessions) & set(test_accessions)) == 0, f"Expected no overlap but got {set(val_accessions) & set(test_accessions)}"


        return (self.subset(train_accessions), self.subset(val_accessions), self.subset(test_accessions)), \
                (train_accessions, val_accessions, test_accessions)


    def split_train_val_test(self, split_method:str='dendrogram',
                             return_indices: bool = False,
                             ) -> List:
        split_method = str(split_method).lower()

        if not split_method in ['dendrogram', 'genera', 'species', 'species_even']:
            raise ValueError(f"Invalid split method: {split_method}")
        
        if split_method == 'dendrogram':
            return self._split_dendrogam(return_indices=return_indices)
        
        if split_method == 'genera':
            return self._split_taxa(return_indices=return_indices, level='genus')
        
        if split_method == 'species':
            return self._split_taxa(return_indices=return_indices, level='species')
        
        if split_method == 'species_even' or 'strain':
            return self._even_taxa_split(return_indices=return_indices, level='species')
        

        
    def subset(self, accessions:List):
        # Make a copy, in this way the sliced similarities can be used to identify the subset
        subset_dataset = copy.deepcopy(self)

        indices = [i for i, x in enumerate(self.all_accessions) if x in accessions]
        # Match indices to accessions
        # mapped_indices = [idx % len(self.all_accessions) for idx in indices]    # TODO: This is untennable. Need to switch to unique IDS
        # accessions = self.all_accessions[mapped_indices]

        # Remove any accessions whose spectra were removed
        if self.sliced_similarities is not None:
            subset_dataset.sliced_similarities = {k: v.loc[v['subject_genbank'].isin(accessions)] for k, v in self.sliced_similarities.items() if str(k) in accessions}       # OVERLAPS FOR EACH SUBSET

        print(f"Subset dataset has {len(accessions)} accessions")
        print(f"Subset dataset has {len(indices)} indices")

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
    def __init__(self, ds, indices, paired=True):
        if paired:
            self.sampler = ExhaustiveSampler(ds, indices)
        else:
            self.sampler = ExhaustiveSingleSampler(ds, indices)
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
    def __init__(self, data: Paired_MALDI_TOF_DS, accessions: torch.Tensor):
        self.data = data

        worker_total_num = torch.utils.data.get_worker_info()
        if worker_total_num is not None:
            worker_total_num = worker_total_num.num_workers
            if worker_total_num > 1:
                raise ValueError("ExhaustiveSampler does not support multi-processing")
        
        # Get unique
        accessions = np.unique(accessions)

        self.accessions = accessions
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
        
    
class ExhaustiveSingleSampler():
    def __init__(self, data: single_MALDI_TOF_DS, accessions: torch.Tensor):
        self.data = data

        worker_total_num = torch.utils.data.get_worker_info()
        if worker_total_num is not None:
            worker_total_num = worker_total_num.num_workers
            if worker_total_num > 1:
                raise ValueError("ExhaustiveSampler does not support multi-processing")
            
        # Get unique
        accessions = np.unique(accessions)

        self.accessions = accessions
        self.metadata = self.data.metadata_table.loc[self.data.metadata_table['accession'].isin(self.accessions)]
        print("Found a total of", len(self.metadata), "strains.")
        print("Found a total of", len(self.accessions), "accessions.")
        self.all_strains = self.metadata['Strain name'].values
        self.num_strains = len(self.all_strains)
        self._iterator = None

    def _iter_strains(self):
        for i in range(self.num_strains):
            try:
                strain, metadata = self.data.get_by_strain_name(self.all_strains[i])
            except Exception as e:
                continue
            
            metadata = {
                'accession': metadata['accession'],
                'strain_name': self.all_strains[i],
                'spectrum': np.array(strain),
                'num_peaks': metadata['num_peaks'],
            }

            yield strain, metadata

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