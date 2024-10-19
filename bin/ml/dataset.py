from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from preprocess import postprocess_files
import os 
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pytest

class MALDI_TOF_DS(Dataset):
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
        metadata_table['accession'] = metadata_table['Genbank accession'].str.split('.').str[0]
        metadata_table = metadata_table[metadata_table['Strain name'].isin(all_spectra_names)]
        self.metadata_table = metadata_table
        self.transform = transform

        self.num_turns = 2
        
        if not process:
            preprocessing_dir_stat = Path(self.preprocessing_dir).stat()
            most_recent_m_time = self.get_most_recent_modified_time()

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
        temp_similarities = temp_similarities.loc[temp_similarities['query_genbank'].isin(post_filtration_accessions) & temp_similarities['subject_genbank'].isin(post_filtration_accessions)]

        self.all_accessions = np.unique(np.concatenate((temp_similarities['query_genbank'].values, temp_similarities['subject_genbank'].values)))

        self.sim_bins = np.linspace(temp_similarities['pident'].min(), temp_similarities['pident'].max(), 11)
        del temp_similarities   # TODO: pass to sliced_similarities for speed

        self.sliced_similarities = self._preslice_similarities(similarities)

    def __len__(self):
        return len(self.all_accessions) * self.num_turns
    
    def __getitem__(self, idx):
        accession_a = self.all_accessions[idx % len(self.all_accessions)]
        strain_name_a = self.sample_strain_from_accession(accession_a)

        strain_name_b, accession_b, similarity = self.find_match_in_range(accession_a, np.random.randint(0, self.sim_bins.shape[0] - 1))
        spectrum_a = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name_a}.pt', weights_only=True)
        spectrum_b = torch.load(Path(self.root_dir) / 'spectra' / f'{strain_name_b}.pt', weights_only=True)

        if self.transform:
            spectrum_a = self.transform(spectrum_a)
            spectrum_b = self.transform(spectrum_b)
            print("After padding", spectrum_a.shape, spectrum_b.shape)
            

        return spectrum_a, spectrum_b, similarity

    def strain_to_accession(self, strain_name):
        return self.metadata_table[self.metadata_table['Strain name'] == strain_name]['accession'].values[0]
    
    def sample_strain_from_accession(self, accession):
        try:
            return self.metadata_table[self.metadata_table['accession'] == accession]['Strain name'].sample(1).values[0]
        except ValueError as ve:
            raise ValueError(f'No strain found for accession {accession}') from ve

    def find_match_in_range(self, accession, bin_index):

        relevant_df = self.sliced_similarities[accession]

        lb = self.sim_bins[bin_index]
        ub = self.sim_bins[bin_index + 1]

        in_range = relevant_df.loc[lb:ub,]
        while len(in_range) == 0:
            lb -= 0.05
            ub += 0.05
            in_range = relevant_df.loc[lb:ub,]

        result_accession = in_range['subject_genbank'].values[0]
        result_similarity = in_range.index.values[0]

        result_strain_name = self.sample_strain_from_accession(result_accession)

        return result_strain_name, result_accession, result_similarity

    def _preslice_similarities(self, similarities:str):
        sims = pd.read_feather(similarities)

        out_dict = {}

        grouped = sims.groupby('query_genbank')
        for key, item in grouped:
            out_dict[key] = item.set_index('pident').sort_index()   # Now we can slice by pident (e.g., df.loc[78:98])

        return out_dict
    
    def preprocess(self,):
        if not Path(self.root_dir).exists():
            Path(self.root_dir).mkdir(parents=True, exist_ok=True)
        postprocess_files(Path(self.preprocessing_dir), Path(self.root_dir))

    
    @staticmethod
    def get_most_recent_modified_time():
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

@pytest.fixture
def ds():
    # Setup code: create the MALDI_TOF_DS instance
    dataset = MALDI_TOF_DS('../../data/idbac_db/preprocessing', 
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
    print(ds.find_match_in_range(accession, 2))

def test_getitem(ds):
    print(ds[0])