from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import utils
import re
from glob import glob
import sys
import pandas as pd

def map_fn(example, tokenizer, dic=None, top=100, max_seq=50):
    ab = th.tensor(example['intensity_array'])
    ab_sort = (-ab).argsort()[:top]
    ab = ab[ab_sort]
    ab /= ab.max()
    spectrum_length = len(ab)
    mz = th.tensor(example['mz_array'])[ab_sort]
    mz_sort = mz.argsort()
    length = len(mz)
    mz_ = th.zeros(top)
    mz_[:len(mz_sort)] = mz[mz_sort]
    ab_ = th.zeros(top)
    ab_[:len(ab_sort)] = ab[mz_sort]
    example['mz_array'] = mz_
    example['intensity_array'] = ab_
    example['precursor_charge'] = th.tensor(example['precursor_charge'], dtype=th.int32)
    example['precursor_mass'] = th.tensor(example['precursor_mass'], dtype=th.float32)
    example['spectrum_length'] = th.tensor(len(example['mz_array']), dtype=th.int32)
    tokenized_sequence = tokenizer(example['modified_sequence'])
    peptide_length = len(tokenized_sequence)
    example['tokenized_sequence'] = th.tensor([dic[m] for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=th.int32)
    example['peptide_length'] = th.tensor(peptide_length, dtype=th.int32)
    example['spectrum_length'] = th.tensor(spectrum_length, dtype=th.int32)

    return example

def collate_fn(batch_list):
    species = [m['experiment_name'] for m in batch_list]
    speclen = th.stack([m['spectrum_length'] for m in batch_list])
    mz = th.stack([m['mz_array'][:speclen.max()] for m in batch_list])
    ab = th.stack([m['intensity_array'][:speclen.max()] for m in batch_list])
    charge = th.stack([m['precursor_charge'] for m in batch_list])
    mass = th.stack([m['precursor_mass'] for m in batch_list])
    peplen = th.stack([m['peptide_length'] for m in batch_list])
    intseq = th.stack([m['tokenized_sequence'][:peplen.max()] for m in batch_list])

    out = {
        'experiment_name': species,
        'mz': mz,
        'ab': ab,
        'charge': charge,
        'mass': mass,
        'length': speclen,
        'intseq': intseq,
        'peplen': peplen,
        #'spectrum_lengths': speclen[:,None],
    }

    return out

exceptions = {
    'C(+57.02)': 'C+57.021',
    'M(+15.99)': 'M+15.995',
    'N(+.98)': 'N+0.984',
    'Q(+.98)': 'Q+0.984',
}

class LoaderHF:
    def __init__(self, 
        dataset_path: str,
        val_species: str=None,
        dictionary_path: str=None,
        tokenizer_path: str=None,
        top_pks: int=100,
        batch_size: int=100,
        num_workers: int=0,
        **kwargs
    ):

        # Scratch directory
        if 'scratch' in kwargs.keys():
            if kwargs['scratch']['use']:
                pth = kwargs['scratch']['path']
                if os.path.exists(pth):
                    # Change the dataset paths
                    dataset_path = {
                        key: pth + dataset_path[key].split("/")[-1]  
                        for key in dataset_path
                    }
                else:
                    print("Scratch directory not found. Using original paths.")

        # Dictionary
        if dictionary_path is not None:
            self.amod_dic = {
                line.split()[0]:m for m, line in enumerate(open(dictionary_path))
            }
            self.amod_dic['X'] = len(self.amod_dic)
            self.amod_dic_rev = {b:a for a,b in self.amod_dic.items()}

        # Dictionary masses
        masses_path = os.path.join(dataset_path, "ns_masses.txt")
        if os.path.exists(masses_path):
            mass_frame = pd.read_csv(masses_path, delimiter=" ", header=None)
            self.massdic = {m:n for m,n in zip(mass_frame[0], mass_frame[1])}

        # Species sizes
        ss_path = os.path.join(dataset_path, "species_sizes.txt")
        if os.path.exists(ss_path):
            species_sizes = pd.read_csv(ss_path, sep=" ", header=None, names=["species", "count"], index_col="species")
            self.val_size = int(species_sizes.query(f"species == '{val_species}'")['count'].iloc[0])
            self.train_size = int(species_sizes.query(f"species != '{val_species}'")['count'].sum())
        else:
            None

        # Dataset
        dataset_path_ = os.path.join(dataset_path, "parquet/processed")
        train_files = glob(os.path.join(dataset_path_, '*'))
        val_files = glob(os.path.join(dataset_path_, f"*{val_species}*"))
        for val_file in val_files:
            train_files.remove(val_file)
        data_files = {'train': train_files, 'val': val_files,}
        dataset = load_dataset(
            'parquet',
            data_files=data_files,
            streaming=True
        )
        dataset['test'] = dataset['val']
        
        # Tokenizer
        tokenizer_path = dataset_path if tokenizer_path==None else tokenizer_path
        sys.path.append(tokenizer_path)
        from enumerate_tokens import partition_modified_sequence
        self.tokenizer = partition_modified_sequence

        # Map to format outputs
        dataset = dataset.map(
            lambda example: 
            map_fn(
                example,
                tokenizer=self.tokenizer,
                dic=self.amod_dic,
                top=top_pks, 
                max_seq=max_seq
            ), 
            remove_columns=kwargs['remove_columns'] if 'remove_columns' in kwargs else None,
        )

        # Filter for length
        if 'pep_length' in kwargs.keys():
            dataset = dataset.filter(
                lambda example: 
                (len(example['tokenized_sequence']) >= kwargs['pep_length'][0]) &
                (len(example['tokenized_sequence']) <= kwargs['pep_length'][1])
            )
            max_seq = kwargs['pep_length'][1]
        else:
            max_seq = None
        
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['precursor_charge'] >= kwargs['charge'][0]) &
                (example['precursor_charge'] <= kwargs['charge'][1])
            )

        # Filter val set for dispersed examples
        if 'val_steps' in kwargs.keys():
            if kwargs['val_steps'] is not None:
                every_n = self.val_size // batch_size // kwargs['val_steps'] - 1 # minus 1 to be safe (charge and length filter make dataset shorter)
                dataset['val'] = dataset['val'].filter(lambda example, idx: idx % every_n == 0, with_indices=True)
        
        # Shuffle the dataset
        if 'buffer_size' in kwargs.keys():
            dataset['train'] = dataset['train'].shuffle(buffer_size=kwargs['buffer_size'])
        else:
            dataset['train'] = dataset['train'].shuffle()
        
        self.dataset = dataset

        # Dataloaders
        num_workers = min(self.dataset['train'].n_shards, num_workers)
        self.dataloader = {
            'train': self.build_dataloader(dataset['train'], batch_size, num_workers),
            'val':   self.build_dataloader(dataset['val']  , batch_size, 0),
            'test':  self.build_dataloader(dataset['test'] , batch_size, 0),
        }

    def build_dataloader(self, dataset, batch_size, num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

