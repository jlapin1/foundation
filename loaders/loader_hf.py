from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import utils
import re
from glob import glob
import sys
import pandas as pd
import numpy as np

def partition_modified_sequence(sequence):
    
    # Split apart letter+number from continuous letters
    #   [A-Z]{0,1}: 0 or 1 letters to start the sequence
    #   [+-]?: one or none of + or -
    #   [0-9]*: any amount of digits 0-9
    #   [.]?: any amount of periods
    #   [0-9]+: 1 to any amount of digits 0-9
    split = re.split("([A-Z]{0,1}[+-]?[0-9]*[.]?[0-9]+)", sequence)
    
    # Split (unmodified) strings into characters, and remove ['']
    list_of_lists = [[x] if re.search("[+-]", x) else list(x) for x in split if x != '']
    
    # Flatten
    tokenized_sequence = [m for n in list_of_lists for m in n]
    
    return tokenized_sequence

def map_fn(
    example,
    idx,
    tokenizer, 
    dic=None, 
    top=100, 
    max_seq=50,
    mz_key='mz',
    ab_key='ab',
    charge_key='charge',
    mass_key='mass',
    name_key='name',
):
    example['iloc'] = idx
    ab = example[ab_key]
    ab_sort = (-ab).argsort()[:top]
    spectrum_length = len(ab_sort)
    ab = ab[ab_sort]
    ab /= ab.max()
    spectrum_length = len(ab)
    mz = example[mz_key][ab_sort]
    mz_sort = mz.argsort()
    length = len(mz)
    #mz_ = th.zeros(top)
    #mz_[:len(mz_sort)] = mz[mz_sort]
    mz_ = np.concatenate([mz[mz_sort], np.zeros((top-len(mz_sort)))])
    #ab_ = th.zeros(top)
    #ab_[:len(ab_sort)] = ab[mz_sort]
    ab_ = np.concatenate([ab[mz_sort], np.zeros((top-len(mz_sort)))])
    example['mz'] = mz_
    example['ab'] = ab_
    example['charge'] = example[charge_key]
    example['mass'] = example[mass_key]
    example['spectrum_length'] = spectrum_length
    example['name'] = f"{example[name_key]}|{example['scan']}"
    if tokenizer is not None and 'modified_sequence' in example:
        tokenized_sequence = tokenizer(example['modified_sequence'])
        peptide_length = len(tokenized_sequence)
        #example['tokenized_sequence'] = th.tensor([dic[m] for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=th.int32)
        example['peptide_length'] = th.tensor(peptide_length, dtype=th.int32)
        #example['spectrum_length'] = th.tensor(spectrum_length, dtype=th.int32)

    return example

def collate_fn(batch_list):
    name = np.array([m['name'] for m in batch_list])
    iloc = np.array([m['iloc'] for m in batch_list])
    speclen = np.stack([m['spectrum_length'] for m in batch_list])
    mz      = np.stack([m['mz'][:speclen.max()] for m in batch_list])
    ab      = np.stack([m['ab'][:speclen.max()] for m in batch_list])
    charge  = np.stack([m['charge'] for m in batch_list])
    mass    = np.stack([m['mass'] for m in batch_list])
    
    if 'peptide_length' in batch_list[0]:
        peplen = np.array([m['peptide_length'].item() for m in batch_list])
        modseq = np.array([m['modified_sequence'] for m in batch_list])
    if 'replicate_counts' in batch_list[0]:
        replicates = np.array([m['replicate_counts'].item() for m in batch_list]).astype(np.int32)

    out = {
        'name': name,
        'iloc': iloc,
        'mz': th.tensor(mz, dtype=th.float32),
        'ab': th.tensor(ab, dtype=th.float32),
        'charge': th.tensor(charge, dtype=th.int32),
        'mass': th.tensor(mass, dtype=th.float32),
        'length': th.tensor(speclen, dtype=th.int32),
        #'intseq': intseq,
        #'peplen': peplen,
        #'spectrum_lengths': speclen[:,None],
    }
    if 'peptide_length' in batch_list[0]:
        out['peplen'] = peplen
        out['modified_sequence'] = modseq
    if 'replicate_counts' in batch_list[0]:
        out['replicate_counts'] = replicates
    
    return out

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
        """masses_path = os.path.join(dataset_path, "ns_masses.txt")
        if os.path.exists(masses_path):
            mass_frame = pd.read_csv(masses_path, delimiter=" ", header=None)
            self.massdic = {m:n for m,n in zip(mass_frame[0], mass_frame[1])}"""

        # Species sizes
        """ss_path = os.path.join(dataset_path, "species_sizes.txt")
        if os.path.exists(ss_path):
            species_sizes = pd.read_csv(ss_path, sep=" ", header=None, names=["species", "count"], index_col="species")
            self.val_size = int(species_sizes.query(f"species == '{val_species}'")['count'].iloc[0])
            self.train_size = int(species_sizes.query(f"species != '{val_species}'")['count'].sum())
        else:
            None"""
        
        # Tokenizer
        """tokenizer_path = dataset_path if tokenizer_path==None else tokenizer_path
        sys.path.append(tokenizer_path)
        from enumerate_tokens import partition_modified_sequence"""
        self.tokenizer = partition_modified_sequence
        self.amod_dic = None
        max_seq = None
        
        # Training Dataset
        data_files = {m:n for m, n in dataset_path.items() if n!=None}
        dataset = load_dataset(
            'parquet',
            data_files=data_files,
            streaming=True,
        ).with_format("numpy")
        
        # Map to format outputs
        base = os.path.split(data_files['train'])[0]
        lambda_function_train = self.create_lambda_function(base, self.tokenizer, self.amod_dic, top_pks, max_seq)
        dataset = dataset.map(
            lambda_function_train,
            with_indices=True, 
            remove_columns=kwargs['remove_columns'] if 'remove_columns' in kwargs else None,
        )

        # NNeval dataset
        dataset_val = load_dataset(
            'parquet',
            data_files=data_files['val'],
            streaming=True,
        ).with_format("numpy")
        
        base = os.path.split(data_files['val'])[0]
        lambda_function_val = self.create_lambda_function(base, None, None, top_pks, None)
        dataset_val = dataset_val.map(
            lambda_function_val,
            with_indices=True,
            remove_columns=kwargs['remove_columns'] if 'remove_columns' in kwargs else None,
        )
        
        # Filter for id'ed or unid'ed spectra
        #dataset = dataset.filter(
        #    lambda example:
        #    example['observed_mz'] != -1
        #)
        """
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
        """
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
            'val':   self.build_dataloader(dataset_val['train']  , batch_size, 0),
        }
        if 'test' in dataset:
            self.dataloader['test'] = self.build_dataloader(dataset['test'], batch_size, 0)

    def build_dataloader(self, dataset, batch_size, num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=False
        )
    
    def create_lambda_function(self, base_file_path, tokenizer, dictionary, top_peaks, max_sequence):
        
        keys = pd.read_csv(os.path.join(base_file_path, 'keys.tsv'), sep='\t', header=None).set_index(0)
        lambda_function = lambda example, idx: map_fn(
            example,
            idx,
            tokenizer=tokenizer,
            dic=dictionary,
            top=top_peaks,
            max_seq=max_sequence,
            mz_key=keys.loc['mz_key'].item(),
            ab_key=keys.loc['ab_key'].item(),
            charge_key=keys.loc['charge_key'].item(),
            mass_key=keys.loc['mass_key'].item(),
            name_key=keys.loc['name_key'].item(),
        )
        return lambda_function
