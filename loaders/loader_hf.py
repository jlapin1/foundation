from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import utils
import re

def map_fn(example, dic=None, top=100, max_seq=50):
    ab = th.tensor(example['ab'])
    ab_sort = (-ab).argsort()[:top]
    ab = ab[ab_sort]
    ab /= ab.max()
    mz = th.tensor(example['mz'])[ab_sort]
    mz_sort = mz.argsort()
    length = len(mz)
    mz_ = th.zeros(top)
    mz_[:len(mz_sort)] = mz[mz_sort]
    ab_ = th.zeros(top)
    ab_[:len(ab_sort)] = ab[mz_sort]
    example['mz'] = mz_
    example['ab'] = ab_
    example['charge'] = th.tensor(example['charge'], dtype=th.int32)
    example['mass'] = th.tensor(example['mass'], dtype=th.float32)
    example['length'] = th.tensor(length, dtype=th.int32)
    if 'sequence' in example.keys():
        intseq = [dic[m] for m in example['sequence']]
        intseq += (max_seq-len(intseq))*[dic['X']]
        example['intseq'] = th.tensor(intseq, dtype=th.int32)
        example['peplen'] = th.tensor(len(example['sequence']), dtype=th.int32)

    return example

def collate_fn(batch_list):
    mz = th.stack([m['mz'] for m in batch_list])
    ab = th.stack([m['ab'] for m in batch_list])
    charge = th.stack([m['charge'] for m in batch_list])
    mass = th.stack([m['mass'] for m in batch_list])
    length = th.stack([m['length'] for m in batch_list])
    if 'peplen' in batch_list[0]:
        peplen = th.stack([m['peplen'] for m in batch_list])
    else:
        peplen = None
    if 'intseq' in batch_list[0]:
        intseq = th.stack([m['intseq'] for m in batch_list])
    else:
        intseq = None

    out = {
        'mz': mz,
        'ab': ab,
        'charge': charge,
        'mass': mass,
        'length': length,
    }
    if intseq is not None:
        out['intseq'] = intseq
    if peplen is not None:
        out['peplen'] = peplen

    return out

exceptions = {
    'C(+57.02)': 'C+57.021',
    'M(+15.99)': 'M+15.995',
    'N(+.98)': 'N+0.984',
    'Q(+.98)': 'Q+0.984',
}

class LoaderHF:
    def __init__(self, 
        dataset_path: dict,
        dictionary_path: str=None,
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
            #for key in exceptions.keys():
            #    if exceptions[key] in self.amod_dic.keys():
            #        self.amod_dic[key] = self.amod_dic[exceptions[key]]
            self.amod_dic_rev = {b:a for a,b in self.amod_dic.items()}
        
        # Dataset
        dataset = load_dataset(
            'parquet',
            data_files=dataset_path,
            streaming=True
        )

        # Filter for length
        if 'pep_length' in kwargs.keys():
            dataset = dataset.filter(
                lambda example: 
                (len(example['sequence']) >= kwargs['pep_length'][0]) &
                (len(example['sequence']) <= kwargs['pep_length'][1])
            )
            max_seq = kwargs['pep_length'][1]
        else:
            max_seq = None
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['charge'] >= kwargs['charge'][0]) &
                (example['charge'] <= kwargs['charge'][1])
            )

        # Map to format outputs
        dataset = dataset.map(
            lambda example: 
            map_fn(
                example,
                self.amod_dic,
                top=top_pks, 
                max_seq=max_seq
            ), 
            remove_columns=kwargs['remove_columns'],
        )
        
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

