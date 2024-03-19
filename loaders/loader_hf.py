from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th

def map_fn(example, dic, top=100, max_seq=50):
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
    peplen = th.stack([m['peplen'] for m in batch_list])
    intseq = th.stack([m['intseq'] for m in batch_list])

    return {
        'mz': mz,
        'ab': ab,
        'charge': charge,
        'mass': mass,
        'length': length,
        'intseq': intseq,
        'peplen': peplen,
    }

class LoaderHF:
    def __init__(self, config, kwargs):
        self.config = config

        # Dictionary
        self.amod_dic = {
            line.split()[0]:m for m, line in enumerate(open(config['dictionary_path']))
        }
        self.amod_dic['X'] = len(self.amod_dic)
        """
        # Scratch directory
        if 'scratch' in kwargs.keys():
            if kwargs['scratch']['use']:
                pth = kwargs['scratch']['path']
                if os.path.exists(pth):
                    self.fn2full = {
                        key: pth + self.fn2full[key].split("/")[-1]  
                        for key in self.fn2full.keys()
                    }
                else:
                    print("Scratch directory not found. Using original paths.")
        """
        # Dataset
        dataset = load_dataset(
            'parquet',
            data_files=config['dataset_path'],
            split='train',
            streaming=True
        )
        # Filter for length
        dataset = dataset.filter(
            lambda example: 
            (len(example['sequence']) >= config['pep_length'][0]) &
            (len(example['sequence']) <= config['pep_length'][1])
        )
        # Map to format outputs
        dataset = dataset.map(
            lambda example: 
            map_fn(
                example,
                self.amod_dic,
                top=config['top_pks'], 
                max_seq=config['pep_length'][1]
            ), 
            remove_columns=['name', 'sequence']
        )

        # Split dataset
        val_dataset = dataset.take(config['split'])
        train_dataset = dataset.skip(config['split']).shuffle(buffer_size=config['buffer_size'])
        self.ds = {
            'train': train_dataset,
            'val': val_dataset,
        }

        # Dataloaders
        self.dl = {
            'train': self.build_dataloader(train_dataset, config['batch_size']),
            'val': self.build_dataloader(val_dataset, config['batch_size'])
        }

    def build_dataloader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn
        )

