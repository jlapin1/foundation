import numpy as np
import torch as th
import os
import pandas as pd
import re
import path
import multiprocessing
import queue
from itertools import cycle

def gather_file_md(filepath, typ=None):
    if typ==None:
        typ = filepath.split('.')[-1].strip().lower()

    with open(filepath) as f:
        _ = f.read()
        end = f.tell()
        f.seek(0)
        
        pos = f.tell()
        pos_prev = f.tell()
        spectra = {}
        spec_ticker = 0
        while pos!=end:

            line = f.readline().strip()
            pos = f.tell()
            
            if typ=='mgf':
                if line == 'BEGIN IONS':
                    # If the last spectrum had no peaks (or no charge), delete it
                    if spec_ticker in spectra:
                        del(spectra[spec_ticker])

                    spectra[spec_ticker] = {}
                elif line.split('=')[0] == 'SCANS':
                    scan = int(line.split('=')[-1])
                    spectra[spec_ticker]['scan'] = scan
                elif line.split('=')[0] == 'RTINSECONDS':
                    rt = float(line.split('=')[-1])
                    spectra[spec_ticker]['rt'] = rt
                elif line.split('=')[0] == 'CHARGE':
                    charge = int(line.split('=')[-1].replace('+',''))
                    spectra[spec_ticker]['charge'] = charge
                elif line.split('=')[0] == 'PEPMASS':
                    mass = float(line.split('=')[-1])
                    spectra[spec_ticker]['mass'] = mass
                elif line.split('=')[0] == 'SEQUENCE':
                    seq = line.split('=')[-1]
                    spectra[spec_ticker]['sequence'] = seq
                elif len(line.split('.')) == 3:
                    peak_ticker = 0
                    spectra[spec_ticker]['pos'] = pos_prev
                    while line != 'END IONS':
                        peak_ticker += 1
                        line = f.readline().strip()
                    spectra[spec_ticker]['nmpks'] = peak_ticker
                    
                    if (
                        'charge' in spectra[spec_ticker] and
                        'mass' in spectra[spec_ticker]
                    ):
                        spec_ticker += 1
                
                pos_prev = pos
                
            elif typ=='msp':

                # Start of a spectrum entry: label
                # - assume labels are {seq}/{charge}_{mods}_{ev}eV_NCE{nce}
                if line[:5]=='Name:':
                    spectra[spec_ticker] = {}
                    spectra[spec_ticker]['label'] = line.split()[-1]
                    seq, other = line.split()[-1].split('/')
                    spectra[spec_ticker]['seq'] = seq
                    charge, mods, ev, nce = other.split('_')
                    spectra[spec_ticker]['charge'] = int(charge)
                    spectra[spec_ticker]['ev'] = float(ev[:-2])
                    spectra[spec_ticker]['nce'] = float(nce[3:])
                    
                    # parsing mod
                    spectra[spec_ticker]['mod_label'] = mods
                    spectra[spec_ticker]['mod_pos'] = []
                    spectra[spec_ticker]['mod_name'] = []
                    spectra[spec_ticker]['mod_aa'] = []
                    if mods != '0':
                        m0 = mods.find('(')
                        mod_amt = int(mods[:m0])
                        for mod in mods[m0+1:-1].split(')('):
                            pos, aa, name = mod.split(',')
                            spectra[spec_ticker]['mod_pos'].append(int(pos))
                            spectra[spec_ticker]['mod_name'].append(name)
                            spectra[spec_ticker]['mod_aa'].append(aa)
                    # Done with label
                    # Search no more than 10 lines for MW
                    for i in range(10):
                        line = f.readline()
                        if line[:3]=='MW:':
                            spectra[spec_ticker]['mw'] = float(line.split()[-1])
                            break
                    # Search no more than 10 lines for Num peaks
                    for i in range(10):
                        line = f.readline()
                        if line[:10] == 'Num peaks:':
                            nmpks = int(line.split()[-1])
                            spectra[spec_ticker]['nmpks'] = nmpks
                            spectra[spec_ticker]['pos'] = f.tell()
                            for _ in range(nmpks): f.readline()
                            break

                    assert len(spectra[spec_ticker].keys()) == 12
                    spec_ticker += 1
            else:
                NotImplementedError("File type not implemented yet.")
    
    # If the very last spectrum had no peaks, delete it
    tick = max(list(spectra.keys()))
    if (
        'pos' not in spectra[tick] or
        'charge' not in spectra[tick]
    ):
        del(spectra[tick])
    return spectra

def gather_evid_data(filepath):
    with open(filepath) as f:
        header = f.readline().strip().split('\t')
        dic = {m: [] for m in header}
        for line in f:
            for m,n in zip(dic.keys(), line.strip().split('\t')):
                dic[m].append(n)

    return dic

def filter_length(df, len_rng):
    assert hasattr(df, 'seq')
    boolean = (
        (np.vectorize(len)(df['seq'])>=len_rng[0]) &
        (np.vectorize(len)(df['seq'])<=len_rng[1])
    )
    df = df.iloc[boolean]

    return df

def filter_charge(df, ch_rng):
    assert hasattr(df, 'charge')
    boolean = np.array(
        (df['charge']>=ch_rng[0]) &
        (df['charge']<=ch_rng[1])
    )
    df = df.iloc[boolean]
    
    return df

def filter_mod(df, mod_list):
    assert hasattr(df, 'mod_name')
    boolean = []
    for i in range(len(df)):
        pepmods = df.iloc[i]['mod_name']
        tick = True
        for j in pepmods:
            if j not in mod_list:
                boolean.append(False)
                tick=False
                break
        if tick:
            boolean.append(True)
    boolean = np.array(boolean)
    df = df.iloc[boolean]

    return df

def worker_fn(dataset, index_queue, output_queue):
    while True:
        # Worker function, simply reads indices from index_queue, and adds the
        # dataset element to the output_queue
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))

class DatasetObj:
    def __init__(self, 
                 paths=None, 
                 train_dirs=None,
                 preopen_files=True, 
                 mdsaved_path='./mdsaved', 
                 top_pks=100, 
                 save_md=True,
                 filter_psms=False,
                 **kwargs
                 ):
        self.is_filter_psms = filter_psms

        # Get paths from train_dirs if paths not provided
        if paths == None:
            self.paths = {fp: path.glob.glob(fp+'/*.mgf') for fp in train_dirs}
            self.dir_lookup = {}
            for d in self.paths.keys():
                for f in self.paths[d]:
                    fn = f.split('/')[-1].split('.')[0]
                    self.dir_lookup[fn] = d
            all_paths = [m for n in self.paths.values() for m in n]
        self.train_dirs = train_dirs

        # cut out possible mdsaved directory from path search
        self.all_paths = [path for path in all_paths if 'evidence' not in path]
        self.preopen = preopen_files
        self.mdsaved_path = mdsaved_path
        self.top_pks = top_pks
        self.save_md = save_md
        
        if save_md and not os.path.exists(mdsaved_path):
            os.mkdir(mdsaved_path)
        
        self.gather_md()

        self.gather_labels()
        
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

        if self.preopen:
            self.open_files()

    
    def gather_md(self):
        self.md = {}
        self.fn2full = {}
        for path in self.all_paths:
            
            # Snip off the filename
            filename = path.split('/')[-1].split('.')[0]
            self.fn2full[filename] = path
            
            # Create the (potential) full path to the saved md data
            fullpath = '%s/%s_md.pkl'%(self.mdsaved_path, filename)
            
            if os.path.exists(fullpath):
                df = pd.read_pickle(fullpath)
            else:
                # Read the file
                spec_dic = gather_file_md(path)
                df = pd.DataFrame(spec_dic).transpose() # transpose alters dtypes
                # Save these (2) entries as integers
                df['scan'] = df['scan'].astype("int32")
                df['pos'] = df['pos'].astype("int32")
                df['nmpks'] = df['nmpks'].astype('int32')
                if self.save_md:
                    df.to_pickle('%s/%s_md.pkl'%(self.mdsaved_path, filename))

            self.md[filename] = df
            if self.is_filter_psms:
                assert hasattr(self, 'dir_lookup')
                self.filter_psms(filename)

        self.filenames = list(self.md.keys())
    
    def filter_psms(self, filename):
        # In place altering of metadata dataframe
        bdir = self.dir_lookup[filename]
        evid_path = bdir + '/evidence/'
        assert os.path.exists(evid_path)
        Evid = evid_path + filename + '.evid'
        assert os.path.exists(Evid)
        dic = gather_evid_data(Evid)
        if "MS/MS Scan Number" in dic.keys():
            sns = np.array(dic['MS/MS Scan Number'], dtype=int)
        else:
            sns = np.array(dic["Scan number" ], dtype=int)
        
        fullpath = '%s/%s_index.txt'%(self.mdsaved_path, filename)
        if os.path.exists(fullpath):
            I = np.loadtxt(fullpath).astype(int)
        else:
            I = np.array(
                [self.md[filename].query('scan == %d'%sn).index for sn in sns]
            ).squeeze()
            if self.save_md:
                np.savetxt(fullpath, I, fmt='%d')

        assert len(I)>0
        self.md[filename] = self.md[filename].iloc[I]
        self.md[filename] = pd.DataFrame({
            'scan': self.md[filename]['scan'],
            'rt': self.md[filename]['rt'],
            'charge': self.md[filename]['charge'],
            'mass': self.md[filename]['mass'],
            'pos': self.md[filename]['pos'],
            'nmpks': self.md[filename]['nmpks'],
            'Seq': dic['Sequence'],
            'Mod': dic['Modifications'],
            'Modseq': dic['Modified sequence'],
            'Mass': np.array(dic['Mass'], dtype=float),
            'MassError': np.array(dic['Mass Error [ppm]'], dtype=float),
            'Score': np.array(dic['Score'], dtype=float)
        })
        
    def gather_labels(self):
        # index.values needs df.loc, enumerate needs df.iloc
        listoflists = [
            ['%s|%d'%(filename,iloc) for iloc, loc in enumerate(self.md[filename].index.values)] 
            for filename in self.md.keys()
        ]
        self.labels = [m for n in listoflists for m in n]

    def open_files(self):
        self.fps = {
            name: open(path) 
            for name, path in zip(self.filenames, self.paths)
        }

    def close_files(self):
        for key in self.fps.keys():
            self.fps[key].close()
    
    def read_spec_(self, filename, index, ann=False):
        fp = self.fps[filename] if self.preopen else open(self.fn2full[filename])
        md = self.md[filename] # calling the sample automatically casts dtypes
        fp.seek(md['pos'].iloc[index])
        pks = np.array([
                [float(m) for m in fp.readline().strip().split()] 
                for _ in range(md['nmpks'].iloc[index])
        ])
   
        #mz, ab = np.split(pks, 2, -1)
        mz = pks[:,0];ab = pks[:,1]
        output = {
                'mz': mz,
                'ab': ab,
                'charge': md['charge'].iloc[index],
                'mass': md['mass'].iloc[index]
        }
        if not self.preopen: fp.close()
        return output
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index, top=None):
        if top==None:
            top=self.top_pks
        mz = np.zeros((top))
        ab = np.zeros((top))
        
        fnm, ind = self.labels[index].split('|')
        spec_dic = self.read_spec_(fnm, int(ind))
        marg = np.argsort(spec_dic['ab'])[-top:]
        mzsort = np.argsort(spec_dic['mz'][marg])

        mz[:len(marg)] = spec_dic['mz'][marg][mzsort]
        ab_ = spec_dic['ab'][marg][mzsort]
        ab[:len(marg)] = ab_ / ab_.max()
        
        return {
            'mz': mz,
            'ab': ab,
            'charge': spec_dic['charge'],
            'mass': spec_dic['mass'],
            'length': len(mzsort)
        }

    def load_batch(self, indices, top=None):
        if top==None: top=self.top_pks
        
        L = len(indices)
        mz = np.zeros((L, top))
        ab = np.zeros((L, top))
        charge = np.zeros((L,))
        mass = np.zeros((L,))
        lengths = np.zeros((L,))
        for i, j in enumerate(indices):
            
            spec_dic = self.__getitem__(j)
                  
            mz[i] = spec_dic['mz']
            ab[i] = spec_dic['ab']
            charge[i] = spec_dic['charge']
            mass[i] = spec_dic['mass']
            lengths[i] = spec_dic['length']

        output = {
                'mz': th.tensor(mz, dtype=th.float32), 
                'ab': th.tensor(ab/ab.max(-1, keepdims=True), dtype=th.float32),
                'charge': th.tensor(charge, dtype=th.int32),
                'mass': th.tensor(mass, dtype=th.float32),
                'length': th.tensor(lengths, dtype=th.int32)
        }

        return output

    def collate_fn(self, list_gis):
        mz = th.stack([th.tensor(L['mz'], dtype=th.float32) for L in list_gis])
        ab = th.stack([th.tensor(L['ab'], dtype=th.float32) for L in list_gis])
        charge = th.tensor([L['charge'] for L in list_gis], dtype=th.int32)
        mass = th.tensor([L['mass'] for L in list_gis], dtype=th.float32)
        lengths = th.tensor([L['length'] for L in list_gis], dtype=th.int32)

        return {
            'mz': mz,
            'ab': ab,
            'charge': charge,
            'mass': mass,
            'length': lengths,
        }

class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size=100,
        num_workers=1,
        prefetch_batches=0.5,
        shuffle=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = dataset.collate_fn
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.shuffle = shuffle

        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = cycle(range(num_workers))
        self.cache = {}
        self.index = 0
        self.prefetch_index = 0
        self.empty_oq = 0
        

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, 
                args=(self.dataset, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

        self.perm = (
            np.random.permutation(len(dataset)) 
            if shuffle else 
            np.arange(len(dataset))
        )

        self.prefetch()
    
    def prefetch(self):
        """
        Add dataset indices to the respective worker index_queues,
        -->>> CONSEQUENTLY, ADD DATA TO OUTPUT QUEUE
        WHY?
        - The worker processes are already started and running in the background.
          Once a index number is put into the index_queue(s), the index will be
          found in worker_init_fn and data will be put into output_queue
        """
        while(
            self.prefetch_index < len(self.dataset) and
            self.prefetch_index < 
            self.index + self.prefetch_batches*self.num_workers*self.batch_size
        ):
            # if the prefetch_index hasn't reached the end of the dataset
            # and it is not 2 batches ahead, add indices to the index queues
            self.index_queues[next(self.worker_cycle)].put(self.perm[self.prefetch_index])
            self.prefetch_index += 1

    def __iter__(self):
        """
        This function is entered once per epoch
        """
        self.index = 0
        self.cache = {}
        self.prefetch_index = 0
        if self.shuffle:
            self.perm = np.random.permutation(len(self.dataset))
        self.prefetch()
        return self
    
    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        batch_size = min(len(self.dataset) - self.index, self.batch_size)
        return self.collate_fn([self.get() for _ in range(batch_size)])

    def get(self):
        """
        This subroutine is getting called by multiple processes, independently.
        We are looking for a specific, global index (self.index), which when
        encountered will return that data structure (item). 
        - Every structure pulled from the output_queue that does not have that 
        index is stored for later calls of get(), at which time it will be pu-
        lled from self.cache.
        """
        I = self.perm[self.index]
        self.prefetch()
        if I in self.cache:
            item = self.cache[I]
            del self.cache[I]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty: # output queue empty, keep trying
                    self.empty_oq += 1
                    continue
                if index == I: # found our item, ready to return
                    item = data
                    break
                else: # item isn't the one we want, cache for later
                    self.cache[index] = data

        self.index += 1
        return item

    def __del__(self):
        try:
            for i, w in enumerate(self.workers):
                self.index_queues[i].put(None)
                w.join(timeout=5.0)
            for q in self.index_queues:
                q.cancel_join_thread()
                q.close()
            self.output_queue.cancel_join_thread()
            self.output_queue.close()
        finally:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()


"""
import yaml

with open("/cmnfs/home/j.lapin/projects/foundational/yaml/datasets.yaml") as stream:
    config = yaml.safe_load(stream)

L = LoadObj(**config['pretrain'])
print(L.load_batch(L.labels[:100]))
"""
