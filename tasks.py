# STARTED MIGRATION
from collections import deque
import numpy as np
from utils import discretize_mz
from copy import deepcopy
import torch as th
F = th.nn.functional

class Task:
    def __init__(self, typ, maxlen=50):
        assert typ.lower() in ['mz', 'ab', 'both', 'charge', 'mass']
        self.typ = typ.lower()
        self.maxlen = maxlen
        self.running_loss = {'main': deque(maxlen=maxlen)}
        self.total_loss = {'main': 0}
        self.total_counter = 0
    
    def add_loss_key(self, key):
        self.running_loss[key] = deque(maxlen=self.maxlen)
        self.total_loss[key] = 0

    def log_loss(self, nploss):
        for key in self.running_loss.keys():
            self.running_loss[key].append(nploss)
            self.total_loss[key] += nploss
        self.total_counter += 1

    def calc_avg_total_loss(self):
        outputs = {}
        for key in self.total_loss.keys():
            outputs[key] = self.total_loss[key] / (self.total_counter+1e-10)
        
        return outputs

    def calc_avg_running_loss(self):
        outputs = {}
        for key in self.running_loss.keys():
            outputs[key] = (
                np.mean(self.running_loss[key]) 
                if len(self.running_loss[key])>0 else 
                0
            )

        return outputs

    def reset_total_loss(self):
        for key in self.total_loss.keys():
            self.total_loss[key] = 0
        self.total_counter = 0

class TrinaryTask(Task):
    def __init__(self, typ, freq=0.15, stdev=5, clip_vals=None):
        super().__init__(typ)
        self.freq = freq
        self.stdev = stdev
        self.clip_op = lambda x: (
            x if clip_vals==None else x.clip(*clip_vals)
        )

    def inptarg(self, batch, freq=None, std=None):
        
        dev = batch['mz'].device

        freq = self.freq if freq==None else freq
        std = self.stdev if std==None else std

        # MODEL INPUT: Corrupt mz
        mzab = deepcopy(batch[self.typ])
        
        # Random sequence indices to change
        inds_boolean = th.empty(mzab.shape, device=dev).uniform_(0, 1) < freq
        inds = th.cat(th.where(inds_boolean)).reshape(2, -1).T
        
        # Get their mz values
        means = mzab[inds_boolean]
        
        # Generate normal distributions for inds, centered on original value
        updates = th.normal(means, std)
        updates = self.clip_op(updates)
        
        # Distribute updates into corrupted indices
        mzab[inds_boolean] = updates
        if self.typ == 'mz':
            mzab_inp = th.cat([mzab[...,None], batch['ab'][...,None]], -1)
        elif self.typ == 'ab':
            mzab_inp = th.cat([batch['mz'][...,None], mzab[...,None]], -1)
        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length']
        }

        # TARGET: Classify all inds
        
        # inds that are below original value (0)
        zero = means > updates # 1d boolean
        zero_inds = inds[zero] # nx2 indices
        
        # inds that are above original value (2)
        two = means < updates # 1d boolean 
        two_inds = inds[two] # nx2 indices
        
        # Create target one-hot classification tensor
        # by default, everything starts with same/1
        target = th.ones(mzab.shape, dtype=th.int64, device=dev)
        
        # Separate nx2 indices into (2,) tuple
        target[zero_inds.split(1,1)] = th.zeros(
            zero.sum(), dtype=th.int64, device=dev
        )[:,None]
        
        target[two_inds.split(1,1)] = 2*th.ones(
            two.sum(), dtype=th.int64, device=dev
        )[:,None]
        
        self.target = F.one_hot(target, 3).type(th.float32)
        
        return inp

    def loss(self, prediction):
        # logits dimension (length 3: trinary) must be after batch
        #target = self.target.to(prediction.device).transpose(-1,-2)
        loss = F.cross_entropy(
            prediction.transpose(-1,-2), self.target.transpose(-1,-2), reduction='none'
        )

        return loss

class HiddenPeak(Task):
    def __init__(self, typ, binsz=0.1, mzlims=[0,2000], freq=0.15, loss_weight=1.):
        super().__init__(typ)
        self.typ = typ
        self.binsz = binsz
        if typ == 'mz':
            totbins = int(np.floor((mzlims[1] - mzlims[0]) / binsz))
        elif typ == 'ab':
            totbins = int(1/binsz)
        self.totbins = totbins
        self.freq = freq
        self.loss_weight = loss_weight

    def inptarg(self, batch, freq=None):
        
        dev = batch['mz'].device

        freq = self.freq if freq==None else freq
        
        mzab = deepcopy(batch[self.typ])

        # Random inds to mask out
        elig_inds_bool = th.tile(
            th.arange(mzab.shape[1], device=dev)[None], [mzab.shape[0], 1]
        ) < batch['length'][:,None]
        elig_inds = th.cat(th.where(elig_inds_bool)).reshape(2,-1).T
        # choose <freq> percent of those eligible inds
        indsinds = th.empty(elig_inds_bool.sum(), device=dev).uniform_(0, 1) < freq
        inds = elig_inds[indsinds].split(1,1)
        # Create a mask that will multiply by mz/ab fourier vectors inside MzAb 
        #mask = tf.ones((tf.shape(mzab)[0], tf.shape(mzab)[1], 2))
        #val = [[0., 1.]] if self.typ=='mz' else [[1., 0.]]
        #mask = tf.tensor_scatter_nd_update(
        #    mask, inds, tf.tile(tf.constant(val), [tf.shape(inds)[0], 1])
        #)
        # Also mask out original values, just to be safe
        inputmzab = deepcopy(mzab)
        inputmzab[inds] = th.zeros(inds[0].shape[0], 1, device=dev)
        #inputmzab = tf.tensor_scatter_nd_update(
        #    mzab, inds, tf.zeros((tf.shape(inds)[0]))
        #)
        
        if self.typ == 'mz':
            mzab_inp = th.cat(
                [inputmzab[...,None], batch['ab'][...,None]], -1
            )
        elif self.typ == 'ab':
            mzab_inp = th.cat(
                [batch['mz'][...,None], inputmzab[...,None]], -1
            )

        inp = {
            'x': mzab_inp, 
            'charge': batch['charge'],
            'mass': batch['mass'],
            'inp_mask': None
        }
        
        self.inds = inds
        target = mzab
        self.target = discretize_mz(
            target, self.binsz, self.totbins
        ).type(th.float32)
        
        return inp

    def loss(self, prediction, weight=None):
        if weight == None:
            weight = self.loss_weight
        pred = prediction[self.typ]
        loss = F.cross_entropy(
            pred.transpose(-1,-2), self.target.transpose(-1,-2)
        )

        return loss


class HiddenAbSpectrum(Task):
    def __init__(self, loss_weight=1.):
        super().__init__('ab')
        self.weight = loss_weight
    
    def inptarg(self, batch):
        bs, sl = batch['ab'].shape

        # Create a mask that will multiply by mz/ab fourier vectors inside MzAb 
        #mask = th.tensor([1., 0.])[None, None] # 1, 1, 2
        
        # Notice that the abundance is all zeros
        mzab_inp = th.cat(
            [batch['mz'][...,None], th.zeros_like(batch['ab'])[...,None]], -1
        )

        inp = {
            'x': mzab_inp, 
            'charge': batch['charge'],
            'mass': batch['mass'],
            'inp_mask': None#mask
        }
        
        self.target = batch['ab']
        
        return inp

    def loss(self, prediction):
        pred = prediction['ab'].sigmoid()
        pred = pred.mean(1)
        loss = F.cosine_similarity(self.target, pred)
        loss = -loss.mean()
        loss *= self.weight

        return loss

class HiddenCharge(Task):
    def __init__(self, max_charge=10, loss_weight=1.):
        super().__init__(typ='charge')
        self.max_charge = max_charge
        self.loss_weight = loss_weight

    def inptarg(self, batch):
        mzab_inp = th.cat(
            [batch['mz'][...,None], batch['ab'][...,None]], 
            dim=-1
        )
        charge = batch['charge']
        inp = {
            'x': mzab_inp, 
            'charge': th.zeros_like(charge),
            'mass': batch['mass'],
            'inp_mask': None
        }
        self.target = F.one_hot(
            (charge-1).type(th.int64), self.max_charge
        ).type(th.float32)

        return inp
    
    def loss(self, prediction):
        loss = F.cross_entropy(prediction, self.target)
        loss *= self.loss_weight

        return loss

class HiddenMass(Task):
    def __init__(self, loss_weight=1.):
        super().__init__(typ='mass')
        self.loss_weight = loss_weight

    def inptarg(self, batch):
        mzab_inp = th.cat(
            [batch['mz'][...,None], batch['ab'][...,None]],
            axis=-1
        )
        mass = batch['mass']
        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': th.zeros_like(mass),
            'inp_mask': None
        }
        self.target = F.one_hot(
            th.minimum(mass.round(), th.fill(mass, 3000)).type(th.int64), 3001
        ).type(th.float32)

        return inp

    def loss(self, prediction):
        loss = F.cross_entropy(prediction, self.target)
        loss *= self.loss_weight

        return loss

class Maldi(Task):
    def __init__(self, freq=0.15):
        super().__init__(typ='both')
        self.freq = freq

    def inptarg(self, batch):
        mz = deepcopy(batch['mz'])
        ab = deepcopy(batch['ab'])
        bs, sl = mz.shape
        
        # Don't shuffle filler peaks
        nonzero_spectrum_indices = th.where(mz != 0)
        nonzero_spectrum_indices = th.cat([m.unsqueeze(1) for m in nonzero_spectrum_indices], 1)
        
        # Choose 15% of the non-zero peaks
        hold = th.rand(nonzero_spectrum_indices.shape[0]) < self.freq
        rand_indices = nonzero_spectrum_indices[hold]
        hold = th.rand(rand_indices.shape[0])

        # Choose half to keep and half to permute
        #keep_indices = rand_indices[hold<0.5]
        change_indices = rand_indices[hold>=0.5]
        
        # Random permutation of (indices of) the peak indices to change
        perm = th.randperm(change_indices.shape[0], device=mz.device)

        # After permuting, which peaks ended up in the same spectrum?
        original_batch_inds = change_indices[:,0]
        same = original_batch_inds[perm] == original_batch_inds

        # Permute amongst same peaks. Hopefully this gets rid of all peaks that
        # ended up in the same spectrum.
        perm2 = th.randperm(sum(same), device=mz.device)
        #same2 = original_batch_inds[same] == original_batch_inds[same][perm2]
        perm[same] = perm[same][perm2]

        # Permute the peaks in the spectrum
        split = [m.squeeze() for m in change_indices.split(1,-1)]
        mz[split] = mz[change_indices[perm].split(1,-1)].squeeze()
        ab[split] = ab[change_indices[perm].split(1,-1)].squeeze()
        
        # Re-sort spectrum from low to high mz, with trailing zeros
        mz[mz==0] = 1e10
        sort = mz.argsort(dim=1)
        mz = th.gather(mz, 1, sort)
        ab = th.gather(ab, 1, sort)
        mz[mz > 1e9] = 0

        # Input
        mzab = th.cat([mz[...,None], ab[...,None]], -1)
        inp = {
            'x': mzab,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length']
        }
        
        # Target
        target = th.zeros_like(mz).long()
        target[change_indices.split(1,-1)] = 1
        self.target = target
        
        # Loss mask
        mask = th.zeros_like(mz)
        mask[rand_indices.split(1,-1)] = 1
        #mask = (th.arange(sl, device=mz.device)[None].tile(bs,1) < batch['length'][:,None]).float()
        self.mask = mask

        return inp

    def loss(self, prediction):
        loss = F.cross_entropy(prediction.transpose(-1,-2), self.target, reduction='none')
        loss *= self.mask
        loss = loss.sum() / self.mask.sum()

        return loss

class ResidualRegression(Task):
    def __init__(self, freq=0.15, stdev=1, clip_vals=None):
        super().__init__('mz')
        self.freq = freq
        self.stdev = stdev
        self.clip_op = lambda x: (
            x if clip_vals==None else x.clip(*clip_vals)
        )

    def inptarg(self, batch):

        dev = batch['mz'].device

        freq = self.freq
        std = self.stdev

        # MODEL INPUT: Corrupt mz
        mzab = deepcopy(batch[self.typ])
        
        # Random sequence indices to change
        inds_boolean = th.empty(mzab.shape, device=dev).uniform_(0, 1) < freq
        inds = th.cat(th.where(inds_boolean)).reshape(2, -1).T
        
        # Get their mz values
        means = mzab[inds_boolean]
        
        # Generate normal distributions for inds, centered on original value
        updates = th.normal(means, std)
        updates = self.clip_op(updates)
        
        # Distribute updates into corrupted indices
        mzab[inds_boolean] = updates
        mzab_inp = th.cat([mzab[...,None], batch['ab'][...,None]], -1)
        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length']
        }

        # TARGET: Classify all inds
        self.target = batch[self.typ] - mzab
        #assert self.target.shape[1] == 100, (
        #    batch[self.typ], mzab
        #)
        
        return inp

    def loss(self, prediction):
        loss = (self.target - prediction).abs()

        return loss

all_tasks = lambda tc: {
    'trinary_mz': TrinaryTask('mz', stdev=tc['trinary_mz']['stdev']),
    'trinary_ab': TrinaryTask(
        'ab', stdev=tc['trinary_ab']['stdev'], clip_vals=[0., 1.]
    ),
    'hidden_mz': HiddenPeak(
        'mz', loss_weight=tc['hidden_mz']['loss_weight'],
        binsz=tc['hidden_mz']['binsz'], mzlims=tc['hidden_mz']['mzlims'],
    ),
    'hidden_ab': HiddenPeak(
        'ab', loss_weight=tc['hidden_ab']['loss_weight'],
        binsz=tc['hidden_ab']['binsz']
    ),
    'hidden_spectrum': HiddenAbSpectrum(
        loss_weight=tc['hidden_spectrum']['loss_weight']
    ),
    'hidden_charge': HiddenCharge(
        max_charge=tc['hidden_charge']['max_charge'],
        loss_weight=tc['hidden_charge']['loss_weight']
    ),
    'hidden_mass': HiddenMass(loss_weight=tc['hidden_charge']['loss_weight']),
    'maldi': Maldi(**tc['maldi']),
    'resid_regr': ResidualRegression(**tc['resid_regr']),
}

