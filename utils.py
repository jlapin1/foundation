"""
Functions that I don't want to define in Pretrainmodel.py
"""
import torch as th
import numpy as np
from difflib import get_close_matches as gcm

def save_optimizer_state(opt, fn):
    th.save(opt.state_dict(), fn)

def load_optimizer_state(opt, fn, device):
    opt.load_state_dict(th.load(fn, map_location=device))

def save_full_model(model, optimizer, svdir):
    th.save(
        model.state_dict(), 
        "save/%s/weights/model_enc.wts"%(svdir)
    )
    save_optimizer_state(
        optimizer, 'save/%s/weights/opt_encopt.wts'%(svdir)
    )

def message_board(line, path):
    with open(path, 'a') as F:
        F.write(line)

def discretize_mz(mz, binsz, totbins):
    indices = th.maximum(
        th.zeros_like(mz), (mz / binsz).round().type(th.int32) - 1
    ).type(th.int64)
    
    return th.nn.functional.one_hot(indices, totbins)

def NonnullInds(SIArray, null_value):
    return th.where( SIArray != null_value )

def AccRecPrec(target, prediction, null_value):
    boolean = (target==prediction).type(th.int32)
    accsum = boolean.sum()
    recall_bool = target != null_value
    #recsum = tf.reduce_sum(tf.gather_nd(boolean, recall_inds))
    recsum = boolean[recall_bool].sum()
    prec_bool = prediction != null_value
    #precsum = tf.reduce_sum(tf.gather_nd(boolean, prec_inds))
    precsum = boolean[prec_bool].sum()
    out = {
        'accuracy': {'sum': accsum, 'total': target.shape[0]*target.shape[1]},
        'recall': {'sum': recsum, 'total': recall_bool.sum()},
        'precision': {'sum': precsum, 'total': prec_bool.sum()},
    }

    return out

def partition_seq(seq, collect_mods=False):
        Seq = []
        if collect_mods: mods = []
        p=0
        while p < len(seq):
            aa = seq[p]
            if aa=='(':
                let = seq[p-1]
                end = seq[p:].find(')')
                mod = seq[p+1 : p+end]
                if collect_mods: mods.append(mod)
                p += end
                aa = '%c_%s'%(let, mod)
                Seq[-1] = aa
            else:
                Seq.append(aa)
            p+=1
        output = {'seq': Seq}
        if collect_mods: output['mods'] = mods

        return output

masses = {
	'A': 71.037113805,
    'R': 156.101111050,
	'N': 114.042927470,
	'D': 115.026943065,
	'C': 103.009184505,
	'Q': 128.058577540,
	'E': 129.042593135,
	'G': 57.021463735,
	'H': 137.058911875,
	'I': 113.084064015,
	'L': 113.084064015,
	'K': 128.094963050,
	'M': 131.040484645,
	'F': 147.068413945,
	'P': 97.052763875,
	'S': 87.032028435,
	'T': 101.047678505,
	'W': 186.079312980,
	'Y': 163.063328575,
	'V': 99.068413945,
}

class Scale:
    def __init__(self, amod_dict):
        self.amod_dict = amod_dict
        int2mass = np.zeros((len(amod_dict)))
        for aa, integer in amod_dict.items():
            if len(aa.split('_')) == 2:
                aa, modwt = aa.split('_')
                int2mass[integer] = masses[aa] + eval(modwt)
            else:
                if aa in masses.keys():
                    int2mass[integer] = masses[aa]
                else:
                    int2mass[integer] = 0

        self.tok2mass = {key: int2mass[amod_dict[key]] for key in amod_dict.keys()}
        self.mp = th.tensor(int2mass, dtype=th.float32)
        """self.mp = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(int2mass.keys()), list(int2mass.values()),
                key_dtype=tf.int64, value_dtype=tf.float32
            ), num_oov_buckets=1
        )"""
    
    def intseq2mass(self, intseq):
        return th.gather(self.mp, 0, intseq).sum(1)

    def modseq2mass(self, modified_sequence):
        return np.sum(
            self.tok2mass[tok] for tok in partition_seq(modified_sequence)['seq']
        )

deltaPPM = lambda mprec, mpred: abs(mprec - mpred) * 1e6 / mprec

def Dict2dev(Dict, device, inplace=False):
    if inplace:
        for key in Dict.keys():
            Dict[key] = Dict[key].to(device)
        return True
    else:
        return {a: b.to(device) for a,b in Dict.items()}
