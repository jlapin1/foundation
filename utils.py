"""
Functions that I don't want to define in Pretrainmodel.py
"""
import torch as th
import numpy as np
from difflib import get_close_matches as gcm
from sklearn.metrics import average_precision_score
from copy import deepcopy
import datetime
import re
import os

def timestamp():
    dt = str(datetime.datetime.now()).split()
    dt[-1] = re.sub(':', '-', dt[-1]).split('.')[0]
    return "_".join(dt)

def create_experiment(directory, svwts=False):
    os.mkdir(directory)
    os.mkdir('%s/yaml'%directory)
    os.system("cp ./yaml/*.yaml %s/yaml/"%directory)
    if svwts: 
        os.mkdir('%s/weights'%directory)

def message_board(line, path):
    with open(path, 'a') as F:
        F.write(line)

def save_optimizer_state(opt, fn):
    th.save(opt.state_dict(), fn)

def load_optimizer_state(opt, fn, device):
    opt.load_state_dict(th.load(fn, map_location=device))

def save_full_model(model, optimizer, svdir, remark=""):
    th.save(
        model.state_dict(), 
        "%s/weights/model_enc_%s.wts"%(svdir, remark)
    )
    save_optimizer_state(
        optimizer, '%s/weights/opt_encopt_%s.wts'%(svdir, remark)
    )

def save_all_weights(svdir, encodern, header, remark="", clear=False):
    if clear:
        os.system("rm %s/weights/*"%svdir)
    encoder, optencoder = encodern
    save_full_model(encoder, optencoder, svdir, remark=remark)
    # Save header optimizer weights individually
    for task_name in header.heads.keys():
        th.save(header.heads[task_name].state_dict(), "%s/weights/head_%s_%s.wts"%(svdir, task_name, remark))
        # optimizer.name should have opt_ already in it (see Header in models)
        fn = '%s/weights/opt_%s_%s.wts'%(svdir, task_name, remark)
        save_optimizer_state(header.opts[task_name], fn)

find_file = lambda Match, loadpath: os.path.join(loadpath, [m for m in os.listdir(loadpath) if Match in m][0])


def discretize_mz(mz, binsz, totbins):
    indices = th.maximum(
        th.zeros_like(mz), (mz / binsz).round().type(th.int32) - 1
    ).type(th.int64)
    
    return th.nn.functional.one_hot(indices, totbins)

def NonnullInds(SIArray, null_value):
    return th.where( SIArray != null_value )

def OldRecall(target, prediction, null_value):
    boolean = (target == prediction).type(th.int32)
    recall_bool = target != null_value
    rec_sum = boolean[recall_bool].sum()

    return rec_sum / recall_bool.sum()

def AccRecPrec(target, prediction, null_value):
    boolean = (target==prediction).type(th.int32)
    accsum = boolean.sum()
    recall_bool = target != null_value
    #recsum = tf.reduce_sum(tf.gather_nd(boolean, recall_inds))
    recsum = boolean[recall_bool].sum()
    prec_bool = prediction != null_value
    #precsum = tf.reduce_sum(tf.gather_nd(boolean, prec_inds))
    precsum = boolean[prec_bool].sum()
    
    boolean[target == null_value] *= 0
    peptide_sum = (boolean.sum(1) == recall_bool.sum(1)).sum()
    out = {
        'accuracy' : {'sum': accsum,      'total': target.numel()   },
        'recall'   : {'sum': recsum,      'total': recall_bool.sum()},
        'precision': {'sum': precsum,     'total': prec_bool.sum()  },
        'peptide'  : {'sum': peptide_sum, 'total': target.shape[0]},
    }

    return out

def RocCurve(target, prediction, probs, null_value=23, typ='aa'):
    bs, sl, pc = probs.shape
    one = th.arange(bs)[:,None].tile(1, sl).reshape(-1,).type(th.int32)
    two = th.arange(sl)[None].tile(bs, 1).reshape(-1,).type(th.int32)
    three = prediction.reshape(-1,).type(th.int32)
    probs = probs.softmax(-1)[(one,two,three)]
    
    if typ == 'aa':
        # Only real experimental tokens
        # - include first null -> <EOS>
        target_ = deepcopy(target)
        target_[th.arange(bs), (target==null_value).int().argmax(1)] = 1000
        bln = (target_ != null_value).reshape(-1,)
        # Predicted correctly?
        eq = (target == prediction).reshape(-1,)
        
        probs_ = probs[bln]
        eq_ = eq[bln]
    elif typ == 'pep':
        # Peptide predicted correctly?
        nonnull = target != null_value
        bln = (target == prediction) & nonnull
        eq_ = bln.sum(1) == nonnull.sum(1) # all aa's correct?

        probs = probs.reshape(bs, sl)
        probs_ = (probs * nonnull).sum(1) / nonnull.sum(1)
        #log_conf = (probs.log() * bln).sum(1)
        #probs_ = log_conf.exp()
    
    # Sort confidence from high to low
    argsort = probs_.argsort(0).flip(0)
    probs_sort = probs_[argsort]
    eq_sort = eq_[argsort]

    #probs_sort = probs_sort[probs_sort > threshhold]
    #eq_sort = eq_sort[probs_sort > threshhold]

    cumsum = th.cumsum(eq_sort, 0)
    precision_denom = th.arange(1, eq_sort.shape[0]+1, 1).to(prediction.device)
    recall_denom = eq_sort.shape[0]
    precision = cumsum / precision_denom
    recall = cumsum / recall_denom
    
    if eq_sort.sum() > 0:
        auprc = average_precision_score(eq_sort.detach().cpu().numpy(), probs_sort.detach().cpu().numpy())
    else:
        auprc = 0
    
    return {
        'precision': precision.detach().cpu().numpy(), 
        'recall': recall.detach().cpu().numpy(),
        'probabilities': probs_sort.detach().cpu().numpy(),
    }, auprc

def roc_apply_threshold(recall, precision, probabilities, threshold=0.9):
    boolean = probabilities > threshold
    if sum(boolean) == 0:
        return {'recall': 0, 'precision': 0}

    recall = recall[boolean]
    precision = precision[boolean]

    return {
        'recall': recall[-1],
        'precision': precision[-1]
    }


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

def partition_modified_sequence(modseq):
    sequence = []
    p = 0
    while p < len(modseq):
        character = modseq[p]
        hx = ord(character)
        
        # Pull out mod, in the form of a floating point number
        if hx < 65:
            mod_lst = []

            # N-terminal modifications precede the amino acid letter
            nterm = True if p == 0 else False

            # All numerals and mathematical symbels are below 65
            while hx < 65:
                mod_lst.append(character)
                p += 1

                # This will happen if we have a C-term modification
                if p == len(modseq):
                    break
                else:
                    character = modseq[p]
                    hx = ord(character)
            mod = "".join(mod_lst)

            # Add the amino acid to the end of the number if N-term
            if nterm:
                # These nterm modifications occur with every amino acid at least once.
                if mod in ["+42.011", "+43.006"]:
                    sequence.append(mod)
                else:
                    token = mod + character
                    sequence.append(token)
            
            # Grab the previously stored sequence AA and add modification to it
            else:
                sequence[-1] += mod
            
            p -= 1

        else:
            sequence.append(character)

        p += 1
    
    return sequence

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
    
    def intseq2mass(self, intseq):
        sumdim = 0 if len(intseq.shape) == 1 else 1
        masses = self.mp.to(intseq.device)
        return th.gather(masses, 0, intseq).sum(sumdim)

    def modseq2mass(self, modified_sequence):
        return np.sum(
            self.tok2mass[tok] for tok in partition_seq(modified_sequence)['seq']
        )

deltaPPM = lambda mprec, mpred: abs(mprec - mpred) * 1e6 / mprec

def Dict2dev(Dict, device, inplace=False):
    if inplace:
        for key in Dict.keys():
            if type(b)==th.Tensor: Dict[key] = Dict[key].to(device)
        return True
    else:
        return {a: b.to(device) for a,b in Dict.items() if type(b)==th.Tensor}

def global_grad_norm(model):
    return sum([m.grad.detach().square().sum().item() for m in model.parameters() if (m.requires_grad and m.grad is not None)])**0.5
