from copy import deepcopy
import numpy as np
from utils import Scale
import models.model_parts as mp
import torch as th
from torch import nn

class Decoder(nn.Module):
    def __init__(self,
                 running_units=512,
                 kv_indim=256,
                 sequence_length=30, # maximum number of amino acids
                 num_inp_tokens=21,
                 depth=9,
                 d=64,
                 h=4,
                 ffn_multiplier=1,
                 ce_units=256,
                 use_charge=True,
                 use_energy=False,
                 use_mass=True,
                 norm_type='layer',
                 prenorm=True,
                 preembed=True,
                 penultimate_units=None,
                 pool=False,
                 ):
        super(Decoder, self).__init__()
        self.run_units = running_units
        self.kv_indim = kv_indim
        self.sl = sequence_length
        self.num_inp_tokens = num_inp_tokens
        self.num_out_tokens = num_inp_tokens - 2 # no need for start or hidden tokens
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass

        self.ce_units = ce_units
        
        # Normalization type
        self.norm = mp.get_norm_type(norm_type)

        # First embeddings
        self.seq_emb = nn.Embedding(num_inp_tokens, running_units)
        self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        
        # charge/energy embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            num = sum([use_charge, use_energy, use_mass])
            self.ce_emb = nn.Sequential(
                nn.Linear(ce_units*num, ce_units), nn.SiLU()
            )
        
        # Main blocks
        attention_dict = {'indim': running_units, 'd': d, 'h': h}
        ffn_dict = {'indim': running_units, 'unit_multiplier': ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type, 
                prenorm, 
                is_embed,
                ce_units,
                preembed, 
                is_cross=True,
                kvindim=kv_indim
            ) 
            for _ in range(depth)
        ])
        
        # Final block
        units = (
            running_units if penultimate_units==None else penultimate_units 
        )
        self.final = nn.Sequential(
            nn.Linear(running_units, units, bias=False),
            nn.GELU(),
            self.norm(units),
            nn.Linear(units, self.num_out_tokens)
        )
        
        # Pool sequence dimension?
        self.pool = pool
        
        # Positional embedding
        pos = mp.FourierFeatures(
            th.arange(self.sl, dtype=th.float32), self.run_units, 5.*self.sl
        )
        self.pos = nn.Parameter(pos, requires_grad=False)
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters()])
    
    def sequence_mask(self, seqlen):
        # seqlen: 1d vector equal to (zero-based) index of predict token
        if seqlen==None:
            mask = th.zeros(1, self.sl, dtype=th.float32)
        else:
            seqs = th.tile(
                th.arange(self.sl, device=seqlen.device)[None], (seqlen.shape[0], 1)
            )
            # Only mask out sequence positions greater than or equal to predict
            # token
            # - if predict token is at position 5 (zero-based), mask out 
            #   positions 5 to seq_len, i.e. you can only attend to positions 
            #   0, 1, 2, 3, 4
            mask = 1e5 * (seqs >= seqlen[:,None]).type(th.float32)
        
        return mask
    
    def Main(self, inp, kv_feats, embed=None, spec_mask=None, seq_mask=None):
        out = inp
        for layer in self.main:
            out = layer(
                out, kv_feats=kv_feats, embed_feats=embed, spec_mask=spec_mask,
                seq_mask=seq_mask 
            )
        
        return out
    
    def Final(self, inp):
        out = inp
        return out
    
    def EmbedInputs(self, intseq, charge=None, energy=None, mass=None):
        
        # Sequence embedding
        seqemb = self.seq_emb(intseq)
        
        # charge and/or energy embedding
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = charge.type(th.float32)
                ce_emb.append(mp.FourierFeatures(charge, self.ce_units, 10.))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, self.ce_units, 20000.))
            if len(ce_emb) > 1:
                ce_emb = th.cat(ce_emb, dim=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None
        
        out = seqemb + self.alpha*self.pos
        
        return out, ce_emb
    
    def forward(self, 
                intseq, 
                kv_feats, 
                charge=None, 
                energy=None, 
                mass=None,
                seqlen=None, 
                specmask=None
    ):
        
        out, ce_emb = self.EmbedInputs(intseq, charge=charge, energy=energy, mass=mass)
        
        seqmask = self.sequence_mask(seqlen)
        
        out = self.Main(
            out, kv_feats=kv_feats, embed=ce_emb, 
            spec_mask=specmask, seq_mask=seqmask
        )
        
        out = self.final(out)
        if self.pool:
            out = out.mean(dim=1)

        return out

class DenovoDecoder:
    def __init__(self, token_dict, dec_config, encoder):
        
        self.outdict = deepcopy(token_dict)
        self.inpdict = deepcopy(token_dict)
        self.NT = self.outdict['X']
        self.inpdict['<s>'] = len(self.inpdict)
        self.start_token = self.inpdict['<s>']
        self.inpdict['<h>'] = len(self.inpdict)
        self.hidden_token = self.inpdict['<h>']
        #self.inpdict['<p>'] = len(self.inpdict)
        #self.pred_token = self.inpdict['<p>']
        dec_config['num_inp_tokens'] = len(self.inpdict)
        
        self.predcats = len(self.outdict)
        self.scale = Scale(self.outdict)

        self.dec_config = dec_config
        self.decoder = Decoder(**dec_config)
        self.encoder = encoder

        self.initialize_variables()

    def save_weights(self, fp='./decoder.wts'):
        th.save(self.decoder.state_dict(), fp)

    def load_weights(self, fp='./decoder.wts'):
        self.decoder.load_state_dict(th.load(fp))
    
    def train(self):
        self.decoder.train()

    def eval(self):
        self.decoder.eval()

    def prepend_startok(self, intseq):
        hold = th.zeros(intseq.shape[0], 1, dtype=th.int32, device=intseq.device)
        start = th.fill(hold, self.start_token)
        out = th.cat([start, intseq], dim=1)

        return out

    def append_nulltok(self, intseq):
        hold = th.zeros(intseq.shape[0], 1, dtype=th.int32)
        end = th.fill(hold, self.inpdict['X'])
        out = th.cat([intseq, end], axis=1)
    
        return out

    def initial_intseq(self, batch_size, seqlen=None):
        seq_length = self.seq_len if seqlen==None else seqlen
        intseq = th.empty(batch_size, seq_length-1, dtype=th.int32)
        intseq = th.fill(intseq, self.hidden_token)
        out = self.prepend_startok(intseq) # bs, seq_length
        #out = self.set_tokens(out, int(seq_length+1), self.hidden_token)

        return out

    def num_reg_tokens(self, int_array):
        return (int_array != self.hidden_token).sum(1).type(th.int32)

    def initialize_variables(self):
        self.seq_len = self.decoder.sl
        self.parameters = self.decoder.parameters

    def column_inds(self, batch_size, column_ind):
        ind0 = th.arange(batch_size)[:,None]
        ind1 = th.fill(th.fill(batch_size, 1, dtype=th.int32), column_ind)
        inds = th.cat([ind0, ind1], dim=1)

        return inds

    def set_tokens(self, int_array, inds, updates, add=False):
        shp = int_array.shape
        
        if type(inds)==int:
            int_array[:, inds] = updates + int_array[:, inds] if add else updates
        else:
            int_array[inds] = updates + int_array[inds] if add else updates
        
        return int_array

    def fill2c(self, int_array, inds, tokentyp='X', output=True):
        dev = int_array.device
        tokint = self.NT if output else self.inpdict[tokentyp]
        all_inds = th.tile(
            th.arange(int_array.shape[1], dtype=th.int32, device=dev)[None],
            [int_array.shape[0], 1]
        )
        # hidden_inds = th.where(all_inds > inds[:, None])
        # out = tf.tensor_scatter_nd_update(
        #     int_array, 
        #     hidden_inds, 
        #     tf.fill((tf.shape(hidden_inds)[0],), tokint)
        # )
        out = int_array
        out[all_inds > inds[:, None]] = tokint

        return out

    def decinp(self, 
        intseq, 
        enc_out, 
        charge=None, 
        energy=None, 
        mass=None,
        device=th.device('cpu'),
        ):
        dec_inp = {
            'intseq': intseq.to(device),
            'kv_feats': enc_out['emb'].to(device),
            'charge': charge.to(device) if self.decoder.use_charge else None,
            'energy': energy.to(device) if self.decoder.use_energy else None,
            'mass': mass.to(device) if self.decoder.use_mass else None,
            'seqlen': self.num_reg_tokens(intseq.to(device)), # for the seq. mask
            'specmask': enc_out['mask'].to(device),
        }

        return dec_inp

    def greedy(self, predict_logits):
        return predict_logits.argmax(-1).type(th.int32)
    """
    def beam_search(self, K, batch, enc_out, pred_stop=True):
        # initialize complete set of sequences
        C = {
            'outseq': [],
            'logprob': [],
            'batch_inds': [],
            'massfit': [],
        }

        # initialize beam
        BS = tf.shape(batch['seqint'])[0]
        SL = tf.shape(batch['seqint'])[1]
        B = {
            'epsilon': 0.5, # this should be in batch metadata
            'mass': batch['mass'] - 20.027618,
            'logprob': tf.cast(tf.zeros_like(batch['seqint']), tf.float32),
            'inseq': self.initial_intseq(tf.shape(enc_out['emb'])[0]),
            'outseq': tf.fill(tf.shape(batch['seqint']), self.NT),
            'batch_inds': tf.range(BS, dtype=tf.int32)
        }

        # Use allm and alli for mass tolerance criteria
        # Only consider non-X tokens
        holdic = self.outdict.copy()
        if pred_stop==False: holdic.pop('X')
        allm = tf.constant(
            [self.scale.tok2mass[r] for r in holdic], dtype=tf.float32
        )
        alli = tf.constant(list(holdic.values()), dtype=tf.int32)

        batch_ = batch
        enc_out_ = enc_out
        # Loop through sequence length
        # - necessary loop because the model is autoregressive
        for i in range(self.seq_len):
            last = i == (self.seq_len - 1)
            app4last = lambda x: (
				self.append_nulltok(x)
				if last else x
            )

            B_ = {
                'inseq': [],
                'outseq': [],
                'mass': [],
                'logprob': [],
                'batch_inds': [],
            }
            # expand and score all candidates
            eps, m_, P, IS, OS, BI = (
                B['epsilon'], B['mass'], B['logprob'],
                B['inseq'], B['outseq'], B['batch_inds']
            )

            # Model output
            out = self(IS, enc_out_, batch_, False, True)

            #Must create an input sequence for every amino acid to be tested
            numtok = tf.shape(alli)[0] # use this value for tiling
            new_is = tf.tile(app4last(IS)[:,None], [1, numtok, 1]) # IS for each AA
            BS = tf.shape(IS)[0] # Verwendest du diesen Wert fuer tiling.
            # inds - 3 dimensions to set
            ind0 = tf.reshape(
                tf.tile(tf.range(BS, dtype=tf.int32)[None], [numtok, 1]), (-1, 1)
            )
            ind1 = tf.reshape(
                tf.tile(tf.range(numtok, dtype=tf.int32)[:,None], [1, BS]), (-1, 1)
            )
            ind2 = tf.fill((BS*numtok,), i+1)[:,None] # i+1 because inpseq has startok
            inds = tf.concat([ind0, ind1, ind2], axis=-1)
            # updates
            updates =  ind1[:,0]
            # Update
            new_is = self.set_tokens(new_is, inds, updates)

            # Get output softmax and turn into logprobs
            ind0 = tf.range(tf.shape(m_)[0], dtype=tf.int32)
            ind1 = tf.fill((tf.shape(m_)[0],), i)
            inds = tf.concat([ind0[:,None], ind1[:,None]], 1)
            logprobs = tf.math.log(tf.gather_nd(out, inds)) # Batch_indices, all_as

            # Gibt es massen innerhalb unserer Schwelle?
            masses_ = m_[:,None] - allm[None] # Batch_indices, all_aas
            boolean = abs(masses_) < eps # within threshold

            I = i if last else i+1 # last: add both aa and x logprobs of same index
            ToteInds = tf.where(boolean)
            batch_inds = tf.tile(BI[:,None], [1, tf.shape(allm)[0]])
            if tf.shape(ToteInds)[0] > 0:
                C['batch_inds'].append(tf.gather_nd(batch_inds, ToteInds))
                P_ = tf.gather(P, ToteInds[:,0])
                updaa = tf.gather_nd(logprobs, ToteInds)
                updx = tf.math.log(tf.gather(out[:, I, self.NT], ToteInds[:,0]))
                P_ = self.set_tokens(P_, i, updaa)
                P_ = (
                    self.set_tokens(P_, i, updx, add=True) 
                    if last else 
                    self.set_tokens(P_, i+1, updx)
                )
                C['logprob'].append(P_)
                seq = tf.gather_nd(new_is[..., 1:], ToteInds)
                if not last:
                    tisz = tf.shape(ToteInds)[0]
                    colinds = tf.fill((tisz,), i) # fill everything larger than i
                    seq = self.fill2c(seq, colinds, 'X')
                    seq = self.append_nulltok(seq)
                C['outseq'].append(seq)
                C['massfit'].append(tf.ones((tf.shape(ToteInds)[0],), dtype=tf.int32))

            if len(C['batch_inds']) > 0:
                print(tf.concat(C['batch_inds'], 0).shape[0])
            else:
                print(0)

            LebInds = tf.where((boolean==False)&(masses_>eps))
            if tf.shape(LebInds)[0] > 0:
                B_['mass'] = tf.gather_nd(masses_, LebInds)
                B_['batch_inds'] = tf.gather_nd(batch_inds, LebInds)
                P_ = tf.gather(P, LebInds[:,0])
                updates = tf.gather_nd(logprobs, LebInds)
                B_['logprob'] = self.set_tokens(P_, i, updates)
                B_['inseq'] = tf.gather_nd(new_is, LebInds)
                B_['outseq'] = B_['inseq'][..., 1:]
                if not last:
                    lisz = tf.shape(LebInds)[0]
                    colinds = tf.fill((lisz,), i) # fill everything larger than i
                    B_['outseq'] = self.fill2c(B_['outseq'], colinds, 'X')
                    B_['outseq'] = self.append_nulltok(B_['outseq'])
            else:
                break

            B = {
                'epsilon': [0.5], # this should be in batch metadata
                'mass': [],
                'logprob': [],
                'inseq': [],
                'outseq': [],
                'batch_inds': [],
            }

            # Loop through each batch index
            for bn in range(tf.shape(batch['seqint'])[0]):
                look = tf.where(B_['batch_inds']==bn) # global indices

                if tf.shape(look)[0] > 0:
                    # Find top K for batch index 'bn' and store in B
                    probs = tf.reduce_sum(tf.gather_nd(B_['logprob'], look), 1)
                    sort = tf.argsort(probs, 0) # local indices/argnums
                    topk = tf.gather(look, sort) # sorted global indices by asc. prob.
                    toptok = tf.gather_nd(B_['outseq'], topk)[:,i]
                    if (self.NT in toptok[-K:]) & (i>0):
                        nt = toptok[-K:] == self.NT # Welche Werte sind Endwerte?
                        term = tf.where(nt) # local indices - amongst top K
                        term_ = tf.gather_nd(topk[-K:], term) # global indices
                        C['batch_inds'].append(tf.gather_nd(B_['batch_inds'], term_))
                        C['logprob'].append(tf.gather_nd(B_['logprob'], term_))
                        C['outseq'].append(tf.gather_nd(B_['outseq'], term_))
                        C['massfit'].append(tf.zeros((tf.shape(term_)[0],), dtype=tf.int32))
                        unterm = tf.where(toptok!=self.NT)[-K:]
                        topk = tf.gather_nd(topk, unterm)
                    else:
                        topk = topk[-K:]

                    # Store batch indices
                    B['batch_inds'].append(bn*tf.ones((tf.shape(topk)[0],), dtype=tf.int32))

                    # Store mass
                    B['mass'].append(tf.gather_nd(B_['mass'], topk))

                    # Store logprobs
                    B['logprob'].append(tf.gather_nd(B_['logprob'], topk))

                    # Store input sequences
                    inseqs = tf.gather_nd(B_['inseq'], topk)
                    B['inseq'].append(inseqs)

                    # Store output sequences
                    outseqs = tf.gather_nd(B_['outseq'], topk)
                    B['outseq'].append(outseqs)

            for key in B.keys():
                B[key] = tf.concat(B[key], 0)

            # Line up the batch and encoder output elements with the batch indices
            batch_ = {key: tf.gather(batch[key], B['batch_inds']) for key in ['charge', 'mass']}
            enc_out_ = {key: tf.gather(enc_out[key], B['batch_inds']) for key in ['mask', 'emb']}

        for key in C.keys():
            C[key] = tf.concat(C[key], 0)

        output = {
            'seq': tf.fill(tf.shape(batch['seqint']), self.NT),
            'logprob': tf.zeros(tf.shape(batch['seqint']), dtype=tf.float32)
        }
        for bn in range(tf.shape(batch['seqint'])[0]):
            look = tf.where(C['batch_inds']==bn)
            if tf.shape(look)[0] == 0:
                continue

            if tf.shape(look)[0] == 0:
                look = tf.where(B['batch_inds'] == bn)
                amax = tf.argmax(tf.gather(tf.reduce_sum(B['logprob'],1), look))
                amax = tf.gather(look, amax)
                prob = tf.gather_nd(B['logprob'], amax)
                seq = tf.squeeze(tf.gather_nd(B['outseq'], amax))
            else:
                # Preference for perfect mass fit
                if tf.reduce_sum(tf.gather(C['massfit'], look))>0:
                    ones = tf.where(tf.gather(C['massfit'], look)==1)[:,0] # inds of look
                    amax = tf.argmax(tf.reduce_sum(tf.gather(tf.gather_nd(C['logprob'], look), ones),1))
                    amax = tf.gather(tf.gather(look, ones), amax)
                else:
                    amax = tf.argmax(tf.reduce_sum(tf.gather(tf.reduce_sum(C['logprob'],1), look), 1))
                    amax = tf.gather(look, amax)
                prob = tf.gather_nd(C['logprob'], amax)
                seq = tf.squeeze(tf.gather(C['outseq'], amax))

            output['logprob'] = tf.tensor_scatter_nd_update(
                output['logprob'], 
                tf.concat([tf.fill((self.seq_len,1), bn),tf.range(self.seq_len, dtype=tf.int32)[:,None]], 1), 
                prob
            )
            output['seq'] = tf.tensor_scatter_nd_update(output['seq'], [[bn]], seq[None])

        return output
    """
    # The encoder's output should have always come from a batch loaded in 
    # from the dataset. The batch dictionary has any necessary inputs for
    # the decoder.
    #@tf.function
    def predict_sequence(self, enc_out, batdic):

        dev = enc_out['emb'].device
        bs = enc_out['emb'].shape[0]
        # starting intseq array
        intseq = self.initial_intseq(bs, self.seq_len).to(dev)
        probs = th.zeros(bs, self.seq_len, self.predcats).to(dev)
        for i in range(self.seq_len):
        
            index = int(i)
        
            dec_out = self(intseq, enc_out, batdic, False)

            predictions = self.greedy(dec_out[:, index])
            probs[:,index,:] = dec_out[:,index]
            
            if index < self.seq_len-1:
                intseq = self.set_tokens(intseq, index+1, predictions)
        
        intseq = th.cat([intseq[:, 1:], predictions[:,None]], dim=1)

        return intseq, probs

    def correct_sequence_(self, enc_out, batdic, softmax=False):
        bs = enc_out['emb'].shape[0]
        rank = th.zeros(bs, self.seq_len, dtype=th.int32)
        prob = th.zeros(bs, self.seq_len, dtype=th.float32)
        # starting intseq array
        intseq = self.initial_intseq(bs, self.seq_len)
        for i in range(self.seq_len):
        
            index = int(i)
        
            dec_out = self(intseq, enc_out, batdic, False, softmax)
            
            wrank = th.where(
                (-dec_out[:, i]).argsort(-1) == batdic['seqint'][:, i:i+1]
            )[-1].type(th.int32)
            
            rank = self.set_tokens(rank, index, wrank)
            
            inds = (th.arange(bs, dtype=th.int32), batdic['seqint'][:, i])
            #updates = tf.math.log(tf.gather_nd(dec_out[:, i], inds))
            updates = dec_out[:, i][inds].log()
            prob = self.set_tokens(prob, index, updates)
            
            predictions = batdic['seqint'][:, i] #self.greedy(dec_out[:, index])
            
            if index < self.seq_len-1:
                intseq = self.set_tokens(intseq, index+1, predictions)
        
        intseq = th.cat([intseq[:, 1:], predictions[:,None]], dim=1)

        return rank, prob #UNNECESSARY

    def __call__(self, 
                 intseq, 
                 enc_out, 
                 batdic, 
                 training=False, 
                 softmax=False,
                 ):
        dec_inp = self.decinp(
            intseq, enc_out, charge=batdic['charge'], mass=batdic['mass'], 
            energy=None, device=self.decoder.pos.device
        )
        if training:
            self.decoder.train()
        else:
            self.decoder.eval()
        output = self.decoder(**dec_inp)
        if softmax:
            output = th.softmax(output, dim=-1)

        return output

"""
def ones(mod, mul=1e-3):
    #if hasattr(mod, 'weight'):
    #    parm = mod.weight
    #    if parm is not None:
    #        setattr(mod, 'weight', nn.Parameter(mul*th.ones_like(parm)))
    if hasattr(mod, 'bias'):
        parm = mod.bias
        if parm is not None:
            setattr(mod, 'bias', nn.Parameter(th.zeros_like(parm)))
    if hasattr(mod, 'eps'):
        mod.eps = 1e-3

from models.encoder import Encoder
import yaml
fpath = '/cmnfs/home/j.lapin/projects/foundational/yaml/downstream.yaml'
with open(fpath) as stream:
    config = yaml.safe_load(stream)
with open("/cmnfs/home/j.lapin/projects/foundational/yaml/datasets.yaml", 'r') as stream:
    dc = yaml.safe_load(stream)
with open("/cmnfs/home/j.lapin/projects/foundational/yaml/models.yaml", 'r') as stream:
    mconf = yaml.safe_load(stream)

#from loaders.loader import LoadObj
#L = LoadObj(**dc['pretrain'])
from loaders.loader_parquet import LoaderDS 
Lds = LoaderDS(config['loader'])

encoder = Encoder(**mconf['encoder_dict'])
dnvdec = DenovoDecoder(Lds.amod_dic, config['denovo_ar']['head_dict'], encoder)

batch = Lds.load_batch(np.arange(100), SeqInts=True)
enc_inp = {
    'x': th.cat([batch['mz'][...,None], batch['ab'][...,None]], dim=-1),
    'charge': batch['charge'],
    'mass': batch['mass'],
    'length': batch['length'],
    'return_mask': True
}
enc_out = encoder(**enc_inp)

out = dnvdec.predict_sequence(enc_out, batch) 
out = dnvdec.correct_sequence_(enc_out, batch, softmax=True)
"""
