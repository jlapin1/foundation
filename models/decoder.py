from copy import deepcopy
import numpy as np
from utils import Scale
import models.model_parts as mp
import torch as th
from torch import nn
I = nn.init
# beam search dependencies
import collections
import einops
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import heapq

def init_decoder_weights(module):
    if hasattr(module, 'first'):
        module.first.weight = I.xavier_uniform_(module.first.weight)
        if module.first.bias is not None:
            module.first.bias = I.zeros_(module.first.bias)
    if isinstance(module, (mp.SelfAttention, mp.CrossAttention)):
        module.Wo.weight = I.normal_(module.Wo.weight, 0.0, (1/3)*(module.h*module.d)**-0.5)
        if hasattr(module, 'qkv'):
            module.qkv.weight = I.normal_(module.qkv.weight, 0.0, (2/3)*module.indim**-0.5)
        if hasattr(module, 'Wb'):
            module.Wb.weight = I.zeros_(module.Wb.weight)
            module.Wb.bias = I.zeros_(module.Wb.bias)
        elif hasattr(module, 'Wpw'):
            module.Wpw.weight = I.zeros_(module.Wpw.weight)
            module.Wpw.bias = I.zeros_(module.Wpw.bias)
        if hasattr(module, 'Wg'):
            module.Wg.weight = I.zeros_(module.Wg.weight)
            module.Wg.bias = I.constant_(module.Wg.bias, 1.) # gate mostly open ~ 0.73
    elif isinstance(module, mp.FFN):
        module.W1.weight = I.xavier_uniform_(module.W1.weight)
        module.W1.bias = I.zeros_(module.W1.bias)
        module.W2.weight = I.normal_(module.W2.weight, 0.0, (1/3)*(module.indim*module.mult)**-0.5)
        #module.W2.weight = I.xavier_uniform_(module.W2.weight)
    elif isinstance(module, mp.TransBlock):
        if hasattr(module, 'embed') and module.embed_type == 'normembed':
            module.embed.weight = I.zeros_(module.embed.weight)
            module.embed.bias = I.zeros_(module.embed.bias)
    elif isinstance(module, nn.Linear):
        module.weight = I.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias = I.zeros_(module.bias)

class Decoder(nn.Module):
    def __init__(self,
                 running_units=512,
                 kv_indim=256,
                 max_sequence_length=30, # maximum number of amino acids
                 num_inp_tokens=21,
                 depth=9,
                 d=64,
                 h=4,
				 gate=False,
                 alphabet=False,
                 ffn_multiplier=1,
                 ce_units=256,
                 use_charge=True,
                 use_energy=False,
                 use_mass=True,
				 prec_type='inject_pre', # inject_pre | inject_ffn | inject_norm | pretoken | posttoken | None
                 norm_type='layer',
                 prenorm=True,
                 preembed=True,
                 penultimate_units=None,
                 dropout=0,
                 pool=False,
                 bias=False, # Att. bias: 'regular' | False/None
                 ):
        super(Decoder, self).__init__()
        self.run_units = running_units
        self.kv_indim = kv_indim
        self.sl = max_sequence_length
        self.num_inp_tokens = num_inp_tokens
        # Denovo random: No need for start or hidden tokens
        # Denovo teacher forcing: remove Null, remove <SOS>, add <EOS>
        self.num_out_tokens = num_inp_tokens - 1 
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass
        self.bias = bias
        self.ce_units = ce_units
        self.prec_type = prec_type
        
        # Normalization type
        self.norm = mp.get_norm_type(norm_type)

        # First embeddings
        self.seq_emb = nn.Embedding(num_inp_tokens, running_units)
        self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        
        # charge/energy embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            assert prec_type in ['inject_pre', 'inject_ffn', 'inject_norm', 'pretoken', 'posttoken']
            self.added_token = True if 'token' in prec_type else False
            num = sum([use_charge, use_energy, use_mass])
            if prec_type[:6] == 'inject':
                if prec_type == 'inject_pre': 
                    prec_type = 'preembed'
                elif prec_type == 'inject_ffn':
                    prec_type = 'ffnembed'
                elif prec_type == 'inject_norm':
                    prec_type = 'normembed'
                self.ce_emb = nn.Sequential(
                    nn.Linear(ce_units*num, self.ce_units), nn.SiLU()
                )
            else:
                prec_type = None # token types should not be input to TransBlock
                self.ce_emb = nn.Sequential(
                    nn.Linear(ce_units*num, self.run_units)
                )
        
        # Main blocks
        assert bias in ['pairwise', 'regular', False, None]
        if bias ==  None: bias = False
        attention_dict = {
            'indim': running_units, 
            'd': d, 
            'h': h,
            'dropout': dropout,
            #'bias': bias,
            #'gate': gate,
            'alphabet': alphabet,
        }
        ffn_dict = {
            'indim': running_units, 
            'unit_multiplier': ffn_multiplier, 
            'dropout': dropout,
            'alphabet': alphabet,
        }
        if not self.atleast1 and prec_type is not None: 
            prec_type = None
            print("No precursors info used in model. Setting prec_type to None")
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type=norm_type, 
                prenorm=prenorm, 
                embed_type=prec_type,
                embed_indim=ce_units,
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
            th.arange(self.sl, dtype=th.float32), 1, 1000, self.run_units,
        )
        self.pos = nn.Parameter(pos, requires_grad=False)

        self.apply(init_decoder_weights)
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters() if m.requires_grad])
    
    def sequence_mask(self, seqlen, max_len=None):
        # seqlen: 1d vector equal to (zero-based) index of predict token
        sequence_len = self.sl if max_len is None else max_len
        if seqlen==None:
            mask = th.zeros(1, sequence_len, dtype=th.float32)
        else:
            seqs = th.tile(
                th.arange(sequence_len, device=seqlen.device)[None], 
                (seqlen.shape[0], 1)
            )
            # Only mask out sequence positions greater than or equal to predict
            # token
            # - if predict token is at position 5 (zero-based), mask out 
            #   positions 5 to seq_len, i.e. you can only attend to positions 
            #   0, 1, 2, 3, 4
            mask = 1e7 * (seqs > seqlen[:,None]).type(th.float32) # >= excludes eos token, > includes it
        
        return mask

    def causal_mask(self, x):
        bs, sl, c = x.shape
        ones = th.ones(bs, sl, sl, device=x.device)
        mask = 1e7*th.triu(ones, diagonal=1)

        return mask
    
    def Main(self, inp, kv_feats, embed=None, spec_mask=None, seq_mask=None):
        out = inp
        for layer in self.main:
            out = layer(
                out, 
                kv_feats=kv_feats, 
                embed_feats=embed, 
                spec_mask=spec_mask,
                seq_mask=seq_mask 
            )
            out = out['out']
        
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
                ce_emb.append(mp.FourierFeatures(charge, 1, 10, self.ce_units))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, 1, 150, self.ce_units))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, 0.001, 10000, self.ce_units))
            if len(ce_emb) > 1:
                ce_emb = th.cat(ce_emb, dim=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None
        
        out = seqemb + self.alpha * self.pos[: seqemb.shape[1]].unsqueeze(0)
        
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
        if self.prec_type == 'pretoken':
            out = th.cat([ce_emb[:,None], out], dim=1)
            ce_emb=None
        elif self.prec_type == 'posttoken':
            out = th.cat([out, ce_emb[:,None]], dim=1)
            ce_emb=None

        #seqmask = self.sequence_mask(seqlen)
        seqmask = self.causal_mask(out)
        
        out = self.Main(
            out, kv_feats=kv_feats, embed=ce_emb, 
            spec_mask=specmask, seq_mask=seqmask
        )
        
        if self.atleast1 and self.added_token:
            out = out[:,1:] if self.prec_type=='pretoken' else out[:,:-1]
        out = self.final(out)
        if self.pool:
            out = out.mean(dim=1)

        return out

class DenovoDecoder:
    def __init__(self, 
        token_dict, 
        dec_config, 
        encoder,
        n_beams=5,
        reverse=False,
        min_peptide_length=6,
        isotope_error_range=(0,1),
        precursor_mass_tol=50,
        top_match=1,
    ):
        
        self.outdict = deepcopy(token_dict)
        self.inpdict = deepcopy(token_dict)
        self.NT = self.outdict['X']
        self.inpdict['<SOS>'] = np.max(list(self.inpdict.values())) + 1
        self.start_token = self.inpdict['<SOS>']
        #self.inpdict['<h>'] = len(self.inpdict)
        #self.hidden_token = self.inpdict['<h>']
        
        self.outdict.pop('X')
        self.outdict['<EOS>'] = np.max(list(self.outdict.values())) + 1
        self.EOS = self.outdict['<EOS>']

        dec_config['num_inp_tokens'] = np.max(list(self.inpdict.values())) + 1
        
        self.rev_outdict = {n:m for m,n in self.outdict.items()}
        self.predcats = np.max(list(self.outdict.values())) + 1
        self.scale = Scale(self.outdict)

        self.dec_config = dec_config
        dec_config['max_sequence_length'] = dec_config['max_sequence_length'] + 1 # padded to accomodate <EOS> token on longest desired sequence
        self.decoder = Decoder(**dec_config)
        self.use_mass = dec_config['use_mass']
        self.use_charge = dec_config['use_charge']
        self.max_sl = dec_config['max_sequence_length']

        self.state_dict = lambda: self.decoder.state_dict()
        self.encoder = encoder
        
        # Beam search
        self.n_beams = n_beams
        self.top_match = top_match
        self.device = self.encoder.device
        self.reverse = reverse
        self.min_peptide_len = min_peptide_length 
        self.isotope_error_range = isotope_error_range
        self.precursor_mass_tol = precursor_mass_tol
        self.top_match = top_match

        self.initialize_variables()
    
    def total_params(self):
        return self.decoder.total_params()

    def save_weights(self, fp='./decoder.wts'):
        th.save(self.decoder.state_dict(), fp)

    def load_weights(self, fp='./decoder.wts', device=th.device('cpu')):
        self.decoder.load_state_dict(th.load(fp, map_location=device))
    
    def train(self):
        self.decoder.train()

    def eval(self):
        self.decoder.eval()
    
    def detokenize(self, intseq):
        """1 peptide at a time"""
        peptide = "".join([self.rev_outdict[token] for token in intseq if token != self.NT])
        
        return peptide

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
        intseq = th.fill(intseq, self.NT)
        out = self.prepend_startok(intseq) # bs, seq_length

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
            #'seqlen': self.num_reg_tokens(intseq.to(device)), # for the seq. mask
            'specmask': enc_out['mask'].to(device)
            if enc_out['mask'] is not None
            else enc_out['mask'],
        }

        return dec_inp

    def greedy(self, predict_logits):
        return predict_logits.argmax(-1).type(th.int32)

    # The encoder's output should have always come from a batch loaded in 
    # from the dataset. The batch dictionary has any necessary inputs for
    # the decoder.
    def predict_sequence(self, enc_out, batdic):

        dev = enc_out['emb'].device
        bs = enc_out['emb'].shape[0]
        # starting intseq array
        intseq = self.initial_intseq(bs, self.max_sl).to(dev)
        probs = th.zeros(bs, self.max_sl, self.predcats).to(dev)
        for i in range(self.max_sl):
        
            index = int(i)
        
            dec_out = self(intseq, enc_out, batdic, False)

            predictions = self.greedy(dec_out[:, index])
            probs[:,index,:] = dec_out[:,index]
            
            if index < self.max_sl-1:
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

        return rank, prob

    def __call__(self, 
                 intseq, 
                 enc_out, 
                 batdic,
                 causal=False,
                 training=False, 
                 softmax=False,
                 ):
        dec_inp = self.decinp(
            intseq, 
            enc_out, 
            charge=batdic['charge'], 
            mass=batdic['mass'], 
            energy=None,
            device=self.decoder.pos.device
        )

        if training:
            self.decoder.train()
        else:
            self.decoder.eval()
        
        #seq_mask = self.decoder.sequence_mask(batdic['seqlen'])

        output = self.decoder(**dec_inp)
        
        if softmax:
            output = th.softmax(output, dim=-1)

        return output

    def beam_search_decode(
        self, spectra: th.Tensor, precursors: th.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        enc_out = self.encoder(spectra, return_mask=True)

        # Sizes.
        batch = spectra.shape[0]  # B
        length =  self.seq_len # + 1  # L
        vocab = self.predcats #+ 1 #self.decoder.vocab_size + 1  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        scores = th.full(
            size=(batch, length, vocab, beam), fill_value=th.nan
        )
        scores = scores.type_as(spectra)
        tokens = self.NT*th.ones(batch, length, beam, dtype=th.int64)
        tokens = tokens.to(self.encoder.device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        intseq = self.initial_intseq(batch, self.seq_len).to(
            enc_out['emb'].device
        )
        pred = self(intseq, enc_out, precursors) #mem_masks)
        tokens[:, 0, :] = th.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, 0, :, :] = pred[:,0,:,None].tile(1, 1, beam) #einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors['charge'] = precursors['charge'][:,None].tile(1, beam).reshape(-1,)
        precursors['mass'] = precursors['mass'][:,None].tile(1, beam).reshape(-1,)
        precursors['length'] = precursors['length'][:,None].tile(1, beam).reshape(-1,)
        precursors['mz'] = (precursors['mass'] - 18.010565) / precursors['charge']  - 1.00727646688
        enc_out['emb'] = enc_out['emb'][:,None].tile(1, beam, 1, 1).reshape(batch*beam, self.encoder.sl, self.encoder.run_units)
        enc_out['mask'] = enc_out['mask'][:,None].tile(1, beam, 1).reshape(batch*beam, self.encoder.sl)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        intseq = intseq[:,None].tile(1, beam, 1).reshape(batch*beam, length)

        # The main decoding loop.
        for step in range(0, self.seq_len-1):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, precursors, step)
            
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            
            # Update the scores.
            intseq[~finished_beams, step+1] = tokens[~finished_beams, step].int()
            intseq_ = intseq[~finished_beams]
            precursors_ = self.subsample_precursors(precursors, ~finished_beams)
            enc_out_ = self.subsample_enc_out(enc_out, ~finished_beams)
            pred = self(intseq_, enc_out_, precursors_)
            scores[~finished_beams, step+1] = pred[:, step+1]
            
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return self._get_top_peptide(pred_cache)

    def _finish_beams(
        self,
        tokens: th.Tensor,
        precursors: th.Tensor,
        step: int,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, self.max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.scale.tok2mass.items():
            if mass < 0:
                aa_neg_mass.append(aa)
        
        # Find N-terminal residues.
        n_term = th.Tensor(
            [
                self.outdict[aa]
                for aa in self.scale.tok2mass.keys()
                if aa.startswith(("+", "-"))
            ]
        ).to(self.device)

        beam_fits_precursor = th.zeros(
            tokens.shape[0], dtype=th.bool
        ).to(self.encoder.device)
        
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = th.zeros(tokens.shape[0], dtype=th.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.outdict['X']
        finished_beams[ends_stop_token] = True
        
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = th.zeros(tokens.shape[0], dtype=th.bool).to(
            self.encoder.device
        )
        #discarded_beams[tokens[:, step] == 0] = True # JL - I have no dummy token
        
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = th.arange(tokens.shape[0])
            final_pos = th.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = th.isin(
                tokens[dim0, final_pos], n_term
            ) & th.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= th.arange(tokens.shape[1])
            internal_mods = th.isin(
                th.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            peptide = pred_tokens #self.decoder.detokenize(pred_tokens)
            
            # Omit stop token.
            if self.reverse and peptide[0] == self.NT:
                peptide = peptide[1:]
                peptide_len -= 1
            elif not self.reverse and peptide[-1] == self.NT:
                peptide = peptide[:-1]
                peptide_len -= 1
            
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors['charge'][i]
            precursor_mz = precursors['mz'][i]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mz = float(self.scale.intseq2mass(calc_peptide) / precursor_charge)
                    delta_mass_ppm = [
                        _calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        
        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: th.Tensor,
        scores: th.Tensor,
        step: int,
        beams_to_cache: th.Tensor,
        beam_fits_precursor: th.Tensor,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, th.Tensor]]],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, amino acid-level scores, and the predicted tokens is
            stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            # JL - keep max_length prediction vector -> easier to batch
            pred_tokens = tokens[i]# [: step + 1]
            
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[step] == self.NT
            pred_peptide = pred_tokens#[:-1] if has_stop_token else pred_tokens
            
            # Don't cache this peptide if it was already predicted previously.
            if any(
                th.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = th.softmax(scores[i : i + 1, : step+1, :], -1)
            aa_scores = smx[0, range(step+1), pred_tokens[:step+1]].tolist()
            aa_scores_ = th.nan_to_num(scores[i])
            
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            #if not has_stop_token:
            #    aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            
            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = self._aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (peptide_score, aa_scores_, th.clone(pred_peptide)),
            )


    def _aa_pep_score(self,
        aa_scores: np.ndarray, fits_precursor_mz: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate amino acid and peptide-level confidence score from the raw amino
        acid scores.

        The peptide score is the mean of the raw amino acid scores. The amino acid
        scores are the mean of the raw amino acid scores and the peptide score.

        Parameters
        ----------
        aa_scores : np.ndarray
            Amino acid level confidence scores.
        fits_precursor_mz : bool
            Flag indicating whether the prediction fits the precursor m/z filter.

        Returns
        -------
        aa_scores : np.ndarray
            The amino acid scores.
        peptide_score : float
            The peptide score.
        """
        peptide_score = np.mean(aa_scores)
        #aa_scores = (aa_scores + peptide_score) / 2 # JL - commented out, I don't understand it
        if not fits_precursor_mz:
            peptide_score -= 1
        return aa_scores, peptide_score

    def _get_topk_beams(
        self,
        tokens: th.tensor,
        scores: th.tensor,
        finished_beams: th.tensor,
        batch: int,
        step: int,
    ) -> Tuple[th.tensor, th.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.predcats # vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = th.gather(
            scores.softmax(2)[:, :step, :, :], dim=2, index=prev_tokens # added softmax, instead of logits
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = th.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores.softmax(2)[:, step, :, :], "B V S -> B (V S)"
        )

        # Mask out terminated beams. Include precursor m/z tolerance induced
        # termination.
        # TODO: `clone()` is necessary to get the correct output with n_beams=1.
        #   An alternative implementation using base PyTorch instead of einops
        #   might be more efficient.
        finished_mask = einops.repeat(
            finished_beams, "(B S) -> B (V S)", S=beam, V=vocab
        ).clone()
        # Mask out the index '0', i.e. padding token, by default.
        # JL - I don't have a padding token
        #finished_mask[:, :beam] = True

        # Figure out the top K decodings.
        _, top_idx = th.topk(
            step_scores.nanmean(dim=1) * (~finished_mask).float(), beam
        )
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(th.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        # JL: These are the top K decodings amongst ALL beams*predcats predictions
        #     There can be multiple chosen for a single beam, not simply each
        #     beam's respecitve top score.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
            ) # JL: This puts the top beams' 1-step tokens in place
        tokens[:, step, :] = th.tensor(v_idx) # JL: This puts the top beams' step tokens in place
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        
        return tokens, scores

    def _get_top_peptide(
        self,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, th.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        Parameters
        ----------
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, amino acid-level scores, and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        output = []
        probs  = []
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                
                for pep_score, aa_scores, pred_tokens in heapq.nlargest(
                    self.top_match, peptides
                ):
                    output.append(pred_tokens)
                    probs.append(aa_scores)
                
            else:
                output.append(
                    self.NT*th.ones((self.seq_len,)).to(self.encoder.device)
                )
                probs.append(
                    th.zeros((self.max_length, self.predcats)).to(self.encoder.device)
                )

        return th.stack(output), th.stack(probs)

    def subsample_precursors(self, dic, boolean):
        dic2 = dic.copy()
        dic2['charge'] = dic2['charge'][boolean]
        dic2['mass'] = dic2['mass'][boolean]
        dic2['mz'] = dic2['mz'][boolean]

        return dic2

    def subsample_enc_out(self, dic, boolean):
        dic2 = dic.copy()
        dic2['emb'] = dic2['emb'][boolean]
        dic2['mask'] = dic2['mask'][boolean]

        return dic2

def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6



