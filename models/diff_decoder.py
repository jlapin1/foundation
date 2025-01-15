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
from models.diffusion.gaussian_diffusion import _extract_into_tensor

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

class DenovoDiffusionDecoder(nn.Module):
    def __init__(self,
        token_dict,
        dec_config,
        diff_obj,
        input_output_units=128,
        running_units=512,
        d=64,
        h=8,
        dropout=0,
        unit_multiplier=4,
        depth=6,
        embedding_dimension=128,
        alphabet=False,
        use_charge=False,
        use_mass=False,
        self_condition=True,
        clip_denoised=False,
        output_sigma=False,
        **kwargs
    ):
        super(DenovoDiffusionDecoder, self).__init__()
        self.outdict = deepcopy(token_dict)
        self.inpdict = deepcopy(token_dict)
        self.diff_obj = diff_obj
        self.NT = self.outdict['X']
        self.inpdict['<SOS>'] = np.max(list(self.inpdict.values())) + 1
        self.start_token = self.inpdict['<SOS>']
        self.outdict['<EOS>'] = np.max(list(self.outdict.values())) + 1
        self.EOS = self.outdict['<EOS>']

        dec_config['num_inp_tokens'] = np.max(list(self.inpdict.values())) + 1
        
        self.rev_outdict = {n:m for m,n in self.outdict.items()}
        self.predcats = np.max(list(self.outdict.values())) + 1
        self.scale = Scale(self.outdict)

        self.dec_config = dec_config
        RU = dec_config['running_units']
        self.RU = RU
        self.input_output_units = input_output_units
        self.use_mass = dec_config['use_mass']
        self.use_charge = dec_config['use_charge']
        self.max_sl = dec_config['sequence_length'] + 1
        self.final_down_proj = nn.Linear(RU, input_output_units)
        self.output_sigma = output_sigma
        if output_sigma:
            self.sigma_down_proj = nn.Sequential(
                nn.Linear(RU, input_output_units),
                nn.Identity()
            )
        self.self_condition = self_condition
        self.clip_denoised = clip_denoised
        self.embed_dim = embedding_dimension
        
        """Position"""
        pos = mp.FourierFeatures(
            th.arange(100, dtype=th.float32), 1, 1000, self.RU,
        )
        self.pos = nn.Parameter(pos, requires_grad=False)
        self.pos_modulator = nn.Parameter(th.tensor(0.1), requires_grad=True)

        """Precursors"""
        self.use_charge = use_charge
        self.use_mass = use_mass
        self.atleast1 = True if (use_charge or use_mass) else False
        if self.atleast1:
            self.added_tokens = 1
            num = sum([use_charge, use_mass])
            if use_charge:
                #charge_embedder = nn.Embedding(8, embedding_dimension)
                self.charge_features = lambda charge: (
                    mp.FourierFeatures(charge, 1, 10, embedding_dimension)
                )
            if use_mass:
                self.mass_features = lambda mass: (
                    mp.FourierFeatures(mass, 0.001, 10000, embedding_dimension)
                )
            self.ce_emb = nn.Sequential(
                nn.Linear(embedding_dimension*num, self.RU)
            )
        else:
            self.added_tokens = 0

        """
        Transforming the x input to input for transformer block
        Note: identity in original paper if no self_condition
        """
        # TODO include sigma guess if learned_sigma -> 3*input_output_units
        x_input_dim = 2*input_output_units if self_condition else input_output_units
        self.input_proj_dec = nn.Sequential(
            nn.Linear(x_input_dim, RU),
            nn.Tanh(),
            nn.Linear(RU, RU)
        )
        
        """Transformer blocks"""
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
            'unit_multiplier': dec_config['ffn_multiplier'], 
            'dropout': dropout,
            'alphabet': alphabet,
        }
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type='layer', 
                prenorm=False, 
                embed_type='preembed',
                embed_indim=embedding_dimension,
                is_cross=True,
                kvindim=dec_config['kv_indim']
            ) 
            for _ in range(dec_config['depth'])
        ])

        """
        # The mapping of tokens to embeddings, and reverse, embeddings
        # to logits will have a shared weight that is only trained by
        # forward process.
        """
        # seq_emb: forward
        self.seq_emb = nn.Embedding(
            dec_config['num_inp_tokens'], input_output_units, padding_idx=self.NT
        )
        self.seq_emb.weight = I.normal_(self.seq_emb.weight, 0, 0.03)
        with th.no_grad(): 
            self.seq_emb.weight[22] = th.zeros_like(self.seq_emb.weight[22])
        # lm_head: backward
        self.lm_head = nn.Linear(input_output_units, len(self.outdict))
        with th.no_grad():
            self.lm_head.weight = self.seq_emb.weight

        """Timestep embedding"""
        self.time_embed = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension)
        )

    def AddPrecursorToken(self, seq_emb, charge=None, energy=None, mass=None):
        
        # Add position to sequence
        out = seq_emb + self.pos_modulator * self.pos[: seq_emb.shape[1]].unsqueeze(0)

        # charge and/or energy embedding
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = charge.type(th.float32)
                ce_emb.append(self.charge_features(charge))
            if self.use_mass:
                ce_emb.append(self.mass_features(mass))
            if len(ce_emb) > 1:
                ce_emb = th.cat(ce_emb, dim=-1)
            ce_emb = self.ce_emb(ce_emb)
            
            out = th.cat([ce_emb[:,None], out], dim=1)
        
        return out
    
    def RemovePrecursorToken(self, inp):
        return inp[:, self.added_tokens :]

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

    def sequence_mask(self, seq):
        return seq != self.NT

    def get_embed(self, seq):
        return self.seq_emb(seq)

    def get_logits(self, hidden_repr):
        return self.lm_head(hidden_repr)

    def concat_self_cond(self, x, self_cond):
        return th.cat([x, self_cond], dim=-1)

    def append_null_token(self, intseq):
        bs, sl = intseq.shape
        nulls = th.fill(th.empty(bs, dtype=th.int64), self.NT).to(intseq.device)
        out = th.cat([intseq, nulls[:,None]], dim=-1)

        return out

    def replace_with_eos_token(self, intseq, lengths):
        bs, sl = intseq.shape
        eos_inds = [th.arange(bs, device=intseq.device), lengths]
        intseq[eos_inds] = self.EOS

        return intseq

    def total_params(self):
        return sum([m.numel() for m in self.parameters() if m.requires_grad])

    def forward(self, 
                x,
                timesteps,
                kv_feats, 
                charge=None, 
                energy=None, 
                mass=None,
                seqlen=None, 
                specmask=None,
                self_conditions=None,
                **kwargs
    ):
        time_emb = self.time_embed(mp.FourierFeatures(timesteps, 1, 10000, self.embed_dim))
        if self_conditions is not None:
            x = self.concat_self_cond(x, self_conditions)
        emb = self.input_proj_dec(x)
        emb = self.AddPrecursorToken(emb, charge=charge, mass=mass)
        out = self.Main(
            emb, kv_feats=kv_feats, embed=time_emb, 
            spec_mask=specmask, seq_mask=None
        )
        out = self.RemovePrecursorToken(out)
        out = self.final_down_proj(out)
        out_dict = {'mean': out}
        if self.output_sigma:
            logvar_fraction = self.sigma_down_proj(out)
            out_dict['var'] = logvar_fraction
        return out_dict

    def predict_sequence(self, embedding, batch, save_xcur=False):
        shape = (
            embedding['emb'].shape[0],
            batch['intseq'].shape[1] + 1,
            self.input_output_units,
        )
        model_kwargs = {
            'kv_feats': embedding['emb'],
            'charge': batch['charge'] if 'charge' in batch else None,
            'mass': batch['mass'] if 'mass' in batch else None,
        }

        # Create fully noised real data
        device = model_kwargs['kv_feats'].device
        noise = th.randn(*shape, device=device)
        """target = self.append_null_token(batch['intseq'])
        target = self.replace_with_eos_token(target, batch['peplen'])
        loss_mask = self.sequence_mask(target)
        model_kwargs['loss_mask'] = loss_mask
        x_start_mean = self.get_embed(target)
        std = _extract_into_tensor(
            self.diff_obj.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]).to(x_start_mean.device),
            x_start_mean.shape,
        )
        x_start = self.diff_obj.get_x_start(x_start_mean, std)
        ts = th.tensor(x_start.shape[0]*[self.diff_obj.num_timesteps-1]).to(x_start.device)
        #ts = th.tensor(x_start.shape[0]*[2000-1]).to(x_start.device)
        noise = self.diff_obj.q_sample(x_start, ts, noise=noise)"""

        units = self.diff_obj.my_p_sample_loop(
            self,
            shape,
            noise=noise,
            #denoised_fn=self.clamp,
            clip_denoised=self.clip_denoised,
            model_kwargs=model_kwargs,
            save_xcur=save_xcur
        )
        logits = self.get_logits(units) # bs, 31, predcats
        final = logits.argmax(dim=-1)
        
        return final, logits

    def clamp(self, x_0, *args):
        embedding = self.lm_head.weight # 24, 512
        dist = (x_0[...,None,:] - embedding[None, None]).square().sum(-1) # bs, 31, 24
        input_ids = dist.argmin(-1)
        #input_ids = self.get_logits(x_0).argmax(-1) # bs, 31
        return self.get_embed(input_ids)

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



