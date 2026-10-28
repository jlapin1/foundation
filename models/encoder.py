from . import model_parts as mp
from . import model_parts_pw as pw
import torch as th
from torch import nn
from utils import sample_lognormal_poisson
import numpy as np
I = nn.init

def init_encoder_weights(module):
    #if hasattr(module, 'MzSeq'):
    #    module.MzSeq[0].weight = I.xavier_uniform_(module.MzSeq[0].weight)
    #    if module.MzSeq[0].bias is not None:
    #        module.MzSeq[0].bias = I.zeros_(module.MzSeq[0].bias)
    if hasattr(module, 'first'):
        module.first.weight = I.xavier_uniform_(module.first.weight)
        if module.first.bias is not None:
            module.first.bias = I.zeros_(module.first.bias)
    if hasattr(module, 'class_token'):
        module.class_token = I.normal_(module.class_token, 0.0, 1.0)
    if isinstance(module, mp.SelfAttention):
        #maxmin = (6 / (module.qkv.in_features + module.d))**0.5
        #module.qkv.weight = I.xavier_uniform_(module.qkv.weight)#, -maxmin, maxmin)
        #module.Wo.weight = I.xavier_uniform_(module.Wo.weight)
        #module.qkv.weight = I.uniform_(module.qkv.weight, -0.03, 0.03)
        #module.Wo.weight = I.uniform_(module.Wo.weight, -0.03, 0.03)
        module.qkv.weight = I.normal_(module.qkv.weight, 0.0, (2/3)*module.indim**-0.5)
        module.Wo.weight = I.normal_(module.Wo.weight, 0.0, (1/3)*(module.h*module.d)**-0.5)
        module.qkv.bias = I.zeros_(module.qkv.bias)
        module.Wo.bias = I.zeros_(module.Wo.bias)
        if hasattr(module, 'Wb'):
            module.Wb.weight = I.zeros_(module.Wb.weight)
            module.Wb.bias = I.zeros_(module.Wb.bias)
    elif isinstance(module, mp.FFN):
        module.W1.weight = I.xavier_uniform_(module.W1.weight)
        module.W1.bias = I.zeros_(module.W1.bias)
        module.W2.weight = I.normal_(module.W2.weight, 0.0, (1/3)*(module.indim*module.mult)**-0.5)
        #module.W2.weight = I.xavier_uniform_(module.W2.weight)
    elif isinstance(module, nn.Linear):
        module.weight = I.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias = I.zeros_(module.bias)
    

class Encoder(nn.Module):
    def __init__(self,
        # 1D options
        in_units=2, # input units from mz/ab tensor
        running_units=512, # num units running throughout model
        sequence_length=100, # maximum number of peaks
        mz_units=512, # units in mz fourier vector
        ab_units=256, # units in ab fourier vector
        subdivide=False, # subdivide mz units in 2s and expand-concat
        use_charge=False, # inject charge into TransBlocks
        use_energy=False, # inject energy into TransBlocks
        use_mass=False, # injuect mass into TransBlocks
        class_token=False, # constant learned class token
        ce_units=256, # units for transformation of mzab fourier vectors
        att_d=64, # attention qkv dimension units
        att_h=4,  # attention qkv heads
        gate=False, # input dependent gate following weights*V
        alphabet=False, # single parameters on residual and skip connections
        ffn_multiplier=4, # multiply inp units for 1st FFN transform
        prenorm=True, # normalization before attention/ffn layers
        norm_type='layer', # normalization type
        prec_type=None, # inject_pre | inject_ffn | inject_norm | None
        depth=9, # number of transblocks
        # Looping
        iterations=1, # loop iterations
        loop_style='one', # one | two
        embed_loop_iter=True,
        # Pairwise options
        bias=False, # use pairwise mz tensor to create SA-bias
        dropout=0, # dropout rate for residuals in attention and feed forward
        pw_mz_units=None, # sinusoidal units to expand pw tensor into
        pw_run_units=None, # units to project pw tensor to after sinusoidal expansion
        pw_attention_ch=32, # triangle attention channels
        pw_attention_h=4, # triangle attention heads
        pw_blocks=2, # number of pairstack blocks for pairwise features
        pw_n=4, # pair transition unit multiplier
        # Miscellaneous
        device=th.device('cpu'),
        masked=False,
        return_all=False,
        mask_absent_peaks=True,
        **kwargs
    ):
        super(Encoder, self).__init__()
        self.run_units = running_units
        self.sl = sequence_length
        self.mz_units = mz_units
        self.ab_units = ab_units
        self.subdivide = subdivide
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass
        self.use_class_token = class_token
        self.ce_units = ce_units
        self.ce_out = running_units if prec_type is not None and 'pretoken' in prec_type else ce_units
        self.d = att_d
        self.h = att_h
        self.bias = bias
        self.pw_mzunits = mz_units if pw_mz_units==None else pw_mz_units
        self.pw_runits = running_units if pw_run_units==None else pw_run_units
        self.depth = depth
        self.prenorm = prenorm
        self.norm_type = norm_type
        self.prec_type = prec_type
        self.its = iterations
        self.loop_style = loop_style
        self.embed_loop_iter = embed_loop_iter
        self.device = device
        self.masked = masked
        self.return_all = return_all
        self.mask_absent_peaks = mask_absent_peaks
        
        # m/z Fourier Features
        mdim = mz_units//4 if subdivide else mz_units
        self.mdim = mdim
        self.MzSeq = nn.Identity() # # nn.Sequential(nn.Linear(mdim, mdim), nn.SiLU())

        # Pairwise mz
        if bias == 'pairwise':
            # - subidvide and expand based on mz_units, transform to pw_units
            mdimpw = self.pw_mzunits//4 if subdivide else self.pw_mzunits
            self.mdimpw = mdimpw
            self.MzpwSeq = nn.Identity()#Sequential(nn.Linear(mdimpw, mdimpw), nn.SiLU())
            self.pwfirst = nn.Identity()#nn.Linear(self.pw_mzunits, self.pw_runits)
            #self.alphapw = nn.Parameter(th.tensor(0.1), requires_grad=True)
            #self.pospw = pw.RelPos(sequence_length, self.pw_runits)
            # Evolve features
            """multdict = {'in_dim': self.pw_runits, 'c': 128}
            attdict = {'in_dim': self.pw_runits, 'c': pw_attention_ch, 'h': pw_attention_h}
            ptdict = {'in_dim': self.pw_runits, 'n': pw_n}
            self.PwSeq = nn.Sequential(*[
                pw.PairStack(multdict, attdict, ptdict, drop_rate=0)
                for m in range(pw_blocks)
            ])"""
            self.PwSeq = nn.Sequential(*[
                #pw.PairStack(multdict, attdict, ptdict, drop_rate=0)
                #for m in range(pw_blocks)
                nn.Linear(self.pw_mzunits, ffn_multiplier*self.pw_runits),
                nn.SiLU(),
                nn.Linear(ffn_multiplier*self.pw_runits, self.pw_runits)
            ])

        # m/z for masks
        if self.masked:
            self.mask_token = nn.Parameter(th.randn(running_units), requires_grad=True)

        # charge/energy/mass embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            if prec_type == 'inject_pre':
                prec_type = 'preembed'
            elif prec_type == 'inject_ffn':
                prec_type = 'ffnembed'
            elif prec_type == 'inject_norm':
                prec_type = 'normembed'
            elif prec_type in ['pretoken_fuse', 'pretoken_separate']:
                prec_type = None
            else:
                raise NotImplementedError("Choose real prec_type")
            num = sum([use_charge, use_energy, use_mass])
            self.ce_emb = nn.Sequential(
                nn.Linear(ce_units*num, self.ce_out), nn.SiLU()
            )
        else:
            self.prec_type = None
        
        # First transformation
        self.first = nn.Linear(mz_units+ab_units, running_units, bias=False)

        # Class token
        if class_token:
            self.class_token = nn.Parameter(th.zeros(running_units), requires_grad=True)

        # Main block
        assert bias in ['pairwise', 'regular', False, None]
        if bias == None: bias = False
        attention_dict = {
            'indim': running_units, 
            'd': att_d, 
            'h': att_h,
            'bias': bias,
            'bias_in_units': self.pw_runits,
            'modulator': False,
            'gate': gate,
            'dropout': dropout,
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
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type=norm_type, 
                prenorm=prenorm, 
                embed_type=prec_type, 
                embed_indim=ce_units,
            ) 
            for _ in range(depth)
        ])
        
        # Normalization type
        self.norm = mp.get_norm_type(norm_type)
        
        # Loopering iterations
        if iterations > 1:
            if embed_loop_iter:
                self.iteration_embedder = nn.Sequential(
                    nn.Linear(running_units, running_units),
                    nn.SiLU(),
                    nn.Linear(running_units, running_units),
                )
            if loop_style == 'two':
                self.adapter = nn.Linear(2*running_units, running_units)
            
        self.global_step = nn.Parameter(th.tensor(0), requires_grad=False)
        
        self.apply(init_encoder_weights)
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters()])

    def MzAb(self, x, inp_mask=None):
        Mz, ab = th.split(x, 1, -1)
        
        Mz = Mz.squeeze()
        if x.shape[0] == 1: Mz = Mz[None] # for batch_size=1
        if self.subdivide:
            mz = mp.subdivide_float(Mz)
            mz_emb = mp.FourierFeatures(mz, 1, 500, self.mdim)
        else:
            mz_emb = mp.FourierFeatures(Mz, 0.001, 10000, self.mz_units)
        mz_emb = self.MzSeq(mz_emb) # multiply sequential to mz fourier feature
        mz_emb = mz_emb.reshape(x.shape[0], x.shape[1], -1)
        ab_emb = mp.FourierFeatures(ab[...,0], 0.000001, 1, self.ab_units)
        
        # Apply input mask, if necessary
        if inp_mask is not None:
            mz_mask, ab_mask = th.split(inp_mask, 1, -1)
            # Zeros out the feature's entire Fourier vector
            mz_emb *= mz_mask
            ab_emb *= ab_mask

        # Pairwise features
        if self.bias == 'pairwise':
            dtsr = mp.delta_tensor(Mz, 0.)
            # expand based on mz_units
            if self.subdivide:
                mzpw = mp.subdivide_float(dtsr)
                mzpw_emb = mp.FourierFeatures(mzpw, 0.001, 10000, self.mdimpw)
            else:
                mzpw_emb = mp.FourierFeatures(dtsr, 0.001, 10000, self.pw_mzunits)
            # transform based on pw_units
            mzpw_emb = self.MzpwSeq(mzpw_emb)
            mzpw_emb = mzpw_emb.reshape(x.shape[0], x.shape[1], x.shape[1], -1)
        else:
            mzpw_emb = None
            
        out = th.cat([mz_emb, ab_emb], dim=-1)

        return {'1d': out, '2d': mzpw_emb}

    def spectrum_level_embeddings(self, charge, energy, mass):
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = charge.type(th.float32)
                ce_emb.append(mp.FourierFeatures(charge, 1, 50, self.ce_units))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, 0.001, 10000, self.ce_units))
            # tf.concat works if list is 1 or multiple members
            ce_emb = th.cat(ce_emb, dim=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None
        
        return ce_emb

    def spectrum_mask(self, length, xshape, spectrum_sl):
        if length != None and self.mask_absent_peaks:
            
            grid = th.tile(
                th.arange(spectrum_sl, dtype=th.int32)[None].to(length.device), 
                (len(length), 1)
            ) # bs, seq_len
            mask = grid >= length[:, None]
            num = xshape[1] - spectrum_sl
            if num != 0:
                prepend = th.full((mask.shape[0], num), fill_value=False, device=mask.device)
                mask = th.cat([prepend, mask], dim=1)
            mask = (1e7*mask).type(th.float32)
        else:
            mask = None

        return mask
    
    def prepend_token(self, token, tensor):
        if token.ndim == 1:
            prepend = token[None, None].tile([tensor.shape[0], 1, 1])
        elif token.ndim == 2:
            prepend = token[:,None]
        elif token.ndim == 3:
            prepend = token
        out = th.cat([prepend, tensor], dim=1)
        return out

    def remove_pretoken(self, tensor):
        prepend = tensor[:,0]
        return tensor[:,1:], prepend

    def add_pretokens(self, tensor, ce_emb):
        if self.prec_type is not None and 'pretoken' in self.prec_type:
            tensor = self.prepend_token(ce_emb, tensor)
        if self.use_class_token:
            if self.prec_type == 'pretoken_fuse':
                tensor[:,0] += self.class_token[None]
                ce_emb = None
            elif self.prec_type == 'pretoken_separate':
                tensor = self.prepend_token(self.class_token, tensor)
                ce_emb = None
            else:
                tensor = self.prepend_token(self.class_token, tensor)

        return tensor, ce_emb
    
    def Prelude(
        self,
        x,
        charge=None,
        energy=None,
        mass=None,
        inp_mask=None,
    ):
        # Spectrum level embeddings
        ce_emb = self.spectrum_level_embeddings(charge, energy, mass)
        
        # Feed forward
        mzab_dic = self.MzAb(x, inp_mask)
        mabemb = mzab_dic['1d'];pwemb = mzab_dic['2d']
        if self.bias == 'pairwise':
            pwemb = self.PwSeq(pwemb)
        else:
            pwemb = None
        
        out = self.first(mabemb)
        if self.masked:
            out[x[...,0] < 0] = self.mask_token

        # Attach pretoken(s)
        out, ce_emb = self.add_pretokens(out, ce_emb)

        return {'out': out, 'ce_emb': ce_emb, 'pw_emb': pwemb}

    def Coda(self, inp):
        return inp

    def Main(self, inp, embed=None, mask=None, pwtsr=None, return_full=False):
        out = inp
        other = []
        all_outs = []
        for layer in self.main:
            out = layer(out, embed_feats=embed, spec_mask=mask, biastsr=pwtsr, return_full=return_full)
            other.append(out['other'])
            out = out['out']
            if self.return_all:
                all_outs.append(out)
        
        return {'out': out, 'all': all_outs, 'other': other}
    
    def UpdateEmbed(self, 
        x, # s_i in paper
        e=None,
        ce_embed=None,
        pwtsr=None,
        length=None,
        inp_mask=None,
        tag_array=None,
        return_mask=False,
        return_full=False,
        sequence_length=None,
        iteration=0,
    ):

        # Reycling the embedding with normalization, perhaps dense transform
        if self.its > 1:
            if e is not None:
                # Unite the prelude (e) and the current embedding (x)
                x = self.adapter(th.cat([x, e], dim=-1))
            if self.embed_loop_iter:
                IT = th.full((1, 1), iteration, device=x.device, dtype=th.float32)
                ITff = mp.FourierFeatures(IT, 1, 10, x.shape[-1]).squeeze()
                iteration_embedding = self.iteration_embedder(ITff)
                x = self.prepend_token(iteration_embedding, x)

        # Create mask
        mask = self.spectrum_mask(length, x.shape, sequence_length)
        
        # Main transformer block
        main = self.Main(
            x,
            embed=ce_embed,
            mask=mask,
            pwtsr=pwtsr,
            return_full=return_full
        )

        if self.its > 1 and self.embed_loop_iter:
            main['out'], _ = self.remove_pretoken(main['out'])
        
        output = {'emb': main['out'], 'mask': mask, 'other': main['other'], 'all': main['all']}
        
        return output

    @th.no_grad()
    def update_embed_without_gradients(
        self, latent, e, ce_embed, pwtsr, length, inp_mask, 
        tag_array, return_mask, return_full, sequence_length, iteration
    ):
        return self.UpdateEmbed(
            latent, e, ce_embed, pwtsr, length, inp_mask, 
            tag_array, return_mask, return_full, sequence_length, iteration
        )

    def forward(
        self,
        x,
        charge=None, 
        energy=None, 
        mass=None,
        length=None,
        emb=None,
        inp_mask=None,
        tag_array=None,
        iterations=None,
        return_mask=False,
        return_full=False,
    ):
        # Set its and divide grad/no_grad its
        if self.training and self.its>1 and self.loop_style=='two':
            its = sample_lognormal_poisson(minimum=1, maximum=self.its, target_mean=self.its//3, sigma=0.01).item()
            grad_its = np.minimum(its, 3).item()
            no_grad_its = its - grad_its
        else:
            its = self.its  if iterations==None else iterations
            no_grad_its = 0
        
        # Prelude
        spectrum_sl = x.shape[1]
        P = self.Prelude(x, charge, energy, mass, inp_mask=inp_mask)
        
        # Set latent and e
        latent = P['out']
        e = None
        if self.its>1 and self.loop_style == 'two':
            latent = th.empty(*P['out'].shape).normal_(0, 0.0632).to(P['out'].device)
            e = P['out']

        # Middle
        latents = []
        for It in range(its):
            if It < no_grad_its:
                output_ = self.update_embed_without_gradients(
                    latent,
                    e=e,
                    ce_embed=P['ce_emb'],
                    pwtsr=P['pw_emb'],
                    length=length,
                    inp_mask=inp_mask,
                    tag_array=tag_array,
                    return_mask=return_mask,
                    return_full=return_full,
                    sequence_length=spectrum_sl,
                    iteration=It,
                )
            else:
                output_ = self.UpdateEmbed(
                    latent,
                    e=e,
                    ce_embed=P['ce_emb'],
                    pwtsr=P['pw_emb'],
                    length=length,
                    inp_mask=inp_mask,
                    tag_array=tag_array,
                    return_mask=return_mask,
                    return_full=return_full,
                    sequence_length=spectrum_sl,
                    iteration=It,
                )
            latents.append(output_['emb'])
            latent = latents[-1]
        
        # Coda
        coda_input = th.cat(latents, dim=0) if its>1 and self.training else latents[-1]
        final = self.Coda(coda_input)
        #if its>1:
        #    other = list(tail_input.split(x.shape[0], 0))
        #    final = other.pop(-1)
        
        return {'emb': final}

