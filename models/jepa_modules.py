import models.model_parts as mp
import torch as th
from torch import nn
from .encoder import Encoder

class JEPATransformer(nn.Module):
    def __init__(
        self,
        in_units,
        running_units=256,
        att_d=64,
        att_h=4,
        depth=4,
        bias=False,
        ffn_multiplier=2,
        dropout=0.25,
    ):
        super(JEPATransformer, self).__init__()
        self.running_units = running_units

        # Main block
        assert bias in ['pairwise', 'regular', False, None]
        if bias == None: bias = False
        attention_dict = {
            'indim': running_units, 
            'd': att_d, 
            'h': att_h,
            #'bias': bias,
            'bias_in_units': None,
            #'modulator': False,
            #'gate': False,
            'dropout': dropout,
            'alphabet': False,
        }
        ffn_dict = {
            'indim': running_units, 
            'unit_multiplier': ffn_multiplier,
            'dropout': dropout,
            'alphabet': False,
        }
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type='layer', 
                prenorm=True, 
                embed_type=None, 
                embed_indim=256,
                is_cross=True,
                kvindim=in_units
            ) 
            for _ in range(depth)
        ])

    def forward(self, context_embedding):
        out = context_embedding
        for block in self.main:
            out = block(out)
        return out

class JEPAPredictor(JEPATransformer):
    def __init__(
        self,
        in_units,
        out_units,
        running_units=256,
        att_d=64,
        att_h=4,
        depth=4,
        bias=False,
        ffn_multiplier=2,
        dropout=0.25,
    ):
        super().__init__(
            in_units=in_units,
            running_units=running_units,
            att_d=att_d,
            att_h=att_h,
            depth=depth,
            bias=bias,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.first = nn.Linear(running_units, running_units)
        self.mask_token = nn.Parameter(th.randn(running_units), requires_grad=True)
        self.final = nn.Linear(running_units, out_units)

    def forward(self, target_mz, emb):
        mz_emb = mp.FourierFeatures(target_mz, 0.001, 10000, self.running_units) + self.mask_token
        out = self.first(mz_emb)
        for block in self.main:
            out = block(out, kv_feats=emb)['out']
        out = self.final(out)

        return out

