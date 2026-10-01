import torch as th
from torch import nn
import models.model_parts as mp

#class Data2VecPredictor(nn.Module):
#    def __init__(
#        self,
#        in_units,
#        running_units=512,
#        att_d=64,
#        att_h=4,
#        depth=4,
#        ffn_multiplier=2,
#        dropout=0.25,
#    ):
#        super(Data2VecPredictor, self).__init__()
#        self.running_units = running_units
#        
#        # Positional embedding
#        position_indices = th.arange(1000, dtype=th.float32)
#        pos = mp.FourierFeatures(position_indices, 1, 2000, in_units)
#        self.pos = nn.Parameter(pos, requires_grad=False)
#        self.alpha = nn.Parameter(th.tensor(1.0), requires_grad=True)
#
#        # First transformation
#        self.first = nn.Sequential(
#            nn.Linear(in_units, running_units),
#            nn.ReLU(),
#        )
#
#        # Main block
#        attention_dict = {
#            'indim': running_units, 
#            'd': att_d, 
#            'h': att_h,
#            'dropout': dropout,
#        }
#        ffn_dict = {
#            'indim': running_units, 
#            'unit_multiplier': ffn_multiplier,
#            'dropout': dropout,
#        }
#        self.main = nn.ModuleList([
#            mp.TransBlock(
#                attention_dict, 
#                ffn_dict, 
#                norm_type='layer', 
#                prenorm=True, 
#                is_cross=False,
#                kvindim=in_units
#            ) 
#            for _ in range(depth)
#        ])
#
#        # Final transformation
#        self.last = nn.Sequential(
#            nn.Linear(running_units, running_units),
#            nn.LayerNorm(running_units),
#            nn.ReLU(),
#            nn.Linear(running_units, in_units),
#        )
#
#    def forward(self, emb):
#        out = emb + self.alpha * self.pos[:emb.shape[1]]
#        out = self.first(out)
#        for block in self.main:
#            out = block(out)['out']
#        out = self.last(out)
#        return out

class Data2VecPredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Data2VecPredictor, self).__init__()
        self.linear = nn.Linear(10,10)

    def forward(self, emb, **kwargs):
        return emb

