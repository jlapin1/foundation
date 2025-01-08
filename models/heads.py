import numpy as np
import models.model_parts as mp
import torch as th
from torch import nn

class ClassifierHead(nn.Module):
    # Outputs a desired number (num_classes) of classes
    # - One number for the ENTIRE SPECTRUM if spectrum_wise=False
    # - One number for EACH PEAK if spectrum_wise=True
    def __init__(self,
                 num_classes,
                 in_units,
                 penult_units,
                 norm_type='layer',
                 prenorm=False,
                 spectrum_wise=False
                 ):
        super(ClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.penult_units = penult_units
        self.norm_type = norm_type
        self.spectrum_wise = spectrum_wise
        
        norm = mp.get_norm_type(norm_type)

        self.penult = nn.Sequential(
            norm(penult_units) if prenorm else nn.Identity(),
            nn.Linear(in_units, penult_units),
            nn.GELU(),
            nn.Identity() if prenorm else norm(penult_units)
        )
        
        self.final = nn.Linear(penult_units, num_classes)

    def forward(self, emb):
        out = self.penult(emb)
        out = self.final(out)
        if self.spectrum_wise:
            out = out.mean(dim=1)

        return out

class RegressorHead(nn.Module):
    # Outputs a single number for regression
    # - spectrum_wise==False -> per peak
    # - spectrum_wise==True -> per spectrum
    def __init__(self,
                 in_units,
                 penult_units,
                 norm_type='layer',
                 spectrum_wise=True,
                 activation=None,
                 scale=1.0,
                 ):
        super(RegressorHead, self).__init__()
        self.norm_type = norm_type
        norm = mp.get_norm_type(norm_type)
        self.spectrum_wise = spectrum_wise
        self.act = lambda x: (
            x.tanh() if activation == 'tanh' else x
        )
        self.scale = scale

        self.penult = nn.Sequential(
            nn.Linear(in_units, penult_units),
            nn.GELU(),
            norm(penult_units),
        )

        self.final = nn.Linear(penult_units, 1)

    def forward(self, emb):
        out = self.penult(emb)
        out = self.final(out)
        if self.spectrum_wise:
            out = out.mean(dim=1)
        out = self.scale * self.act(out)

        return out.squeeze()

class SequenceHead(nn.Module):
    def __init__(self, 
                 in_units,
                 final_units,
                 in_seq_len,
                 final_seq_len,
                 penult_units=512,
                 drop_rate=0,
                 norm_type='layer'
                 ):
        super(SequenceHead, self).__init__()
        self.final_units = final_units
        self.final_seq_len = final_seq_len
        self.penult_units = penult_units
        self.norm_type = norm_type
        
        norm = mp.get_norm_type(norm_type)

        self.penult = nn.Sequential(
            nn.Linear(in_units, penult_units), 
            nn.GELU(), 
            norm(penult_units),
        )

        self.final_sl = (
            nn.Sequential(
                nn.Linear(in_seq_len, final_seq_len),
                nn.GELU(),
                norm(final_seq_len),
            )
            if final_seq_len is not None else
            None
        )
        self.drop = nn.Identity() if drop_rate==0 else nn.Dropout(drop_rate)
        self.final_ch = ( 
            nn.Linear(penult_units, final_units) 
            if final_units is not None else 
            nn.Identity()
        )
    
    def forward(self, emb):
        out = self.penult(emb)
        if self.final_sl is not None:
            out = self.final_sl(out.permute([0,2,1]))
            out = out.permute([0,2,1])
        out = self.drop(out)
        out = self.final_ch(out)
        
        return out

class SpectrumHead(nn.Module):
    def __init__(self,
                 in_units,
                 bins,
                 penult_units=512,
                 sequence_length=100,
                 out_types=['mz', 'ab'],
                 norm_type='layer',
                 ):
        super(SpectrumHead, self).__init__()
        self.in_units = in_units
        self.bins = bins
        self.out_types = out_types
        self.penult_units = penult_units
        self.norm_type = norm_type
        
        norm = mp.get_norm_type(norm_type)

        self.penult = nn.Sequential(
            nn.Linear(in_units, penult_units),
            nn.GELU(),
            norm(penult_units),
        )
        
        # Do not accomodate rank with mz/ab
        if 'mz' in out_types:
            self.final_mz = nn.Linear(penult_units, bins)
        if 'ab' in out_types:
            self.final_ab = nn.Linear(penult_units, bins)
    
    def forward(self, emb):
        out = self.penult(emb)
            
        if 'mz' in self.out_types:
            mz = self.final_mz(out)
        else:
            mz = None
            
        if 'ab' in self.out_types:
            ab = self.final_ab(out)
        else:
            ab = None

        return {'mz': mz, 'ab': ab}

class Header(nn.Module):
    def __init__(self, 
                 head_dic,
                 final_seq_len=None,
                 final_ch=None,
                 lr=1e-4
                 ):
        super(Header, self).__init__()
        self.head_dic = head_dic
        self.name = 'head'

        IU = head_dic['in_units']
        self.heads = {}
        if 'pepseq' in head_dic.keys():
            dic = head_dic['pepseq']
            self.heads['pepseq'] = SequenceHead(**dic, in_units=IU)
        
        if 'trinary_ab' in head_dic.keys():
            dic = head_dic['trinary_ab']
            self.heads['trinary_ab'] = ClassifierHead(**dic, in_units=IU)

        if 'trinary_mz' in head_dic.keys():
            dic = head_dic['trinary_mz']
            self.heads['trinary_mz'] = ClassifierHead(**dic, in_units=IU)

        if 'hidden_ab' in head_dic.keys():
            dic = head_dic['hidden_ab']
            self.heads['hidden_ab'] = SpectrumHead(**dic, in_units=IU)

        if 'hidden_mz' in head_dic.keys():
            dic = head_dic['hidden_mz']
            self.heads['hidden_mz'] = SpectrumHead(**dic, in_units=IU)
        
        if 'hidden_spectrum' in head_dic.keys():
            dic = head_dic['hidden_spectrum']
            self.heads['hidden_spectrum'] = SpectrumHead(**dic, in_units=IU)
        
        if 'hidden_charge' in head_dic.keys():
            dic = head_dic['hidden_charge']
            self.heads['hidden_charge'] = ClassifierHead(**dic, in_units=IU)
        
        if 'hidden_mass' in head_dic.keys():
            dic = head_dic['hidden_mass']
            dic['num_classes'] = 3001
            self.heads['hidden_mass'] = ClassifierHead(**dic, in_units=IU)

        if 'maldi' in head_dic.keys():
            dic = head_dic['maldi']
            self.heads['maldi'] = ClassifierHead(**dic, in_units=IU)
        
        if 'resid_regr' in head_dic.keys():
            dic = head_dic['resid_regr']
            self.heads['resid_regr'] = RegressorHead(**dic, in_units=IU)

        self.opts = {
            key: th.optim.Adam(self.heads[key].parameters(), lr=lr)
            for key in 
            self.heads.keys()
        }

    def forward(self, emb, outs='all'):
        out = {}
        if outs=='all': outs = self.heads.keys()
        
        for head_type in outs:
            out[head_type] = self.heads[head_type](emb)
        
        return out


