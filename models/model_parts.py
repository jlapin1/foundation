import torch as th
from torch import nn
from math import log

class BatchTorch1d(nn.Module):
    def __init__(self, units):
        super(BatchTorch1d, self).__init__()
        self.norm = nn.BatchNorm1d(units)
    def forward(self, x):
        return self.norm(x.transpose(-1,-2)).transpose(-1,-2)

def get_norm_type(string):
    if string.lower() == 'layer':
        return nn.LayerNorm
    elif string.lower() == 'batch':
        return BatchTorch1d

class QKVAttention(nn.Module):
    def __init__(self,
                 heads,
                 dim,
                 sl=None,
                 is_relpos=False, 
                 max_rel_dist=None
    ):
        super(QKVAttention, self).__init__()
        self.heads = heads
        self.dim = dim
        self.sl = sl
        self.is_relpos = is_relpos
        self.maxd = max_rel_dist
        
        self.scale = dim**-0.25
        if is_relpos:
            assert sl is not None
            self.maxd = sl if max_rel_dist==None else max_rel_dist
            self.ak = self.build_relpos_tensor(sl, self.maxd) # sl, sl, dim
            self.av = self.build_relpos_tensor(sl, self.maxd) # same
    
    def build_relpos_tensor(self, seq_len, maxd=None):
        # This is the equivalent of an input dependent ij bias, rather than one
        # that is a directly learned ij tensor
        maxd = seq_len-1 if maxd==None else (maxd-1 if maxd==seq_len else maxd)
        a = th.arange(seq_len, dtype=th.int32)
        b = th.arange(seq_len, dtype=th.int32)
        relpos = a[:, None] - b[None]
        tsr = th.zeros(
            2*seq_len-1, self.dim
        ).normal_(0, seq_len**-0.5).type(th.float32)
        # set maximum distance
        relpos = relpos.clamp(-maxd, maxd)
        relpos += maxd
        relpos_tsr = tsr[relpos]
        relpos_tsr.requires_grad = True
        
        return relpos_tsr
    
    def forward(self, Q, K, V, mask=None, bias=None):
        bsp, sl, dim = Q.shape
        # shape: batch/heads/etc, sequence_length, dim
        QK = th.einsum('abc,adc->abd', self.scale*Q, self.scale*K)
        if self.is_relpos:
            QK += th.einsum('abc,bec->abe', Q, self.ak)
        QK = QK.reshape(-1, self.heads, sl, K.shape[1])
        if bias is not None:
            QK += bias
        
        # mask.shape: bs, 1, 1, sl
        mask = th.zeros_like(QK) if mask==None else mask[:, None, None, :]
        weights = th.softmax(QK-mask, dim=-1)
        weights = weights.reshape(-1, sl, V.shape[1])
        
        att = th.einsum('abc,acd->abd', weights, V)
        if self.is_relpos:
            att += th.einsum('abc,bcd->abd', weights, self.av)
        
        return att

class BaseAttentionLayer(nn.Module):
    def __init__(self, indim, d, h, out_units=None):
        super(BaseAttentionLayer, self).__init__()
        self.indim = indim
        self.d = d
        self.h = h
        self.out_units = indim if out_units==None else out_units
        
        self.attention_layer = QKVAttention(h, d)
        shape = (d*h, self.out_units)
        self.Wo = nn.Linear(*shape, bias=False)
        self.Wo.weight = nn.Parameter( 
            nn.init.normal_(th.empty(self.Wo.weight.shape), 0.0, 0.3*(h*d)**-0.5)
        )
        self.shortcut = (
            nn.Identity()
            if self.out_units == indim else
            nn.Linear(indim, out_units)
        )
        
class SelfAttention(BaseAttentionLayer):
    def __init__(self, 
                 indim, 
                 d, 
                 h, 
                 out_units=None, 
                 pairwise_bias=False, 
                 bias_in_units=None
    ):
        super().__init__(indim=indim, d=d, h=h, out_units=out_units)
        self.pairwise_bias = pairwise_bias

        self.qkv = nn.Linear(indim, 3*d*h, bias=False)
        if pairwise_bias:
            self.pwbias = nn.Linear(bias_in_units, h)
        
    def get_qkv(self, qkv):
        bs, sl, units = qkv.shape
        Q, K, V = qkv.split(units//3, -1)
        Q = Q.reshape(-1, sl, self.d, self.h)
        Q = Q.permute([0,3,1,2]).reshape(-1, sl, self.d)
        K = K.reshape(-1, sl, self.d, self.h)
        K = K.permute([0,3,1,2]).reshape(-1, sl, self.d)
        V = V.reshape(-1, sl, self.d, self.h)
        V = V.permute([0,3,1,2]).reshape(-1, sl, self.d)
        
        return Q, K, V # bs*h, sl, d
    
    def forward(self, x, mask=None, pwtsr=None):
        bs, sl, units = x.shape
        qkv = self.qkv(x) # bs, sl, 3*d*h
        Q, K, V = self.get_qkv(qkv) # bs*h, sl, d
        if self.pairwise_bias:
            bias = self.pwbias(pwtsr) # bs, sl, sl, h
            bias = bias.permute([0,3,1,2]) # bs, h, sl, sl
        else:
            bias = None
        att = self.attention_layer(Q, K, V, mask, bias=bias) # bs*h, sl, d
        att = att.reshape(-1, self.h, sl, self.d)
        att = att.permute([0,2,3,1])
        att = att.reshape(-1, sl, self.d*self.h) # bs, sl, d*h
        resid = self.Wo(att)
        
        return self.shortcut(x) + resid

class CrossAttention(BaseAttentionLayer):
    def __init__(self, indim, kvindim, d, h, out_units=None):
        super().__init__(indim=indim, d=d, h=h, out_units=out_units)
        
        self.Wq = nn.Linear(indim, d*h, bias=False)
        self.Wkv = nn.Linear(kvindim, 2*d*h, bias=False)
        
    def get_qkv(self, q, kv):
        bs, sl, units = q.shape
        bs, sl2, kvunits = kv.shape
        Q = q.reshape(bs, sl, self.d, self.h)
        Q = Q.permute([0,3,1,2]).reshape(-1, sl, self.d)
        K, V = kv.split(kvunits//2, -1)
        K = K.reshape(bs, sl2, self.d, self.h)
        K = K.permute([0,3,1,2]).reshape(-1, sl2, self.d)
        V = V.reshape(bs, sl2, self.d, self.h)
        V = V.permute([0,3,1,2]).reshape(-1, sl2, self.d)
        
        return Q, K, V
    
    def forward(self, q_feats, kv_feats, mask=None):
        _, slq, _ = q_feats.shape
        Q = self.Wq(q_feats)
        KV = self.Wkv(kv_feats)
        Q, K, V = self.get_qkv(Q, KV)
        att = self.attention_layer(Q, K, V, mask)
        att = att.reshape(-1, self.h, slq, self.d)
        att = att.permute([0,2,1,3])
        att = att.reshape(-1, slq, self.h*self.d)
        resid = self.Wo(att)
        
        return self.shortcut(q_feats) + resid

class FFN(nn.Module):
    def __init__(self, indim, unit_multiplier=1, out_units=None):
        super(FFN, self).__init__()
        self.indim = indim
        self.mult = unit_multiplier
        self.out_units = indim if out_units==None else out_units
        
        self.W1 = nn.Linear(indim, indim*self.mult)
        self.W2 = nn.Linear(indim*self.mult, self.out_units, bias=False)
        shape = self.W2.weight.shape
        self.W2.weight = nn.Parameter( 
            nn.init.normal_(th.empty(shape), 0.0, 0.3*(indim*self.mult)**-0.5)
        )
    
    def forward(self, x, embed=None):
        out = self.W1(x)
        out = th.relu(out + (0 if embed==None else embed))
        out = self.W2(out)
        
        return x + out

class TransBlock(nn.Module):
    def __init__(self,
                 attention_dict,
                 ffn_dict,
                 norm_type='layer',
                 prenorm=True,
                 is_embed=False,
                 embed_indim=256,
                 preembed=True,
                 is_cross=False,
                 kvindim=256
    ):
        super(TransBlock, self).__init__()
        self.norm_type = norm_type
        self.mult = ffn_dict['unit_multiplier']
        self.prenorm = prenorm
        self.is_embed = is_embed
        self.preembed = preembed
        self.is_cross = is_cross
        
        if preembed: self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        norm = get_norm_type(norm_type)
        
        indim = attention_dict['indim']
        self.norm1 = norm(indim)
        self.norm2 = norm(ffn_dict['indim'])
        self.selfattention = SelfAttention(**attention_dict)
        if is_cross:
            cross_dict = attention_dict.copy()
            if 'pairwise_bias' in cross_dict.keys(): cross_dict.pop('pairwise_bias')
            if 'bias_in_units' in cross_dict.keys(): cross_dict.pop('bias_in_units')
            cross_dict['kvindim'] = kvindim
            self.crossnorm = norm(indim)
            self.crossattention = CrossAttention(**cross_dict)
        self.ffn = FFN(**ffn_dict)
        
        if self.is_embed:
            assert type(embed_indim) == int
            units = indim if self.preembed else indim*self.mult
            self.embed = nn.Linear(embed_indim, units)
        
    def forward(self, 
                x, 
                kv_feats=None, 
                embed_feats=None, 
                spec_mask=None, 
                seq_mask=None,
                pwtsr=None,
    ):
        selfmask = seq_mask if self.is_cross else spec_mask
        Emb = self.embed(embed_feats)[:,None,:] if self.is_embed else 0
        
        out = x + self.alpha*Emb if self.preembed else x
        out = self.norm1(out) if self.prenorm else out
        out = self.selfattention(out, selfmask, pwtsr)
        if self.is_cross:
            out = self.crossnorm(out) if self.prenorm else out
            out = self.crossattention(out, kv_feats, spec_mask)
            out = out if self.prenorm else self.crossnorm(out)
        out = self.norm2(out) if self.prenorm else self.norm1(out)
        out = self.ffn(out, None) if self.preembed else self.ffn(out, Emb)
        out = out if self.prenorm else self.norm2(out)
        
        return out

class ActModule(nn.Module):
    def __init__(self, activation):
        super(ActModule, self).__init__()
        self.act = activation
    def forward(self, x):
        return self.act(x)

def FourierFeatures(t, embedsz, freq=10000.):
    embed = ( 
        t[..., None] * 
        th.exp(
            -log(float(freq)) * 
            th.arange(embedsz//2).type(th.float32).to(t.device) / 
            (embedsz//2)
        )[None]
    )
    
    return th.cat([embed.cos(), embed.sin()], dim=-1)

def subdivide_float(x):
    mul = -1*(x < 0).type(th.float32) + (x >= 0).type(th.float32)
    X = abs(x)
    a = X.floor_divide(100)
    b = (X-a*100).floor_divide(1)
    X_ = ((X - X.floor_divide(1))*10000).round()
    c = X_.floor_divide(100)
    d = (X_ - c*100).floor_divide(1)
    
    return mul[...,None] * th.cat([a[...,None], b[...,None], c[...,None], d[...,None]], -1)

def delta_tensor(mz, shift=0):
    return mz[..., None] - mz[:, None] + shift # bs, sl, sl


