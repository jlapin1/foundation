import torch as th
nn = th.nn
F = nn.functional

def weight_init(module):
    if isinstance(module, TriangleAttention):
        dim = module.c * module.h
        module.Wo.weight = nn.init.normal_(module.Wo.weight, 0., 0.3*dim**-0.5)
        module.Wo.bias = nn.init.zeros_(module.Wo.bias)
    if isinstance(module, TriangleMultiplicative):
        module.Wgg.bias = nn.init.constant_(module.Wgg.bias, -4)
    if isinstance(module, PairTransition):
        dim = module.in_dim * module.n
        module.Wo.weight = nn.init.normal_(module.Wo.weight, 0., 0.05*dim**-0.5)
        module.Wo.bias = nn.init.zeros_(module.Wo.bias)

class QKVAttention(nn.Module):
    def __init__(self, nheads, typ='start'):
        super(QKVAttention, self).__init__()
        self.h = nheads
        self.typ = typ

        if typ == 'start':
            op1 = 'abcd,abed->abce'
            op2 = 'abcd,abde->abce'
            self.ad = lambda bias: bias[:,None].permute([0,4,1,2,3])
        elif typ == 'end':
            op1 = 'abcd,aecd->abce'
            op2 = 'abcd,adce->abce'
            self.ad = lambda bias: bias[...,None,:].permute([0,4,3,2,1])
        
        self.es1 = lambda q, k: th.einsum(op1, q, k)
        self.es2 = lambda a, v: th.einsum(op2, a, v)

    def forward(self, q, k, v, bias=None, gate=None):
        (bsh, slq1, slq2, c) = q.shape
        (bsh, slk1, slk2, c) = k.shape
        qk = self.es1(q, k)
        qk = qk.reshape(-1, self.h, slq1, slq2, slk2)
        if bias is not None:
            qk += self.ad(bias)
        
        a = th.softmax(qk, -1)
        a = a.reshape(-1, slq1, slq2, slk2)
        o = self.es2(a, v)
        if gate is not None:
            gate = (
                gate.reshape(-1,slq1,slq2,c,self.h)
                .permute([0,4,1,2,3])
                .reshape(-1,slq1,slq2,c)
            )
            o *= gate
        
        return o

class TriangleAttention(nn.Module):
    def __init__(self,
                 in_dim,
                 c=32,
                 h=4,
                 typ='end'
                 ):
        super(TriangleAttention, self).__init__()
        self.c = c
        self.h = h
        
        self.attention = QKVAttention(h, typ=typ)
        
        self.norm = nn.LayerNorm(in_dim)
        self.Wqkv = nn.Linear(in_dim, 3*c*h, bias=False)
        self.Wbias = nn.Linear(in_dim, h, bias=False)
        self.Wg = nn.Linear(in_dim, c*h)
        self.Wo = nn.Linear(c*h, in_dim)
        self.scale = c**-0.25

    def get_qkv(self, qkv):
        (bs, sl1, sl2, ch) = qkv.shape
        q, k, v = qkv.split(ch//3, -1)
        q = q.reshape(bs, sl1, sl2, self.h, self.c)
        q = q.permute([0,3,1,2,4])
        q = q.reshape(-1, sl1, sl2, self.c)
        k = k.reshape(bs, sl1, sl2, self.h, self.c)
        k = k.permute([0,3,1,2,4])
        k = k.reshape(-1, sl1, sl2, self.c)
        v = v.reshape(bs, sl1, sl2, self.h, self.c)
        v = v.permute([0,3,1,2,4])
        v = v.reshape(-1, sl1, sl2, self.c)

        return q, k, v

    def forward(self, zij):
        (bs, sl1, sl2, c) = zij.shape
        # Input projections
        zij_ = self.norm(zij)
        qkv = self.Wqkv(zij_)
        q, k, v = self.get_qkv(qkv)
        b = self.Wbias(zij_)
        g = th.sigmoid(self.Wg(zij_))
        # Attention
        q *= self.scale
        k *= self.scale
        o = self.attention(q, k, v, b, g)
        # Output projection
        o = (
            o.reshape(-1, self.h, sl1, sl2, self.c)
            .permute([0, 2, 3, 4, 1])
            .reshape(bs, sl1, sl2, -1)
        )
        zij__ = self.Wo(o)
        
        return zij__

class TriangleMultiplicative(nn.Module):
    def __init__(self,
                 in_dim,
                 c=128,
                 typ='out'
                 ):
        super(TriangleMultiplicative, self).__init__()
        self.c = c
        op = (
            # Outgoing edges
            # All combos of 2 rows (a:b, b:e) become ij's
            'abcd,aecd->abed'
            if typ == 'out' else
            # Incoming edges
            # All combos of 2 cols (a:c, b:e) become ij's
            'abcd,abed->aced'
        )
        self.es = lambda a,b: th.einsum(op, a, b)

        self.norm1 = nn.LayerNorm(in_dim)
        self.Wag = nn.Linear(in_dim, c)
        self.Wa = nn.Linear(in_dim, c)
        self.Wbg = nn.Linear(in_dim, c)
        self.Wb = nn.Linear(in_dim, c)
        self.norm2 = nn.LayerNorm(c)
        self.Wgg = nn.Linear(in_dim, in_dim)
        self.Wg = nn.Linear(c, in_dim)

    def forward(self, zij):
        # Step 1
        zij_ = self.norm1(zij)
        # Step 2
        a_gate = th.sigmoid(self.Wag(zij_))
        a = self.Wa(zij_)
        b_gate = th.sigmoid(self.Wbg(zij_))
        b = self.Wb(zij_)
        aij = a*a_gate
        bij = b*b_gate
        # Step 3
        gij = th.sigmoid(self.Wgg(zij_))
        # Step 4
        ab = self.es(aij, bij)
        zij__ = gij * self.Wg(self.norm2(ab))

        return zij__

class PairTransition(nn.Module):
    def __init__(self,
                 in_dim,
                 n=4
                 ):
        super(PairTransition, self).__init__()
        self.in_dim = in_dim
        self.n = n
        
        self.norm = nn.LayerNorm(in_dim)
        self.Wa = nn.Linear(in_dim, n*in_dim)
        self.Wo = nn.Linear(n*in_dim, in_dim)

    def forward(self, zij):
        zij_ = self.norm(zij)
        aij = self.Wa(zij_)
        zij__ = self.Wo(th.relu(aij))

        return zij__

class PairStack(nn.Module):
    def __init__(self,
                 multdict,
                 attdict,
                 ptdict,
                 drop_rate=0.25
                 ):
        super(PairStack, self).__init__()
        self.drop_rate = drop_rate
        
        #self.TMout = TriangleMultiplicative(**multdict, typ='out')
        #self.TMin = TriangleMultiplicative(**multdict, typ='in')
        self.TAstart = TriangleAttention(**attdict, typ='start')
        self.TAend = TriangleAttention(**attdict, typ='end')
        self.PT = PairTransition(**ptdict)

        self.dropTMout = nn.Identity() if drop_rate==0 else nn.Dropout(drop_rate)
        self.dropTMin = nn.Identity() if drop_rate==0 else nn.Dropout(drop_rate)
        self.dropTAstart = nn.Identity() if drop_rate==0 else nn.Dropout(drop_rate)
        self.dropTAend = nn.Identity() if drop_rate==0 else nn.Dropout(drop_rate)

        self.apply(weight_init)
                
    def forward(self, zij):
        # Beware of in_place changes to tensors needed for backprop
        # -> don't use '+=' operation
        #zij = zij  +     self.dropTMout(self.TMout(zij))
        #zij = zij  +       self.dropTMin(self.TMin(zij))
        zij = zij  + self.dropTAstart(self.TAstart(zij))
        zij = zij  +     self.dropTAend(self.TAend(zij))
        zij = zij  +                       self.PT(zij)

        return zij

class RelPos(nn.Module):
    def __init__(self, seq_len, cz):
        super(RelPos, self).__init__()
        self.cz = cz

        a = th.arange(seq_len)
        A = a[None] - a[:,None]
        A += seq_len-1
        self.ResInd = nn.Parameter(A, requires_grad=False)
        self.vbins = 2 * (seq_len - 1) + 1
        self.Wp = nn.Linear(self.vbins, cz)
    
    def forward(self):
        pij = self.Wp(F.one_hot(self.ResInd, self.vbins).type(th.float32))

        return pij

"""
inp = th.randn(10,100,100,256)
lay = TriangleAttention(256)
lay = TriangleMultiplicative(256)
lay = PairTransition(256)

multdict = {'in_dim': 256, 'c': 128}
attdict = {'in_dim': 256, 'c': 32, 'h': 4}
ptdict = {'in_dim': 256, 'n': 4}
lay = PairStack(multdict, attdict, ptdict, drop_rate=0.25)
print(sum([m.numel() for m in lay.parameters() if m.requires_grad]))

lay.apply(weight_init)
out = lay(inp)
print(out.mean(), out.std())
"""
