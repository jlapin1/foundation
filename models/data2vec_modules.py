import torch as th
from torch import nn

class Data2VecPredictor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Data2VecPredictor, self).__init__()
        self.nn = nn.Linear(10,10)

    def forward(self, emb, **kwargs):
        return emb
