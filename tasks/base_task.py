from collections import deque
import numpy as np
import torch as th

MzAbInp = lambda batch: th.cat(
    [batch['mz'][...,None], batch['ab'][...,None]],
    axis=-1
)

beta_ab_sampler = lambda ab, c0=1.0: 1 - th.distributions.Beta(concentration1=ab.clamp(1e-5, 1), concentration0=c0).sample()

custom_sampler = lambda ab, num=1.0: th.rand_like(ab)**(num / ab - 1) if num>=1 else th.rand_like(ab) # num: higher means flatter sampler

class Task:
    def __init__(self, typ, maxlen=50):
        assert typ.lower() in ['mz', 'ab', 'both', 'charge', 'mass']
        self.typ = typ.lower()
        self.maxlen = maxlen
        self.running_loss = {'main': deque(maxlen=maxlen)}
        self.total_loss = {'main': 0}
        self.total_counter = 0
    
    def add_loss_key(self, key):
        self.running_loss[key] = deque(maxlen=self.maxlen)
        self.total_loss[key] = 0

    def log_loss(self, nploss):
        for key in self.running_loss.keys():
            self.running_loss[key].append(nploss)
            self.total_loss[key] += nploss
        self.total_counter += 1

    def calc_avg_total_loss(self):
        outputs = {}
        for key in self.total_loss.keys():
            outputs[key] = self.total_loss[key] / (self.total_counter+1e-10)
        
        return outputs

    def calc_avg_running_loss(self):
        outputs = {}
        for key in self.running_loss.keys():
            outputs[key] = (
                np.mean(self.running_loss[key]) 
                if len(self.running_loss[key])>0 else 
                0
            )

        return outputs

    def reset_total_loss(self):
        for key in self.total_loss.keys():
            self.total_loss[key] = 0
        self.total_counter = 0

    def inptarg(self, *args, **kwargs):
        pass
