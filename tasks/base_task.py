from collections import deque
import numpy as np

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
