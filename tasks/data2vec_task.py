from .base_task import Task, MzAbInp, custom_sampler
import torch
import numpy as np
from copy import deepcopy
from utils import compute_embedding_variance_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

local_MzAbInp = lambda batch: MzAbInp(batch), None
local_custom_sampler = lambda ab, num: torch.where(ab==0, 0, custom_sampler(ab, num))

class EMAUpdater:
    def __init__(self, online_model, target_model, decay_0=0.999, decay_e=0.9999, steps=1e5, **kwargs):
        self.online_model = online_model
        self.target_model = target_model
        self.decay_0 = decay_0
        self.decay_e = decay_e
        self.steps = steps
        
        self.step_size = (decay_e - decay_0) / steps
        self.decay_current = decay_0

    @torch.no_grad()
    def copy_weights(self):
        for online_param, target_param in zip(self.online_model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(online_param.data)

    @torch.no_grad()
    def update(self):
        for online_param, target_param in zip(self.online_model.parameters(), self.target_model.parameters()):
            if online_param.requires_grad:
                target_param.data.mul_(self.decay_current).add_(online_param.data, alpha=1.0-self.decay_current)
        self.decay_current = np.minimum(self.decay_e, self.decay_current+self.step_size)

class Data2Vec(Task):
    def __init__(self, encoder, mask_prop=0.15, top_blocks=5, ema_params={}):
        super().__init__('both')
        self.target_encoder = deepcopy(encoder)
        self.target_encoder.return_all = True
        for param in self.target_encoder.parameters(): param.requires_grad=False
        self.target_encoder.to(device)
        self.target_encoder.eval()
        self.ema = EMAUpdater(encoder, self.target_encoder, **ema_params)
        
        self.mask_prop = mask_prop
        self.K = top_blocks
        self.layer_norm = torch.nn.LayerNorm(self.target_encoder.run_units, elementwise_affine=False, bias=False)

    def masking_function(self, batch, rate, samp_num=0):
        device = batch['mz'].device
        
        # How many peaks should we keep
        mask_amount = (batch['length']*rate).round().int()
        
        # Index array
        array = torch.arange(
            batch['ab'].shape[1], device=device
        )[None].tile([batch['ab'].shape[0], 1])
        batch_array = torch.arange(batch['ab'].shape[0], device=device)[:,None].tile([1, batch['ab'].shape[1]])
        
        # Sample
        sample = custom_sampler(batch['ab'], samp_num)
        # but don't select any non-peaks
        sample[array>=batch['length'][:,None]] = 2
        # Sort top sampled peaks
        inds = sample.argsort(dim=1, descending=False)
        inds = (batch_array[array < mask_amount[:,None]], inds[array < mask_amount[:,None]])
        
        mz = batch['mz'].clone()
        mz[inds] = -1
        ab = batch['ab'].clone()
        ab[inds] = -1
        
        return {
            'mz': mz,
            'ab': ab,
        }, inds

    def inptarg(self, batch):
        BS, SL = batch['mz'].shape
        self.SL = SL

        mzab, inds = local_MzAbInp(batch) if self.mask_prop==0 else self.masking_function(batch, self.mask_prop)
        inp = {
            'x': MzAbInp(mzab),
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length']
        }
        
        target_input = {
            'x': MzAbInp(batch),
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length'],
        }
        
        target_outs = self.target_encoder(**target_input)['all']
        concat = self.layer_norm(torch.cat(target_outs[-self.K:])[:self.K*BS, -SL:])
        self.target = concat.reshape(self.K, BS, SL, -1).mean(0)
        
        self.mask = torch.zeros_like(batch['mz']).bool()
        self.mask[inds] = True
        
        return inp

    def loss(self, prediction):
        BS, _, UNITS = prediction.shape
        loss = ((self.target - prediction[:, -self.SL:])*self.mask[...,None]).square().reshape(BS, -1).sum(-1) / (self.mask.sum(-1)*UNITS+1e-7)
        return loss

    def on_train_step_end(self):
        self.ema.update()

    def on_save(self, save_directory, remark):
        torch.save(self.target_encoder.state_dict(), f"{save_directory}/weights/target_encoder_JEPA_{remark}.wts")

    def log_dict(self):
        dic = compute_embedding_variance_metrics(self.target)
        return {
            'ema_tau': self.ema.decay_current.item()
        } | dic

