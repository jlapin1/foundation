from .base_task import Task, MzAbInp, custom_sampler
import torch
import numpy as np
from copy import deepcopy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

local_MzAbInp = lambda batch: MzAbInp(batch), None
local_custom_sampler = lambda ab, num: torch.where(ab==0, 0, custom_sampler(ab, num))

class EMAUpdater:
    def __init__(self, online_model, target_model, decay=0.996):
        self.online_model = online_model
        self.target_model = target_model
        self.decay = decay

    @torch.no_grad()
    def copy_weights(self):
        for online_param, target_param in zip(self.online_model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(online_param.data)

    @torch.no_grad()
    def update(self):
        for online_param, target_param in zip(self.online_model.parameters(), self.target_model.parameters()):
            if online_param.requires_grad:
                target_param.data.mul_(self.decay).add_(online_param.data, alpha=1.0-self.decay)

class JEPA(Task):
    def __init__(self, encoder, context_prop, target_prop, target_blocks):
        super().__init__('both')
        self.target_encoder = deepcopy(encoder)
        for param in self.target_encoder.parameters(): param.requires_grad=False
        self.target_encoder.to(device)
        self.target_encoder.eval()
        self.ema = EMAUpdater(encoder, self.target_encoder)

        self.context_prop = context_prop
        self.target_prop = target_prop
        self.target_blocks = target_blocks

    def subsample(self, batch, rate, samp_num=5):
        device = batch['mz'].device
        
        # How many peaks should we keep
        keep_amount = (batch['length']*rate).round().int()
        max_seq_length = keep_amount.max()

        # Index array
        array = torch.arange(
            batch['ab'].shape[1], device=device
        )[None].tile([batch['ab'].shape[0], 1])
        
        # Sample
        sample = custom_sampler(batch['ab'], samp_num)
        # but don't select any non-peaks
        sample[array>=batch['length'][:,None]] = 0
        # Sort top sampled peaks
        a = sample.argsort(dim=1, descending=True)
        
        # Get new subsampled spectra, sorted by m/z
        sampled_mzs = batch['mz'].gather(1, a)
        keep_bool_array = torch.where(array >= keep_amount[:,None])
        sampled_mzs[keep_bool_array] = 9e9
        sampled_intensities = batch['ab'].gather(1, a)
        sorted_mzs, b = sampled_mzs.sort(dim=1)
        sorted_mzs[keep_bool_array] = 0
        sorted_intensities = sampled_intensities.gather(1, b)

        return {
            'mz': sorted_mzs[:, :max_seq_length],
            'ab': sorted_intensities[:, :max_seq_length],
        }, a[:,:max_seq_length]

    def inptarg(self, batch):
        BS, SL = batch['mz'].shape
        target_sl = int(SL*self.target_prop)

        mzab, inds = local_MzAbInp(batch) if self.context_prop==0 else self.subsample(batch, self.context_prop)
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
        target_rep = self.target_encoder(**target_input)['emb']
        
        intensity_array = batch['ab'][:,None].tile([1, self.target_blocks, 1]).reshape(-1, SL)
        sample = local_custom_sampler(intensity_array, 0)
        batch_indices = torch.arange(intensity_array.shape[0])[:,None].tile([1, inds.shape[1]])
        inds_indices = inds[:,None].tile([1, self.target_blocks, 1]).reshape(-1, inds.shape[1])
        sample[batch_indices, inds_indices] = 0
        M = sample.argsort(dim=1)[:, -target_sl:]
        mz_array = batch['mz'][:,None].tile([1, self.target_blocks, 1]).reshape(-1, SL)
        target_mz = mz_array.gather(-1, M)
        self.head_kwargs['target_mz'] = target_mz
        
        target_rep = target_rep[:,None].tile([1, self.target_blocks, 1, 1]).reshape(-1, *target_rep.shape[1:])
        self.targets = target_rep.gather(1, M[...,None].expand(-1, -1, target_rep.shape[-1]))

        """
        masses = batch['mz']
        intensities = batch['ab']
        
        # Filter data to only inside m/z range
        mask = (masses > self.mzlims[0]) & (masses < self.mzlims[1])
        masses[mask==False] = 0
        intensities[mask==False] = 0
        
        bin_indices = torch.bucketize(masses, self.bin_edges.to(masses.device))
        bin_indices = torch.clamp(bin_indices, 0, self.totbins-1)

        sum_intensities = torch.zeros(masses.shape[0], self.totbins, device=masses.device, dtype=masses.dtype)
        counts = torch.zeros(masses.shape[0], self.totbins, device=masses.device, dtype=masses.dtype)

        sum_intensities.scatter_add_(1, bin_indices, intensities)
        counts.scatter_add_(1, bin_indices, torch.ones_like(intensities))

        avg_intensities = torch.where(counts>0, sum_intensities / counts, 0.0)
        avg_intensities = avg_intensities / avg_intensities.max(1, keepdims=True)[0]
        
        self.target = avg_intensities
        """

        return inp

    def loss(self, prediction):
        loss = (self.targets - prediction).square().reshape(prediction.shape[0], -1).mean(-1)
        return loss

    def on_train_step_end(self):
        self.ema.update()

    def on_save(self, save_directory, remark):
        torch.save(self.target_encoder.state_dict(), f"{save_directory}/weights/target_encoder_JEPA_{remark}.wts")

