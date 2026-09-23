from .base_task import Task, MzAbInp, custom_sampler
import torch
import numpy as np

class VAE(Task):
    def __init__(self, binsz=0.1, mzlims=[0, 2000], kl_weight=1.0, subsample_rate=0):
        super().__init__('both')
        self.binsz = binsz
        self.mzlims = mzlims
        self.totbins = int(np.floor((mzlims[1] - mzlims[0]) / binsz))
        self.bin_edges = torch.linspace(mzlims[0], mzlims[1], self.totbins)
        self.bin_centers = self.bin_edges[:-1] + binsz / 2.0
        self.kl_weight = kl_weight
        self.subsample_rate = subsample_rate
    
    def subsample(self, batch):
        device = batch['mz'].device
        
        # How many peaks should we keep
        keep_amount = (batch['length']*self.subsample_rate).round().int()
        max_seq_length = keep_amount.max()

        # Index array
        array = torch.arange(
            batch['ab'].shape[1], device=device
        )[None].tile([batch['ab'].shape[0], 1])
        
        # Sample
        sample = custom_sampler(batch['ab'], 5.)
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
        sorted_intensities = sampled_intensities.gather(1, b)

        return {
            'mz': sorted_mzs[:, :max_seq_length],
            'ab': sorted_intensities[:, :max_seq_length],
        }

    def inptarg(self, batch):
        mzab = MzAbInp(batch if self.subsample_rate==0 else self.subsample(batch))
        inp = {
            'x': mzab,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length']
        }

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

        return inp

    def loss(self, prediction):
        spectrum = prediction['spectrum']
        recon_loss = 1 - (self.target*spectrum).sum(1) / self.target.norm(dim=1) / spectrum.norm(dim=1)

        mu = prediction['mu']
        std = prediction['std']

        kl_loss = -0.5 * torch.sum(1+torch.log(std**2) - mu**2 - std**2, dim=1).mean()

        loss = recon_loss.mean() + (self.kl_weight * kl_loss)

        return loss
