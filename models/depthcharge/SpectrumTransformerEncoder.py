from collections.abc import Callable

import torch
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/foundational")
from models.depthcharge.sinusoidal import PeakEncoder


class SpectrumTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
        sequence_length: int = 100
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self.run_units = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout
        self.sl = sequence_length

        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:
            self.peak_encoder = PeakEncoder(d_model)
        else:
            self.peak_encoder = torch.nn.Linear(2, d_model)

        # The Transformer layer
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )
        
        # JL: Required member of encoder model
        self.global_step = torch.nn.Parameter(torch.tensor(0), requires_grad=False)

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward
    
    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def dropout(self) -> float:
        return self._dropout
    
    # JL: Renamed forward to forward_. Need to create forward() that works with my model's inputs
    def forward_(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        spectra = torch.stack([mz_array, intensity_array], dim=2)
        n_batch = spectra.shape[0]
        # JL: zeros is the new mask, commenting out prepending of False's for precursor mask
        zeros = ~spectra.sum(dim=2).bool()
        mask = zeros#torch.cat(
        #    [torch.tensor([[False]] * n_batch).type_as(zeros), zeros], dim=1
        #)
        peaks = self.peak_encoder(spectra)

        # Add the precursor information:
        # JL: Commenting out the prepending of precursor information
        """latent_spectra = self.precursor_hook(
            mz_array=mz_array,
            intensity_array=intensity_array,
            **kwargs,
        )
        peaks = torch.cat([latent_spectra[:, None, :], peaks], dim=1)"""
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask
    
    # JL: added function that takes my model's expected input and turns it 
    #     into Casanovo's expected input. Returns my model's expected output.
    def forward(self, x, return_mask=False, **kwargs):
        [mz, ab] = [m[...,0] for m in x.split(1, -1)]
        out, mask = self.forward_(mz_array=mz, intensity_array=ab)
        output = {
            'emb': out,
            'mask': mask.type(torch.float32)*1e5 if return_mask else None
        }

        return output

    def precursor_hook(
        self,
        mz_array: torch.Tensor,
        **kwargs: dict
    ) -> torch.Tensor:

        return torch.zeros((mz_array.shape[0], self.d_model)).type_as(mz_array)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def total_params(self):
        return sum([m.numel() for m in self.parameters()])

#####################
### Encoder model ###
#####################
def dc_encoder(sequence_length):
    enc_dict = {
        'd_model': 512,
        'nhead': 8,
        'dim_feedforward': 2048,
        'n_layers': 9,
        'dropout': 0.0,
        'sequence_length': sequence_length,
    }
    encoder = SpectrumTransformerEncoder(**enc_dict)

    return encoder
"""
##############
### Loader ###
##############
import sys
sys.path.append("/cmnfs/home/j.lapin/projects/foundational")
import yaml
with open("/cmnfs/home/j.lapin/projects/foundational/yaml/datasets.yaml", 'r') as f:
    dc = yaml.safe_load(f)
dc['pretrain']['mdsaved_path'] = "/cmnfs/home/j.lapin/projects/foundational/save/mdsaved"
from loaders.loader import DatasetObj, DataLoader
dataset = DatasetObj(**dc['pretrain'])
L = DataLoader(
    dataset=dataset,
    num_workers=dc['num_workers'],
    batch_size=100,
    shuffle=True,
)

batch = next(iter(L))

encoder = dc_encoder()

enc_inp = {
    'x': torch.cat([batch['mz'][...,None], batch['ab'][...,None]], -1),
    'charge': batch['charge'],
    'mass': batch['mass'],
    'return_mask': True
}
out = encoder(**enc_inp)
print(out['mask'])
"""
