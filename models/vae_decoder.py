import torch as th
from torch import nn
from torch.nn import functional as F

def decoder_block(input_dim, output_dim, use_norm=False, activation=True):
    layers = [
        nn.Linear(input_dim, output_dim, bias=use_norm==False),
    ]
    #if use_norm:
        #layers.append(nn.LayerNorm(output_dim))
        #layers.append(nn.RMSNorm(output_dim))
    if activation:
        layers.append(nn.GELU())
    
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, use_norm=False, activation=True):
        super(ResBlock, self).__init__()
        output_dim = input_dim if output_dim==None else output_dim
        self.block1 = decoder_block(input_dim, output_dim, use_norm=use_norm, activation=True)
        self.block2 = decoder_block(output_dim, input_dim, use_norm=use_norm, activation=False)

    def forward(self, x):
        resid = self.block1(x)
        resid = self.block2(resid)
        return F.gelu(x+resid)

class VAEDecoder(nn.Module):
    def __init__(
        self,
        in_units,
        latent_dim=16,
        hidden_dim=256,
        output_dim=20000,
        depth=4,
        use_softplus_std=False,
        class_token_input=False,
    ):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_softplus_std = use_softplus_std
        self.class_token_input = class_token_input

        # Layers
        self.fc = nn.Linear(in_units, 2*latent_dim)
        layers = [decoder_block(latent_dim, hidden_dim)] + [ResBlock(hidden_dim, hidden_dim, use_norm=True) for _ in range(depth)]
        self.decoder_block = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=True),
            nn.Sigmoid(),
        )

    def _sigma_to_std(self, sigma: th.Tensor, eps: float=1e-8) -> th.Tensor:
        if self.use_softplus_std:
            return F.softplus(sigma) + eps
        else:
            return th.exp(0.5 * sigma)

    def reparameterize(self, mu: th.Tensor, std: th.Tensor) -> th.Tensor:
        eps = th.randn_like(std)
        return mu + eps * std

    def decode(self, z: th.Tensor) -> th.Tensor:
        out = self.decoder_block(z)
        return self.final(out)

    def forward(self, x):
        if self.class_token_input:
            x = x[:,0]
        func = lambda x: x.mean(dim=1) if x.ndim == 3 else x
        
        mu, sigma = func(self.fc(x)).chunk(2, dim=-1)
        std = self._sigma_to_std(sigma)
        z = self.reparameterize(mu, std)

        decoded = self.decode(z)

        return {'spectrum': decoded, "mu": mu, 'std': std}

    def sample(self, batch_size: int) -> th.Tensor:
        z = th.randn(batch_size, self.latent_dim)
        decoded = self.decode(z)

        return decoded

