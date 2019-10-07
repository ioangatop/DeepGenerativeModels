"""
Auto-Encoding Variational Bayes

https://arxiv.org/abs/1312.6114
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 1, 28, 28)


class Encoder(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.encoder_nn = nn.Sequential(
            Flatten(),
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2*z_dim)
        )

    def forward(self, x):
        """
        Perform forward pass of encoder.
        """
        mean, log_var = self.encoder_nn(x).chunk(2, dim=1)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.nn_decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid(),
            UnFlatten()
        )

    def forward(self, x):
        """
        Perform forward pass of encoder.
        """
        return self.nn_decoder(x)


class VAE(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def sample(self, n_samples=1, device='cpu'):
        z = torch.randn((n_samples, self.z_dim)).to(device)
        return self.decoder(z)

    def reparameterize(self, z_mu, z_log_var):
        epsilon = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5*z_log_var)*epsilon

    def elbo(self, x, x_hat, z_mu, z_log_var):
        kl = - 0.5*torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        return (kl + bce)/x.shape[0]

    def forward(self, **kwargs):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        x = kwargs['x']
        z_mu, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mu, z_log_var)
        x_hat = self.decoder(z)
        loss = self.elbo(x, x_hat, z_mu, z_log_var)
        return loss
