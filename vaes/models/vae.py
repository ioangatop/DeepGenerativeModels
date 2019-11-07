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
        return self.nn_decoder(x)


class VAE(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20, recon_loss='MSE'):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        self.recon_loss = recon_loss

    def sample(self, n_samples=1, device='cpu'):
        z = torch.randn((n_samples, self.z_dim)).to(device)
        return self.decoder(z)

    def reparameterize(self, z_mu, z_log_var):
        epsilon = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5*z_log_var)*epsilon

    def reconstuction_loss(self, x, x_hat):
        if self.recon_loss == 'MSE':
            return F.mse_loss(x_hat, x, reduction='sum')
        elif self.recon_loss == 'Binary':
            return F.binary_cross_entropy(x_hat, x, reduction='sum')
        else:
            raise NotImplementedError

    def elbo(self, x, x_hat, z_mu, z_log_var):
        kl = - 0.5*torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        recon = self.reconstuction_loss(x, x_hat)
        return (kl + recon)/x.shape[0]

    def reconstruct(self, **kwargs):
        z_mu, z_log_var = self.encoder(kwargs['x'])
        z = self.reparameterize(z_mu, z_log_var)
        return self.decoder(z)

    def forward(self, **kwargs):
        """
        Returns: Negative average elbo for given batch.
        """
        x = kwargs['x']
        z_mu, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mu, z_log_var)
        x_hat = self.decoder(z)
        loss = self.elbo(x, x_hat, z_mu, z_log_var)
        return loss
