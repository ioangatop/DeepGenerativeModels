"""



"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import GatedConv2d, GatedConvTranspose2d

class Encoder(nn.Module):
    def __init__(self, n_chanels=1, batch_size=256, z_dim=20):
        super().__init__()
        act = None

        self.encoder_nn = nn.Sequential(
            GatedConv2d(n_chanels, 32, 5, 1, 2, activation=act),
            GatedConv2d(32, 32, 5, 2, 2, activation=act),
            GatedConv2d(32, 64, 5, 1, 2, activation=act),
            GatedConv2d(64, 64, 5, 2, 2, activation=act),
            GatedConv2d(64, 64, 5, 1, 2, activation=act),
            GatedConv2d(64, batch_size, 7, 1, 0, activation=act)
        )
        self.encoder_mean = nn.Linear(batch_size, z_dim)
        self.encoder_var = nn.Sequential(
            nn.Linear(batch_size, z_dim),
            nn.Softplus(),
            nn.Hardtanh(min_val=0.01, max_val=7.)

        )

    def forward(self, x):
        out = self.encoder_nn(x)
        out = out.view(out.size(0), -1)
        mean, log_var = self.encoder_mean(out), self.encoder_var(out)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, n_chanels=1, batch_size=256, z_dim=20):
        super().__init__()
        self.z_dim = z_dim

        act = None
        self.decoder_nn = nn.Sequential(
            GatedConvTranspose2d(z_dim, 64, 7, 1, 0, activation=act),
            GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act),
            GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act),
            GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act),
            GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act),
            GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act)
        )

        self.decoder_mean = nn.Sequential(
            nn.Conv2d(32, batch_size, 5, 1, 2),
            nn.Conv2d(batch_size, n_chanels * batch_size, 1, 1, 0),
            # output shape: batch_size, num_channels * num_classes, pixel_width, pixel_height
        )

    def forward(self, z):
        z = z.view(z.size(0), self.z_dim, 1, 1)
        return self.decoder_mean(self.decoder_nn(z))


class GatedVAE(nn.Module):
    def __init__(self, n_chanels=1, batch_size=256, z_dim=20, recon_loss='MSE'):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(n_chanels, batch_size, z_dim)
        self.decoder = Decoder(n_chanels, batch_size, z_dim)
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
