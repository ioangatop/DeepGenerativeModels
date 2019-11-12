import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import Flatten, UnFlattenConv2D


class Encoder(nn.Module):
    def __init__(self, n_chanels, h_dim=32, z_dim=20):
        super().__init__()
        self.encoder_nn = nn.Sequential(
            # i_dim x 28 x 28
            nn.Conv2d(n_chanels, h_dim, 3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # h_dim x 14 x 14
            nn.Conv2d(h_dim, 2*h_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2*h_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
            # 2*h_dim x 7 x 7
            Flatten(),
            nn.Linear(4096, 2*z_dim)
        )

    def forward(self, x):
        mean, log_var = self.encoder_nn(x).chunk(2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, n_chanels, h_dim=32, z_dim=20):
        super().__init__()
        self.nn_decoder = nn.Sequential(
            UnFlattenConv2D(z_dim),
            # z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, 4*h_dim, 5, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(4*h_dim),
            nn.ReLU(True),
            # 4*h_dim x 1 x 1
            nn.ConvTranspose2d(4*h_dim, 2*h_dim, 5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(2*h_dim),
            nn.ReLU(True),
            # 2*h_dim x 1 x 1
            nn.ConvTranspose2d(2*h_dim, h_dim, 5, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(True),
            # h_dim x 1 x 1
            nn.ConvTranspose2d(h_dim, n_chanels, 4, stride=1, padding=0, bias=False),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.nn_decoder(x)


class ConvVAE(nn.Module):
    def __init__(self, n_chanels, z_dim=20, h_dim=32, recon_loss='MSE'):
        super().__init__()
        self.z_dim = z_dim
        self.recon_loss = recon_loss
        self.encoder = Encoder(n_chanels, h_dim, z_dim)
        self.decoder = Decoder(n_chanels, h_dim, z_dim)

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
