from functools import partial 

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import args
from src.modules import *


class VAE(nn.Module):
    """
    Variational AutoEncoder.
    """
    def __init__(self, x_shape):
        super().__init__()
        self.x_shape = x_shape

        self.x_dim = np.prod(self.x_shape)
        self.z_dim = args.z_dim

        self.beta = args.beta

        # Reconstruction loss
        if args.reconstraction_loss == 'mse_loss':
            self.recon_loss = partial(mse_loss, reduction='none')
            self.sample_distribution = lambda x: x
        elif args.reconstraction_loss == 'discretized_logistic_loss':
            self.num_classes = 256
            self.recon_loss = partial(discretized_logistic_loss)
            self.sample_distribution = partial(sample_from_discretized_logistic_loss, nc=self.x_shape[0], random_sample=False)
        elif args.reconstraction_loss == 'discretized_mix_logistic_loss':
            self.nmix = 10
            self.recon_loss = partial(discretized_mix_logistic_loss, nc=self.x_shape[0], nmix=self.nmix)
            self.sample_distribution = partial(sample_from_discretized_mix_logistic, nc=self.x_shape[0], nmix=self.nmix, random_sample=False)

        # Prior: p(z)
        if args.prior == 'std_normal':
            self.prior = StandardNormal(self.z_dim)
        elif args.prior == 'mog':
            self.prior = MixtureOfGaussians(self.z_dim, num_mixtures=10)
        elif args.prior == 'vampprior':
            self.prior = VampPrior(self.x_shape, n_components=512)

        # Encoder: q(z|x)
        self.encoder = DensenetEncoder(self.z_dim, self.x_shape)

        # Decoder: p(x|z)
        self.decoder = ResidualDecoder(self.x_shape, self.z_dim)


    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5*z_log_var)*epsilon


    def generate(self, n_samples=20):
        z = self.prior.sample(n_samples=n_samples, encoder=self.encoder).to(args.device)
        x_logits = self.decoder.forward(z)
        return x_logits if args.reconstraction_loss == 'mse_loss' else self.sample_distribution(x_logits, random_sample=True)


    def reconstruct(self, x):
        z_q_mean, z_q_logvar = self.encoder(x)
        z = self.reparameterize(z_q_mean, z_q_logvar)
        x_logits = self.decoder(z)
        x_hat = self.sample_distribution(x_logits)
        return x_hat


    def nelbo(self, x, x_mean, z_q, z_q_mean, z_q_logvar):
        z_q = z_q.view(x.shape[0], -1)
        z_q_mean = z_q_mean.view(x.shape[0], -1)
        z_q_logvar = z_q_logvar.view(x.shape[0], -1)

        # Reconstraction loss
        RE = self.recon_loss(x, x_mean)

        # Regularization loss
        log_q_z = log_normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
        log_p_z = self.prior.forward(z_q, encoder=self.encoder, dim=1)
        KL = log_p_z - log_q_z

        # Total lower bound loss
        loss = RE + self.beta * KL
        nelbo = -loss.mean(dim=0)

        losses = {
            "nelbo" : nelbo.item(),
            "RE"    : - RE.mean(dim=0).item(),
            "KL"    : - KL.mean(dim=0).item()
        }
        return nelbo, losses


    def forward(self, x):
        z_q_mean, z_q_logvar = self.encoder.forward(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)
        x_mean = self.decoder.forward(z_q)
        nelbo, losses = self.nelbo(x, x_mean, z_q, z_q_mean, z_q_logvar)
        return nelbo, losses


if __name__ == "__main__":
    pass
