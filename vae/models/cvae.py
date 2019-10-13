"""
Learning Structured Output Representation using Deep Conditional Generative Models

https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_encode(x, n_labels):
    one_hot = torch.zeros((*x.shape, n_labels), dtype=torch.float).to(x.device)
    one_hot[torch.arange(one_hot.shape[0]), x.long().flatten()] = 1.
    one_hot = one_hot.reshape((*x.shape, n_labels))
    return one_hot


class Encoder(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20, n_labels=10):
        super().__init__()
        self.encoder_nn = nn.Sequential(
            nn.Linear(784 + n_labels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2*z_dim)
        )

    def forward(self, x):
        mean, log_var = self.encoder_nn(x).chunk(2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20, n_labels=10):
        super().__init__()
        self.nn_decoder = nn.Sequential(
            nn.Linear(z_dim+n_labels, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.nn_decoder(z)


class CVAE(nn.Module):
    def __init__(self, hidden_dim=500, z_dim=20, n_labels=10, recon_loss='MSE'):
        super().__init__()
        self.z_dim = z_dim
        self.n_labels = n_labels
        self.encoder = Encoder(hidden_dim, z_dim, n_labels)
        self.decoder = Decoder(hidden_dim, z_dim, n_labels)
        self.recon_loss = recon_loss

    def sample(self, n_samples=10, device='cpu'):
        z = torch.randn((n_samples, self.z_dim)).to(device)
        one_hot_c = one_hot_encode(torch.LongTensor(n_samples).random_(0, 10), self.n_labels)
        joint_z = torch.cat((z, one_hot_c), dim=-1)
        sample =  self.decoder(joint_z)
        sample = sample.view(sample.size(0), 1, 28, 28)
        return sample

    def reparameterize(self, z_mu, z_log_var):
        epsilon = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5*z_log_var)*epsilon

    def reconstuction_loss(self, recon_x, x):
        if self.recon_loss == 'MSE':
            return F.mse_loss(recon_x, x, reduction='sum')
        elif self.recon_loss == 'Binary':
            return F.binary_cross_entropy(recon_x, x, reduction='sum')
        else:
            raise NotImplementedError

    def elbo(self, x, x_hat, z_mu, z_log_var):
        kl = - 0.5*torch.sum(1 + z_log_var - z_mu.pow(2) - z_log_var.exp())
        recon = self.reconstuction_loss(x_hat, x)
        return (kl + recon)/x.shape[0]

    def reconstruct(self, **kwargs):
        x, c = kwargs['x'], kwargs['c']
        x = x.view(x.size(0), -1)
        one_hot_c = one_hot_encode(c, self.n_labels)
        joint_x = torch.cat((x, one_hot_c), dim=-1)
        z_mu, z_log_var = self.encoder(joint_x)
        z = self.reparameterize(z_mu, z_log_var)
        joint_z = torch.cat((z, one_hot_c), dim=-1)
        return self.decoder(joint_z).view(x.size(0), 1, 28, 28)

    def forward(self, **kwargs):
        """
        Returns: Negative average elbo for given batch.
        """
        x, c = kwargs['x'], kwargs['c']
        x = x.view(x.size(0), -1)
        one_hot_c = one_hot_encode(c, self.n_labels)

        joint_x = torch.cat((x, one_hot_c), dim=-1)
        z_mu, z_log_var = self.encoder(joint_x)
        z = self.reparameterize(z_mu, z_log_var)

        joint_z = torch.cat((z, one_hot_c), dim=-1)
        x_hat = self.decoder(joint_z)

        loss = self.elbo(x, x_hat, z_mu, z_log_var)
        return loss
