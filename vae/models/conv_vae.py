import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import Flatten, UnFlattenConv2D


def create_encoder(input_dims, base_filters=64, layers=4, latent=512):
    h = input_dims[1]//2**layers
    w = input_dims[2]//2**layers
    c = base_filters*2**(layers-2)
    channels = [input_dims[0] if i==0 else base_filters*2**i for i in range(layers+1)]

    encoder = []
    for i in range(layers):
        encoder.append(nn.Conv2d(channels[i], channels[i+1], 
                                 kernel_size=5, stride=2, padding=1, bias=False))
        encoder.append(nn.BatchNorm2d(channels[i+1]))
        encoder.append(nn.LeakyReLU(0.2, inplace=True))
    encoder.append(Flatten())
    encoder.append(nn.Linear(w*h*c, latent*2))

    return nn.Sequential(*encoder)


def create_decoder(output_dims, base_filters=32, layers=3, latent=512):
    #[channels, height, width]
    h = output_dims[1]//2**(layers+1)
    w = output_dims[2]//2**(layers+1)
    c = base_filters*2**(layers-1)
    channels = [c if i==layers-1 else base_filters*2**i for i in range(layers)]

    decoder = []
    decoder.append(nn.Linear(latent, w*h*c))
    decoder.append(UnFlattenConv2D([c,h,w]))
    for i in reversed(range(layers-1)):
        decoder.append(nn.ConvTranspose2d(channels[i+1], channels[i], 
                                          kernel_size=5, stride=2, padding=1, bias=False))
        decoder.append(nn.BatchNorm2d(channels[i]))
        decoder.append(nn.LeakyReLU(0.2, inplace=True))

    decoder.append(nn.ConvTranspose2d(channels[i], output_dims[0], 
                                      kernel_size=4, stride=2, padding=0, bias=False))
    return nn.Sequential(*decoder)


class Encoder(nn.Module):
    def __init__(self, input_dims, layers=3, base_filters=32, latent=20):
        super().__init__()
        self.encoder_nn = create_encoder(input_dims, base_filters=base_filters, layers=layers, latent=latent)

    def forward(self, x):
        mean, log_var = self.encoder_nn(x).chunk(2, dim=1)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, output_dims, layers, base_filters=32, latent=20):
        super().__init__()
        self.nn_decoder = create_decoder(output_dims, base_filters=base_filters, layers=layers, latent=latent)

    def forward(self, x):
        return self.nn_decoder(x)


class ConvVAE(nn.Module):
    def __init__(self, img_dim, latent=20, base_filters=16, recon_loss='MSE'):
        super().__init__()
        self.z_dim = latent
        layers=3
        self.recon_loss = recon_loss
        self.encoder = Encoder(img_dim, layers, base_filters, latent)
        self.decoder = Decoder(img_dim, layers, base_filters, latent)

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
