import math

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from .prior import Prior
from src.modules.nn_layers import *
from src.modules.distributions import *
from src.utils import args


class VampPrior(Prior):
    """
    VAE with a VampPrior

    https://arxiv.org/abs/1705.07120

    Parameters
    ----------
        • n_components: number of compoments. Default: 512
        • p_mean: Initialization of pseudo inputs mean. Default: -0.05
        • p_std: Initialization of pseudo inputs std. Default: 0.01

    """
    def __init__(self, output_shape, n_components=512, p_mean=0.5, p_std=0.5):
        super().__init__()
        self.output_shape = output_shape
        self.n_components = n_components

        # init pseudo-inputs
        self.means = nn.Sequential(
            nn.Linear(self.n_components, np.prod(self.output_shape)),
            nn.Hardtanh(min_val=-1., max_val=1.)
        )

        self.normal_init(self.means[0], p_mean, p_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = Variable(torch.eye(self.n_components, self.n_components),
                                   requires_grad=False).to(args.device)

    def normal_init(self, m, mean=-0.05, std=0.01):
        m.weight.data.normal_(mean, std)

    def sample(self, n_samples, encoder):
        means = self.means(self.idle_input)[0:n_samples]
        means = means.view(means.shape[0], *self.output_shape)
        z_sample_gen_mean, z_sample_gen_logvar = encoder.forward(means)
        z_sample_rand = reparameterize(z_sample_gen_mean, z_sample_gen_logvar)
        return z_sample_rand

    def forward(self, x, encoder, dim=1):
        y = self.means(self.idle_input)
        y = y.view(y.shape[0], *self.output_shape)
        u_q_mean, u_q_logvar = encoder.forward(y)

        # expand z
        u_expand = x.unsqueeze(1)
        means = u_q_mean.unsqueeze(0)
        logvars = u_q_logvar.unsqueeze(0)

        a = log_normal_diag(u_expand, means, logvars, dim=2) \
            - math.log(self.n_components)

        a_max, _ = torch.max(a, 1)

        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), dim=dim))
        return log_prior

    def __str__(self):
      return "VampPrior"

if __name__ == "__main__":
    pass
