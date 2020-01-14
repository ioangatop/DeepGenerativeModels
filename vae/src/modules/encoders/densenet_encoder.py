import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from src.modules.nn_layers import *
from src.utils.args import args


class DensenetEncoder(Encoder):
    def __init__(self, output_dim, input_shape):
        super().__init__()

        nc = input_shape[0]
        if nc == 3:
            H_OUT, W_OUT = input_shape[-2] // 4, input_shape[-1] // 4
        else:
            H_OUT, W_OUT = 7, 7     # HACK only for 28x28 dimentions!

        self.main_nn = nn.Sequential(
            DenseNet(nc, 15, 3),

            Conv2d(45+nc, 48, 
                kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            DenseNet(48, 16, 3),

            Conv2d(96, 96,
                kernel_size=3, stride=2, padding=1),
            nn.ELU(),

            DenseNet(96, 16, 6),

            Conv2d(192, 96, 
                kernel_size=1, stride=1, padding=0),
            nn.ELU(),

            Conv2d(96, 24,
                kernel_size=1, stride=1, padding=0),
            nn.ELU(),

            Flatten(),

            nn.Linear(24 * H_OUT * W_OUT, 2 * output_dim)
        )


    def forward(self, input):
        mu, logvar = self.main_nn(input).chunk(2, 1)
        return mu, F.softplus(logvar)


if __name__ == "__main__":
    pass
