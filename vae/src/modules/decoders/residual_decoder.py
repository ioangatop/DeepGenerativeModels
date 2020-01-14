import torch
import torch.nn as nn

from .decoder import Decoder
from src.modules.nn_layers import *
from src.utils.args import args


class ResidualDecoder(Decoder):
    def __init__(self, output_shape, input_dim):
        super().__init__()

        # Output Channels
        if args.reconstraction_loss=='mse_loss':
            nc = output_shape[0]
        elif args.reconstraction_loss=='discretized_logistic_loss':
            nc = 256 * output_shape[0]
        elif args.reconstraction_loss=='discretized_mix_logistic_loss':
            nmix = 10
            nc = (output_shape[0] * 3 + 1) * nmix

        H_IN, W_IN = output_shape[-2] // 4, output_shape[-1] // 4

        self.main_nn = nn.Sequential(
            nn.Linear(input_dim, 24 * H_IN * W_IN),

            UnFlatten((24, H_IN, W_IN)),

            ConvTranspose2d(24, 96,
                kernel_size=1, stride=1, padding=0),
            nn.ELU(),

            DeResidualNetwork(96, [96, 96, 96], [1, 1, 1], [0, 0, 0]),

            ConvTranspose2d(96, 96,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),

            DeResidualNetwork(96, [96, 96], [1, 1, 1], [0, 0]),

            ConvTranspose2d(96, 48,
                kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),

            DeResidualNetwork(48, [48, 48], [1, 1], [0, 0]),

            Conv2d(48, nc,
                kernel_size=1, stride=1, padding=0),
        )
        
    def forward(self, input):
        input_hat = self.main_nn(input)
        return input_hat


if __name__ == "__main__":
    pass
