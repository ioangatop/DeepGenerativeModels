import torch
import torch.nn as nn


class Prior(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        return torch.randn((n_samples, self.z_dim))

    def forward(self, input):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


if __name__ == "__main__":
    pass
