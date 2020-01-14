import torch

class StandardNormal:
    def __init__(self, z_dim):
        self.z_dim = z_dim

    def sample(self, n_samples=1, **kwargs):
        return torch.randn((n_samples, self.z_dim))

    def forward(self, x, dim=None, **kwargs):
        if dim is None:
            return torch.sum(-0.5 * torch.pow(x, 2))
        else:
            return torch.sum(-0.5 * torch.pow(x, 2), dim)

    def __str__(self):
      return "StandardNormal"


if __name__ == "__main__":
    pass
