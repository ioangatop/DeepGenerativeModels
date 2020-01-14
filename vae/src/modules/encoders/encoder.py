import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder: q(z|x)

    Performs amortized inference using Residual Neural Networks.

    Parameters
    ----------
        • output_dim (int): Dimentionality of latent space

        • input_shape (tuple): Shape of input data.

        • n_hidden (int): Number of hidden channels for the network.

        • n_blocks (int): Number of residual box.

        • n_block_hidden (int): Number of hidden channels for the network
                                inside the block.

    Returns
    -------
        • mean (torch.tensor): Tensor of with the mean of the latent 
                               representations.

        • log_var (torch.tensor): Tensor of with the log variance of the 
                                  latent representations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        raise NotImplementedError


if __name__ == "__main__":
    pass
