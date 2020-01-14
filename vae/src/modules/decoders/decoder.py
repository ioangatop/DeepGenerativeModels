import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder base class: p(x|z)

    Performs amortized inference using Residual Neural Networks.

    Parameters
    ----------
        • output_shape (tuple): Shape of the output data.

        • input_dim (int): Dimentionality of the input latent space.

        • n_hidden (int): Number of hidden channels for the network.

        • n_blocks (int): Number of residual box.

        • n_block_hidden (int): Number of hidden channels for the network
                                inside the block.

    Returns
    -------
        • mean (torch.tensor): Tensor of with the mean of the reconstructed
                               image.

        • log_var (torch.tensor): Tensor of with the log variance of the 
                                  reconstructed image.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        raise NotImplementedError