import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


#######################################
#######         FLATTEN        ########
#######################################
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, unflatten_size):
        super().__init__()
        if isinstance(unflatten_size, tuple):
            self.c = unflatten_size[0]
            self.h = unflatten_size[1]
            self.w = unflatten_size[2]
        elif isinstance(unflatten_size, int):
            self.c = unflatten_size
            self.h = 1
            self.w = 1

    def forward(self, x):
        return x.view(x.size(0), self.c, self.h, self.w)


#######################################
########     NEURAL NETWORKS   ########
#######################################
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1, bias=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        return self.conv(input)


class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, output_padding=0, dilation=1, groups=1, bias=True):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, output_padding=output_padding,
                                       dilation=dilation, groups=groups, bias=bias)
    def forward(self, input):
        return self.conv(input)


#######################################
########   RESIDUAL NETWORKS   ########
#######################################
class ResidualBlock(nn.Module):
    """
    Residual Block.
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.activation = nn.ELU()
        self._block = nn.Sequential(
            Conv2d(in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1),
            self.activation,
            Conv2d(out_channels, out_channels,
                kernel_size=3, stride=1, padding=1)
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        x = self.activation(x)
        residual = x if self.downsample is None else self.downsample(x)
        out = self._block(x)
        return out + residual


class ResidualNetwork(nn.Module):
    """
    Stacks of Residual blocks make a Residual Network.
    """
    def __init__(self, in_channels, out_channels, strides):
        super().__init__()

        _blocks = []
        for out_channel, stride in zip(out_channels, strides):
            _blocks.append(ResidualBlock(in_channels, out_channel, stride))
            in_channels = out_channel

        self._blocks = nn.Sequential(*_blocks)

    def forward(self, input):
        return self._blocks(input)


class DeResidualBlock(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        super().__init__()

        self.activation = nn.ELU()
        self._block = nn.Sequential(
            ConvTranspose2d(in_channels, out_channels,
                kernel_size=3, stride=stride, padding=1, output_padding=output_padding),
            self.activation,
            ConvTranspose2d(out_channels, out_channels,
                kernel_size=3, stride=1, padding=1, output_padding=0)
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = ConvTranspose2d(in_channels, out_channels,
                                            kernel_size=1, stride=stride, 
                                            padding=0, output_padding=output_padding)

    def forward(self, x):
        x = self.activation(x)
        residual = x if self.downsample is None else self.downsample(x)
        out = self._block(x)
        return out + residual


class DeResidualNetwork(nn.Module):
    """
    Stacks of Residual blocks make a Residual Network.
    """
    def __init__(self, in_channels, out_channels, strides, output_paddings):
        super().__init__()

        _blocks = []
        for out_channel, stride, output_padding in zip(out_channels, strides, output_paddings):
            _blocks.append(DeResidualBlock(in_channels, out_channel, stride, output_padding))
            in_channels = out_channel

        self._blocks = nn.Sequential(*_blocks)

    def forward(self, input):
        return self._blocks(input)


#######################################
########   DENSELY  NETWORKS   ########
#######################################
class DenseNetBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.main_nn = nn.Sequential(
            Conv2d(in_channels, 4 * growth_rate,
                kernel_size=1, stride=1, padding=0),
            nn.ELU(),
            Conv2d(4 * growth_rate, growth_rate,
                kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.main_nn(x)
        return torch.cat([x, out], dim=1)

class DenseNet(nn.Module):
    def __init__(self, in_channels, growth_rate, steps):
        super().__init__()
        blocks = []
        for step in range(steps):
            blocks.append(DenseNetBlock(in_channels, growth_rate))
            in_channels += growth_rate
        self.main_nn = nn.Sequential(*blocks)

    def forward(self, x):
        return self.main_nn(x)


if __name__ == "__main__":
    pass
