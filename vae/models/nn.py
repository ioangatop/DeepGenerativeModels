import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 1, 28, 28)


# class UnFlattenConv2D(nn.Module):
#     def __init__(self, unflatten_size):
#         super().__init__()
#         self.unflatten_size = unflatten_size

#     def forward(self, x):
#         return x.view(x.size(0), self.unflatten_size, 1, 1)


class UnFlattenConv2D(nn.Module):
    def __init__(self, unflatten_size):
        super().__init__()
        self.c = unflatten_size[0]
        self.h = unflatten_size[1]
        self.w = unflatten_size[2]

    def forward(self, x):
        return x.view(x.size(0), self.c, self.h, self.w)


class GatedConv2d(nn.Module):
    """
    https://arxiv.org/pdf/1612.08083.pdf
    """
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g


class GatedConvTranspose2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1,
                 activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding,
                                    dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))

        g = self.sigmoid(self.g(x))

        return h * g
