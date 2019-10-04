"""
Generative Adversarial Networks

https://arxiv.org/abs/1406.2661
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, dropout_prob):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        hidden = [128, 256, 512, 1024, 28*28]

        net = []
        net.append(nn.Linear(latent_dim, hidden[0]))
        for i in range(len(hidden)-1):
            if i != 0:  # Do not apply Batch Norm in the first layer
                net.append(nn.BatchNorm1d(hidden[i]))
            net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            if i != 0:  # Do not apply Dropout in the first layer
                net.append(nn.Dropout(dropout_prob))
            net.append(nn.Linear(hidden[i], hidden[i+1]))
        net.append(nn.Tanh())
        self.net = nn.Sequential(*net)

    def forward(self, z):
        """ Forward pass: Generate images from z
        """
        return self.net(z)



class Discriminator(nn.Module):
    def __init__(self, dropout_prob):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(nn.Linear(28*28, 512),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Dropout(dropout_prob),
                                 nn.Linear(512, 256),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Linear(256, 1),
                                 nn.Sigmoid()
                                )

    def forward(self, img):
        """ Forward pass: Returns discriminator score for img
        """
        # Flatten image
        img = img.view(-1, 28*28)
        return self.net(img)



class GAN(nn.Module):
    def __init__(self, args):
        super(GAN, self).__init__()
        self.latent_dim = args.latent_dim
        self.device = args.device

        self.generator = nn.DataParallel(Generator(args.latent_dim, args.dropout_G).to(self.device))
        self.discriminator = nn.DataParallel(Discriminator(args.dropout_D).to(self.device))

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(),
                                            lr=args.lr, betas=(args.b1, args.b2))

        self.criterion = nn.BCEWithLogitsLoss()

    def sample(self, n_samples, eval_mode=True):
        if eval_mode: self.generator.eval()
        z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        return self.generator(z).data.view(-1, 1, 28, 28)

    def forward(self, **kwargs):
        self.generator.train()
        x, real_labels, fake_labels = kwargs['x'], kwargs['real_labels'], kwargs['fake_labels']
        z = torch.randn((x.shape[0], self.latent_dim)).to(self.device)
        gen_data = self.generator(z)

        # Discriminator
        self.optimizer_D.zero_grad()
        d_real = self.discriminator(x)
        d_real_loss = self.criterion(d_real, real_labels)

        d_fake = self.discriminator(gen_data)
        d_fake_loss = self.criterion(d_fake, fake_labels)
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward(retain_graph=True)
        self.optimizer_D.step()

        # Generator
        self.optimizer_G.zero_grad()
        d_fake = self.discriminator(gen_data)
        g_loss = self.criterion(d_fake, real_labels)

        g_loss.backward()
        self.optimizer_G.step()

        return d_loss.item(), g_loss.item()
