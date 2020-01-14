"""
Variational AutoEncoder.

TODO
    • Make sure that the loss function sums up to one.

NOTE
To run with cuda on cluster execude:
    • srun --gres=gpu:1 --time=10:00:00 python -u main.py

Copyright © 2019 Ioannis Gatopoulos.
"""

from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.models import VAE
from src.utils import *


@torch.no_grad()
def val(model, test_loader):
    model.eval()
    acc_losses = {}
    for i, (x_imgs, labels) in enumerate(test_loader):
        x_imgs, labels = x_imgs.to(args.device), labels.to(args.device)
        nelbo, losses = model.module.forward(x_imgs)
        acc_losses = Counter(acc_losses) + Counter(losses)
    avg_losses = {k: acc_losses[k] / len(test_loader) for k in acc_losses}
    return avg_losses


def train(model, optimizer, train_loader):
    model.train()
    acc_losses = {}
    for i, (x_imgs, labels) in enumerate(train_loader):
        x_imgs, labels = x_imgs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        nelbo, losses = model.module.forward(x_imgs)
        nelbo.backward()
        optimizer.step()
        acc_losses = Counter(acc_losses) + Counter(losses)
        log_interval(i+1, len(train_loader), acc_losses)
    avg_losses = {k: acc_losses[k] / len(train_loader) for k in acc_losses}
    return avg_losses


def main():
    data_shape = get_data_shape(train_loader)

    model     = nn.DataParallel(VAE(data_shape).to(args.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps)

    n_parameters(model)

    for epoch in range(1, args.epochs+1):
        # Train and Validate
        train_losses = train(model, optimizer, train_loader)
        test_losses = val(model, test_loader)
        # Visualizations and logging
        reconstruct_data(model, test_loader, epoch, writer)
        generate_data(model, epoch, writer)
        project_latent_space(model, test_loader, writer, epoch)
        logging(epoch, train_losses, test_losses, writer)


if __name__ == "__main__":
    fix_random_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='./logs/' +
                           args.dataset + '_' + args.tags +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))

    writer.add_text('args', namespace2markdown(args))

    main()

    writer.close()

    print('\n'+24*'='+' Experiment Ended '+24*'=')
