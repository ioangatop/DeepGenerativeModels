from datetime import datetime
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from vae import VAE
from args import args
from data import train_loader, test_loader


def logging(epoch, train_loss, test_loss):
    print('Epoch [{:4d}/{:4d}] | Train loss: {:6.2f} | Validation loss: {:6.2f}'.format(
            epoch+1, ARGS.epochs, train_loss, test_loss))


def generate_data(model, epoch, n_samples=1):
    fake_img = model.module.sample(n_samples=n_samples, device=ARGS.device)
    grid = torchvision.utils.make_grid(fake_img)
    writer.add_image(ARGS.name, grid, epoch)


def train(model, optimizer):
    model.train()
    avg_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(ARGS.device)
        optimizer.zero_grad()
        loss = model(imgs)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= len(train_loader)
    return avg_loss


def val(model):
    model.eval()
    avg_loss = 0.0
    for imgs, _ in test_loader:
        loss = model(imgs)
        avg_loss += loss.item()
    avg_loss /= len(test_loader)
    return avg_loss


def main():
    model = nn.DataParallel(VAE(z_dim=ARGS.zdim).to(ARGS.device))
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(ARGS.epochs):
        train_loss = train(model, optimizer)
        test_loss = val(model)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        generate_data(model, epoch)

        logging(epoch, train_loss, test_loss)

    writer.close()


if __name__ == "__main__":
    ARGS = args()

    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir='logs/'+ARGS.name+datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))

    main()