import numpy as np
from datetime import datetime

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from args import args
from data import train_loader, test_loader


def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logging(epoch, train_loss, test_loss):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    print('Epoch [{:4d}/{:4d}] | Train loss: {:6.2f} | Validation loss: {:6.2f}'.format(
        epoch, args.epochs, train_loss, test_loss))


def generate_data(model, epoch, n_samples=9):
    fake_img = model.module.sample(n_samples=n_samples, device=args.device)
    grid = torchvision.utils.make_grid(fake_img, nrow=int(n_samples**0.5))
    writer.add_image(args.model.module.__class__.__name__, grid, epoch)


def train(model, optimizer):
    model.train()
    avg_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        loss = model(x=imgs, c=labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= len(train_loader)
    return avg_loss


def val(model):
    model.eval()
    avg_loss = 0.0
    for imgs, labels in test_loader:
        loss = model(x=imgs, c=labels)
        avg_loss += loss.item()
    avg_loss /= len(test_loader)
    return avg_loss


def main():
    model = args.model
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, args.epochs+1):
        train_loss = train(model, optimizer)
        test_loss = val(model)
        generate_data(model, epoch)
        logging(epoch, train_loss, test_loss)


if __name__ == "__main__":
    fix_random_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='logs/' +
                           args.model.module.__class__.__name__ +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S")
                           )
    main()
    writer.close()
