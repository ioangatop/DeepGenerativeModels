from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from args import args
from data import train_loader, test_loader
from utils import fix_random_seed, logging, generate_data


def train(model, optimizer):
    model.train()
    avg_loss = 0.0
    for i, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        loss = model(x=imgs, c=labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        print('{:6}/{:3d} | Loss: {:4.2f}'.format(
            i+1, len(train_loader), loss.item()), end='\r')

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
    print("Training started.")
    for epoch in range(1, args.epochs+1):
        train_loss = train(model, optimizer)
        test_loss = val(model)
        generate_data(model, epoch, writer)
        logging(epoch, train_loss, test_loss, writer)


if __name__ == "__main__":
    fix_random_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='logs/' +
                           args.model.module.__class__.__name__ +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S")
                           )
    main()
    writer.close()
