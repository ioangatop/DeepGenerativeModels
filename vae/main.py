from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

from args import args
from data import train_loader, test_loader
from utils import fix_random_seed, logging, load_model, get_data_shape
from plotting import generate_data, reconstruct_data, project_latent_space


def train(model, optimizer):
    model.train()
    avg_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        loss = model.module.forward(x=imgs, c=labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss /= len(train_loader)
    return avg_loss


def val(model):
    model.eval()
    avg_loss = 0.0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(args.device), labels.to(args.device)
        loss = model.module.forward(x=imgs, c=labels)
        avg_loss += loss.item()
    avg_loss /= len(test_loader)
    return avg_loss


def main():
    data_dim = get_data_shape(train_loader)
    model = load_model(args.model, data_dim)
    optimizer = torch.optim.Adam(model.parameters())
    print("Training started.")
    for epoch in range(1, args.epochs+1):
        # Train and Validate
        train_loss = train(model, optimizer)
        test_loss = val(model)
        # Visualizations and logging
        generate_data(model, epoch, writer)
        reconstruct_data(model, test_loader, epoch, writer)
        project_latent_space(model, test_loader, writer, epoch)
        logging(epoch, train_loss, test_loss, writer)


if __name__ == "__main__":
    fix_random_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='logs/' +
                           args.model +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S"))
    main()
    writer.close()
