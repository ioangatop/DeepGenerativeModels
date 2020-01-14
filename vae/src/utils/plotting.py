import numpy as np
import torch
from torchvision.utils import make_grid
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .args import args


normalize = True


def generate_data(model, epoch, writer):
    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'generate data...'), end='\r')

    fake_img = model.module.generate(n_samples=args.n_samples)
    grid = make_grid(fake_img, nrow=int(args.n_samples**0.5), normalize=normalize)
    writer.add_image('sampling', grid, epoch)


def reconstruct_data(model, dataloader, epoch, writer):
    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'reconstract data...'), end='\r')

    model.eval()
    imgs, labels = next(iter(dataloader))
    imgs, labels = imgs[:args.n_samples].to(args.device), labels[:args.n_samples].to(args.device)
    imgs_recon = model.module.reconstruct(imgs)
    grid = make_grid(imgs_recon, nrow=int(imgs.shape[0]**0.5), normalize=normalize)
    if epoch == 1:
        writer.add_image('reconstruction/original_data',
                         make_grid(imgs, nrow=int(imgs.shape[0]**0.5), normalize=normalize))
    writer.add_image('reconstruction/reconstructed_data', grid, epoch)


def project_latent_space(model, dataloader, writer, epoch):
    if args.dataset == 'Freyfaces':
        return

    if args.log_interval: 
        print('{:<2} {:<4}'.format('', 'projection of latent space...'), end='\r')

    # Get latent embeddings
    imgs, latent_emb, labels = [], [], []
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(args.device), label.to(args.device)
        z = model.module.encoder.forward(img)[0]
        imgs.append(img)
        latent_emb.append(z.detach())
        labels.extend(label.tolist())

    imgs, latent_emb = torch.cat(imgs), torch.cat(latent_emb)

    # Get labels
    if args.dataset == 'MNIST':
        labels = [str(label) for label in labels]
    elif args.dataset == 'FashionMNIST':
        classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        labels = [classes[label] for label in labels]
    elif args.dataset == 'CIFAR10':
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        labels = [classes[label] for label in labels]
    else:
        labels = [str(label) for label in labels]

    writer.add_embedding(latent_emb, metadata=labels, label_img=imgs, global_step=epoch, tag='default', metadata_header=None)

if __name__ == "__main__":
    pass
