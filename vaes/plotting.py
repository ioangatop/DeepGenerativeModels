import numpy as np
import torch
import torchvision
from scipy.stats import norm
import matplotlib.pyplot as plt

from args import args


def generate_data(model, epoch, writer):
    fake_img = model.module.sample(n_samples=args.n_samples, device=args.device)
    grid = torchvision.utils.make_grid(fake_img, nrow=int(args.n_samples**0.5))
    writer.add_image('sampling', grid, epoch)


def reconstruct_data(model, dataloader, epoch, writer):
    model.eval()
    imgs, labels = next(iter(dataloader))
    imgs, labels = imgs.to(args.device), labels.to(args.device)
    imgs_recon = model.module.reconstruct(x=imgs, c=labels)
    grid = torchvision.utils.make_grid(imgs_recon, nrow=int(imgs.shape[0]**0.5))
    if epoch == 1:
        writer.add_image('reconstruction/original_data',
                         torchvision.utils.make_grid(imgs, nrow=int(imgs.shape[0]**0.5)))
    writer.add_image('reconstruction/reconstructed_data', grid, epoch)


def plot_manifold(model, writer, epoch, n_manifold=19):
    eps = norm.ppf(np.linspace(0.01, 0.99, n_manifold + 2)[1:-1])
    z = torch.FloatTensor(np.dstack(np.meshgrid(eps, -eps)).reshape(-1, 2)).to(args.device)
    images = model.module.decoder(z).view(-1, 1, 28, 28)
    manifold_grid = torchvision.utils.make_grid(images, nrow=int(images.shape[0]**0.5))
    writer.add_image('latent_space_visualization/manifold', manifold_grid, epoch)


def plot_scatter_plot(vis_data, labels, writer, epoch, c=10, dpi=100):
    vis_x, vis_y = vis_data[:, 0], vis_data[:, 1]

    fig, ax = plt.subplots(1, dpi=dpi)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.scatter(vis_x, vis_y, marker='.', c=list(labels), cmap=plt.cm.get_cmap("jet", c))
    plt.axis('off')
    plt.colorbar(ticks=range(c))
    plt.clim(-0.5, 9.5)
    fig.tight_layout()
    writer.add_figure('latent_space_visualization/scatter_plot', fig, epoch)


def project_latent_space(model, dataloader, writer, epoch):
    if args.zdim > 2:
        return

    # Get data
    latent_emb = None
    for imgs, label in dataloader:
        imgs, label = imgs.to(args.device), label.to(args.device)
        z = model.module.encoder.forward(imgs)[0]
        if latent_emb is None:
            latent_emb = z.detach()
            labels = label
        else:
            latent_emb = torch.cat([latent_emb, z.detach()], dim=0)
            labels = torch.cat([labels, label], dim=0)

    vis_data, labels = latent_emb.cpu().numpy(), labels.cpu().numpy()

    # Plot
    plot_scatter_plot(vis_data, labels, writer, epoch)
    plot_manifold(model, writer, epoch)
