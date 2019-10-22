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
    imgs_recon = model.module.reconstruct(x=imgs, c=labels)
    grid = torchvision.utils.make_grid(imgs_recon, nrow=int(imgs.shape[0]**0.5))
    if epoch == 1:
        writer.add_image('reconstruction/original_data',
                         torchvision.utils.make_grid(imgs, nrow=int(imgs.shape[0]**0.5)))
    writer.add_image('reconstruction/reconstructed_data', grid, epoch)


def plot_manifold(model, writer, epoch, n_manifold=19):
    z1 = torch.from_numpy(norm.ppf(np.linspace(0.01, 0.99, n_manifold))).float().to(args.device)
    z2 = torch.from_numpy(norm.ppf(np.linspace(0.01, 0.99, n_manifold))).float().to(args.device)

    manifold_grid = torch.stack(torch.meshgrid(z1, z2))
    manifold_grid = manifold_grid.permute(2, 1, 0).contiguous().view(-1, args.zdim)
    manifold_imgs = model.module.decoder(manifold_grid).data.view(-1, 1, 28, 28)
    writer.add_image('latent_space_visualization/manifold',
                     torchvision.utils.make_grid(manifold_imgs,
                                                 nrow=int(manifold_imgs.shape[0]**0.5)),
                     epoch)


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
        z = model.module.encoder.forward(imgs)[0]
        if latent_emb is None:
            latent_emb = z.detach()
            labels = label
        else:
            latent_emb = torch.cat([latent_emb, z.detach()], dim=0)
            labels = torch.cat([labels, label], dim=0)

    vis_data, labels = latent_emb.numpy(), labels.numpy()

    # Plot
    plot_scatter_plot(vis_data, labels, writer, epoch)
    plot_manifold(model, writer, epoch)
