import torch
from torchvision import datasets
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt


def dataloader(root_name='data', batch_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    train_data = datasets.MNIST(
        root=root_name, train=True,
        download=True, transform=transform)

    test_data = datasets.MNIST(
        root=root_name, train=False,
        download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    return train_loader, test_loader


def visual_data(dataloader):
    dataiter = iter(dataloader)
    images, _ = dataiter.next()
    images = images.numpy()

    # get one image from the batch
    img = np.squeeze(images[0])

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    fig.savefig('fig.png', format='png')
    plt.close(fig)


train_loader, test_loader = dataloader()
