import torch
from torchvision import datasets
import torchvision.transforms as transforms

def get_dataloader(root_name='data', batch_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])

    data = datasets.MNIST(
        root=root_name, train=True,
        download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return dataloader

dataloader = get_dataloader()
