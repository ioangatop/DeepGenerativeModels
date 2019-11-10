import torch
from torchvision import datasets
import torchvision.transforms as transforms

from args import args

def dataloader(root_name='data', batch_size=64, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor()
        ])

    train_data = getattr(datasets, args.data)(
        root=root_name, train=True,
        download=True, transform=transform)

    test_data = getattr(datasets, args.data)(
        root=root_name, train=False,
        download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = dataloader()

if __name__ == "__main__":
    pass
