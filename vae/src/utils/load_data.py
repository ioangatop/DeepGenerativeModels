import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
from functools import partial
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset

from .args import args


def load_pickle(pickle_file):
    """
    Helper function to load pickle data.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data[0]


class Freyfaces(Dataset):
    def __init__(self, root, transform, split="train", **kwargs):
        self.transform = transform

        TRAIN = 1565 + 200
        TEST  = 200

        # start processing
        data = load_pickle(root + '/freyfaces.pkl')

        # get data
        if split=="train":
            self.data = data[0:TRAIN].reshape(-1, 28, 20)
        elif split=="test":
            self.data = data[(TRAIN):(TRAIN + TEST)].reshape(-1, 28, 20)

        self.targets = np.zeros((self.data.shape[0], 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, downscaled_image, target)
                    where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_pil = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img_pil)

        return img, target


def dataloader(dataset=args.dataset, batch_size=args.batch_size, root_name='/var/scratch/igatopou/data/'):
    """
    The folder structure will be: ./data/dataset. e.g. ./data/CIFAR10

    Transformation of data
        • [-1, 1] : mean = [0.5, 0.5, 0.5]; std = [0.5, 0.5, 0.5]
        • [0, 1]  : mean = [0, 0, 0]; std = [1, 1, 1]
    """

    # Data Normalizaton
    nc = 3 if dataset in ['CIFAR10', 'CelebA'] else 1
    if args.reconstraction_loss == 'discretized_mix_logistic_loss':
        mean, std = nc*[0.5], nc*[0.5]
    else:
        mean, std = nc*[0], nc*[1]

    # Data transformation
    if args.img_resize is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((args.img_resize, args.img_resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Dataset and Loader kwargs
    kwargs = {} if args.device=='cpu' else {'num_workers': 2, 'pin_memory': True}
    dataset_kwargs = {'root':root_name+dataset, 'download':True, 'transform':transform}
    loader_kwargs = {'batch_size':batch_size, **kwargs}

    # Load datasets
    if dataset in ['MNIST', 'CIFAR10']:
        train_data = getattr(datasets, dataset)(train=True, **dataset_kwargs)
        test_data = getattr(datasets, dataset)(train=False, **dataset_kwargs)
    elif dataset in ['CelebA']:
        train_data = getattr(datasets, dataset)(split="train", **dataset_kwargs)
        test_data = getattr(datasets, dataset)(split="test", **dataset_kwargs)
    elif dataset=='Freyfaces':
        train_data = Freyfaces(split="train", **dataset_kwargs)
        test_data = Freyfaces(split="test", **dataset_kwargs)
    else:
        raise NotImplementedError

    # Subset
    if args.subset != 1.0:
        train_subset = list(np.random.choice(len(train_data), int(len(train_data)*args.subset), replace=False))
        test_subset  = list(np.random.choice(len(test_data), int(len(test_data)*args.subset), replace=False))

        train_data = torch.utils.data.Subset(train_data, train_subset)
        test_data  = torch.utils.data.Subset(test_data, test_subset)

    # Build dataloaders
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_data, shuffle=False, **loader_kwargs)

    return train_loader, test_loader


train_loader, test_loader = dataloader()


if __name__ == "__main__":
    pass
