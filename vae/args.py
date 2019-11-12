import argparse
import torch
import torch.nn as nn


def print_(ARGS):
    print('\n'+64*'-')
    print('Training model: {}'.format(ARGS.model))
    print('Dataset: {}'.format(ARGS.data))
    print('Training epochs: {}'.format(ARGS.epochs))
    print('Dimensionality of latent space: {}'.format(ARGS.zdim))
    print('Batch size: {}'.format(ARGS.batch_size))
    print('Training on: {}'.format(str(ARGS.device)))
    print(64*'-'+'\n')


def parser():
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--data', default='MNIST', type=str,
                        help="Data to be used. Valid options: \
                        'MNIST', 'CIFAR10', 'ImageNet'.")
    PARSER.add_argument('--model', default='conv_vae', type=str,
                        help="Model to be used. Valid options: \
                        'vae', 'gated_vae', 'cvae'.")

    PARSER.add_argument('--epochs', default=40, type=int,
                        help='Number of training epochs.')
    PARSER.add_argument('--zdim', default=2, type=int,
                        help='Dimensionality of latent space.')
    PARSER.add_argument('--batch_size', default=64, type=int,
                        help='Batch size.')

    PARSER.add_argument('--n_samples', type=int, default=64,
                        help='The number of the generated images.')
    PARSER.add_argument('--seed', default=None, type=int,
                        help='Fix random seed.')
    PARSER.add_argument('--device', default=None, type=str,
                        help='Device to run the experiment. \
                              Valid options: "cpu", "cuda".')
    ARGS = PARSER.parse_args()

    # check device
    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print_(ARGS)
    return ARGS

args = parser()

if __name__ == "__main__":
    pass
