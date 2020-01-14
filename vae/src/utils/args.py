import json
import argparse

import torch
import torch.nn as nn


def print_args(ARGS):
    print('\n'+26*'='+' Configuration '+26*'=')
    for name, var in vars(ARGS).items():
        print('{} : {}'.format(name, var))
    print('\n'+25*'='+' Training Starts '+25*'='+'\n')


def parser():
    PARSER = argparse.ArgumentParser(description='Training parameters.')
    PARSER.add_argument('--config', default=True, type=bool,
                        help="Use config file.")

    # dataset
    PARSER.add_argument('--dataset', default='Freyfaces', type=str,
                        choices=['MNIST', 'CIFAR10', 'CelebA', 'Freyfaces'],
                        help="Data to be used.")

    PARSER.add_argument('--img_resize', default=None,
                        help='Image resize dimentions. "None" for original shape.')
    PARSER.add_argument('--subset', default=1.0, type=float,
                        help="Proportion of data to be used.")

    # prior
    PARSER.add_argument('--prior', default='mog', type=str,
                        choices=['std_normal', 'vampprior', 'mog'],
                        help='Prior type.')
    PARSER.add_argument('--z_dim', default=2, type=int,
                        help='Dimensionality of z latent space.')

    # learning rate
    PARSER.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate.')
    PARSER.add_argument('--betas', type=float, default=(0.9, 0.999),
                        help='Betas of learning rate.')
    PARSER.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon of learning rate.')

    # training parameters
    PARSER.add_argument('--epochs', default=40, type=int,
                        help='Number of training epochs.')
    PARSER.add_argument('--batch_size', default=48, type=int,
                        help='Batch size.')
    PARSER.add_argument('--beta', default=1., type=float,
                        help='Beta value.')
    PARSER.add_argument('--warmup', default=0, type=int,
                        help='Number of epochs for warmu-up.')
    PARSER.add_argument('--reconstraction_loss', default='discretized_logistic_loss', type=str,
                        choices=['mse_loss', 'discretized_logistic_loss', 'discretized_mix_logistic_loss'],
                        help="Reconstruction loss.")

    # general settings
    PARSER.add_argument('--n_samples', default=25, type=int,
                        help='Number of generated samples.')

    PARSER.add_argument('--seed', default=111, type=int,
                        help='Fix random seed.')
    PARSER.add_argument('--log_interval', default=True, type=bool,
                        help='Print progress on every batch.')
    PARSER.add_argument('--tags', default='logs', type=str,
                        help='Run tags.')
    PARSER.add_argument('--device', default=None, type=str,
                        choices=['cpu', 'cuda'],
                        help='Device to run the experiment.')

    ARGS = PARSER.parse_args()

    # Check device
    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Apply configs of specific dataset
    if ARGS.config==True:
        config_path = 'src/configs/' + ARGS.dataset + '.json'
        params = json.load(open(config_path, 'r'))
        for param, value in params.items():
            vars(ARGS)[param] = value

    print_args(ARGS)

    return ARGS


args = parser()


if __name__ == "__main__":
    pass
