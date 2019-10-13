import argparse
import torch
import torch.nn as nn


def print_(ARGS):
    print('\n'+64*'-')
    print('Training model: {}'.format(ARGS.model.module.__class__.__name__))
    print('Dataset: {}'.format('MNIST'))
    print('Training epochs: {}'.format(ARGS.epochs))
    print('Dimensionality of latent space: {}'.format(ARGS.zdim))
    print('Training on: {}'.format(str(ARGS.device)))
    print(64*'-'+'\n')


def parser():
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--epochs', default=40, type=int,
                        help='Number of training epochs.')
    PARSER.add_argument('--zdim', default=20, type=int,
                        help='Dimensionality of latent space.')
    PARSER.add_argument('--device', default=None, type=str,
                        help='Device to run the experiment. \
                              Valid options: "cpu", "cuda".')
    PARSER.add_argument('--seed', default=None, type=int,
                        help='Fix random seed.')
    PARSER.add_argument('--n_samples', type=int, default=64,
                        help='The number of the generated images.')
    PARSER.add_argument('--model', default='cvae', type=str,
                        help="Model to be used. Valid options: \
                        'vae', 'conv_vae', 'cvae'.")

    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if ARGS.model == 'vae':
        from models.vae import VAE
        ARGS.model = nn.DataParallel(VAE(z_dim=ARGS.zdim).to(ARGS.device))
    elif ARGS.model == 'conv_vae':
        from models.conv_vae import ConvVAE
        ARGS.model = nn.DataParallel(ConvVAE(z_dim=ARGS.zdim).to(ARGS.device))
    elif ARGS.model == 'cvae':
        from models.cvae import CVAE
        ARGS.model = nn.DataParallel(CVAE(z_dim=ARGS.zdim, n_labels=10).to(ARGS.device))
    else:
        print('Model {} is not implimented'.format(ARGS.model))
        quit()

    print_(ARGS)
    return ARGS

args = parser()

if __name__ == "__main__":
    pass
