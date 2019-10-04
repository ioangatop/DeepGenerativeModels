import argparse
import torch
import torch.nn as nn


def print_(ARGS):
    print('\n'+64*'-')
    print('Training model: {}'.format(ARGS.model.module.__class__.__name__))
    print('Dataset: {}'.format('MNIST'))
    print('Training epochs: {}'.format(ARGS.epochs))
    print('Dimensionality of latent space: {}'.format(ARGS.latent_dim))
    print('Dropout probability on the Discriminator: {}'.format(ARGS.dropout_D))
    print('Dropout probability on the Generator: {}'.format(ARGS.dropout_G))
    print('Label Smoothing: {}'.format(str(ARGS.label_smoothing)))
    print('Flipped Labbels: {}'.format(str(ARGS.flipped_labbels)))
    print('Training on: {}'.format(str(ARGS.device)))
    print(64*'-'+'\n')


def args():
    PARSER = argparse.ArgumentParser()

    # Training parameters
    PARSER.add_argument('--epochs', default=40, type=int,
                        help='Number of training epochs.')
    PARSER.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    PARSER.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    PARSER.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    PARSER.add_argument("--b1", type=float, default=0.5,
                        help="momentum; beta1 in Adam optimizer.")
    PARSER.add_argument("--b2", type=float, default=0.999,
                        help="decay; beta2 in Adam optimizer.")
    PARSER.add_argument('--dropout_D', type=float, default=0.2,
                        help='Dropout probability on the Discriminator.')
    PARSER.add_argument('--dropout_G', type=float, default=0.2,
                        help='Dropout probability on the Generator.')
    PARSER.add_argument('--label_smoothing', type=bool, default=True,
                        help='Label Smoothing.')
    PARSER.add_argument('--flipped_labbels', type=bool, default=True,
                        help='Flipped Labbels.')

    PARSER.add_argument('--eval_mode', type=bool, default=True,
                        help='Evaluation mode On/Off when sampling.')
    PARSER.add_argument('--n_samples', type=int, default=25,
                        help='The number of the generated images.')

    PARSER.add_argument('--device', default=None, type=str,
                        help='Device to run the experiment. \
                              Valid options: "cpu", "cuda".')
    PARSER.add_argument('--seed', default=None, type=int,
                        help='Fix random seed.')
    PARSER.add_argument('--model', default='gan', type=str,
                        help="Model to be used. Valid options: \
                        'gan'.")

    ARGS = PARSER.parse_args()

    if ARGS.device is None:
        ARGS.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if ARGS.model == 'gan':
        from models.gan import GAN
        ARGS.model = nn.DataParallel(GAN(args=ARGS).to(ARGS.device))
    else:
        print('Model {} is not implimented'.format(ARGS.model))
        quit()

    print_(ARGS)

    return ARGS
