import numpy as np
import torch
import torch.nn as nn

from args import args


def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logging(epoch, train_loss, test_loss, writer):
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    print('Epoch [{:4d}/{:4d}] | Train loss: {:6.2f} | Validation loss: {:6.2f}'.format(
        epoch, args.epochs, train_loss, test_loss))


def get_data_shape(data_loader):
    return tuple(next(iter(data_loader))[0].shape[1:])


def load_model(model_name, data_dim):
    # load model
    if model_name == 'vae':
        from models.vae import VAE as MODEL
        return nn.DataParallel(VAE(z_dim=args.zdim).to(args.device))
    elif model_name == 'cvae':
        from models.cvae import CVAE
        return nn.DataParallel(CVAE(z_dim=args.zdim, n_labels=10).to(args.device))
    elif model_name == 'conv_vae':
        from models.conv_vae import ConvVAE
        return nn.DataParallel(ConvVAE(data_dim, latent=args.zdim).to(args.device))
    elif model_name == 'gated_vae':
        from models.gated_conv2d_vae import GatedVAE
        return nn.DataParallel(GatedVAE(n_chanels=data_dim[0], batch_size=args.batch_size,
                                        z_dim=args.zdim).to(args.device))
    else:
        print('Model {} is not implimented'.format(model_name))
        quit()
