import numpy as np
import torch
import torchvision

from args import args


def get_labels(batch_size):
    """
    'Improved Techniques for Training GANs', Salimans et. al. 2016
    https://arxiv.org/abs/1606.03498
    """
    if args.label_smoothing:
        if args.flipped_labbels:
            real_labels = torch.ones(batch_size, 1).to(args.device) \
                        * torch.FloatTensor(1).uniform_(0.0, 0.3).to(args.device)
            fake_labels = torch.ones(batch_size, 1).to(args.device) \
                        * torch.FloatTensor(1).uniform_(0.7, 1.2).to(args.device)
        else:
            real_labels = torch.ones(batch_size, 1).to(args.device) \
                        * torch.FloatTensor(1).uniform_(0.7, 1.2).to(args.device)
            fake_labels = torch.ones(batch_size, 1).to(args.device) \
                        * torch.FloatTensor(1).uniform_(0.0, 0.3).to(args.device)
    else:
        if args.flipped_labbels:
            real_labels = torch.ones(batch_size, 1).to(args.device)
            fake_labels = torch.zeros(batch_size, 1).to(args.device)
        else:
            real_labels = torch.zeros(batch_size, 1).to(args.device)
            fake_labels = torch.ones(batch_size, 1).to(args.device)

    return real_labels, fake_labels


def generate_data(model, epoch, writer):
    gen_data = model.module.sample(n_samples=args.n_samples, eval_mode=args.eval_mode)
    grid = torchvision.utils.make_grid(gen_data, nrow=int(args.n_samples**0.5))
    writer.add_image(args.model.module.__class__.__name__, grid, epoch)


def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logging(epoch, d_loss, g_loss, writer):
    writer.add_scalar('Loss/discriminator', d_loss, epoch)
    writer.add_scalar('Loss/generator', g_loss, epoch)
    print('Epoch [{:4d}/{:4d}] | D_loss: {:6.2f} | G_loss: {:6.2f}'.format(
        epoch, args.epochs, d_loss, g_loss))

