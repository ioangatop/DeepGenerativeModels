import numpy as np
import torch

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
