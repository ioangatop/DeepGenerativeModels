from markdown import markdown

import torch
import torch.nn as nn

import numpy as np

from .args import args


def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True


def get_data_shape(data_loader):
    data_shape = tuple(next(iter(data_loader))[0].shape[1:])
    return data_shape


def log_interval(i, len_data_loader, acc_losses, enable=args.log_interval):
    if args.log_interval:
        print('{:6d}/{:4d} batch | nelbo: {:4.2f}'.format(
            i, len_data_loader, acc_losses['nelbo']/i), end='\r')


def logging(epoch, train_losses, test_losses, writer):
    for loss in train_losses:
        writer.add_scalar('Train Loss/' + loss, train_losses[loss], epoch)

    for loss in test_losses:
        writer.add_scalar('Test Loss/' + loss, test_losses[loss], epoch)

    print('Epoch [{:4d}/{:4d}] | Train loss: {:4.2f} | Test loss: {:4.2f}'.format(
        epoch, args.epochs, train_losses['nelbo'], test_losses['nelbo']))


def warm_up(epoch, warmup=args.warmup):
    if warmup==0:
        beta = args.beta
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    return beta


def namespace2markdown(args):
    """
    Turns parsed arguments into markdown table.
    while formats them into a table.

    The purpuse is to output a table format of the
    given arguments that can be stored and viewed
    in tensorboad.

    Dependences: 
    -------
    Please make sure you import the following:
        from markdown import markdown

    Example:
    -------
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', default=40, type=int,
                            help="Training Epochs.")

        args = parser.parse_args()

        writer.add_text('args', namespace2markdown(args))

    params:
    ------
        args (namespace): Parsed arguments

    returns:
    ------
        markdown

    """
    txt = '<table> <thead> <tr> <td> <strong> Hyperparameter </strong> </td> <td> <strong> Values </strong> </td> </tr> </thead>'
    txt += ' <tbody> '
    for name, var in vars(args).items():
        txt += '<tr> <td> <code>' + str(name) + ' </code> </td> ' + '<td> <code> ' + str(var) + ' </code> </td> ' + '<tr> '
    txt += '</tbody> </table>'
    return markdown(txt)


def n_parameters(model):
    M_parameters = (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
    print(f'# Total Number of Parameters: {M_parameters:.3f}M')


if __name__ == "__main__":
    pass
