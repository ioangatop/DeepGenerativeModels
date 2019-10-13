import numpy as np
import torch
import torchvision


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


def generate_data(model, epoch, writer):
    fake_img = model.module.sample(n_samples=args.n_samples, device=args.device)
    grid = torchvision.utils.make_grid(fake_img, nrow=int(args.n_samples**0.5))
    writer.add_image('Sampling', grid, epoch)

def reconstruct_data(model, data, epoch, writer):
    model.eval()
    imgs, labels = next(iter(data))
    imgs_recon = model.module.reconstruct(x=imgs, c=labels)
    grid = torchvision.utils.make_grid(imgs_recon, nrow=int(imgs.shape[0]**0.5))
    writer.add_image('Reconstruction', grid, epoch)
