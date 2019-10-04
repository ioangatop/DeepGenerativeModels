import sys
from datetime import datetime

import numpy as np

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from args import args
from data import dataloader


def get_labels(batch_size, label_smoothing, flipped_labbels):
    """
    'Improved Techniques for Training GANs', Salimans et. al. 2016
    https://arxiv.org/abs/1606.03498
    """
    if label_smoothing:
        if flipped_labbels:
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
        if flipped_labbels:
            real_labels = torch.ones(batch_size, 1).to(args.device)
            fake_labels = torch.zeros(batch_size, 1).to(args.device)
        else:
            real_labels = torch.zeros(batch_size, 1).to(args.device)
            fake_labels = torch.ones(batch_size, 1).to(args.device)

    return real_labels, fake_labels



def fix_random_seed(seed=0):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def generate_data(model, epoch, n_samples=9):
    gen_data = model.module.sample(n_samples=n_samples, eval_mode=args.eval_mode)
    grid = torchvision.utils.make_grid(gen_data, nrow=int(n_samples**0.5))
    writer.add_image(args.model.module.__class__.__name__, grid, epoch)



def logging(epoch, d_loss, g_loss):
    writer.add_scalar('Loss/discriminator', d_loss, epoch)
    writer.add_scalar('Loss/generator', g_loss, epoch)
    print('Epoch [{:4d}/{:4d}] | D_loss: {:6.2f} | G_loss: {:6.2f}'.format(
        epoch, args.epochs, d_loss, g_loss))



def main():
    model = args.model
    print("Training started.")
    for epoch in range(1, args.epochs+1):
        avg_d_loss, avg_g_loss = 0.0, 0.0
        for n_iter, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(args.device)
            real_labels, fake_labels = get_labels(imgs.shape[0],
                                                  args.label_smoothing,
                                                  args.flipped_labbels)

            d_loss, g_loss = model(x=imgs, c=labels,
                                   real_labels=real_labels, fake_labels=fake_labels)

            avg_d_loss += d_loss
            avg_g_loss += g_loss
            print('{:6}/{:3d} D_loss: {:4.2f} | G_loss: {:2.2f}'.format(n_iter, len(dataloader), d_loss, g_loss), end='\r')
            generate_data(model, n_iter*epoch, n_samples=args.n_samples)

        avg_d_loss, avg_g_loss = avg_d_loss/n_iter, avg_g_loss/n_iter
        logging(epoch, avg_d_loss, avg_g_loss)


if __name__ == "__main__":
    args = args()

    fix_random_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='logs/' +
                           args.model.module.__class__.__name__ +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S")
                           )

    main()

    writer.close()
