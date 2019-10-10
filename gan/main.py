from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from args import args
from data import dataloader
from utils import fix_random_seed, get_labels, logging, generate_data


def main():
    model = args.model
    print("Training started.")
    for epoch in range(1, args.epochs+1):
        avg_d_loss, avg_g_loss = 0.0, 0.0
        for n_iter, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(args.device)
            real_labels, fake_labels = get_labels(imgs.shape[0])
            d_loss, g_loss = model(x=imgs, c=labels, real_labels=real_labels, fake_labels=fake_labels)

            avg_d_loss += d_loss
            avg_g_loss += g_loss
            print('{:6}/{:3d} D_loss: {:4.2f} | G_loss: {:2.2f}'.format(
                n_iter, len(dataloader), d_loss, g_loss), end='\r')

            generate_data(model, n_iter*epoch, writer)

        avg_d_loss, avg_g_loss = avg_d_loss/n_iter, avg_g_loss/n_iter
        logging(epoch, avg_d_loss, avg_g_loss, writer)


if __name__ == "__main__":
    fix_random_seed(seed=args.seed)
    writer = SummaryWriter(log_dir='logs/' +
                           args.model.module.__class__.__name__ +
                           datetime.now().strftime("/%d-%m-%Y/%H-%M-%S")
                           )
    main()
    writer.close()
