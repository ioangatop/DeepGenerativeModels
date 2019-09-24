import argparse

def args():
    PARSER = argparse.ArgumentParser()

    # Training
    PARSER.add_argument('--epochs', default=40, type=int,
                        help='Number of training epochs.')
    PARSER.add_argument('--zdim', default=2, type=int,
                        help='Dimensionality of latent space.')
    PARSER.add_argument('--device', default=None, type=str,
                        help='Device to run the experiment. \
                              Valid options: "cpu", "cuda".')

    PARSER.add_argument('--name', default='VAE', type=str,
                    help="Experiment's name.")

    return PARSER.parse_args()
