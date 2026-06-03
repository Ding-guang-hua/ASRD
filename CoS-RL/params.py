import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='CoS-RL Model Params')

    parser.add_argument('--data', default='VolumeGenre', type=str, help='name of dataset')
    parser.add_argument('--gpu', default=1, type=int, help='indicates which gpu to use')
    parser.add_argument('--latdim', default=32, type=int, help='embedding size')
    parser.add_argument('--epoch', default=500, type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    return parser.parse_args()

args = ParseArgs()
