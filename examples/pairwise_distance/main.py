import argparse
import torch
from fml.functional import pairwise_distances

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(
            description='Pairwise distance between two batchs of points.')
    parser.add_argument('--batch_size', '-bz', type=int, default=3,
            help='Batch size.')
    parser.add_argument('--set_size', '-sz', type=int, default=10,
            help='Set size.')
    parser.add_argument('--point_dim', '-pd', type=int, default=4,
            help='Point dimension.')
    parser.add_argument('--lp_distance', '-p', type=int, default=2,
            help='p for the Lp-distance.')

    args = parser.parse_args()

    # Set the parameters
    minibatch_size = args.batch_size
    set_size = args.set_size
    point_dim = args.point_dim

    # Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
    set_a = torch.rand([minibatch_size, set_size, point_dim])
    set_b = torch.rand([minibatch_size, set_size, point_dim])

    # Compute the pairwise distances between each pair of sets in the minibatch
    # distances is a tensor of shape [minibatch_size, set_size, set_size] where each
    # distances[k, i, j] = ||set_a[k, i] - set_b[k, j]||^2
    distances = pairwise_distances(set_a, set_b, p=args.lp_distance)

    print('Set A')
    print(set_a)

    print('Set B')
    print(set_b)

    print('Distance')
    print(distances)

