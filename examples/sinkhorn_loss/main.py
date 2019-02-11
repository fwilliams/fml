import argparse
import torch
from fml.nn import SinkhornLoss

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(
            description='SikhornLoss between two batchs of points.')
    parser.add_argument('--batch_size', '-bz', type=int, default=3,
            help='Batch size.')
    parser.add_argument('--set_size', '-sz', type=int, default=10,
            help='Set size.')
    parser.add_argument('--point_dim', '-pd', type=int, default=4,
            help='Point dimension.')
    parser.add_argument('--transport_matrix', '-tm', action='store_true',
            help='Return transport matrix.')

    args = parser.parse_args()

    # Set the parameters
    minibatch_size = args.batch_size
    set_size = args.set_size
    point_dim = args.point_dim

    # Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
    set_a = torch.rand([minibatch_size, set_size, point_dim])
    set_b = torch.rand([minibatch_size, set_size, point_dim])

    print('Set A')
    print(set_a)

    print('Set B')
    print(set_b)

    # Create a loss function module with default parameters. See the class documentation for optional parameters.
    loss_fun = SinkhornLoss(return_transport_matrix=args.transport_matrix)

    # Compute the loss between each pair of sets in the minibatch
    # loss is a tensor with [minibatch_size] elements which can be backpropagated through
    if args.transport_matrix:
        loss, P = loss_fun(set_a, set_b)
        print('Transport Matrix')
        print(P)
    else:
        loss = loss_fun(set_a, set_b)

    print('Loss')
    print(loss)

