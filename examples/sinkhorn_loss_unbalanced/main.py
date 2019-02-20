import argparse
import torch
from fml.functional import pairwise_distances, sinkhorn

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser(
            description='Sinkhorn loss using the functional interface.')
    parser.add_argument('--batch_size', '-bz', type=int, default=3,
            help='Batch size.')
    parser.add_argument('--set1_size', '-sz1', type=int, default=5,
            help='Set size.')
    parser.add_argument('--set2_size', '-sz2', type=int, default=10,
            help='Set size.')
    parser.add_argument('--point_dim', '-pd', type=int, default=4,
            help='Point dimension.')
    parser.add_argument('--lp_distance', '-p', type=int, default=2,
            help='p for the Lp-distance.')

    args = parser.parse_args()

    # Set the parameters
    minibatch_size = args.batch_size
    set1_size = args.set1_size
    set2_size = args.set2_size
    point_dim = args.point_dim

    # Create two minibatches of point sets where each batch item set_a[k, :, :] is a set of `set_size` points
    set_a = torch.rand([minibatch_size, set1_size, point_dim])
    set_b = torch.rand([minibatch_size, set2_size, point_dim])

    print('Set A')
    print(set_a)

    print('Set B')
    print(set_b)

    # Condition P*1 = a and P^T*1 = b
    a = torch.ones(set_a.shape[0:2],
            requires_grad=False,
            device=set_a.device)

    b = torch.ones(set_b.shape[0:2],
            requires_grad=False,
            device=set_b.device)
    # Have the same total mass than set_a
    b = b * a.sum(1, keepdim=True) / b.sum(1, keepdim=True)

    # Compute the cost matrix 
    M = pairwise_distances(set_a, set_b, p=args.lp_distance)

    print('Distance')
    print(M)

    # Compute the transport matrix between each pair of sets in the minibatch with default parameters
    P = sinkhorn(a, b, M, 1e-3, max_iters=500, stop_thresh=1e-8)
    
    print('Transport Matrix')
    print(P)

    print('Condition error')

    aprox_a = P.sum(2) 
    aprox_b = P.sum(1) 

    print('\t P*1_d mean error: {}'.format(torch.mean((aprox_a - a).abs()).item()))
    print('\t P^T*1_d mean error: {}'.format(torch.mean((aprox_b - b).abs()).item()))
    
    # Compute the loss
    loss = (M * P).sum(2).sum(1)

    print('Loss')
    print(loss)

