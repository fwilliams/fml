import torch
import torch.nn as nn
from .functional import sinkhorn, pairwise_distances, chamfer


class SinkhornLoss(nn.Module):
    def __init__(self, eps=1e-3, max_iters=300, stop_thresh=1e-3, return_transport_matrix=False):
        """
        A pytorch Module for evaluating the Sinkhorn divergence: https://arxiv.org/abs/1306.0895

        This uses a numerically stable version of the Sinkhorn algorithm that works with 32 bit precision floats.

        :param eps: The reciprocal of the Sinkhorn regularization parameter
        :param max_iters: The maximum number of Sinkhorn iterations
        :param stop_thresh: Stop iterating if the max-norm change in iterates is below this value
        :param return_transport_matrix: If true, will return the transport matrix as well as the loss
        """
        super().__init__()
        self.eps = eps
        self.max_iters = max_iters
        self.return_transport_matrix = return_transport_matrix
        self.stop_thresh = stop_thresh

    def forward(self, predicted, expected, a=None, b=None):
        """
        Evaluate the Sinkhorn divergence between two minibatches of sets.
        i.e. predicted and expected are minibatches [m, n, d] where each [i, :, :] is a set of up to n vectors of
             dimension d. The vectors represent a distribution of weighted dirac delta functions centered at each
             vector. The weights are optionally specified by the a, and, b parameters and must integrate to 1.
        :param predicted: A minibatch of sets of shape [m, n, d] where each [i, :, :] is a set of up to n dim-d vectors
        :param expected: A minibatch of sets of shape [m, n, d] where each [i, :, :] is a set of up to n dim-d vectors
        :param a: An optional tensor of shape [m, n], representing the weights assigned to each set entry in predicted.
                  Use zero weights to ignore set elements.
        :param b: An optional tensor of shape [m, n], representing the weights assigned to each set entry in expected.
                  Use zero weights to ignore set elements.
        :return: A [m] tensor of Sinkhorn losses for each set and possibly an [m, n, ] tensor of transport matrices if
                 return_transport_matrix was specified.
        """
        if predicted.shape != expected.shape:
            raise ValueError("Invalid tensor shapes in SinkhornLoss()")

        if a is None:
            a = torch.ones(predicted.shape[0:2],
                           requires_grad=False,
                           device=predicted.device)
        else:
            a = a.to(predicted.device)

        if b is None:
            b = torch.ones(predicted.shape[0:2],
                           requires_grad=False,
                           device=predicted.device)
        else:
            b = b.to(predicted.device)

        M = pairwise_distances(predicted, expected)

        P = sinkhorn(a, b, M, self.eps, max_iters=self.max_iters, stop_thresh=self.stop_thresh)
        loss = (M * P).sum(2).sum(1)
        if self.return_transport_matrix:
            return loss, P
        else:
            return loss


class ChamferLoss(nn.Module):
    """
    A pytorch Module for evaluating the Chamfer distance between sets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted, expected):
        """
        Return an [m] shaped tensor representing the chamfer loss between a minibatch of m sets of vectors.
        :param predicted: A [m, n, d] tensor of m sets of up to n vectors of dimension d
        :param expected: A [m, n, d] tensor of m sets of up to n vectors of dimension d
        :return: The chamfer distance between each pair of sets in the minibatch
        """
        return chamfer(predicted, expected)
