import torch


def pairwise_distances(a: torch.Tensor, b: torch.Tensor, p=2):
    """
    Compute the pairwise distance_tensor matrix between a and b which both have size [m, n, d]. The result is a tensor of
    size [m, n, n] whose entry [m, i, j] contains the distance_tensor between a[m, i, :] and b[m, j, :].
    :param a: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param b: A tensor containing m batches of n points of dimension d. i.e. of size [m, n, d]
    :param p: Norm to use for the distance_tensor
    :return: A tensor containing the pairwise distance_tensor between each pair of inputs in a batch.
    """

    if len(a.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", a.shape)
    if len(b.shape) != 3:
        raise ValueError("Invalid shape for a. Must be [m, n, d] but got", b.shape)
    return (a.unsqueeze(2) - b.unsqueeze(1)).abs().pow(p).sum(3)


def chamfer(a, b):
    """
    Compute the chamfer distance between two sets of vectors, a, and b
    :param a: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_a, d]
    :param b: A m-sized minibatch of point sets in R^d. i.e. shape [m, n_b, d]
    :return: A [m] shaped tensor storing the Chamfer distance between each minibatch entry
    """
    M = pairwise_distances(a, b)
    return M.min(1)[0].sum(1) + M.min(2)[0].sum(1)


def sinkhorn(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor, eps: float,
             max_iters: int = 100, stop_thresh: float = 1e-3):
    """
    Compute the Sinkhorn divergence between two sum of dirac delta distributions, U, and V.
    This implementation is numerically stable with float32.
    :param a: A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = [m, n]
    :param b: A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = [m, n]
    :param M: A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V.
             i.e. shape = [m, n, n] and each i.e. M[k, i, j] = ||u[k,_i] - v[k, j]||
    :param eps: The reciprocal of the sinkhorn regularization parameter
    :param max_iters: The maximum number of Sinkhorn iterations
    :param stop_thresh: Stop if the change in iterates is below this value
    :return:
    """
    # a and b are tensors of size [m, n]
    # M is a tensor of size [m, n, n]

    nb = M.shape[0]
    m = M.shape[1]
    n = M.shape[2]

    if a.dtype != b.dtype or a.dtype != M.dtype:
        raise ValueError("Tensors a, b, and M must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(M) = %s"
                         % (str(a.dtype), str(b.dtype), str(M.dtype)))
    if a.device != b.device or a.device != M.device:
        raise ValueError("Tensors a, b, and M must be on the same device got: "
                         "device(a) = %s, device(b) = %s, device(M) = %s"
                         % (a.device, b.device, M.device))
    if len(M.shape) != 3:
        raise ValueError("Got unexpected shape for M (%s), should be [nb, m, n] where nb is batch size, and "
                         "m and n are the number of samples in the two input measures." % str(M.shape))
    if torch.Size(a.shape) != torch.Size([nb, m]):
        raise ValueError("Got unexpected shape for tensor a (%s). Expected [nb, m] where M has shape [nb, m, n]." %
                         str(a.shape))

    if torch.Size(b.shape) != torch.Size([nb, n]):
        raise ValueError("Got unexpected shape for tensor b (%s). Expected [nb, n] where M has shape [nb, m, n]." %
                         str(b.shape))

    # Initialize the iteration with the change of variable
    u = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
    v = eps * torch.log(b)

    M_t = torch.transpose(M, 1, 2)

    def stabilized_log_sum_exp(x):
        max_x = torch.max(x, dim=2)[0]
        x = x - max_x.unsqueeze(2)
        ret = torch.log(torch.sum(torch.exp(x), dim=2)) + max_x
        return ret

    for current_iter in range(max_iters):
        u_prev = u
        v_prev = v

        summand_u = (-M + v.unsqueeze(1)) / eps
        u = eps * (torch.log(a) - stabilized_log_sum_exp(summand_u))

        summand_v = (-M_t + u.unsqueeze(1)) / eps
        v = eps * (torch.log(b) - stabilized_log_sum_exp(summand_v))

        err_u = torch.max(torch.sum(torch.abs(u_prev-u), dim=1))
        err_v = torch.max(torch.sum(torch.abs(v_prev-v), dim=1))

        if err_u < stop_thresh and err_v < stop_thresh:
            break

    log_P = (-M + u.unsqueeze(2) + v.unsqueeze(1)) / eps
    
    P = torch.exp(log_P)

    return P

