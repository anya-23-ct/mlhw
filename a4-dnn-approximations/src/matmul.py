import torch
import torch.nn.functional as F


def matmul(A, B, method='naive', **kwargs):
    """
    Multiply two matrices.
    :param A: (N, M) torch tensor.
    :param B: (M, K) torch tensor.
    :param method:
    :return:
        Output matrix with shape (N, K)
    """
    method = method.lower()
    if method in ['naive', 'pytorch', 'torch']:
        return naive(A, B)
    elif method == 'svd':
        return svd(A, B, **kwargs)
    elif method in ['log', 'logmatmul']:
        return logmatmul(A, B, **kwargs)
    else:
        raise valueError("Invalid [method] value: %s" % method)


def naive(A, B, **kwargs):
    return A @ B


def svd(A, B, rank_A=None, rank_B=None):
    """
    Apply low-rank approximation (SVD) to both matrix A and B with rank rank_A
    and rank_B respectively.
    :param A: (N, M) pytorch tensor
    :param B: (M, K) pytorch tensor
    :param rank_A: None or int. None means use original A matrix.
    :param rank_B: None or int. None means use original B matrix.
    :return: a (N, K) pytorch tensor
    """
    # raise NotImplementedError
    def apply_svd(M, rank=None):
        if rank is None:
            U, S, V = torch.svd(M)
        else:
            U, S, V = torch.svd(M, some=True)
            U = U[:, :rank]
            S = S[:rank]
            V = V[:, :rank]
        return U @ torch.diag(S) @ V.T

    A_svd = apply_svd(A, rank_A)
    B_svd = apply_svd(B, rank_B)
    out = A_svd @ B_svd
    return out


def logmatmul(A, B, **kwargs):
    """ TODO: use log multiplication for matrix-matrix multiplication """
    # raise NotImplementedError

    s1_a, s2_a = A.shape 
    s_A = torch.sign(A)
    log_A = torch.log2(torch.abs(A))

    s1_b, s2_b = B.shape 
    s_B = torch.sign(B)
    log_B = torch.log2(torch.abs(B))

    sum_logs = log_A.view(s1_a, s2_a, 1) + log_B.view(1, s1_b, s2_b) 
    prod_signs = s_A.view(s1_a, s2_a, 1) * s_B.view(1, s1_b, s2_b) 
    prod_out = prod_signs*(2**sum_logs)
    out = torch.sum(prod_out, dim=1)

    return out