import torch
import torch.nn.functional as F


def conv2d(x, k, b, method='naive'):
    """
    Convolution of single instance and single input and output channel
    :param x:  (H, W) PyTorch Tensor
    :param k:  (Hk, Wk) PyTorch Tensor
    :param b:  (1,) PyTorch tensor or scalar
    :param method: Which method do we use to implement it. Valid choices include
                   'naive', 'torch', 'pytorch', 'im2col', 'winograd', and 'fft'
    :return:
        Output tensor should have shape (H_out, W_out)
    """
    method = method.lower()
    if method == 'naive':
        return naive(x, k, b)
    elif method in ['torch', 'pytorch']:
        return pytorch(x, k, b)
    elif method == 'im2col':
        return im2col(x, k, b)
    elif method == 'winograd':
        return winograd(x, k, b)
    elif method == 'fft':
        return fft(x, k, b)
    else:
        raise ValueError("Invalid [method] value: %s" % method)


def naive(x, k, b):
    """ Sliding window solution. """
    output_shape_0 = x.shape[0] - k.shape[0] + 1
    output_shape_1 = x.shape[1] - k.shape[1] + 1
    result = torch.zeros(output_shape_0, output_shape_1)
    for row in range(output_shape_0):
        for col in range(output_shape_1):
            window = x[row: row + k.shape[0], col: col + k.shape[1]]
            result[row, col] = torch.sum(torch.multiply(window, k))
    return result + b


def pytorch(x, k, b):
    """ PyTorch solution. """
    return F.conv2d(
        x.unsqueeze(0).unsqueeze(0),  # (1, 1, H, W)
        k.unsqueeze(0).unsqueeze(0),  # (1, 1, Hk, Wk)
        b   # (1, )
    ).squeeze(0).squeeze(0)  # (H_out, W_out)


def im2col(x, k, b):
    """ TODO: implement `im2col`"""
    # raise NotImplementedError

    # Extract dimensions
    Hk, Wk = k.shape
    H, W = x.shape
    H_out = H - Hk + 1
    W_out = W - Wk + 1

    # Extract patches and reshape into columns
    col = []
    for i in range(H_out):
        for j in range(W_out):
            patch = x[i:i+Hk, j:j+Wk]
            col.append(patch.flatten())
    col = torch.stack(col, dim=1)

    # Perform convolution using torch.matmul
    result = torch.matmul(k.flatten().unsqueeze(0), col).squeeze(0)

    # Reshape result to (H_out, W_out)
    result = result.reshape(H_out, W_out)

    # Add bias
    return result + b


def winograd(x, k, b):
    """ TODO: implement `winograd`"""
    # raise NotImplementedError
    def winograd_helper(padded_inp, kernel):
        # Winograd transformation matrices
        B_T = torch.tensor([[1, 0, -1, 0], 
                            [0, 1, 1, 0], 
                            [0, -1, 1, 0], 
                            [0, 1, 0, -1]], dtype=padded_inp.dtype, device=padded_inp.device)
        G = torch.tensor([[1, 0, 0],
                          [0.5, 0.5, 0.5],
                          [0.5, -0.5, 0.5],
                          [0, 0, 1]], dtype=padded_inp.dtype, device=padded_inp.device)
        A_T = torch.tensor([[1, 1, 1, 0],
                            [0, 1, -1, -1]], dtype=padded_inp.dtype, device=padded_inp.device)
        B = B_T.T
        G_T = G.T
        A = A_T.T

        # Winograd algorithm steps
        U = G @ kernel @ G_T
        V = B_T @ padded_inp @ B
        Y = U * V
        out = A_T @ Y @ A
        return out
    
    # Validate kernel shape
    Hk, Wk = k.shape
    
    # Compute required dimensions
    H, W = x.shape
    H_padding = 1 if H % 2 == 1 else 0
    W_padding = 1 if W % 2 == 1 else 0
    H_out = H - Hk + 2
    W_out = W - Wk + 2
    
    # Pad input tensor
    x_padded = torch.nn.functional.pad(x, (0, W_padding, 0, H_padding), mode='constant', value=0)
    
    # Initialize result tensor
    result = torch.zeros((H_out, W_out), dtype=x.dtype, device=x.device)
    
    # Perform convolution using Winograd algorithm
    for h in range(0, H_out, 2):
        for w in range(0, W_out, 2):
            padded_inp = x_padded[h:h+4, w:w+4]
            out = winograd_helper(padded_inp, k)
            result[h:h+2, w:w+2] += out
    
    return result[:-1, :-1] + b



def fft(x, k, b):
    """ TODO: implement `fft`"""
    # raise NotImplementedError
    H, W = x.shape
    Hk, Wk = k.shape
    padded_H = H + Hk - 1
    padded_W = W + Wk - 1

    x_padded = torch.zeros(padded_H, padded_W, dtype=x.dtype, device=x.device)
    k_padded = torch.zeros(padded_H, padded_W, dtype=k.dtype, device=k.device)

    x_padded[:H, :W] = x
    flipped_filter = k.flip((0,1))
    k_padded[:Hk, :Wk] = flipped_filter

    x_fft = torch.fft.fft2(x_padded)
    k_fft = torch.fft.fft2(k_padded)

    conv_fft = x_fft * k_fft

    conv = torch.fft.ifft2(conv_fft).real
    result = conv[Hk-1:H, Wk-1:W]
    return result + b