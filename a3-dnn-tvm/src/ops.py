import os
import tvm
from tvm import te


# def make_conv1d_cpu_scheduler(M, N):
#     A = te.placeholder((M,), name="A")
#     W = te.placeholder((N,), name="W")

#     k = te.reduce_axis((0, M + N - 1), "k")

#     B = te.compute(
#         (M + N - 1,),
#         lambda n: te.sum(tvm.tir.if_then_else(
#             tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
#             tvm.tir.const(0.0, "float32"),
#             A[k] * W[n - k]), axis=k),
#         name="B",
#     )

#     s = te.create_schedule(B.op)

#     return s, A, W, B


def make_conv1d_cpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    pad_size = N - 1

    A_pad = te.compute(
        (M + 2 * pad_size,),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.any(n < pad_size, n >= (M + pad_size)),
            tvm.tir.const(0.0, "float32"),
            A[n - pad_size]
        ),
        name="A_pad",
    )

    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            A_pad[n + pad_size - k] * W[k], axis=k
        ),
        name="B",
    )

    s = te.create_schedule(B.op)
    b_out, b_in = s[B].split(B.op.axis[0], factor=16)
    s[B].vectorize(b_in)

    return s, A, W, B


# def make_conv1d_gpu_scheduler(M, N):
#     A = te.placeholder((M,), name="A")
#     W = te.placeholder((N,), name="W")
#     # TODO: fill-in start
#     # B = None
#     # s = None
#     k = te.reduce_axis((0, M + N - 1), "k")
#     B = te.compute(
#         (M + N - 1,),
#         lambda n: te.sum(tvm.tir.if_then_else(
#             tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
#             tvm.tir.const(0.0, "float32"),
#             A[k] * W[n - k]), axis=k),
#         name="B",
#     )
#     s = te.create_schedule(B.op)
#     bx, tx = s[B].split(s[B].op.axis[0], factor=64)
#     block = te.thread_axis("blockIdx.x")
#     thread = te.thread_axis("threadIdx.x")
#     s[B].bind(bx, block)
#     s[B].bind(tx, thread)
#     # TODO: fill-in end
#     return s, A, W, B


# optimised
def make_conv1d_gpu_scheduler(M, N):
    A = te.placeholder((M,), name="A")
    W = te.placeholder((N,), name="W")

    # TODO: fill-in start
    # B = None
    # s = None
    pad_size = N - 1
    A_pad = te.compute(
        (M + 2 * pad_size,),
        lambda n: tvm.tir.if_then_else(
            tvm.tir.any(n < pad_size, n >= (M + pad_size)),
            tvm.tir.const(0.0, "float32"),
            A[n - pad_size]
        ),
        name="A_pad",
    )
    k = te.reduce_axis((0, N), "k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            A_pad[n + pad_size - k] * W[k], axis=k
        ),
        name="B",
    )
    s = te.create_schedule(B.op)

    block = te.thread_axis("blockIdx.x")
    thread = te.thread_axis("threadIdx.x")

    ba, ta = s[A_pad].split(s[A_pad].op.axis[0], factor = 64)

    s[A_pad].parallel(ba)
    s[A_pad].parallel(ta)
    s[A_pad].bind(ba, block)
    s[A_pad].bind(ta, thread)

    bx, tx = s[B].split(s[B].op.axis[0], factor=64)

    s[B].parallel(bx)
    s[B].parallel(tx)
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    s[B].bind(tx, te.thread_axis("threadIdx.x"))

    # TODO: fill-in end
    return s, A, W, B


# def make_gemm_gpu_scheduler(M, K, N):
#     A = te.placeholder((M, K), name="A")
#     B = te.placeholder((K, N), name="B")

#     # TVM Matrix Multiplication using TE
#     k = te.reduce_axis((0, K), "k")
#     A = te.placeholder((M, K), name="A")
#     B = te.placeholder((K, N), name="B")
#     C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
#     # Default schedule
#     s = te.create_schedule(C.op)

#     # the i-th block is indexed by blockIdx.x.
#     # the number of threads in each block is blockDim.x
#     # and the i-th thread within a block is indexed by threadIdx.x
#     # overall index of a thread can be calculated as
#     # 𝑖=blockIdx.x×blockDim.x+threadIdx.x
#     block_x = te.thread_axis("blockIdx.y")
#     block_y = te.thread_axis("blockIdx.x")

#     x, y = s[C].op.axis
#     (k,) = s[C].op.reduce_axis
#     s[C].bind(y, block_y)
#     s[C].bind(x, block_x)

#     return s, A, B, C


def make_gemm_gpu_scheduler(M, K, N):
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    k = te.reduce_axis((0, K), "k")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
    s = te.create_schedule(C.op)

    y_o, x_o, y_i, x_i = s[C].tile(s[C].op.axis[0], s[C].op.axis[1], 32, 32)    
    
    bx, tx = s[C].split(x_i, nparts=32)
    by, ty = s[C].split(y_i, nparts=32)
    s[C].bind(bx, te.thread_axis((0, 32), "threadIdx.x"))    
    s[C].bind(by, te.thread_axis((0, 32), "threadIdx.y"))

    s[C].parallel(y_o)
    s[C].parallel(x_o)
    s[C].bind(y_o, te.thread_axis("blockIdx.y"))
    s[C].bind(x_o, te.thread_axis("blockIdx.x"))

    return s, A, B, C


# def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
#     assert K % 2 == 1
#     inp = te.placeholder((B, C, H, W), name="A")
#     ker = te.placeholder((C, 1, K, K), name="W")

#     # TODO: fill-in start
#     # out = None
#     # s = None

#     kern_h = (K-1)//2
#     kern_w = (K-1)//2
#     ki = te.reduce_axis((0, K), 'ki')
#     kj = te.reduce_axis((0, K), 'kj')

#     pad_in = te.compute((B, C, H + K - 1, W + K - 1),
#                             lambda *i: tvm.tir.if_then_else(tvm.tir.any(
#                                 i[-2] < kern_h, i[-2]>= H + kern_h, i[-1] < kern_w, i[-1] >= W + kern_w),
#                                 tvm.tir.const(0.0, "float32"),
#                                 inp[i[:-2] + (i[-2] - kern_h, i[-1] - kern_w)]),
#                                 name='pad_in')

#     out = te.compute((B, C, H, W),
#                      lambda b, c, h, w: te.sum(pad_in[b,c,h+ki,w+kj]*ker[c,0,ki,kj],
#                      axis=[ki,kj]),
#                      name='out')

#     s = te.create_schedule(out.op)

#     b_padin, c_padin, h_padin, w_padin = s[pad_in].op.axis
#     h_padin_o, h_padin_i = s[pad_in].split(h_padin, factor=2)
#     w_padin_o, w_padin_i = s[pad_in].split(w_padin, factor=2)

#     s[pad_in].bind(h_padin_i, te.thread_axis("threadIdx.z"))
#     s[pad_in].bind(w_padin_i, te.thread_axis("threadIdx.y"))
#     s[pad_in].bind(h_padin_o, te.thread_axis("blockIdx.z"))
#     s[pad_in].bind(w_padin_o, te.thread_axis("blockIdx.y"))

#     b_out, c_out, h_out, w_out = s[out].op.axis
#     h_out_o, h_out_i = s[out].split(h_out, factor=2)
#     w_out_o, w_out_i = s[out].split(w_out, factor=2)

#     s[out].bind(h_out_i, te.thread_axis("threadIdx.x"))
#     s[out].bind(w_out_i, te.thread_axis("threadIdx.y"))
#     s[out].bind(h_out_o, te.thread_axis("blockIdx.x"))
#     s[out].bind(w_out_o, te.thread_axis("blockIdx.y"))

#     # TODO: fill-in end

#     return s, inp, ker, out


def make_dwsp_conv2d_gpu_scheduler(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    # TODO: fill-in start
    # out = None
    # s = None

    kern_h = (K-1)//2
    kern_w = (K-1)//2
    ki = te.reduce_axis((0, K), 'ki')
    kj = te.reduce_axis((0, K), 'kj')

    pad_in = te.compute((B, C, H + K - 1, W + K - 1),
                            lambda *i: tvm.tir.if_then_else(tvm.tir.any(
                                i[-2] < kern_h, i[-2]>= H + kern_h, i[-1] < kern_w, i[-1] >= W + kern_w),
                                tvm.tir.const(0.0, "float32"),
                                inp[i[:-2] + (i[-2] - kern_h, i[-1] - kern_w)]),
                                name='pad_in')

    out = te.compute((B, C, H, W),
                     lambda b, c, h, w: te.sum(pad_in[b,c,h+ki,w+kj]*ker[c,0,ki,kj],
                     axis=[ki,kj]),
                     name='out')

    s = te.create_schedule(out.op)

    b_padin, c_padin, h_padin, w_padin = s[pad_in].op.axis
    h_padin_o, h_padin_i = s[pad_in].split(h_padin, factor=4)
    w_padin_o, w_padin_i = s[pad_in].split(w_padin, factor=4)

    b_out, c_out, h_out, w_out = s[out].op.axis
    h_out_o, h_out_i = s[out].split(h_out, factor=4)
    w_out_o, w_out_i = s[out].split(w_out, factor=4)

    s[pad_in].reorder(b_padin, c_padin, h_padin_o, w_padin_o, h_padin_i, w_padin_i)
    s[out].reorder(b_out, c_out, h_out_o, w_out_o, h_out_i, w_out_i)

    s[pad_in].parallel(h_padin_i)
    s[pad_in].parallel(w_padin_i)
    s[pad_in].bind(h_padin_i, te.thread_axis("threadIdx.z"))
    s[pad_in].bind(h_padin_o, te.thread_axis("blockIdx.z"))

    s[pad_in].parallel(h_padin_o)
    s[pad_in].parallel(w_padin_o)
    s[pad_in].bind(w_padin_i, te.thread_axis("threadIdx.y"))
    s[pad_in].bind(w_padin_o, te.thread_axis("blockIdx.y"))

    s[out].parallel(w_out_i)
    s[out].parallel(h_out_o)
    s[out].bind(w_out_i, te.thread_axis("threadIdx.y"))
    s[out].bind(w_out_o, te.thread_axis("blockIdx.y"))

    s[out].parallel(h_out_i)
    s[out].parallel(w_out_o)
    s[out].bind(h_out_i, te.thread_axis("threadIdx.x"))
    s[out].bind(h_out_o, te.thread_axis("blockIdx.x"))

    # TODO: fill-in end

    return s, inp, ker, out