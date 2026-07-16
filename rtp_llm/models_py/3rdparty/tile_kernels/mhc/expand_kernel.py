import tilelang
import torch
from tilelang import language as T


@tilelang.jit
def expand_to_mhc_fwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128

    @T.prim_func
    def expand_to_mhc_fwd_kernel(
        x: T.Tensor[(n, h), T.bfloat16],
        o: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xl = T.alloc_fragment((blk_n, blk_h), T.bfloat16)
                T.copy(x[pid_i * blk_n, pid_j * blk_h], xl)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            o[i, m, j] = xl[ti, tj]

    return expand_to_mhc_fwd_kernel


@tilelang.jit
def expand_to_mhc_bwd_tl(hidden: int, mhc_mult: int) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden
    mhc = mhc_mult

    blk_n = 32
    blk_h = 128

    @T.prim_func
    def expand_to_mhc_bwd_kernel(
        o_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        x_grad: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n), T.ceildiv(h, blk_h)) as (pid_i, pid_j):
            if n > 0:
                xgl = T.alloc_fragment((blk_n, blk_h), T.float32)
                T.fill(xgl, 0)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            xgl[ti, tj] += o_grad[i, m, j]
                T.copy(xgl, x_grad[pid_i * blk_n, pid_j * blk_h])

    return expand_to_mhc_bwd_kernel
