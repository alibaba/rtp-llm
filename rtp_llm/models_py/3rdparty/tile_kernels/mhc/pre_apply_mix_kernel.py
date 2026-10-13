import math

import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_apply_mix_fwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic("n")
    h = hidden
    mhc = mhc_mult

    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def _mhc_pre_apply_mix_fwd_kernel(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                os = T.alloc_shared(h_blk, T.bfloat16)
                ol = T.alloc_fragment(h_blk, T.float32)
                T.clear(ol)

                for i_mhc in T.serial(mhc):
                    for i1_h in T.Parallel(h_blk):
                        ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]

                T.copy(ol, os, disable_tma=True)
                T.copy(os, o[pid_n, i0_h * h_blk], disable_tma=True)

    return _mhc_pre_apply_mix_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS, out_idx=[4])
def _mhc_pre_apply_mix_bwd(
    mhc_mult: int,
    hidden: int,
    n_thr: int = 128,
    h_blk: int = 1024,
) -> tilelang.JITKernel:
    n = T.dynamic("n")
    h = hidden
    mhc = mhc_mult

    h_blk = math.gcd(h_blk, hidden)

    @T.prim_func
    def _mhc_pre_apply_mix_bwd_kernel(
        o_grad: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        x_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        mix_grad: T.Tensor[(n, mhc), T.float32],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl, disable_tma=True)

            mgl = T.alloc_reducer(mhc, T.float32, replication="all")
            T.fill(mgl, 0)

            for i0_h in T.Pipelined(h // h_blk, num_stages=2):
                ogs = T.alloc_shared(h_blk, T.bfloat16)
                ogl = T.alloc_fragment(h_blk, T.float32)
                T.copy(o_grad[pid_n, i0_h * h_blk], ogs, disable_tma=True)
                T.copy(ogs, ogl, disable_tma=True)

                xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x[pid_n, 0, i0_h * h_blk], xs, disable_tma=True)
                T.copy(xs, xl, disable_tma=True)

                xgs = T.alloc_shared((mhc, h_blk), T.bfloat16)
                xgl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(x_grad[pid_n, 0, i0_h * h_blk], xgs, disable_tma=True)
                T.copy(xgs, xgl, disable_tma=True)

                for i_mhc, i1_h in T.Parallel(mhc, h_blk):
                    mgl[i_mhc] += ogl[i1_h] * xl[i_mhc, i1_h]
                    xgl[i_mhc, i1_h] += mixl[i_mhc] * ogl[i1_h]

                T.copy(xgl, x_grad[pid_n, 0, i0_h * h_blk], disable_tma=True)

            T.finalize_reducer(mgl)
            T.copy(mgl, mix_grad[pid_n, 0], disable_tma=True)

    return _mhc_pre_apply_mix_bwd_kernel
