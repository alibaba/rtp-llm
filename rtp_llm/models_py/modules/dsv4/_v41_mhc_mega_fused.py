"""R30/A (DSV41_MHC_MEGA_FUSED): delayed-mHC fused stats kernel.

The vendored ``mhc_pre_big_fuse`` (3rdparty tile_kernels) is proven
bitwise-identical to the delayed split chain on post/comb (r30_a_bitwise
gate, 3/3 production shapes), but it applies the mix internally and never
exposes pre_mix — unusable for the delayed chain, which consumes pre_mix
at the NEXT boundary. This variant fuses exactly the delayed boundary's
stats tail: the prenorm GEMM (shared producer, unchanged) + split_mixes +
sinkhorn + the pre_mix global write, in ONE kernel — and leaves the
collapse (y) to the existing eager op (the fused apply would risk
FMA-vs-mul+add divergence vs the eager collapse; gated separately).

Byte-exactness contract: the epilogue reproduces the vendored kernel's
FP32 accumulation and Sinkhorn iteration order verbatim (same TileLang
primitives, same order); the pre_mix write is an exact copy of the
on-chip values. The GEMM producer is shared with the unfused path, so
mixes are identical by construction.
"""

import os
import sys

import torch

_MHC_MEGA_FUSED_FLAG = "DSV41_MHC_MEGA_FUSED"
_ANNOUNCED = [False]


def mhc_mega_fused_enabled() -> bool:
    if os.environ.get(_MHC_MEGA_FUSED_FLAG, "0") == "1":
        if not _ANNOUNCED[0]:
            _ANNOUNCED[0] = True
            print(
                f"[MHCMEGA] engaged ({_MHC_MEGA_FUSED_FLAG}=1): fused stats "
                "(prenorm GEMM + split_mixes + sinkhorn + pre_mix write)",
                file=sys.stderr,
                flush=True,
            )
        return True
    return False


def mhc_delayed_stats_fused(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Fused delayed-boundary stats: shared GEMM producer (identical to the
    unfused path) + one fused epilogue kernel (split_mixes + sinkhorn +
    pre_mix write; no apply — the collapse stays eager). Returns
    (pre_mix, post_mix, comb_mix) as fp32 tensors of shapes
    [T, hc], [T, hc], [T, hc*hc]."""
    import importlib

    from rtp_llm.models_py.modules.dsv4._v41_mhc_mega_fused_kernel import (
        _mhc_delayed_stats_fused,
    )
    from rtp_llm.models_py.utils.arch import mhc_pre_gemm_backend

    _pbf = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
    )
    _ceil_div = _pbf._ceil_div
    _compute_num_split = _pbf._compute_num_split
    _run_deepgemm_splitk_gemm = _pbf._run_deepgemm_splitk_gemm
    _run_tilelang_single_gemm = _pbf._run_tilelang_single_gemm

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult3 = mhc_mult * 2 + mhc_mult * mhc_mult
    mhc_hidden_size = mhc_mult * hidden_size

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    backend = mhc_pre_gemm_backend(residual.device)
    block_k = 64
    block_m = 64

    n_splits = (
        _compute_num_split(block_k, mhc_hidden_size, _ceil_div(num_tokens, block_m))
        if backend in ("deepgemm", "tilelang_splitk")
        else 1
    )

    pre_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult * mhc_mult, dtype=torch.float32, device=residual.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    if backend == "deepgemm":
        _run_deepgemm_splitk_gemm(
            residual_flat.view(num_tokens, mhc_hidden_size),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    elif backend == "tilelang_single":
        n_splits = _run_tilelang_single_gemm(
            residual_flat, fn, gemm_out_mul, gemm_out_sqrsum, mhc_mult3, mhc_hidden_size
        )
        gemm_out_mul = gemm_out_mul[:1]
        gemm_out_sqrsum = gemm_out_sqrsum[:1]
    else:
        raise ValueError(f"R30/A fused stats: unsupported backend {backend!r}")

    _mhc_delayed_stats_fused(
        hidden_size,
        norm_eps,
        pre_eps,
        sinkhorn_eps,
        post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        scale,
        base,
        pre_mix,
        post_mix,
        comb_mix,
    )
    return pre_mix, post_mix, comb_mix
