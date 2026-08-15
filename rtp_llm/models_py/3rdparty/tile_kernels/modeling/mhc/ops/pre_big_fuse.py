import functools
import os

import torch

from ....mhc.norm_fn_kernel import (
    _mhc_pre_norm_fn_fwd_mul,
    _mhc_pre_norm_fn_fwd_mul_splitk,
    round_to_tf32,
)
from ....mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _largest_divisor_le(n: int, want: int) -> int:
    """Largest divisor of ``n`` that is <= ``want`` (>= 1).

    ``_compute_num_split`` returns ``n_sms // grid_size`` capped at
    ``num_block_k // 4``, which is not in general a divisor of the TileLang
    kernel's K block count -- at 78 SMs and m=512 it returns 9, and the split-K
    kernel needs the K blocks to divide evenly. Rounding down to a divisor keeps
    every block's slice the same length, which is what lets the ``pz`` loop stay
    a compile-time constant.
    """
    d = min(max(int(want), 1), n)
    while d > 1 and n % d:
        d -= 1
    return max(d, 1)


@functools.cache
def _compute_num_split(block_k: int, k: int, grid_size: int) -> int:
    device_props = torch.cuda.get_device_properties(0)
    n_sms = device_props.multi_processor_count
    split_k = n_sms // max(grid_size, 1)
    num_block_k = _ceil_div(k, block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


@functools.cache
def _has_deepgemm_prenorm() -> bool:
    """Whether DeepGEMM in this build exports the split-K mHC pre GEMM.

    ``tf32_hc_prenorm_gemm`` is optional in DeepGEMM: the wrapper leaves the impl
    None rather than raising when it is absent, which is exactly the case that has
    to be detected before selecting the backend that needs it.
    """
    try:
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

        return getattr(deepgemm_wrapper, "_tf32_hc_prenorm_gemm_impl", None) is not None or hasattr(
            __import__("deep_gemm"), "tf32_hc_prenorm_gemm"
        )
    except Exception:
        return False


def _requested_backend() -> str:
    requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if requested in ("", "auto"):
        # DeepGEMM's split-K kernel when the build has it -- it is the fastest of
        # the three and the SM100 smoke suite validates DSV4 greedy/golden
        # semantics against it. When the build does NOT export
        # tf32_hc_prenorm_gemm, "auto" used to resolve to it anyway and every call
        # raised, which is why deployments pinned tilelang_single by hand. Fall
        # back to tilelang_splitk instead: same math as tilelang_single, same
        # number of kernels, but the K loop is spread over the grid rather than
        # walked by one CUDA block, which is worth ~20% of decode TPOT.
        return "deepgemm" if _has_deepgemm_prenorm() else "tilelang_splitk"
    aliases = {
        "dg": "deepgemm",
        "tilelang": "tilelang_single",
        "single": "tilelang_single",
        "splitk": "tilelang_splitk",
    }
    return aliases.get(requested, requested)


def _run_tilelang_single_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_mult3: int,
    mhc_hidden_size: int,
) -> int:
    fn = round_to_tf32(fn)
    fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
    fwd_mul_kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul[:1].view(-1, 1, mhc_mult3),
        gemm_out_sqrsum[:1].view(-1, 1),
    )
    return 1


def _run_tilelang_splitk_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_mult3: int,
    mhc_hidden_size: int,
    n_splits: int,
) -> None:
    """Split-K TileLang mHC pre GEMM: same math as the single-GEMM path.

    Writes partials into all ``n_splits`` slots; ``_mhc_pre_big_fuse`` below
    already sums the leading axis, so nothing else changes. Unlike
    ``_run_tilelang_single_gemm`` this does not narrow the buffers to ``[:1]``.
    """
    fn = round_to_tf32(fn)
    kernel = _mhc_pre_norm_fn_fwd_mul_splitk(
        mhc_mult3, 1, mhc_hidden_size, n_splits
    )
    kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul.view(n_splits, -1, 1, mhc_mult3),
        gemm_out_sqrsum.view(n_splits, -1, 1),
    )


def _run_deepgemm_splitk_gemm(
    residual_flat: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    n_splits: int,
) -> None:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    tf32_hc_prenorm_gemm(
        residual_flat,
        fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    backend = _requested_backend()
    block_k = 64
    block_m = 64
    n_splits = (
        _compute_num_split(block_k, mhc_hidden_size, _ceil_div(num_tokens, block_m))
        if backend in ("deepgemm", "tilelang_splitk")
        else 1
    )
    if backend == "tilelang_splitk":
        # The TileLang kernel slices K in units of its hidden_block (256), so
        # the split count has to divide that block count evenly. At decode
        # (num_tokens <= a couple of dozen) grid_size is 1 and this lands on the
        # cap, 64 for a 16384-wide fn; at prefill grid_size already fills the
        # GPU and _compute_num_split returns 1, which makes this kernel identical
        # in shape to the single-GEMM one.
        n_splits = _largest_divisor_le(mhc_hidden_size // 256, n_splits)

    post_mix = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
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
            fn_flat,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    elif backend == "tilelang_single":
        n_splits = _run_tilelang_single_gemm(
            residual_flat,
            fn_flat,
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_mult3,
            mhc_hidden_size,
        )
        gemm_out_mul = gemm_out_mul[:1]
        gemm_out_sqrsum = gemm_out_sqrsum[:1]
    elif backend == "tilelang_splitk":
        _run_tilelang_splitk_gemm(
            residual_flat,
            fn_flat,
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_mult3,
            mhc_hidden_size,
            n_splits,
        )
    else:
        raise ValueError(
            "Unsupported DSV4_MHC_PRE_GEMM_BACKEND="
            f"{backend!r}; expected deepgemm, tilelang_splitk, or tilelang_single."
        )

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input
