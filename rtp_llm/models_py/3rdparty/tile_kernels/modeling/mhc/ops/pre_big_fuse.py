import functools
import logging
import os

import torch

from ....mhc.norm_fn_kernel import (
    _mhc_pre_norm_fn_fwd_mul,
    _mhc_pre_norm_fn_fwd_mul_splitk,
    round_to_tf32,
)
from ....mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse


_logger = logging.getLogger(__name__)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@functools.cache
def _policy():
    """The module that owns backend selection and the split-K arithmetic.

    Imported function-locally, as this file already does for ``deepgemm_wrapper``:
    that module holds no TileLang dependency and must stay importable in a CPU-only
    test lane, which importing it from this file's module scope would defeat.
    """
    from rtp_llm.models_py.modules.dsv4 import mhc_prenorm_backend

    return mhc_prenorm_backend


def _largest_divisor_le(n: int, want: int) -> int:
    """See :func:`mhc_prenorm_backend.largest_divisor_le`."""
    return _policy().largest_divisor_le(n, want)


@functools.cache
def device_num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def resolve_n_splits(**kwargs) -> int:
    """See :func:`mhc_prenorm_backend.resolve_n_splits`."""
    return _policy().resolve_n_splits(**kwargs)


def resolve_backend() -> str:
    """See :func:`mhc_prenorm_backend.resolve_backend`."""
    return _policy().resolve_backend()


# Historical name, kept because the ablation scripts and tests reference it.
_requested_backend = resolve_backend


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

    backend = resolve_backend()
    # At decode (num_tokens <= a couple of dozen) grid_size is 1 and the split
    # count lands on the cap, 64 for a 16384-wide fn; at prefill grid_size
    # already fills the GPU and the heuristic returns 1, which makes the split-K
    # kernel identical in shape to the single-GEMM one. Every distinct count is a
    # distinct tl.constexpr and so a distinct compile, which is why the JIT
    # warmup enumerates them through this same function.
    n_splits = resolve_n_splits(
        mhc_hidden_size=mhc_hidden_size,
        num_tokens=num_tokens,
        num_sms=device_num_sms(),
        backend=backend,
    )

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
