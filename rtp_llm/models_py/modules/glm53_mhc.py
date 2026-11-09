# Source: RTP-LLM e1106ec4b506f93f38429e499751ec522b58328e.

"""Model-local GLM53 mHC post/pre composition.

The model's existing DSV4_MHC_FUSED_POST_PRE gate chooses this helper at its two
non-Mega seams. This module adds no environment switch or generic HC dispatch.
Eligible inference uses the source fixed-FP32 projection and owns all outputs.
Unsupported source shapes keep main's post+pre semantics, including its possible
in-place residual alias; do not infer owned fallback outputs from this helper.
"""

from __future__ import annotations

import functools
import importlib
from typing import Tuple

import torch

from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit


@functools.lru_cache(maxsize=1)
def _kernel_builders():
    # Match main's prerequisite ordering before either TileLang JIT module.
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401
    from rtp_llm.models_py.modules.glm53_mhc_kernel import _mhc_fused_post_pre_fwd

    main_stage = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.mhc.pre_big_fuse_kernel"
    )
    return _mhc_fused_post_pre_fwd, main_stage._mhc_pre_big_fuse


def _source_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Inference fusion for ``mhc_post`` + next ``mhc_pre``.

    Keep the per-token FP32 reduction identical to one-token Decode, including
    multi-request MTP batches. Changing split-K with the batch size can cross
    a BF16 rounding boundary in the next layer input.
    """

    _mhc_fused_post_pre_fwd, _mhc_pre_big_fuse = _kernel_builders()

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_mix.dtype == torch.float32
    assert comb_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    mhc_mult3 = mhc_mult * (mhc_mult + 2)
    outer_shape = residual.shape[:-2]
    num_tokens = residual.numel() // (mhc_mult * hidden_size)
    assert x.shape == (*outer_shape, hidden_size)
    assert post_mix.shape in (
        (*outer_shape, mhc_mult, 1),
        (*outer_shape, mhc_mult),
    )
    assert comb_mix.shape == (*outer_shape, mhc_mult, mhc_mult)
    assert fn.shape == (mhc_mult3, mhc_mult * hidden_size)
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    # The vendored TileLang kernel uses one warp per block for a correct
    # cross-warp reduction. Split-K supplies enough blocks to fill 148 SMs;
    # its reduction order must not depend on the number of batched tokens.
    tile_n = 2
    n_splits = 16 if hidden_size <= 4096 else 8
    if hidden_size % (n_splits * 32) != 0:
        return None

    residual_flat = residual.view(num_tokens, mhc_mult, hidden_size)
    x_flat = x.view(num_tokens, hidden_size)
    post_flat = post_mix.view(num_tokens, mhc_mult)
    comb_flat = comb_mix.view(num_tokens, mhc_mult, mhc_mult)

    residual_cur = torch.empty_like(residual_flat)
    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        mhc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    post_cur = torch.empty(
        num_tokens, mhc_mult, dtype=torch.float32, device=residual.device
    )
    comb_cur = torch.empty(
        num_tokens,
        mhc_mult * mhc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    _mhc_fused_post_pre_fwd(
        mhc_mult,
        hidden_size,
        mhc_mult3,
        tile_n=tile_n,
        n_splits=n_splits,
    )(
        comb_flat,
        residual_flat,
        post_flat,
        x_flat,
        fn.view(mhc_mult3, mhc_mult, hidden_size),
        gemm_out_mul,
        gemm_out_sqrsum,
        residual_cur,
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
        residual_cur,
        post_cur,
        comb_cur,
        layer_input_cur,
    )

    return (
        residual_cur.view(*outer_shape, mhc_mult, hidden_size),
        layer_input_cur.view(*outer_shape, hidden_size),
        post_cur.view(*outer_shape, mhc_mult, 1),
        comb_cur.view(*outer_shape, mhc_mult, mhc_mult),
    )


def _validate_abi(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    unit: HCUnitBase,
) -> None:
    """Check the complete typed, dense ABI before launch or fallback writes.

    Invalid shapes, weights, mixed devices and non-contiguous inputs are errors,
    as in source's typed HC outer wrapper; never silently cast or copy them.
    The explicit same-device/weight-contiguity checks make the source JIT's
    dense Tensor ABI assumptions visible before main's in-place post fallback.
    """
    leading = unit._check_residual_shape(residual, "fused_post_pre residual")
    hidden = int(unit.dim)
    hc = int(unit.hc_mult)
    mixes = hc * (hc + 2)
    tensors = (
        ("x", x, leading + (hidden,), None),
        ("residual", residual, leading + (hc, hidden), None),
        ("post", post, leading + (hc, 1), torch.float32),
        ("comb", comb, leading + (hc, hc), torch.float32),
        ("unit.fn", unit.fn, (mixes, hc * hidden), torch.float32),
        ("unit.scale", unit.scale, (3,), torch.float32),
        ("unit.base", unit.base, (mixes,), torch.float32),
    )
    for name, tensor, shape, dtype in tensors:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"GLM53 fused_post_pre {name} expected shape={shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if dtype is not None and tensor.dtype != dtype:
            raise ValueError(
                f"GLM53 fused_post_pre {name} expected dtype={dtype}, "
                f"got {tensor.dtype}"
            )
        if tensor.device != residual.device:
            raise ValueError(
                f"GLM53 fused_post_pre {name} expected device={residual.device}, "
                f"got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"GLM53 fused_post_pre {name} must be contiguous; "
                f"got shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
            )


def _can_use_source_fusion(
    x: torch.Tensor, residual: torch.Tensor, unit: HCUnitBase
) -> bool:
    # The model factory constructs this concrete typed implementation. Keep
    # custom HC subclasses on their own public post/pre implementation.
    if type(unit) is not TileLangHCUnit:
        return False
    if (
        int(unit.hc_mult) != 4
        or residual.dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or not residual.is_cuda
        or not x.is_cuda
        or residual.numel() == 0
    ):
        return False
    hidden = int(unit.dim)
    if (hidden * int(unit.hc_mult)) % 256 != 0:
        return False
    n_splits = 16 if hidden <= 4096 else 8
    return hidden % (n_splits * 32) == 0


def fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    unit: HCUnitBase,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one HC writeback and the next HC unit's readout.

    Flat inputs use x[T,H], residual[T,hc,H], post[T,hc,1], comb[T,hc,hc].
    Batched inputs retain the real [B,S] leading axes. No input is mutated by
    the eligible source path. Unsupported source dimensions/device/dtype use
    the existing unit.post+unit.pre composition and retain its alias behavior.
    Source fusion itself enters inference_mode, regardless of the outer grad
    context. This serving helper makes no training/autograd support promise.
    Runtime import/JIT/launch errors propagate with their original cause;
    they must not turn into an unsupported-shape fallback.
    """
    _validate_abi(x, residual, post, comb, unit)
    if not _can_use_source_fusion(x, residual, unit):
        updated = unit.post(x, residual, post, comb)
        y, next_post, next_comb = unit.pre(updated)
        return updated, y, next_post, next_comb

    try:
        with torch.inference_mode():
            result = _source_fused_post_pre(
                x,
                residual,
                post,
                comb,
                unit.fn,
                unit.scale,
                unit.base,
                rms_eps=unit.norm_eps,
                mhc_pre_eps=unit.hc_eps,
                mhc_sinkhorn_eps=unit.hc_eps,
                mhc_post_mult_value=2.0,
                sinkhorn_repeat=unit.hc_sinkhorn_iters,
            )
    except Exception as error:
        raise RuntimeError(
            "GLM53 source mHC fused_post_pre failed: "
            f"x.shape={tuple(x.shape)}, residual.shape={tuple(residual.shape)}, "
            f"dtype={residual.dtype}, device={residual.device}"
        ) from error
    if result is None:
        # The public guard already checked source's only shape rejection.
        # An unexpected None is not permission to run an in-place fallback.
        raise RuntimeError("GLM53 source mHC fusion unexpectedly rejected its ABI")
    updated, y, next_post, next_comb = result
    if tuple(updated.shape) != tuple(residual.shape):
        raise ValueError("GLM53 fused_post_pre returned an invalid residual shape")
    unit._check_pre_output(tuple(residual.shape[:-2]), y, next_post, next_comb)
    return updated, y, next_post, next_comb
