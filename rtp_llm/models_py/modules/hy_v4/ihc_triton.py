"""Fused inference kernels for HY4 independent Hyper-Connections.

On SM100, DeepGEMM directly consumes the BF16 residual and jointly computes the
small FP32 projection plus square sum. Triton reduces those split-K partials and
fuses gate activation, four-channel mixing, and the following RMSNorm. Other
CUDA devices use an FP32 GEMM surrounded by Triton cast/reduction epilogues.

Set ``RTP_LLM_HY4_IHC_TRITON=0`` to force eager execution, or set
``RTP_LLM_HY4_IHC_PRE_BACKEND=triton`` to bypass the TF32 DeepGEMM path.
Unsupported inputs return ``None`` so callers preserve the eager fallback.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_IHC_TRITON_ENV = "RTP_LLM_HY4_IHC_TRITON"
_IHC_PRE_BACKEND_ENV = "RTP_LLM_HY4_IHC_PRE_BACKEND"
_HC_MULT = 4
_PRE_BLOCK_K = 4096
_EPILOGUE_BLOCK_H = 1024
_DEEPGEMM_EPILOGUE_BLOCK_H = 2048
_POST_BLOCK_H = 512


def _triton_enabled() -> bool:
    return os.environ.get(_IHC_TRITON_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _requested_pre_backend() -> str:
    requested = os.environ.get(_IHC_PRE_BACKEND_ENV, "auto").strip().lower()
    aliases = {"": "auto", "dg": "deepgemm"}
    requested = aliases.get(requested, requested)
    if requested not in ("auto", "deepgemm", "triton"):
        raise ValueError(
            f"invalid {_IHC_PRE_BACKEND_ENV}={requested!r}; expected auto, "
            "deepgemm, or triton"
        )
    return requested


@cache
def _has_deepgemm_prenorm() -> bool:
    try:
        import deep_gemm

        return hasattr(deep_gemm, "tf32_hc_prenorm_gemm")
    except ImportError:
        return False


@cache
def _device_num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _use_deepgemm_prenorm(channels: torch.Tensor) -> bool:
    requested = _requested_pre_backend()
    if requested == "triton":
        return False
    device_index = channels.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    is_sm100 = torch.cuda.get_device_capability(device_index)[0] == 10
    flat_size = _HC_MULT * channels.shape[2]
    available = is_sm100 and flat_size % 64 == 0 and _has_deepgemm_prenorm()
    return available


def _deepgemm_num_splits(m: int, k: int, device_index: int) -> int:
    grid_size = triton.cdiv(m, 64)
    split_k = _device_num_sms(device_index) // max(grid_size, 1)
    split_k = min(split_k, triton.cdiv(k, 64) // 4)
    return max(split_k, 1)


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return False
    device = tensors[0].device
    return device.type == "cuda" and all(t.device == device for t in tensors)


def _has_enabled_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in tensors)


def ihc_pre_is_supported(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> bool:
    if not _triton_enabled() or _has_enabled_grad(channels, fn_weight, scale, base):
        return False
    if channels.dim() != 3 or channels.shape[1] != _HC_MULT:
        return False
    if channels.dtype != torch.bfloat16 or not channels.is_contiguous():
        return False
    hidden_size = int(channels.shape[2])
    flat_size = _HC_MULT * hidden_size
    if hidden_size <= 0 or channels.shape[0] <= 0:
        return False
    if tuple(fn_weight.shape) != (2 * _HC_MULT, flat_size):
        return False
    if fn_weight.dtype != torch.float32 or not fn_weight.is_contiguous():
        return False
    if scale.dtype != torch.float32 or scale.numel() != 2 or not scale.is_contiguous():
        return False
    if (
        base.dtype != torch.float32
        or base.numel() != 2 * _HC_MULT
        or not base.is_contiguous()
    ):
        return False
    return _same_cuda_device(channels, fn_weight, scale, base)


def ihc_post_is_supported(
    block_output: torch.Tensor,
    channels: torch.Tensor,
    post_gate: torch.Tensor,
) -> bool:
    if not _triton_enabled() or _has_enabled_grad(block_output, channels, post_gate):
        return False
    if channels.dim() != 3 or channels.shape[1] != _HC_MULT:
        return False
    if channels.dtype != torch.bfloat16 or not channels.is_contiguous():
        return False
    if channels.shape[0] <= 0 or channels.shape[2] <= 0:
        return False
    expected_output_shape = (channels.shape[0], channels.shape[2])
    if tuple(block_output.shape) != expected_output_shape:
        return False
    if block_output.dtype != channels.dtype or not block_output.is_contiguous():
        return False
    if tuple(post_gate.shape) != (channels.shape[0], _HC_MULT):
        return False
    if post_gate.dtype != torch.float32 or not post_gate.is_contiguous():
        return False
    return _same_cuda_device(block_output, channels, post_gate)


def ihc_head_is_supported(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> bool:
    if not _triton_enabled() or _has_enabled_grad(channels, fn_weight, scale, base):
        return False
    if channels.dim() != 3 or channels.shape[1] != _HC_MULT:
        return False
    if channels.dtype != torch.bfloat16 or not channels.is_contiguous():
        return False
    hidden_size = int(channels.shape[2])
    flat_size = _HC_MULT * hidden_size
    if hidden_size <= 0 or channels.shape[0] <= 0:
        return False
    if tuple(fn_weight.shape) != (_HC_MULT, flat_size):
        return False
    if fn_weight.dtype != torch.float32 or not fn_weight.is_contiguous():
        return False
    if scale.dtype != torch.float32 or scale.numel() != 1 or not scale.is_contiguous():
        return False
    if (
        base.dtype != torch.float32
        or base.numel() != _HC_MULT
        or not base.is_contiguous()
    ):
        return False
    return _same_cuda_device(channels, fn_weight, scale, base)


@triton.jit(do_not_specialize=["M"])
def _ihc_cast_square_sum_kernel(
    channels_ptr,
    flat_ptr,
    partial_sum_ptr,
    M,
    K: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    offsets = split * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offsets < K
    flat_offsets = row * K + offsets
    values = tl.load(channels_ptr + flat_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(flat_ptr + flat_offsets, values, mask=mask)
    square_sum = tl.sum(values * values, axis=0)
    tl.store(partial_sum_ptr + row * NUM_PARTIALS + split, square_sum)


@triton.jit(do_not_specialize=["M"])
def _ihc_finalize_rstd_kernel(
    partial_sum_ptr,
    rstd_ptr,
    M,
    K: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_PARTIALS)
    partials = tl.load(
        partial_sum_ptr + row * NUM_PARTIALS + offsets,
        mask=offsets < NUM_PARTIALS,
        other=0.0,
    ).to(tl.float32)
    mean_square = tl.sum(partials, axis=0) / K
    tl.store(rstd_ptr + row, tl.rsqrt(mean_square + NORM_EPS))


@triton.jit(do_not_specialize=["M"])
def _ihc_cast_rstd_kernel(
    channels_ptr,
    flat_ptr,
    rstd_ptr,
    M,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NORM_EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    flat_offsets = row * K + offsets
    values = tl.load(channels_ptr + flat_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(flat_ptr + flat_offsets, values, mask=mask)
    mean_square = tl.sum(values * values, axis=0) / K
    tl.store(rstd_ptr + row, tl.rsqrt(mean_square + NORM_EPS))


@triton.jit(do_not_specialize=["M"])
def _ihc_pre_epilogue_kernel(
    channels_ptr,
    projection_ptr,
    rstd_ptr,
    scale_ptr,
    base_ptr,
    read_ptr,
    post_gate_ptr,
    M,
    HC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    HC_EPS: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1)
    channel_offsets = tl.arange(0, HC)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < H

    inv_rms = tl.load(rstd_ptr + row).to(tl.float32)
    scale_pre = tl.load(scale_ptr).to(tl.float32)
    scale_post = tl.load(scale_ptr + 1).to(tl.float32)
    pre_raw = tl.load(projection_ptr + row * (2 * HC) + channel_offsets).to(
        tl.float32
    )
    post_raw = tl.load(
        projection_ptr + row * (2 * HC) + HC + channel_offsets
    ).to(tl.float32)
    pre_base = tl.load(base_ptr + channel_offsets).to(tl.float32)
    post_base = tl.load(base_ptr + HC + channel_offsets).to(tl.float32)
    pre_gate = tl.sigmoid(pre_raw * inv_rms * scale_pre + pre_base) + HC_EPS
    post_gate = (
        MAGNITUDE * tl.sigmoid(post_raw * inv_rms * scale_post + post_base) + HC_EPS
    )

    value_offsets = (
        row * K + channel_offsets[:, None] * H + hidden_offsets[None, :]
    )
    values = tl.load(
        channels_ptr + value_offsets, mask=hidden_mask[None, :], other=0.0
    ).to(tl.float32)
    read = tl.sum(values * pre_gate[:, None], axis=0)
    tl.store(read_ptr + row * H + hidden_offsets, read, mask=hidden_mask)
    tl.store(
        post_gate_ptr + row * HC + channel_offsets,
        post_gate,
        mask=hidden_block == 0,
    )


@triton.jit(do_not_specialize=["M"])
def _ihc_deepgemm_pre_epilogue_kernel(
    channels_ptr,
    projection_partials_ptr,
    square_sum_partials_ptr,
    scale_ptr,
    base_ptr,
    read_ptr,
    post_gate_ptr,
    M,
    HC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    channel_offsets = tl.arange(0, HC)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < H

    projection_offsets = (
        split_offsets[:, None] * M * (2 * HC)
        + row * (2 * HC)
        + channel_offsets[None, :]
    )
    pre_raw = tl.sum(
        tl.load(
            projection_partials_ptr + projection_offsets,
            mask=split_offsets[:, None] < NUM_SPLITS,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    post_raw = tl.sum(
        tl.load(
            projection_partials_ptr + projection_offsets + HC,
            mask=split_offsets[:, None] < NUM_SPLITS,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    square_sum = tl.sum(
        tl.load(
            square_sum_partials_ptr + split_offsets * M + row,
            mask=split_offsets < NUM_SPLITS,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    inv_rms = tl.rsqrt(square_sum / K + NORM_EPS)
    scale_pre = tl.load(scale_ptr).to(tl.float32)
    scale_post = tl.load(scale_ptr + 1).to(tl.float32)
    pre_base = tl.load(base_ptr + channel_offsets).to(tl.float32)
    post_base = tl.load(base_ptr + HC + channel_offsets).to(tl.float32)
    pre_gate = tl.sigmoid(pre_raw * inv_rms * scale_pre + pre_base) + HC_EPS
    post_gate = (
        MAGNITUDE * tl.sigmoid(post_raw * inv_rms * scale_post + post_base) + HC_EPS
    )

    value_offsets = (
        row * K + channel_offsets[:, None] * H + hidden_offsets[None, :]
    )
    values = tl.load(
        channels_ptr + value_offsets, mask=hidden_mask[None, :], other=0.0
    ).to(tl.float32)
    read = tl.sum(values * pre_gate[:, None], axis=0)
    tl.store(read_ptr + row * H + hidden_offsets, read, mask=hidden_mask)
    tl.store(
        post_gate_ptr + row * HC + channel_offsets,
        post_gate,
        mask=hidden_block == 0,
    )


@triton.jit(do_not_specialize=["M"])
def _ihc_deepgemm_pre_rmsnorm_kernel(
    channels_ptr,
    projection_partials_ptr,
    square_sum_partials_ptr,
    scale_ptr,
    base_ptr,
    norm_weight_ptr,
    read_ptr,
    post_gate_ptr,
    M,
    HC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    IHC_NORM_EPS: tl.constexpr,
    READ_NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    channel_offsets = tl.arange(0, HC)
    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < H

    projection_offsets = (
        split_offsets[:, None] * M * (2 * HC)
        + row * (2 * HC)
        + channel_offsets[None, :]
    )
    pre_raw = tl.sum(
        tl.load(
            projection_partials_ptr + projection_offsets,
            mask=split_offsets[:, None] < NUM_SPLITS,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    post_raw = tl.sum(
        tl.load(
            projection_partials_ptr + projection_offsets + HC,
            mask=split_offsets[:, None] < NUM_SPLITS,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    square_sum = tl.sum(
        tl.load(
            square_sum_partials_ptr + split_offsets * M + row,
            mask=split_offsets < NUM_SPLITS,
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    inv_rms = tl.rsqrt(square_sum / K + IHC_NORM_EPS)
    scale_pre = tl.load(scale_ptr).to(tl.float32)
    scale_post = tl.load(scale_ptr + 1).to(tl.float32)
    pre_base = tl.load(base_ptr + channel_offsets).to(tl.float32)
    post_base = tl.load(base_ptr + HC + channel_offsets).to(tl.float32)
    pre_gate = tl.sigmoid(pre_raw * inv_rms * scale_pre + pre_base) + HC_EPS
    post_gate = (
        MAGNITUDE * tl.sigmoid(post_raw * inv_rms * scale_post + post_base) + HC_EPS
    )

    value_offsets = (
        row * K + channel_offsets[:, None] * H + hidden_offsets[None, :]
    )
    values = tl.load(
        channels_ptr + value_offsets, mask=hidden_mask[None, :], other=0.0
    ).to(tl.float32)
    read = tl.sum(values * pre_gate[:, None], axis=0)
    # Preserve the existing iHC-pre -> BF16 -> RMSNorm rounding boundary.
    rounded_read = read.to(tl.bfloat16).to(tl.float32)
    read_square_sum = tl.sum(
        tl.where(hidden_mask, rounded_read * rounded_read, 0.0), axis=0
    )
    read_inv_rms = tl.rsqrt(read_square_sum / H + READ_NORM_EPS)
    norm_weight = tl.load(
        norm_weight_ptr + hidden_offsets, mask=hidden_mask, other=0.0
    ).to(tl.float32)
    normalized = rounded_read * read_inv_rms * norm_weight
    tl.store(read_ptr + row * H + hidden_offsets, normalized, mask=hidden_mask)
    tl.store(post_gate_ptr + row * HC + channel_offsets, post_gate)


@triton.jit(do_not_specialize=["M"])
def _ihc_head_epilogue_kernel(
    channels_ptr,
    projection_ptr,
    rstd_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    M,
    HC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    HC_EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1)
    channel_offsets = tl.arange(0, HC)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < H

    inv_rms = tl.load(rstd_ptr + row).to(tl.float32)
    gate_scale = tl.load(scale_ptr).to(tl.float32)
    raw = tl.load(projection_ptr + row * HC + channel_offsets).to(tl.float32)
    gate_base = tl.load(base_ptr + channel_offsets).to(tl.float32)
    gates = tl.sigmoid(raw * inv_rms * gate_scale + gate_base) + HC_EPS

    value_offsets = (
        row * K + channel_offsets[:, None] * H + hidden_offsets[None, :]
    )
    values = tl.load(
        channels_ptr + value_offsets, mask=hidden_mask[None, :], other=0.0
    ).to(tl.float32)
    output = tl.sum(values * gates[:, None], axis=0)
    tl.store(output_ptr + row * H + hidden_offsets, output, mask=hidden_mask)


@triton.jit(do_not_specialize=["M"])
def _ihc_post_kernel(
    block_output_ptr,
    channels_ptr,
    post_gate_ptr,
    output_ptr,
    M,
    HC: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1)
    channel_offsets = tl.arange(0, HC)
    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < H

    block_output = tl.load(
        block_output_ptr + row * H + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    gates = tl.load(post_gate_ptr + row * HC + channel_offsets).to(tl.float32)
    value_offsets = (
        row * K + channel_offsets[:, None] * H + hidden_offsets[None, :]
    )
    channels = tl.load(
        channels_ptr + value_offsets, mask=hidden_mask[None, :], other=0.0
    ).to(tl.float32)
    output = channels + gates[:, None] * block_output[None, :]
    tl.store(output_ptr + value_offsets, output, mask=hidden_mask[None, :])


def _prepare_flat_and_rstd_two_pass(
    channels: torch.Tensor, norm_eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, _, hidden_size = channels.shape
    flat_size = _HC_MULT * hidden_size
    num_partials = triton.cdiv(flat_size, _PRE_BLOCK_K)
    block_partials = triton.next_power_of_2(num_partials)
    flat = torch.empty((m, flat_size), dtype=torch.float32, device=channels.device)
    partial_sum = torch.empty(
        (m, num_partials), dtype=torch.float32, device=channels.device
    )
    rstd = torch.empty((m,), dtype=torch.float32, device=channels.device)
    with torch.cuda.device(channels.device.index):
        _ihc_cast_square_sum_kernel[(m, num_partials)](
            channels,
            flat,
            partial_sum,
            M=m,
            K=flat_size,
            NUM_PARTIALS=num_partials,
            BLOCK_K=_PRE_BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        _ihc_finalize_rstd_kernel[(m,)](
            partial_sum,
            rstd,
            M=m,
            K=flat_size,
            NUM_PARTIALS=num_partials,
            BLOCK_PARTIALS=block_partials,
            NORM_EPS=norm_eps,
            num_warps=1,
            num_stages=1,
        )
    return flat, rstd


def _prepare_flat_and_rstd_one_pass(
    channels: torch.Tensor, norm_eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, _, hidden_size = channels.shape
    flat_size = _HC_MULT * hidden_size
    flat = torch.empty((m, flat_size), dtype=torch.float32, device=channels.device)
    rstd = torch.empty((m,), dtype=torch.float32, device=channels.device)
    with torch.cuda.device(channels.device.index):
        _ihc_cast_rstd_kernel[(m,)](
            channels,
            flat,
            rstd,
            M=m,
            K=flat_size,
            BLOCK_K=triton.next_power_of_2(flat_size),
            NORM_EPS=norm_eps,
            num_warps=8,
            num_stages=2,
        )
    return flat, rstd


def _prepare_flat_and_rstd(
    channels: torch.Tensor, norm_eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_size = _HC_MULT * channels.shape[2]
    if flat_size <= 32768:
        return _prepare_flat_and_rstd_one_pass(channels, norm_eps)
    return _prepare_flat_and_rstd_two_pass(channels, norm_eps)


def _maybe_deepgemm_ihc_pre(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None = None,
    read_norm_eps: float | None = None,
    split_reference_m: int | None = None,
    read_out: torch.Tensor | None = None,
    post_gate_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if not _use_deepgemm_prenorm(channels):
        return None

    import deep_gemm

    m, _, hidden_size = channels.shape
    flat_size = _HC_MULT * hidden_size
    device_index = channels.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_splits = _deepgemm_num_splits(
        m if split_reference_m is None else split_reference_m,
        flat_size,
        device_index,
    )
    projection_partials = torch.empty(
        (num_splits, m, 2 * _HC_MULT),
        dtype=torch.float32,
        device=channels.device,
    )
    square_sum_partials = torch.empty(
        (num_splits, m), dtype=torch.float32, device=channels.device
    )
    read = (
        torch.empty((m, hidden_size), dtype=channels.dtype, device=channels.device)
        if read_out is None
        else read_out
    )
    post_gate = (
        torch.empty((m, _HC_MULT), dtype=torch.float32, device=channels.device)
        if post_gate_out is None
        else post_gate_out
    )

    with torch.cuda.device(device_index):
        deep_gemm.tf32_hc_prenorm_gemm(
            channels.view(m, flat_size),
            fn_weight,
            projection_partials,
            square_sum_partials,
            num_splits,
        )
        block_splits = triton.next_power_of_2(num_splits)
        if norm_weight is None:
            _ihc_deepgemm_pre_epilogue_kernel[
                (m, triton.cdiv(hidden_size, _DEEPGEMM_EPILOGUE_BLOCK_H))
            ](
                channels,
                projection_partials,
                square_sum_partials,
                scale,
                base,
                read,
                post_gate,
                M=m,
                HC=_HC_MULT,
                H=hidden_size,
                K=flat_size,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=block_splits,
                NORM_EPS=norm_eps,
                HC_EPS=hc_eps,
                MAGNITUDE=magnitude,
                BLOCK_H=_DEEPGEMM_EPILOGUE_BLOCK_H,
                num_warps=4,
                num_stages=2,
            )
        else:
            _ihc_deepgemm_pre_rmsnorm_kernel[(m,)](
                channels,
                projection_partials,
                square_sum_partials,
                scale,
                base,
                norm_weight,
                read,
                post_gate,
                M=m,
                HC=_HC_MULT,
                H=hidden_size,
                K=flat_size,
                NUM_SPLITS=num_splits,
                BLOCK_SPLITS=block_splits,
                IHC_NORM_EPS=norm_eps,
                READ_NORM_EPS=read_norm_eps,
                HC_EPS=hc_eps,
                MAGNITUDE=magnitude,
                BLOCK_H=triton.next_power_of_2(hidden_size),
                num_warps=8,
                num_stages=2,
            )
    return read, post_gate


def maybe_fused_ihc_pre_normed_grouped(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    ihc_norm_eps: float,
    read_norm_eps: float,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """Coalesce adjacent DeepGEMM chunks without changing split-K signatures.

    The historical path selected split-K independently for each ``chunk_size``
    token block.  Rows are independent, so equal-signature full chunks can be
    launched together.  A final short chunk remains separate because it can
    select a different split count.  Both groups write directly into slices of
    the final outputs, removing the two concatenation kernels.
    """
    if not ihc_pre_is_supported(channels, fn_weight, scale, base):
        return None
    hidden_size = channels.shape[2]
    if (
        tuple(norm_weight.shape) != (hidden_size,)
        or norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or not norm_weight.is_contiguous()
        or norm_weight.device != channels.device
        or hidden_size > 8192
        or not _use_deepgemm_prenorm(channels)
    ):
        return None

    m = int(channels.shape[0])
    chunk_size = max(int(chunk_size), 1)
    read = torch.empty((m, hidden_size), dtype=channels.dtype, device=channels.device)
    post_gate = torch.empty((m, _HC_MULT), dtype=torch.float32, device=channels.device)

    full_chunk_tokens = (m // chunk_size) * chunk_size
    groups = []
    if full_chunk_tokens > 0:
        groups.append((0, full_chunk_tokens, chunk_size))
    if full_chunk_tokens < m:
        groups.append((full_chunk_tokens, m, m - full_chunk_tokens))

    for start, end, split_reference_m in groups:
        result = _maybe_deepgemm_ihc_pre(
            channels[start:end],
            fn_weight,
            scale,
            base,
            magnitude,
            hc_eps,
            ihc_norm_eps,
            norm_weight,
            read_norm_eps,
            split_reference_m=split_reference_m,
            read_out=read[start:end],
            post_gate_out=post_gate[start:end],
        )
        if result is None:
            return None
    return read, post_gate


def maybe_fused_ihc_pre_normed(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    ihc_norm_eps: float,
    read_norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if not ihc_pre_is_supported(channels, fn_weight, scale, base):
        return None
    hidden_size = channels.shape[2]
    if (
        tuple(norm_weight.shape) != (hidden_size,)
        or norm_weight.dtype not in (torch.bfloat16, torch.float32)
        or not norm_weight.is_contiguous()
        or norm_weight.device != channels.device
        or hidden_size > 8192
    ):
        return None
    return _maybe_deepgemm_ihc_pre(
        channels,
        fn_weight,
        scale,
        base,
        magnitude,
        hc_eps,
        ihc_norm_eps,
        norm_weight,
        read_norm_eps,
    )


def maybe_fused_ihc_pre(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    if not ihc_pre_is_supported(channels, fn_weight, scale, base):
        return None
    deepgemm_result = _maybe_deepgemm_ihc_pre(
        channels,
        fn_weight,
        scale,
        base,
        magnitude,
        hc_eps,
        norm_eps,
    )
    if deepgemm_result is not None:
        return deepgemm_result
    m, _, hidden_size = channels.shape
    flat_size = _HC_MULT * hidden_size
    flat, rstd = _prepare_flat_and_rstd(channels, norm_eps)
    projection = F.linear(flat, fn_weight)
    read = torch.empty((m, hidden_size), dtype=channels.dtype, device=channels.device)
    post_gate = torch.empty((m, _HC_MULT), dtype=torch.float32, device=channels.device)
    with torch.cuda.device(channels.device.index):
        _ihc_pre_epilogue_kernel[
            (m, triton.cdiv(hidden_size, _EPILOGUE_BLOCK_H))
        ](
            channels,
            projection,
            rstd,
            scale,
            base,
            read,
            post_gate,
            M=m,
            HC=_HC_MULT,
            H=hidden_size,
            K=flat_size,
            HC_EPS=hc_eps,
            MAGNITUDE=magnitude,
            BLOCK_H=_EPILOGUE_BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
    return read, post_gate


def maybe_fused_ihc_post(
    block_output: torch.Tensor,
    channels: torch.Tensor,
    post_gate: torch.Tensor,
) -> torch.Tensor | None:
    if not ihc_post_is_supported(block_output, channels, post_gate):
        return None
    m, _, hidden_size = channels.shape
    flat_size = _HC_MULT * hidden_size
    output = torch.empty_like(channels)
    with torch.cuda.device(channels.device.index):
        _ihc_post_kernel[(m, triton.cdiv(hidden_size, _POST_BLOCK_H))](
            block_output,
            channels,
            post_gate,
            output,
            M=m,
            HC=_HC_MULT,
            H=hidden_size,
            K=flat_size,
            BLOCK_H=_POST_BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
    return output


def maybe_fused_ihc_head(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor | None:
    if not ihc_head_is_supported(channels, fn_weight, scale, base):
        return None
    m, _, hidden_size = channels.shape
    flat_size = _HC_MULT * hidden_size
    flat, rstd = _prepare_flat_and_rstd(channels, norm_eps)
    projection = F.linear(flat, fn_weight)
    output = torch.empty((m, hidden_size), dtype=channels.dtype, device=channels.device)
    with torch.cuda.device(channels.device.index):
        _ihc_head_epilogue_kernel[
            (m, triton.cdiv(hidden_size, _EPILOGUE_BLOCK_H))
        ](
            channels,
            projection,
            rstd,
            scale,
            base,
            output,
            M=m,
            HC=_HC_MULT,
            H=hidden_size,
            K=flat_size,
            HC_EPS=hc_eps,
            BLOCK_H=_EPILOGUE_BLOCK_H,
            num_warps=4,
            num_stages=2,
        )
    return output
