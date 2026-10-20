"""Opt-in native SM103 merged-QB / SWA producer; import remains CUDA-lazy."""

from __future__ import annotations

import math
import os
from functools import lru_cache

import torch

# Qualified shapes only. M is B*S, not the number of requests.
# M384 retains DeepGEMM until its pure-GEMM regression is resolved.
_CONFIGS = {128: (0, 72), 192: (1, 72), 256: (3, 72), 512: (3, 72)}


def _ops():
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops


@lru_cache(maxsize=None)
def _device_supported(device):
    return (
        torch.cuda.get_device_capability(device) == (10, 3)
        and torch.cuda.get_device_properties(device).multi_processor_count == 148
    )


def _config(merged, data, scales):
    if (
        torch.version.hip is not None
        or not data.is_cuda
        or data.ndim not in (2, 3)
        or data.shape[-1] != 1280
        or not data.is_contiguous()
        or data.dtype != torch.float8_e4m3fn
        or getattr(merged, "main_width", None) != 32768
        or getattr(merged, "index_width", None) != 4096
    ):
        return None
    rows = data.numel() // 1280
    config = _CONFIGS.get(rows)
    if config is None:
        return None
    weight, sf = merged.weight, merged.weight_scales
    if not (
        weight.device == sf.device == scales.device == data.device
        and weight.shape == (36864, 1280)
        and weight.dtype == data.dtype
        and weight.is_contiguous()
        and scales.shape == (rows, 10)
        and sf.shape == (36864, 10)
        and all(
            t.dtype == torch.int32
            and t.stride(0) == 1
            and t.stride(1) >= t.shape[0]
            and t.stride(1) % 4 == 0
            for t in (scales, sf)
        )
        and all(t.data_ptr() % 16 == 0 for t in (data, scales, weight, sf))
        and not any(t.requires_grad for t in (data, scales, weight, sf))
        and _device_supported(data.device)
    ):
        return None
    available = getattr(_ops(), "dsv41_qb_kv_available", None)
    return config if available is not None and available() else None


def try_project(merged, data, scales, *, wait_pdl=False):
    if os.environ.get("DSV41_CUSTOM_QB", "0") != "1":
        return None
    config = _config(merged, data, scales)
    if config is None:
        return None
    output = torch.empty(
        (*data.shape[:-1], 36864), dtype=torch.bfloat16, device=data.device
    )
    _ops().dsv41_qb_out(
        data.view(-1, 1280),
        scales,
        merged.weight,
        merged.weight_scales,
        output.view(-1, 36864),
        *config,
        wait_pdl,
    )
    return output.split((32768, 4096), dim=-1)


def try_project_kv(
    merged,
    data,
    scales,
    raw_kv,
    gamma,
    positions,
    frequencies,
    pool,
    slots,
    *,
    eps,
    wait_pdl=False,
):
    """Return (main_q, index_q, kv), or None before any output/cache write."""
    if os.environ.get("DSV41_FUSED_QB_KV", "0") != "1":
        return None
    config = _config(merged, data, scales)
    if config is None or any(t is None for t in (pool, slots)):
        return None
    from ._trap_utils import (
        invalid_kv_access_validation_enabled,
        trap_invalid_kv_access_enabled,
    )

    if invalid_kv_access_validation_enabled():
        return None
    rows = data.numel() // 1280
    if not (
        raw_kv.shape == (*data.shape[:-1], 512)
        and raw_kv.dtype == gamma.dtype == torch.bfloat16
        and raw_kv.stride(-1) == 1
        and raw_kv.stride(-2) >= 512
        and (
            raw_kv.ndim == 2
            or raw_kv.shape[0] == 1
            or raw_kv.shape[1] == 1
            or raw_kv.stride(0) == raw_kv.shape[1] * raw_kv.stride(1)
        )
        and gamma.shape == (512,)
        and gamma.is_contiguous()
        and positions.shape == slots.shape == (rows,)
        and positions.dtype in (torch.int32, torch.int64)
        and slots.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
        and slots.is_contiguous()
        and frequencies.dtype == torch.complex64
        and frequencies.ndim == 2
        and frequencies.shape[1] == 32
        and frequencies.shape[0] > 0
        and frequencies.is_contiguous()
        and not frequencies.is_conj()
        and pool.dtype == torch.uint8
        and pool.ndim == 3
        and pool.shape[0] > 0
        and pool.shape[1] > 0
        and pool.shape[2] == 528
        and pool.stride(2) == 1
        and pool.stride(1) == 528
        and pool.stride(0) >= pool.shape[1] * 528
        and all(
            t.device == data.device and not t.requires_grad
            for t in (raw_kv, gamma, positions, frequencies, pool, slots)
        )
        and math.isfinite(eps)
        and eps >= 0
    ):
        return None
    output = torch.empty(
        (*data.shape[:-1], 36864), dtype=torch.bfloat16, device=data.device
    )
    kv = torch.empty(raw_kv.shape, dtype=torch.bfloat16, device=data.device)
    _ops().dsv41_qb_kv_out(
        data.view(rows, 1280),
        scales,
        merged.weight,
        merged.weight_scales,
        output.view(rows, 36864),
        raw_kv.view(rows, 512),
        gamma,
        positions,
        frequencies.view(torch.float32),
        pool,
        slots,
        kv.view(rows, 512),
        eps,
        *config,
        wait_pdl,
        trap_invalid_kv_access_enabled(),
    )
    main, index = output.split((32768, 4096), dim=-1)
    return main, index, kv
