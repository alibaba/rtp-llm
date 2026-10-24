"""Opt-in DeepGEMM SM103 merged-QB / SWA producer; import remains CUDA-lazy."""

from __future__ import annotations

import math
import os
from functools import lru_cache
from types import SimpleNamespace

import torch

# Qualified shapes only. M is B*S, not the number of requests.
# M384 retains DeepGEMM until its pure-GEMM regression is resolved.
_CONFIGS = {128: (0, 72), 192: (1, 72), 256: (3, 72), 512: (3, 72)}
_READY_CONFIGS = {128: (0, 64), 192: (1, 64), 256: (2, 74), 384: (1, 64), 512: (2, 74)}


def _ops():
    import deep_gemm

    return deep_gemm


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
        or getattr(merged, "index_width", None) not in (0, 4096)
    ):
        return None
    rows = data.numel() // 1280
    width = 32768 + merged.index_width
    config = (_READY_CONFIGS if width == 32768 else _CONFIGS).get(rows)
    if config is None:
        return None
    weight, sf = merged.weight, merged.weight_scales
    if not (
        weight.device == sf.device == scales.device == data.device
        and weight.shape == (width, 1280)
        and weight.dtype == data.dtype
        and weight.is_contiguous()
        and scales.shape == (rows, 10)
        and sf.shape == (width, 10)
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
    ops = _ops()
    return (
        config
        if all(
            callable(getattr(ops, name, None))
            for name in (
                "dsv41_qb_out",
                "dsv41_qb_kv_out",
                "dsv41_qb_kv_index_out",
            )
        )
        else None
    )


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
    _ready=None,
):
    """Return Q/KV (and optional slots), or None before any cache write."""
    flag = "DSV41_FUSED_READY_QKV" if _ready is not None else "DSV41_FUSED_QB_KV"
    if os.environ.get(flag, "0") != "1":
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
    if _ready is not None and (
        not _ready.supports(data.device, rows)
        or not callable(getattr(_ops(), "dsv41_qb_kv_index_out", None))
    ):
        return None
    width = merged.weight.shape[0]
    output = torch.empty(
        (*data.shape[:-1], width), dtype=torch.bfloat16, device=data.device
    )
    kv = torch.empty(raw_kv.shape, dtype=torch.bfloat16, device=data.device)
    arguments = (
        data.view(rows, 1280),
        scales,
        merged.weight,
        merged.weight_scales,
        output.view(rows, width),
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
    if _ready is not None:
        slots_out = torch.empty(
            _ready.local.shape, dtype=torch.int32, device=data.device
        )
        _ops().dsv41_qb_kv_index_out(
            *arguments,
            _ready.req,
            _ready.table,
            _ready.local,
            slots_out,
            _ready.logical_entries,
            _ready.entries,
        )
        return output, kv, slots_out
    _ops().dsv41_qb_kv_out(*arguments)
    main, index = output.split((32768, 4096), dim=-1)
    return main, index, kv


def try_project_kv_index(
    mega,
    data,
    scales,
    raw_kv,
    gamma,
    positions,
    frequencies,
    pool,
    slots,
    plan,
    *,
    eps,
    wait_pdl=False,
):
    """Q-only N32768 + SWA + already-ready TopK mapping, exactly one launch."""
    state = SimpleNamespace(
        main_width=32768,
        index_width=0,
        weight=mega.query_weight,
        weight_scales=mega.query_scale,
    )
    return try_project_kv(
        state,
        data,
        scales,
        raw_kv,
        gamma,
        positions,
        frequencies,
        pool,
        slots,
        eps=eps,
        wait_pdl=wait_pdl,
        _ready=plan,
    )
