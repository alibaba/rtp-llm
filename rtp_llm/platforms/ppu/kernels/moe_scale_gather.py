"""Change H: fused gather of the MXFP4 activation scales into mn-major order.

``compact_mxfp4_routes_nopad`` permutes routes by expert with

    scale_storage = torch.empty((scale.shape[1], route_count), ...)
    scale_out = scale_storage.t()
    scale_out.copy_(torch.index_select(scale, 0, source_rows))

``scale`` is UINT16 ``[M, K/64]`` in DeepGEMM's mn-major layout, stride ``(1, M)``.
``index_select`` produces a row-major temporary and ``copy_`` then transposes it
into the mn-major destination, so the permutation costs an extra 6 MiB buffer plus
two passes.  A prefill trace attributes 27.0 ms over 308 launches to that
``copy_`` alone, at 144 GB/s -- 6% of what a dedicated copy reaches on this device.

Reading mn-major and writing mn-major in one kernel keeps the writes fully
coalesced (consecutive routes are adjacent within a column) and drops the
temporary.  This is pure data movement, so the result is bitwise identical.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ENV = "DSV4_MOE_SCALE_GATHER_FUSED"
_BLOCK_R = 256


def scale_gather_fused_enabled() -> bool:
    """Off by default; the deployed path is index_select plus a transposing copy."""

    requested = os.environ.get(_ENV, "").strip().lower()
    if requested in ("", "0", "off", "false", "no"):
        return False
    if requested in ("1", "on", "true", "yes"):
        return True
    raise ValueError(f"invalid {_ENV}={requested!r}; expected 1 or 0")


@triton.jit
def _gather_mn_major_kernel(
    src_ptr,  # mn-major [M, C] uint16, storage [C, M]
    rows_ptr,  # [route_count] int64 source row per route
    dst_ptr,  # mn-major [route_count, C] uint16, storage [C, route_count]
    tokens,
    route_count,
    BLOCK_R: tl.constexpr,
):
    """One program per (column, route block): coalesced writes, gathered reads."""

    col = tl.program_id(0).to(tl.int64)
    route = tl.program_id(1).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    route_mask = route < route_count

    row = tl.load(rows_ptr + route, mask=route_mask, other=0)
    value = tl.load(src_ptr + col * tokens + row, mask=route_mask, other=0)
    tl.store(dst_ptr + col * route_count + route, value, mask=route_mask)


def gather_scale_mn_major(scale: torch.Tensor, source_rows: torch.Tensor):
    """Return ``scale[source_rows]`` as an mn-major ``[route_count, C]`` view.

    Bitwise identical to ``out.copy_(torch.index_select(scale, 0, source_rows))``
    on an mn-major ``out``, without the row-major temporary.
    """

    if scale.dim() != 2 or scale.dtype != torch.uint16:
        raise TypeError(f"scale must be uint16 [M, C], got {scale.dtype}")
    tokens, columns = (int(v) for v in scale.shape)
    if scale.stride() != (1, tokens):
        raise ValueError("scale must use the DeepGEMM mn-major stride (1, M)")
    if source_rows.dim() != 1:
        raise ValueError("source_rows must be rank 1")
    if source_rows.dtype != torch.int64 or not source_rows.is_contiguous():
        raise TypeError("source_rows must be contiguous int64 indices")
    if not scale.is_cuda or source_rows.device != scale.device:
        raise ValueError("scale and source_rows must share a CUDA-compatible device")

    route_count = int(source_rows.shape[0])
    storage = torch.empty(
        (columns, route_count), dtype=torch.uint16, device=scale.device
    )
    out = storage.t()
    if route_count:
        _gather_mn_major_kernel[(columns, triton.cdiv(route_count, _BLOCK_R))](
            scale,
            source_rows,
            storage,
            tokens,
            route_count,
            BLOCK_R=_BLOCK_R,
            num_warps=4,
        )
    return out


__all__ = ["gather_scale_mn_major", "scale_gather_fused_enabled"]
