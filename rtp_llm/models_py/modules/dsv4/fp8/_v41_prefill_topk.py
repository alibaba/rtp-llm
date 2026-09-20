"""Bounded, strided token K512 selection for the V4.1 prefill indexer.

The public helper preserves ``torch.topk(...).values.isfinite()`` semantics:
NaN/+inf can occupy top-k slots, but every selected nonfinite entry becomes
-1, including short rows. Output ordering and membership within exact ties
are unspecified. Candidate-block selection has a separate contract.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8.indexer import (
    _get_topk_workspace,
    _topk_v3_enabled,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


def is_supported(logits: torch.Tensor, visible: torch.Tensor, topk: int = 512) -> bool:
    return (
        os.environ.get("DSV41_FUSED_PREFILL_TOPK", "1") != "0"
        and _topk_v3_enabled()
        and torch.version.hip is None
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.shape[0] > 0
        and topk == 512
        and topk <= logits.shape[1] <= 2**31 - 1
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and visible.device == logits.device
        and visible.ndim == 1
        and visible.numel() == logits.shape[0]
        and visible.dtype in (torch.int32, torch.int64)
    )


@triton.jit
def _prefill_topk_bounds_kernel(
    visible,
    starts,
    ends,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    VISIBLE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    length = tl.load(visible + rows * VISIBLE_STRIDE, rows < ROWS, other=0)
    # Clamp in the input integer dtype before converting int64 positions.
    length = tl.minimum(tl.maximum(length, 0), WIDTH).to(tl.int32)
    tl.store(starts + rows, 0, rows < ROWS)
    tl.store(ends + rows, length, rows < ROWS)


@triton.jit
def _prefill_topk_finite_kernel(
    logits,
    ends,
    output,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, K)
    end = tl.load(ends + row)
    index = tl.load(output + row * K + slot)
    in_range = (index >= 0) & (index < end)
    value = tl.load(logits + row * STRIDE + index, in_range, other=-float("inf"))
    keep = in_range & (tl.abs(value) < float("inf"))
    tl.store(output + row * K + slot, tl.where(keep, index, -1))


def _select_tokens(
    logits: torch.Tensor, visible: torch.Tensor, topk: int = 512, *, backend: str = "v3"
) -> torch.Tensor:
    """Internal candidate entry point, also used by the standalone benchmark.

    The public path uses v3, whose integer keys explicitly canonicalize both
    NaN signs. The existing row-prefill implementations remain benchmark
    candidates; they are not exposed by the public helper.
    """
    if backend not in ("v3", "insertion", "radix"):
        raise ValueError(f"Unknown prefill TopK backend: {backend}")
    rows, width = logits.shape
    bounds = torch.empty((2, rows), device=logits.device, dtype=torch.int32)
    starts, ends = bounds.unbind(0)
    output = torch.empty((rows, topk), device=logits.device, dtype=torch.int32)
    _prefill_topk_bounds_kernel[(triton.cdiv(rows, 256),)](
        visible, starts, ends, rows, width, visible.stride(0), 256
    )
    if backend == "v3":
        rtp_llm_ops.topk_v3(
            logits, ends, output, _get_topk_workspace(logits.device), topk, width
        )
    else:
        rtp_llm_ops.dsv4_top_k_per_row_prefill(
            logits,
            starts,
            ends,
            output,
            rows,
            logits.stride(0),
            logits.stride(1),
            topk,
            backend == "radix",
        )
    _prefill_topk_finite_kernel[(rows,)](logits, ends, output, logits.stride(0), topk)
    return output


def try_select_tokens(
    logits: torch.Tensor, visible: torch.Tensor, topk: int = 512
) -> torch.Tensor | None:
    """Return selected int32 indices, or None before launching if unsupported."""
    if not is_supported(logits, visible, topk):
        return None
    return _select_tokens(logits, visible, topk)
