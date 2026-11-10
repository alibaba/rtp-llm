"""Bounded, strided token K512 selection for the V4.1 prefill indexer.

The public helper preserves ``torch.topk(...).values.isfinite()`` semantics:
NaN/+inf can occupy top-k slots, but every selected nonfinite entry becomes
-1, including short rows. Output ordering and membership within exact ties
are unspecified. Candidate-block selection has a separate contract.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.fp8.indexer import _TOPK_V3_OK, _get_topk_workspace
from rtp_llm.ops.compute_ops import rtp_llm_ops


def supports_bf16() -> bool:
    """The finite selector entry also identifies native BF16 input support."""
    return _TOPK_V3_OK and hasattr(rtp_llm_ops, "dsv41_topk_v3_finite")


def is_supported(logits: torch.Tensor, visible: torch.Tensor, topk: int = 512) -> bool:
    return (
        _TOPK_V3_OK
        and torch.version.hip is None
        and logits.is_cuda
        and (
            logits.dtype == torch.float32
            or (logits.dtype == torch.bfloat16 and supports_bf16())
        )
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


@triton.jit(do_not_specialize=["ROWS", "WIDTH"])
def _prefill_topk_bounds_kernel(
    visible,
    ends,
    ROWS,
    WIDTH,
    VISIBLE_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    rows = tl.program_id(0) * TILE + tl.arange(0, TILE)
    length = tl.load(visible + rows * VISIBLE_STRIDE, rows < ROWS, other=0)
    # Clamp in the input integer dtype before converting int64 positions.
    length = tl.minimum(tl.maximum(length, 0), WIDTH).to(tl.int32)
    tl.store(ends + rows, length, rows < ROWS)


@triton.jit(do_not_specialize=["STRIDE"])
def _prefill_topk_finite_kernel(
    logits,
    ends,
    output,
    STRIDE,
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


@triton.jit
def _write_torch_topk_results_kernel(
    scores,
    chosen,
    out,
    SCORE_STRIDE: tl.constexpr,
    CHOSEN_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    K: tl.constexpr,
    TILE: tl.constexpr,
    ALLOW_POS_INF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, TILE)
    active = slot < K
    value = tl.load(scores + row * SCORE_STRIDE + slot, active, other=-float("inf"))
    index = tl.load(chosen + row * CHOSEN_STRIDE + slot, active, other=-1)
    value = value.to(tl.float32)
    if ALLOW_POS_INF:
        keep = value > -float("inf")
    else:
        keep = tl.abs(value) < float("inf")
    result = tl.where(keep, index.to(tl.int32), -1)
    tl.store(out + row * OUT_STRIDE + slot, result, active)


def write_torch_topk_results(
    scores: torch.Tensor,
    chosen: torch.Tensor,
    out: torch.Tensor,
    *,
    allow_pos_inf: bool = False,
) -> bool:
    """Write Torch TopK results in slot order, leaving any output tail untouched.

    Token scores require finite values. Candidate block scores also retain
    positive infinity, which pins the newest visible block.
    """
    if not (
        scores.is_cuda
        and scores.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and chosen.dtype == torch.int64
        and out.dtype == torch.int32
        and scores.ndim == 2
        and scores.shape[0] > 0
        and 0 < scores.shape[1] <= 4096
        and chosen.shape == out.shape == scores.shape
        and chosen.device == out.device == scores.device
        and all(
            tensor.stride(1) == 1 and tensor.stride(0) >= scores.shape[1]
            for tensor in (scores, chosen, out)
        )
    ):
        return False
    width = scores.shape[1]
    _write_torch_topk_results_kernel[(scores.shape[0],)](
        scores,
        chosen,
        out,
        scores.stride(0),
        chosen.stride(0),
        out.stride(0),
        width,
        triton.next_power_of_2(width),
        allow_pos_inf,
    )
    return True


def _select_tokens(
    logits: torch.Tensor,
    visible: torch.Tensor,
    topk: int = 512,
    *,
    bounds=None,
    out=None,
    filter_finite=True,
) -> torch.Tensor:
    """Select native BF16 or FP32 scores without a full-matrix conversion."""
    from . import _v41_prefill_deepselect as deepselect

    rows, width = logits.shape
    if bounds is None:
        ends = torch.empty((rows,), device=logits.device, dtype=torch.int32)
        _prefill_topk_bounds_kernel[(triton.cdiv(rows, 256),)](
            visible, ends, rows, width, visible.stride(0), 256
        )
    else:
        ends = bounds[1]
    if deepselect.is_available(logits.device, logits.dtype):
        selected = deepselect.try_select_tokens(
            logits, ends, out=out, filter_finite=filter_finite
        )
        if selected is not None:
            return selected
    output = (
        out
        if out is not None
        else torch.empty((rows, topk), device=logits.device, dtype=torch.int32)
    )
    finite_select = (
        getattr(rtp_llm_ops, "dsv41_topk_v3_finite", None) if filter_finite else None
    )
    select = finite_select if finite_select is not None else rtp_llm_ops.topk_v3
    select(logits, ends, output, _get_topk_workspace(logits.device), topk, width)
    if filter_finite and finite_select is None:
        finish_tokens(logits, ends, output)
    return output


def finish_tokens(logits, ends, output):
    """Finish a previously qualified raw selection with the original predicate."""
    _prefill_topk_finite_kernel[(logits.shape[0],)](
        logits, ends, output, logits.stride(0), output.shape[1]
    )
    return output


def try_select_tokens(
    logits: torch.Tensor,
    visible: torch.Tensor,
    topk: int = 512,
    *,
    bounds=None,
    out=None,
    filter_finite=True,
) -> torch.Tensor | None:
    """Return selected int32 indices, or None before launching if unsupported.

    A caller disabling finite filtering must supply bounds/output and complete
    the same filter before exposing the result, including on epilogue fallback.
    """
    if not is_supported(logits, visible, topk):
        return None
    if not filter_finite and (bounds is None or out is None):
        return None
    if bounds is not None and not (
        len(bounds) == 2
        and all(
            tensor.device == logits.device
            and tensor.dtype == torch.int32
            and tensor.shape == (logits.shape[0],)
            and tensor.is_contiguous()
            for tensor in bounds
        )
    ):
        return None
    if out is not None and not (
        out.device == logits.device
        and out.dtype == torch.int32
        and out.shape == (logits.shape[0], topk)
        and out.is_contiguous()
    ):
        return None
    return _select_tokens(
        logits, visible, topk, bounds=bounds, out=out, filter_finite=filter_finite
    )
