"""Disposable BF16 token K512 selection for local eager PPU Prefill.

The caller must finish candidate publication/masking first and exclude
Decode, CP, diagnostics and idle execution. Exact ties and output order have
the existing Prefill selector's unspecified membership/order contract.
NaN/+inf consume slots before the unchanged finite filter; no slot is refilled.
"""

from __future__ import annotations

import os

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_prefill_topk as _topk
from rtp_llm.models_py.modules.dsv41.ppu import _quant


ENV = "RTP_V41_PPU_LOCAL_PREFILL_TOKEN_TOPK"
HIERARCHICAL_ENV = "RTP_V41_PPU_LOCAL_PREFILL_HIERARCHICAL_TOPK"
K = 512
MAX_WIDTH = 262144


def _device_supported(logits):
    if not logits.is_cuda or torch.version.hip is not None or not _quant.is_ppu():
        return False
    with torch.cuda.device(logits.device):
        return (
            torch.cuda.get_device_capability(logits.device) == (8, 9)
            and not torch.cuda.is_current_stream_capturing()
        )


def is_supported(logits, visible, *, bounds=None, out=None, disposable=False):
    """Metadata-only admission. Unsupported calls do not launch or mutate."""
    if not (
        os.environ.get(ENV, "0") == "1"
        and disposable is True
        and not torch.is_grad_enabled()
        and isinstance(logits, torch.Tensor)
        and isinstance(visible, torch.Tensor)
        and logits.layout == visible.layout == torch.strided
        and not any(t.is_neg() or t.is_conj() for t in (logits, visible))
        and logits.dtype == torch.bfloat16
        and not logits.requires_grad
        and logits.ndim == 2
        and 0 < logits.shape[0] < 2**31
        and K <= logits.shape[1] <= MAX_WIDTH
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.stride(0) % 8 == 0
        and logits.data_ptr() % 16 == 0
        and visible.device == logits.device
        and visible.dtype in (torch.int32, torch.int64)
        and visible.ndim == 1
        and visible.shape == (logits.shape[0],)
        and visible.stride(0) > 0
    ):
        return False
    metadata = [visible]
    if bounds is not None:
        if not (
            isinstance(bounds, (tuple, list)) and len(bounds) == 2
            and all(
                isinstance(t, torch.Tensor) and t.device == logits.device
                and t.layout == torch.strided and not t.is_neg() and not t.is_conj()
                and t.dtype == torch.int32 and t.shape == (logits.shape[0],)
                and t.is_contiguous() for t in bounds
            )
        ):
            return False
        metadata.extend(bounds)
    storage = logits.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() == storage for t in metadata):
        return False
    if out is not None:
        if not (
            isinstance(out, torch.Tensor) and out.device == logits.device
            and out.layout == torch.strided and not out.is_neg() and not out.is_conj()
            and out.dtype == torch.int32 and out.shape == (logits.shape[0], K)
            and out.is_contiguous() and out.data_ptr() % 16 == 0
        ):
            return False
        if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
               for t in (logits, *metadata)):
            return False
    return _device_supported(logits)


def _native_ops(radix):
    if radix:
        from rtp_llm.models_py.kernels.ppu.attention.ppu_bf16_radix_topk import (
            bf16_radix_topk,
            canonicalize_topk_indices,
        )

        return bf16_radix_topk, canonicalize_topk_indices
    from rtp_llm.models_py.kernels.ppu.jit.ppu_fp4_indexer import topk_bf16

    return topk_bf16, None


def try_select_tokens(logits, visible, *, bounds=None, out=None, disposable=False):
    """Return contiguous int32 [M,512], or None before unsupported work.

    Prepared bounds restrict the interval further: start/end are clamped to
    [0,width], end is intersected with visible, and start is capped at end.
    They use the existing radix implementation to support nonzero starts.
    The score tile is modified only with explicit disposable=True. Once
    admitted, execution errors propagate; the caller must not serially retry.
    """
    if not is_supported(logits, visible, bounds=bounds, out=out, disposable=disposable):
        return None
    try:
        select, canonicalize = _native_ops(bounds is not None)
        hierarchy = None
        if bounds is None and os.environ.get(HIERARCHICAL_ENV, "0") == "1":
            from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_hierarchical_topk
            from rtp_llm.models_py.kernels.ppu.attention.ppu_bf16_radix_topk import canonicalize_topk_indices

            hierarchy = local_prefill_hierarchical_topk
    except ImportError:
        return None
    rows, width = logits.shape
    ends = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    _topk._prefill_topk_bounds_kernel[((rows + 255) // 256,)](
        visible, ends, rows, width, visible.stride(0), 256,
    )
    if bounds is None:
        starts = torch.zeros_like(ends)
    else:
        ends = torch.minimum(ends, bounds[1].clamp(0, width))
        starts = torch.minimum(bounds[0].clamp(0, width), ends)
    output = out if out is not None else torch.empty((rows, K), dtype=torch.int32, device=logits.device)
    hierarchical = hierarchy is not None and hierarchy.is_supported(logits, ends, output)
    if hierarchical:
        canonicalize = canonicalize_topk_indices
    # Candidate block pooling/publication must already be complete. Collapse
    # both NaN signs above finite values, preserving the legacy finite count.
    logits.nan_to_num_(nan=torch.inf, posinf=torch.inf, neginf=-torch.inf)
    if hierarchical:
        if hierarchy.select(logits, starts, ends, output) is not output:
            raise RuntimeError("local Prefill hierarchical TopK admission changed before launch")
    else:
        select(logits, starts, ends, output)
    if canonicalize is not None:
        canonicalize(output)
    # This ABI addresses output as row*K; strided destinations are rejected.
    # Native selection already enforces start <= index < end.
    return _topk.finish_tokens(logits, ends, output)
