"""ROCm bridge for the sparse-MLA primitives DSV4.1 calls FlashMLA for.

RTP-LLM's V4.1 prefill path already materialises the per-query KV rows it
needs (compressed pool + sliding window + the current chunk) into a dense
BF16 workspace before calling :func:`flash_mla_sparse_fwd`.  On ROCm there
is no FlashMLA wheel, but AITER ships a portable Triton sparse-attention
kernel (``_sparse_attn_prefill_kernel``) that consumes exactly that shape:
``q[M, H, D]``, ``kv[N, D]``, per-query indices and a per-head sink.

This module adapts the RTP call contract to that kernel.  It deliberately
keeps the CUDA path untouched: :mod:`rtp_llm.models_py.modules.dsv41.flash_mla`
dispatches here only when ``torch.version.hip`` is set.  The kernel is
imported lazily so importing this module on a CUDA/CPU host never pulls in
AITER.

No approximation is introduced: the kernel carries the same online-softmax
recurrence, the same ``-1`` invalid-index sentinel semantics and the same
learned per-head attention sink as the CUDA reference.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

_LOGGER = logging.getLogger(__name__)


def _load_kernel():
    try:
        from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
            _sparse_attn_prefill_kernel,
        )
    except Exception as error:  # pragma: no cover - depends on ROCm image
        raise RuntimeError(
            "DSV4.1 ROCm sparse attention requires the pinned AITER wheel "
            "(aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4); "
            f"import failed: {error!r}"
        ) from error
    return _sparse_attn_prefill_kernel


def _ragged_indices(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    num_queries: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(flat_indices, indptr)`` for the Triton sparse kernel.

    ``indices`` is ``[num_queries, width]``.  The kernel walks
    ``flat_indices[indptr[q]:indptr[q+1]]``.  We use a fixed ``width`` stride
    so no host-side compaction is needed; entries outside ``topk_length``
    are rewritten to the ``-1`` sentinel that the kernel masks out.  The
    padding rewrite is a device-side ``where`` and therefore capture-safe.
    """
    if indices.dim() != 2 or indices.shape != (num_queries, width):
        raise ValueError(
            f"indices must be [{num_queries}, {width}], got {tuple(indices.shape)}"
        )
    flat = indices.to(torch.int32)
    if topk_length is not None:
        lengths = topk_length.to(device=indices.device, dtype=torch.int32).reshape(-1)
        if lengths.numel() != num_queries:
            raise ValueError(
                f"topk_length must have {num_queries} entries, got {lengths.numel()}"
            )
        positions = torch.arange(width, device=indices.device, dtype=torch.int32)
        keep = positions.unsqueeze(0) < lengths.unsqueeze(1)
        flat = torch.where(keep, flat, torch.full_like(flat, -1))
    flat = flat.contiguous().reshape(-1)
    # int64 before the multiply: 1M-token prefills overflow int32 here.
    indptr = (
        torch.arange(0, num_queries + 1, dtype=torch.int64, device=indices.device)
        * width
    ).to(torch.int32)
    return flat, indptr


def sparse_mla_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, None, None]:
    """ROCm equivalent of ``flash_mla_sparse_fwd``.

    Args:
        q: ``[M, H, D]`` BF16 queries.
        kv: ``[N, D]`` or ``[N, 1, D]`` BF16 key/value rows.
        indices: ``[M, 1, K]`` or ``[M, K]`` int32 indices into ``kv``.
        sm_scale: softmax scale.
        d_v: value head dimension used by the caller's contract.
        attn_sink: optional ``[H]`` fp32 per-head sink logit.
        topk_length: optional ``[M]`` int32 valid prefix length per query.

    Returns:
        ``(output, None, None)``.  The second/third slots mirror FlashMLA's
        ``maximum``/``lse`` returns.  LSE is not produced by this kernel and
        is intentionally returned as ``None`` rather than fabricated; the
        only V4.1 consumer is the CP merge, which this lane does not enable.
    """
    import triton

    if q.dim() != 3:
        raise ValueError(f"q must be [M, H, D], got {tuple(q.shape)}")
    if kv.dim() == 3:
        if kv.shape[1] != 1:
            raise ValueError(f"kv must be [N, 1, D] or [N, D], got {tuple(kv.shape)}")
        kv = kv[:, 0, :]
    if kv.dim() != 2:
        raise ValueError(f"kv must be [N, D], got {tuple(kv.shape)}")
    if indices.dim() == 3:
        if indices.shape[1] != 1:
            raise ValueError(
                f"indices must be [M, 1, K] or [M, K], got {tuple(indices.shape)}"
            )
        indices = indices[:, 0, :]
    if indices.dim() != 2:
        raise ValueError(f"indices must be [M, K], got {tuple(indices.shape)}")

    if not q.is_contiguous():
        q = q.contiguous()
    if not kv.is_contiguous():
        kv = kv.contiguous()

    num_queries, num_heads, head_dim = q.shape
    if int(d_v) != int(head_dim):
        raise ValueError(
            f"ROCm sparse-MLA requires d_v == head_dim, got d_v={d_v}, "
            f"head_dim={head_dim}"
        )
    width = int(indices.shape[1])
    if indices.shape[0] != num_queries:
        raise ValueError(
            f"indices rows {indices.shape[0]} != q rows {num_queries}"
        )
    if kv.shape[0] == 0 or width == 0:
        return torch.zeros_like(q), None, None

    flat_indices, indptr = _ragged_indices(
        indices, topk_length, num_queries, width
    )

    # ``attn_sink=None`` means "no virtual key".  The kernel expresses that
    # with exp(sink) == 0, i.e. a -inf sink logit.  Never pass a zero sink,
    # which would add a real exp(0)=1 term to every denominator.
    if attn_sink is None:
        sink = torch.full(
            (num_heads,), float("-inf"), dtype=torch.float32, device=q.device
        )
    else:
        sink = attn_sink.to(device=q.device, dtype=torch.float32).reshape(-1)
        if sink.numel() != num_heads:
            raise ValueError(
                f"attn_sink must have {num_heads} entries, got {sink.numel()}"
            )
        sink = sink.contiguous()

    kernel = _load_kernel()
    out = torch.empty_like(q)
    grid = lambda meta: (num_queries, triton.cdiv(num_heads, meta["BLOCK_H"]))
    kernel[grid](
        q,
        kv,
        flat_indices,
        indptr,
        sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        int(kv.shape[0]),
        float(sm_scale),
        HAS_ATTN_SINK=True,
        BLOCK_D=int(d_v),
    )
    _LOGGER.debug(
        "[dsv41-rocm] sparse prefill q=%s kv=%s indices=%s sink=%s",
        tuple(q.shape),
        tuple(kv.shape),
        tuple(indices.shape),
        attn_sink is not None,
    )
    return out, None, None