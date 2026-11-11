"""FlashMLA head padding for arbitrary valid V4.1 tensor-parallel widths.

Each attention head is independent. Padding unused query heads to the kernel's
64/128-head tiles keeps the model's head partition unrestricted and slices only
unobserved heads from the result.

Pre-SM100 devices (e.g. Ada sm89) are rejected by the packaged sparse-prefill
kernel itself ("only supported on SM100f architectures"); there the wrapper
falls back to an exact query-sub-chunked gather implementation with the same
softmax+sink semantics as ``utils._sparse_attn``.
"""

import os
from functools import lru_cache

import torch
import torch.nn.functional as F


def _pad_heads(q, attn_sink):
    heads = q.shape[-2]
    kernel_heads = ((heads + 63) // 64) * 64
    if kernel_heads == heads:
        return q, attn_sink, heads
    q = F.pad(q, (0, 0, 0, kernel_heads - heads))
    if attn_sink is not None:
        attn_sink = F.pad(attn_sink, (0, kernel_heads - heads), value=float("inf"))
    return q, attn_sink, heads


@lru_cache(maxsize=None)
def _sparse_native_supported(index) -> bool:
    """The native sparse-prefill kernel ships for SM100 families only."""
    major, _ = torch.cuda.get_device_capability(index)
    return major >= 10


def _sparse_ref_rows() -> int:
    """Query rows per reference sub-chunk; bounds the fp32 gather temporary."""
    return max(1, int(os.environ.get("DSV41_SPARSE_REF_ROWS", "16")))


def _sparse_fwd_reference(q, kv, indices, sm_scale, attn_sink, topk_length):
    """Exact gather-based sparse attention for pre-SM100 devices.

    Mirrors ``utils._sparse_attn`` semantics on the flat prefill layout:
    q ``[..., H, D]``, kv ``[T, 1, D]`` bf16 rows, ``indices`` ``[M, K]``
    with -1 padding bounded by ``topk_length`` ``[M]``, and the per-head
    attention sink contributing exp(sink - max) to the softmax denominator
    only. fp32 accumulation; query rows are sub-chunked so the gathered
    ``[rows, K, D]`` temporary stays bounded on cards whose free memory is
    already tight. Returns ``(output, maximum, lse)`` like the native call.
    """
    shape = q.shape
    heads, dim = shape[-2], shape[-1]
    q_flat = q.reshape(-1, heads, dim)
    rows = q_flat.shape[0]
    # ``kv`` is either the dense [T, 1, D] value source or a callable row
    # source (pool-paged long-context path) returning [N, D] for flat
    # indices; both feed the identical per-sub-chunk math below.
    kv_flat = None if callable(kv) else kv.reshape(-1, dim)
    idx_full = indices.reshape(rows, -1).to(torch.long)
    width = idx_full.shape[1]
    lengths = (
        topk_length.reshape(rows).to(torch.long) if topk_length is not None else None
    )
    sink = attn_sink.reshape(heads).to(torch.float32) if attn_sink is not None else None
    out = torch.empty_like(q_flat)
    maximum = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    column = torch.arange(width, device=q.device)
    step = _sparse_ref_rows()
    for begin in range(0, rows, step):
        end = min(begin + step, rows)
        idx = idx_full[begin:end]
        valid = idx >= 0
        if lengths is not None:
            valid = valid & (column[None, :] < lengths[begin:end, None])
        if kv_flat is None:
            selected = (
                kv(idx.clamp_min(0).reshape(-1))
                .to(torch.float32)
                .view(idx.shape[0], width, dim)
            )
        else:
            selected = kv_flat[idx.clamp_min(0)].to(torch.float32)
        queries = q_flat[begin:end].to(torch.float32)
        logits = torch.einsum("mhd,mkd->mhk", queries, selected) * sm_scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        scores_max = logits.amax(dim=-1).clamp_min(-1e30)
        exp_logits = torch.exp(logits - scores_max.unsqueeze(-1))
        sum_exp = exp_logits.sum(dim=-1)
        if sink is not None:
            sum_exp = sum_exp + torch.exp(sink[None, :] - scores_max)
        acc = torch.einsum("mhk,mkd->mhd", exp_logits, selected)
        out[begin:end] = (acc / sum_exp.unsqueeze(-1)).to(out.dtype)
        maximum[begin:end] = scores_max
        lse[begin:end] = scores_max + torch.log(sum_exp)
    return out.reshape(shape), maximum, lse


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    if q.is_cuda and not _sparse_native_supported(q.device.index):
        return _sparse_fwd_reference(
            q, kv, indices, sm_scale, attn_sink, topk_length
        )
    if callable(kv):
        raise RuntimeError(
            "paged sparse-prefill KV source requires the pre-SM100 reference path"
        )

    from flash_mla import flash_mla_sparse_fwd as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return output[:, :heads], maximum[:, :heads], lse[:, :heads]


def flash_mla_with_kvcache(*, q, attn_sink=None, **kwargs):
    from flash_mla import flash_mla_with_kvcache as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, lse = native(q=q, attn_sink=sink, **kwargs)
    return output[..., :heads, :], lse[:, :heads, :]
