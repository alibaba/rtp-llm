"""FlashMLA head padding for arbitrary valid V4.1 tensor-parallel widths.

Each attention head is independent. Padding unused query heads to the kernel's
64/128-head tiles keeps the model's head partition unrestricted and slices only
unobserved heads from the result.
"""

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


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    if q.is_cuda and torch.cuda.get_device_capability(q.device) == (12, 0):
        from .fp8.sm120_sparse_attention import sparse_prefill

        return sparse_prefill(q, kv, indices, sm_scale, d_v, attn_sink, topk_length)
    from flash_mla import flash_mla_sparse_fwd as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return output[:, :heads], maximum[:, :heads], lse[:, :heads]


def flash_mla_with_kvcache(*, q, attn_sink=None, **kwargs):
    if q.is_cuda and torch.cuda.get_device_capability(q.device) == (12, 0):
        from .fp8.sm120_sparse_attention import sparse_decode

        return sparse_decode(
            q,
            attn_sink,
            kwargs["k_cache"],
            kwargs["indices"],
            kwargs["softmax_scale"],
            kwargs.get("topk_length"),
            kwargs.get("extra_k_cache"),
            kwargs.get("extra_indices_in_kvcache"),
            kwargs.get("extra_topk_length"),
        )
    from flash_mla import flash_mla_with_kvcache as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, lse = native(q=q, attn_sink=sink, **kwargs)
    return output[..., :heads, :], lse[:, :heads, :]
