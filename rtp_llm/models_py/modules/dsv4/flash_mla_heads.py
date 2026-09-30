"""Adapt tensor-parallel DSV4 heads to sparse FlashMLA's kernel width."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def flash_mla_num_heads(num_heads: int) -> int:
    """The sparse kernels require at least 64 heads per KV head."""
    if num_heads <= 0:
        raise ValueError(f"invalid FlashMLA query head count: {num_heads}")
    return max(64, num_heads)


def pad_flash_mla_heads(
    q: torch.Tensor, attn_sink: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pad independent heads; the caller must crop outputs to the original width."""
    heads = q.shape[-2]
    padding = flash_mla_num_heads(heads) - heads
    if not padding:
        return q, attn_sink
    if attn_sink is not None:
        if attn_sink.shape != (heads,):
            raise ValueError(
                f"attention sink shape {attn_sink.shape} does not match {heads} heads"
            )
        attn_sink = F.pad(attn_sink, (0, padding))
    return F.pad(q, (0, 0, 0, padding)), attn_sink


def flash_mla_sparse_fwd(*, q, attn_sink=None, **kwargs):
    """Sparse BF16 attention with the logical head count preserved in all outputs."""
    from flash_mla import flash_mla_sparse_fwd as kernel

    heads = q.shape[-2]
    padded_q, padded_sink = pad_flash_mla_heads(q, attn_sink)
    out, max_logits, lse = kernel(q=padded_q, attn_sink=padded_sink, **kwargs)
    if padded_q is q:
        return out, max_logits, lse
    return (
        out[:, :heads, :].contiguous(),
        max_logits[:, :heads].contiguous(),
        lse[:, :heads].contiguous(),
    )
