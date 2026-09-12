"""TP4 Pro compatibility for FlashMLA wheels with a 64-head minimum."""

import torch.nn.functional as F


def _pad_heads(q, attn_sink):
    heads = q.shape[-2]
    if heads != 32:
        return q, attn_sink
    # Heads have independent softmax reductions. Extra heads cannot affect
    # the original outputs, and padding keeps the native packed-KV kernel.
    q = F.pad(q, (0, 0, 0, 32))
    if attn_sink is not None:
        attn_sink = F.pad(attn_sink, (0, 32))
    return q, attn_sink


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    from flash_mla import flash_mla_sparse_fwd as native

    heads = q.shape[-2]
    q, attn_sink = _pad_heads(q, attn_sink)
    options = {"d_v": d_v} if d_v != 512 else {}
    output, maximum, lse = native(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        **options,
    )
    if heads == 32:
        return (
            output[:, :heads].contiguous(),
            maximum[:, :heads].contiguous(),
            lse[:, :heads].contiguous(),
        )
    return output, maximum, lse


def flash_mla_with_kvcache(*, q, attn_sink=None, **kwargs):
    from flash_mla import flash_mla_with_kvcache as native

    heads = q.shape[-2]
    q, attn_sink = _pad_heads(q, attn_sink)
    output, lse = native(q=q, attn_sink=attn_sink, **kwargs)
    if heads == 32:
        return output[:, :, :heads].contiguous(), lse[:, :heads].contiguous()
    return output, lse


def get_mla_metadata(**kwargs):
    from flash_mla import get_mla_metadata as native

    if kwargs.get("num_heads_q") == 32:
        kwargs["num_heads_q"] = 64
        if "num_q_tokens_per_head_k" in kwargs:
            kwargs["num_q_tokens_per_head_k"] *= 2
    return native(**kwargs)
