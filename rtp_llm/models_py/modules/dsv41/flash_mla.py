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


def _sparse_fwd_reference(q, kv, indices, sm_scale, attn_sink, topk_length):
    """Pre-SM100 reference for ``flash_mla_sparse_fwd`` over a BF16 workspace.

    Contract (matches the native kernel as consumed by dsv41):
      * q ``[T, H, D]`` bf16, kv ``[rows, 1, D]`` bf16 (already-dequantized
        workspace rows; the window/index band is pre-encoded in ``indices``),
      * indices ``[T, K]`` int32 row ids into kv dim 0, ``-1`` masked;
        ``topk_length`` ``[T]`` bounds the leftmost valid entries (both the
        length bound and the ``-1`` test are applied — holes may exist inside
        the bound),
      * attn_sink ``[H]``: extra per-head denominator logit with a zero value
        vector,
      * returns ``(output [T,H,D], maximum [T,H], lse [T,H])`` with the raw
        sink-free row max/log-partition (empty rows report +inf lse, like the
        native kernel).
    Row-chunked to bound the fp32 gather temporary; rows are independent so
    chunking is numerically neutral.
    """
    import os

    T, H, D = q.shape
    K = indices.shape[-1]
    kv2 = kv.reshape(-1, kv.shape[-1])
    valid = indices >= 0
    if topk_length is not None:
        cols = torch.arange(K, device=indices.device).view(1, -1)
        valid = valid & (cols < topk_length.view(T, 1).to(torch.int64))
    chunk = max(int(os.environ.get("DSV41_SM80_SPARSE_Q_CHUNK", "1024")), 1)
    outs, maxes, lses = [], [], []
    for start in range(0, T, chunk):
        stop = min(start + chunk, T)
        rows = indices[start:stop].to(torch.int64).clamp_min(0)
        keys = kv2.index_select(0, rows.reshape(-1)).view(stop - start, K, D).float()
        logits = (
            torch.einsum("thd,tkd->thk", q[start:stop].float(), keys) * sm_scale
        )
        logits = logits.masked_fill(~valid[start:stop].unsqueeze(1), float("-inf"))
        row_max = logits.amax(dim=-1).clamp_min(-1e30)  # [t, H]
        exp_logits = torch.exp(logits - row_max.unsqueeze(-1))
        raw_sum = exp_logits.sum(dim=-1)  # [t, H], sink-free
        denom = raw_sum
        if attn_sink is not None:
            denom = denom + torch.exp(attn_sink.float().view(1, H) - row_max)
        acc = torch.einsum("thk,tkd->thd", exp_logits, keys)
        any_valid = valid[start:stop].any(dim=-1).unsqueeze(1)  # [t, 1]
        out = torch.where(any_valid, acc / denom.clamp_min(1e-38).unsqueeze(-1), acc)
        outs.append(out.to(q.dtype))
        maxes.append(row_max)
        lses.append(
            torch.where(
                any_valid,
                torch.log(raw_sum.clamp_min(1e-38)),
                torch.full_like(row_max, float("inf")),
            )
        )
    return torch.cat(outs, 0), torch.cat(maxes, 0), torch.cat(lses, 0)


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    from rtp_llm.models_py.modules.dsv41.sm80_support import (
        has_sm100_native_kernels,
    )

    if not has_sm100_native_kernels():
        return _sparse_fwd_reference(
            q, kv, indices, sm_scale, attn_sink, topk_length
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
