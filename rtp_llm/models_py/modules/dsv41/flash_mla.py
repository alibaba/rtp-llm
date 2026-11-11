"""FlashMLA head padding and non-SM100 sparse-prefill reference path.

Each attention head is independent. Padding unused query heads to the kernel's
64/128-head tiles keeps the model's head partition unrestricted and slices only
unobserved heads from the result.

``flash_mla_sparse_fwd`` is a SM100-only kernel.  A100 (SM80) and other
pre-Blackwell devices therefore execute an explicit gather/online-softmax
reference for sparse prefill.  The reference consumes the same packed
``(q, kv, indices, attn_sink, topk_length)`` contract and keeps the attention
sink contribution in the denominator, so the runtime fallback preserves the
model's semantics instead of changing the workload.
"""

import os

import torch
import torch.nn.functional as F

_S_SPLIT = 256
_K_SPLIT = 256
_FORCE_REFERENCE_ENV = "RTP_LLM_FLASH_MLA_SPARSE_REFERENCE"


def _pad_heads(q, attn_sink):
    heads = q.shape[-2]
    kernel_heads = ((heads + 63) // 64) * 64
    if kernel_heads == heads:
        return q, attn_sink, heads
    q = F.pad(q, (0, 0, 0, kernel_heads - heads))
    if attn_sink is not None:
        attn_sink = F.pad(attn_sink, (0, kernel_heads - heads), value=float("inf"))
    return q, attn_sink, heads


def _slice_heads(tensor, heads):
    """Slice the head axis (``-2``) for both ``[S,H,D]`` and ``[B,S,H,D]``."""
    return tensor[..., :heads, :]


def _slice_lse(tensor, heads):
    """Slice the head axis (``-1``) for both ``[S,H]`` and ``[B,S,H]``."""
    return tensor[..., :heads]


def _reference_backend(q):
    forced = os.environ.get(_FORCE_REFERENCE_ENV, "").strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    if not q.is_cuda:
        return True
    try:
        major, _ = torch.cuda.get_device_capability(q.device)
    except Exception:
        return True
    if major < 10:
        return True
    try:
        import flash_mla  # noqa: F401
    except Exception:
        return True
    return False


def _normalize_inputs(q, kv, indices, topk_length):
    """Return batch-major ``(q4, kv4, idx4, lens, ndim)``.

    Native prefill takes ``q=[S,H,D]``, ``kv=[N,1,D]`` and ``indices=[S,1,K]``.
    The V4.1 runtime also uses the 4D batch-major form.  Both are normalized
    here so the reference sees one shape contract.
    """
    ndim = q.dim()
    if ndim == 3:
        q4 = q.unsqueeze(0)
        if kv.dim() == 3:
            kv4 = kv.squeeze(1).unsqueeze(0) if kv.shape[1] == 1 else kv.unsqueeze(0)
        else:
            kv4 = kv.unsqueeze(0)
        if indices.dim() == 3:
            idx4 = (
                indices.squeeze(1).unsqueeze(0)
                if indices.shape[1] == 1
                else indices.unsqueeze(0)
            )
        else:
            idx4 = indices.unsqueeze(0)
        lens = None
        if topk_length is not None:
            lens = topk_length.reshape(1, -1)
    elif ndim == 4:
        q4 = q
        kv4 = kv[:, :, 0, :] if kv.dim() == 4 else kv
        idx4 = indices[:, :, 0, :] if indices.dim() == 4 else indices
        lens = topk_length
        if lens is not None:
            lens = lens.reshape(q4.shape[0], -1)
    else:
        raise ValueError(f"sparse prefill q must be 3D or 4D, got {tuple(q.shape)}")
    return q4, kv4, idx4, lens, ndim


def _sparse_prefill_reference(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    """Gather-based sparse MLA prefill with sink, for pre-SM100 devices.

    This is deliberately written as a streaming/online-softmax kernel: the
    full ``[S, topk, D]`` gathered K tensor is never materialized, so the same
    path is usable for a 1M-token prefill.  Each query row only reads its own
    top-k slots, exactly like the native sparse kernel.
    """
    q4, kv4, idx4, lens, ndim = _normalize_inputs(q, kv, indices, topk_length)
    batch, s_q, heads, _ = q4.shape
    s_kv = kv4.shape[1]
    topk = idx4.shape[-1]
    out = torch.zeros(
        (batch, s_q, heads, d_v), dtype=torch.float32, device=q4.device
    )
    max_logits = torch.full(
        (batch, s_q, heads), float("-inf"), dtype=torch.float32, device=q4.device
    )
    lse = torch.full_like(max_logits, float("-inf"))
    sink = None
    if attn_sink is not None:
        sink = attn_sink.reshape(-1).to(torch.float32)
        if sink.numel() != heads:
            raise ValueError(
                f"attn_sink has {sink.numel()} entries for {heads} heads"
            )

    k_offsets = torch.arange(topk, device=q4.device)
    for b in range(batch):
        kv_b = kv4[b, :, :d_v]
        for s0 in range(0, s_q, _S_SPLIT):
            s1 = min(s0 + _S_SPLIT, s_q)
            rows = s1 - s0
            q_b = q4[b, s0:s1, :, :d_v].to(torch.float32)
            idx_b = idx4[b, s0:s1].to(torch.int64)
            valid = (idx_b >= 0) & (idx_b < s_kv)
            if lens is not None:
                row_len = lens[b, s0:s1].to(q4.device).to(torch.int64)
                valid = valid & (k_offsets.unsqueeze(0) < row_len.unsqueeze(1))

            # ``m`` is the running max including the sink logit; ``l_den`` is
            # the attention denominator including exp(sink - m).  ``l_sum``
            # tracks only real key logits so ``lse`` matches the native
            # operator (the sink never affects max_logits/lse).
            m = torch.full((rows, heads), -1e30, dtype=torch.float32, device=q4.device)
            l_den = torch.zeros((rows, heads), dtype=torch.float32, device=q4.device)
            l_sum = torch.zeros_like(l_den)
            acc = torch.zeros((rows, heads, d_v), dtype=torch.float32, device=q4.device)
            if sink is not None:
                sink_row = sink.view(1, heads)
                m = torch.maximum(m, sink_row)
                l_den = torch.exp(sink_row - m)

            for k0 in range(0, topk, _K_SPLIT):
                k1 = min(k0 + _K_SPLIT, topk)
                idx_blk = idx_b[:, k0:k1]
                valid_blk = valid[:, k0:k1]
                if not bool(valid_blk.any()):
                    continue
                safe = idx_blk.masked_fill(~valid_blk, 0).reshape(-1)
                sel = torch.index_select(kv_b, 0, safe).view(rows, k1 - k0, d_v)
                sel = sel.to(torch.float32)
                logits = torch.einsum("rhd,rkd->rhk", q_b, sel) * sm_scale
                logits = logits.masked_fill(~valid_blk.unsqueeze(1), float("-inf"))
                block_max = logits.amax(dim=-1).clamp_min(-1e30)
                new_m = torch.maximum(m, block_max)
                alpha = torch.exp(m - new_m)
                p = torch.exp(logits - new_m.unsqueeze(-1))
                p_sum = p.sum(dim=-1)
                l_den = l_den * alpha + p_sum
                l_sum = l_sum * alpha + p_sum
                acc = acc * alpha.unsqueeze(-1) + torch.einsum("rhk,rkd->rhd", p, sel)
                m = new_m

            out_b = acc / l_den.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
            out_b = torch.where(l_den.unsqueeze(-1) > 0, out_b, torch.zeros_like(out_b))
            out[b, s0:s1] = out_b
            max_logits[b, s0:s1] = torch.where(
                l_sum > 0, m, torch.full_like(m, float("-inf"))
            )
            lse[b, s0:s1] = torch.where(
                l_sum > 0,
                m + torch.log(l_sum.clamp_min(torch.finfo(torch.float32).tiny)),
                torch.full_like(m, float("-inf")),
            )

    out = out.to(q.dtype)
    if ndim == 3:
        return out.squeeze(0), max_logits.squeeze(0), lse.squeeze(0)
    return out, max_logits, lse


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    if _reference_backend(q):
        # No head padding on the reference path: it computes exactly the
        # observed heads and needs no 64/128-head kernel tile.
        return _sparse_prefill_reference(
            q,
            kv,
            indices,
            sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

    from flash_mla import flash_mla_sparse_fwd as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return (
        _slice_heads(output, heads),
        _slice_lse(maximum, heads),
        _slice_lse(lse, heads),
    )


def flash_mla_with_kvcache(*, q, attn_sink=None, **kwargs):
    from flash_mla import flash_mla_with_kvcache as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, lse = native(q=q, attn_sink=sink, **kwargs)
    return output[..., :heads, :], lse[..., :heads, :]
