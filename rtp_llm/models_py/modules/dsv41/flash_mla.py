"""FlashMLA dispatch with a Triton sparse fallback for non-SM100 GPUs.

The installed FlashMLA wheel only ships ``sm_100a``/``sm_103a`` cubins.  On
L20 (SM89) the sparse MLA paths are served by
``fp8._v41_sparse_attn_triton`` after the packed FP8/FP4 pools are dequantized
to BF16 with the same codecs used by the prefill paths.  Dense attention is
never substituted: both sparse prefill and sparse decode keep their original
O(top-k) work.
"""

import functools

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


@functools.lru_cache(maxsize=None)
def _native_arch(device_index):
    major, _ = torch.cuda.get_device_capability(device_index)
    return major == 10


def _native_supported(device):
    if device.type != "cuda":
        return False
    try:
        import flash_mla  # noqa: F401
    except Exception:
        return False
    return _native_arch(device.index if device.index is not None else torch.cuda.current_device())


def _dequant_pool(pool_4d, indices):
    pool = pool_4d.squeeze(-2) if pool_4d.dim() == 4 else pool_4d
    entry = int(pool.shape[-1])
    flat = indices.reshape(-1).to(torch.int64)
    if entry == 528:
        from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
            dequantize_swa_k_cache,
        )

        return dequantize_swa_k_cache(pool, flat)
    if entry == 288:
        from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
            dequantize_k_cache_slots_fp4,
        )

        return dequantize_k_cache_slots_fp4(pool, flat)
    if entry == 584:
        from rtp_llm.models_py.modules.dsv41.fp8._swa_dequant_triton import (
            dequantize_slots_to_bf16,
        )

        return dequantize_slots_to_bf16(pool, flat)
    raise ValueError(f"unsupported packed KV entry width {entry} on non-SM100 GPU")


def _decode_sparse_fallback(
    *,
    q,
    k_cache,
    indices,
    attn_sink,
    topk_length,
    extra_k_cache,
    extra_indices_in_kvcache,
    extra_topk_length,
    softmax_scale,
):
    from rtp_llm.models_py.modules.dsv41.fp8._v41_sparse_attn_triton import (
        sparse_mla_fwd,
    )

    primary = _dequant_pool(k_cache, indices)
    kv = primary
    raw_idx = indices.reshape(indices.shape[0], -1).to(torch.int64)
    rows = torch.arange(raw_idx.shape[1], device=raw_idx.device, dtype=torch.int64)
    batch_rows = torch.arange(
        raw_idx.shape[0], device=raw_idx.device, dtype=torch.int64
    ).view(-1, 1) * raw_idx.shape[1]
    idx = torch.where(
        raw_idx >= 0,
        rows.view(1, -1) + batch_rows,
        torch.full_like(raw_idx, -1),
    )
    length = topk_length
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        extra = _dequant_pool(extra_k_cache, extra_indices_in_kvcache)
        raw_eidx = extra_indices_in_kvcache.reshape(
            extra_indices_in_kvcache.shape[0], -1
        ).to(torch.int64)
        erows = torch.arange(
            raw_eidx.shape[1], device=raw_eidx.device, dtype=torch.int64
        )
        ebatch = torch.arange(
            raw_eidx.shape[0], device=raw_eidx.device, dtype=torch.int64
        ).view(-1, 1) * raw_eidx.shape[1]
        eidx = torch.where(
            raw_eidx >= 0,
            erows.view(1, -1) + ebatch + int(primary.shape[0]),
            torch.full_like(raw_eidx, -1),
        )
        kv = torch.cat((primary, extra), dim=0)
        idx = torch.cat((idx, eidx), dim=1)
        # Per-pool lengths cannot be represented by one prefix length once the
        # primary list has tail padding.  -1 entries are authoritative.
        length = None
    out, _, lse = sparse_mla_fwd(
        q,
        kv,
        idx,
        softmax_scale,
        attn_sink=attn_sink,
        topk_length=length,
    )
    b, s, h, d = q.shape
    out = out.view(b, s, h, d)
    lse = lse.view(b, s, h).permute(0, 2, 1).contiguous()
    return out, lse


def flash_mla_sparse_fwd(
    q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None
):
    if not _native_supported(q.device):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sparse_attn_triton import (
            sparse_mla_fwd,
        )

        output, maximum, lse = sparse_mla_fwd(
            q,
            kv,
            indices,
            sm_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        return output[..., :d_v], maximum, lse

    from flash_mla import flash_mla_sparse_fwd as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return output[:, :heads], maximum[:, :heads], lse[:, :heads]


def flash_mla_with_kvcache(
    *,
    q,
    attn_sink=None,
    k_cache=None,
    indices=None,
    topk_length=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    extra_topk_length=None,
    softmax_scale=None,
    **kwargs,
):
    if not _native_supported(q.device):
        return _decode_sparse_fallback(
            q=q,
            k_cache=k_cache,
            indices=indices,
            attn_sink=attn_sink,
            topk_length=topk_length,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            softmax_scale=softmax_scale,
        )

    from flash_mla import flash_mla_with_kvcache as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, lse = native(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=topk_length,
        attn_sink=sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
        softmax_scale=softmax_scale,
        **kwargs,
    )
    return output[..., :heads, :], lse[:, :heads, :]
