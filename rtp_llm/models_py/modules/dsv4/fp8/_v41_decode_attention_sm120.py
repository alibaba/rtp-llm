"""Adapt upstream decode pool views to the qualified SM120 sparse provider."""

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts
from rtp_llm.models_py.modules.dsv4.v41_attention import V41SparseAttention
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache

_attention = V41SparseAttention()


def _cache_view(pool, tag, gen):
    layout = dsv41_page_layouts(gen)[tag]
    width = layout.payload_bytes_per_entry + layout.scale_bytes_per_entry
    if (
        pool.dtype != torch.uint8
        or pool.ndim != 3
        or pool.shape[0] < 1
        or pool.shape[1:] != (layout.entries, width)
        or pool.stride(2) != 1
        or pool.stride(1) != width
        or pool.stride(0) < layout.stride_bytes
    ):
        raise ValueError(f"Unsupported SM120 V4.1 {tag} decode pool geometry")
    raw = pool.as_strided((pool.shape[0], layout.stride_bytes), (pool.stride(0), 1))
    return V41PagedCache(raw, tag, gen_num_per_cycle=gen)


def sparse_decode_attention(
    q, swa_pool, swa_slots, sinks, global_pool=None, global_slots=None
):
    if q.ndim != 4 or not q.is_contiguous():
        raise ValueError("V4.1 decode Q must be contiguous [B,S,H,512]")
    batch, span, heads, dim = q.shape
    if batch < 1 or span != 1:
        raise ValueError(
            "SM120 V4.1 decode currently requires nonempty single-token rows"
        )
    if (global_pool is None) != (global_slots is None):
        raise ValueError("V4.1 GLOBAL pool and indices must be supplied together")
    pools = [(swa_pool, swa_slots)]
    if global_pool is not None:
        pools.append((global_pool, global_slots))
    for _, slots in pools:
        if (
            slots.ndim != 3
            or slots.shape[:2] != (batch, span)
            or not slots.is_contiguous()
        ):
            raise ValueError("V4.1 decode indices must be contiguous [B,S,TopK]")
    gen = 0 if swa_pool.ndim == 3 and swa_pool.shape[1] == 128 else 1
    swa = _cache_view(swa_pool, "swa_kv", gen)
    global_cache = None
    if global_pool is not None:
        tag = (
            "csa_kv"
            if global_pool.ndim == 3 and global_pool.shape[1] == 128
            else "hca_kv"
        )
        global_cache = _cache_view(global_pool, tag, gen)
    output = _attention(
        q.view(batch, heads, dim),
        sinks,
        swa,
        swa_slots.view(batch, -1),
        valid_rows=torch.ones(batch, device=q.device, dtype=torch.bool),
        global_cache=global_cache,
        global_slots=None if global_slots is None else global_slots.view(batch, -1),
    )
    return output.view_as(q)
