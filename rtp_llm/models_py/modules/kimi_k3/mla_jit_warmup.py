"""Prepare K3's Page-RR MLA kernels using private scratch allocations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _MlaWarmupSpec:
    device: torch.device
    cache_dtype: torch.dtype
    cache_stride: tuple[int, ...]
    page_tokens: int
    heads: int
    latent: int
    rope: int
    q_dim: int
    v_dim: int
    softmax_scale: float
    physical_page_tokens: int
    shards: int
    rank: int


def _mla_warmup_specs(model):
    """Extract compile geometry, never retain a tensor from the live KV pool."""
    prepared = set()
    parallelism = model.parallelism_config
    if not (
        parallelism.tp_size > 1 and parallelism.prefill_cp_config.kv_cache_sharded
    ):
        return
    for layer in model.layers:
        attention = layer.attention
        if attention.__class__.__name__ != "KimiK3MLA":
            continue
        for cache in model.kv_cache.get_layer_cache_groups(layer.index):
            page = int(cache.seq_size_per_block)
            width = attention.kv_rank + attention.suffix_dim
            raw = cache.kv_cache_base.view(-1, page, width)
            spec = _MlaWarmupSpec(
                raw.device,
                raw.dtype,
                tuple(raw.stride()),
                page,
                attention.heads,
                attention.kv_rank,
                attention.suffix_dim,
                attention.q_dim,
                attention.v_dim,
                float(model.config.attn_config.softmax_extra_scale)
                / attention.q_dim**0.5,
                int(model.kv_cache.get_seq_size_per_block(cache.tag)),
                int(parallelism.tp_size),
                int(parallelism.tp_rank),
            )
            if spec not in prepared:
                prepared.add(spec)
                yield spec


def _warmup_merge(spec: _MlaWarmupSpec) -> None:
    from rtp_llm.models_py.triton_kernels.mla.lse_merge import (
        merge_attention_states_in_place,
    )

    shape = (2, spec.heads, spec.v_dim)
    partial = torch.zeros(shape, device=spec.device, dtype=torch.bfloat16)
    lse = torch.zeros(shape[:2], device=spec.device, dtype=torch.float32)
    for dtype in (torch.bfloat16, torch.float32):
        canonical = torch.zeros(shape, device=spec.device, dtype=dtype)
        merge_attention_states_in_place(canonical, lse.clone(), partial, lse)


def _warmup_page_rr_prefix(spec: _MlaWarmupSpec) -> None:
    from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
        MlaPageRRCacheGather,
        _restore_mla_page_rr_prefix,
    )

    subpages = spec.physical_page_tokens // spec.page_tokens
    width = spec.latent + spec.rope
    scratch = torch.empty_strided(
        (subpages + 1, spec.page_tokens, width),
        spec.cache_stride,
        device=spec.device,
        dtype=spec.cache_dtype,
    ).zero_()
    table = torch.arange(1, subpages + 1, device=spec.device, dtype=torch.int32)[
        None, :
    ]
    adapter = MlaPageRRCacheGather(
        spec.physical_page_tokens,
        spec.shards,
        spec.rank,
        kernel_page_tokens=spec.page_tokens,
    )
    # This rank owns the sole requested physical page. Stage peers locally;
    # prefix preparation must never introduce a distributed rendezvous.
    descriptor = adapter.build_prefix_chunk_descriptor(
        (0,),
        (spec.rank * spec.physical_page_tokens,),
        (1,),
        feature_width=width,
    )
    payload = adapter._pack_prefix_chunk(scratch, table, descriptor)
    gathered = torch.zeros(
        (spec.shards, *payload.shape),
        device=spec.device,
        dtype=spec.cache_dtype,
    )
    gathered[spec.rank].copy_(payload)
    _restore_mla_page_rr_prefix(gathered, descriptor)


def _warmup_tokenspeed(spec: _MlaWarmupSpec) -> None:
    if spec.cache_dtype != torch.float8_e4m3fn:
        return
    from rtp_llm.models_py.modules.kimi_k3.tokenspeed_prefill import (
        TokenspeedMlaPrefill,
    )

    wrapper = TokenspeedMlaPrefill(spec.heads, spec.softmax_scale)
    q = torch.zeros(
        (2, spec.heads, spec.q_dim), device=spec.device, dtype=spec.cache_dtype
    )
    k = torch.zeros_like(q)
    v = torch.zeros(
        (2, spec.heads, spec.v_dim), device=spec.device, dtype=spec.cache_dtype
    )
    indptr = torch.tensor((0, 2), device=spec.device, dtype=torch.int32)
    for causal in (True, False):
        wrapper.run_partial(
            q,
            k,
            v,
            qo_indptr=indptr,
            kv_indptr=indptr,
            max_q=2,
            max_k=2,
            causal=causal,
        )


@torch.inference_mode()
def _warmup_spec(spec: _MlaWarmupSpec) -> None:
    with torch.cuda.device(spec.device):
        _warmup_page_rr_prefix(spec)
        _warmup_merge(spec)
        _warmup_tokenspeed(spec)
        torch.cuda.synchronize(spec.device)


def warmup_kimi_k3_mla_jit(model, init_resource) -> None:
    if init_resource.is_decode_role or model.kv_cache is None:
        return
    for spec in _mla_warmup_specs(model):
        started = time.perf_counter()
        _warmup_spec(spec)
        logging.info(
            "K3 MLA Page-RR prefix JIT prepared: dtype=%s page=%d heads=%d cost=%.3fs",
            spec.cache_dtype,
            spec.page_tokens,
            spec.heads,
            time.perf_counter() - started,
        )
