"""Compile K3.1 DSA kernels with bounded, private scratch allocations."""

import logging
import time

import torch

from rtp_llm.models_py.kernels.cuda.fp8_indexer_cache import pack_indexer_pages64
from rtp_llm.models_py.modules.dsv41.dsv4_kernel_jit_warmup import (
    _run_deepgemm_warmup_launches_serialized,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
    MlaPageRRCacheGather,
    _restore_mla_page_rr_prefix,
)
from rtp_llm.models_py.modules.kimi_k3.mla_jit_warmup import _warmup_decode_metadata
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    grouped_q_kv_rmsnorm,
    independent_q_kv_rmsnorm,
)
from rtp_llm.models_py.triton_kernels.common.validate_zigzag import validate_zigzag


def _warmup_grouped_norm(attention, prepared):
    if getattr(attention, "_qkv_norm", None) is not grouped_q_kv_rmsnorm:
        return
    if (attention.q_rank, attention.kv_rank) != (1536, 512):
        return
    q_weight, kv_weight = attention.q_norm.weight, attention.kv_norm.weight
    dimensions = (attention.q_rank, attention.kv_rank, attention.suffix_dim)
    width = attention.input.weight.shape[0]
    eps = attention.q_norm.variance_epsilon
    natural = getattr(attention, "natural_q_gate_input", None)
    key = (
        q_weight.device,
        q_weight.dtype,
        dimensions,
        width,
        attention.input.weight.dtype,
        eps,
        natural is not None,
    )
    if key in prepared:
        return
    # Preserve the real fused projection's row stride and split offsets.
    source = torch.zeros((128, width), device=q_weight.device, dtype=q_weight.dtype)
    layouts = [source[:, : sum(dimensions)].split(dimensions, dim=-1)]
    if natural is not None:
        restore = torch.arange(16, device=q_weight.device, dtype=torch.int64)
        padding = torch.ones_like(restore, dtype=torch.bool)
        flags = torch.empty((1,), device=q_weight.device, dtype=torch.bool)
        validate_zigzag(restore, padding, restore, padding, flags)
        # Natural Q+gate and KV are separate producers, including FP8 output
        # padding. Use their actual views instead of the fused input's stride.
        hidden_size = getattr(
            attention.input, "hidden_size", attention.input.weight.shape[1]
        )
        hidden = torch.zeros(
            (128, hidden_size), device=q_weight.device, dtype=q_weight.dtype
        )
        q, _ = natural(hidden)
        kv, suffix = attention.natural_kv_input(hidden).split(dimensions[1:], dim=-1)
        layouts.append((q, kv, suffix))
    for index, (q, kv, suffix) in enumerate(layouts):
        # Preserve the short-input and grouped reduction policies.
        for rows in (1, 128):
            for packed in (False, True):
                grouped_q_kv_rmsnorm(
                    q[:rows],
                    kv[:rows],
                    q_weight,
                    kv_weight,
                    eps,
                    kv_suffix=suffix[:rows] if packed else None,
                )
            if index:
                # Multi-chunk natural routing normalizes Q and KV independently.
                for value, weight, tail in (
                    (q, q_weight, None),
                    (kv, kv_weight, suffix),
                ):
                    independent_q_kv_rmsnorm(
                        value[:rows],
                        weight,
                        eps,
                        q_width=attention.q_rank,
                        kv_width=attention.kv_rank,
                        full_rows=rows,
                        suffix=None if tail is None else tail[:rows],
                    )
    torch.cuda.synchronize(q_weight.device)
    prepared.add(key)


def _warmup_prefix(cache, physical_page, shards, rank, *, indexer_fp8):
    page, features = cache.shape[1:]
    subpages = physical_page // page
    scratch = torch.zeros(
        (subpages + 1, page, features), device=cache.device, dtype=cache.dtype
    )
    table = torch.arange(1, subpages + 1, device=cache.device, dtype=torch.int32)[None]
    adapter = MlaPageRRCacheGather(physical_page, shards, rank, page)
    descriptor = adapter.build_prefix_chunk_descriptor(
        (0,), (rank * physical_page,), (1,), feature_width=features
    )
    for allow_zero in (False, True):
        payload = adapter._pack_prefix_chunk(
            scratch,
            table,
            descriptor,
            indexer_fp8=indexer_fp8,
            allow_reserved_block_zero=allow_zero,
        )
        # Stage peers locally: warmup must not introduce an AllGather or read
        # any live KV payload. This descriptor selects only this rank's page.
        gathered = torch.zeros(
            (shards, *payload.shape), device=cache.device, dtype=cache.dtype
        )
        gathered[rank].copy_(payload)
        _restore_mla_page_rr_prefix(gathered, descriptor, indexer_fp8=indexer_fp8)


def _warmup_native64(page, device):
    cache = torch.zeros((3, page, 132), device=device, dtype=torch.uint8)
    table = torch.tensor([[1, 2]], device=device, dtype=torch.int32)
    output = torch.empty((3, 64, 1, 132), device=device, dtype=torch.uint8)
    pack_indexer_pages64(cache, table, 0, output)


def _warmup_native_score(score, device, max_queries, physical_page=64):
    # SM10 native MQA specializes on heads/dim/mode, not M, N or batch.
    # Paged MQA and its metadata additionally specialize on query width.
    query = torch.zeros((4, 32, 128), device=device, dtype=torch.float8_e4m3fn)
    keys = torch.zeros((256, 128), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.ones(256, device=device, dtype=torch.float32)
    weights = torch.ones((4, 32), device=device, dtype=torch.float32)
    starts = torch.zeros(4, device=device, dtype=torch.int32)
    ends = torch.full((4,), 256, device=device, dtype=torch.int32)
    positions = torch.arange(256, device=device, dtype=torch.int32)
    score.dense(query, keys, scales, weights, starts, ends, positions)

    page = 128 if physical_page == 128 and score.supports_direct_pages(128, device) else 64
    pages = 256 // page
    cache = torch.zeros((1 + pages, page, 1, 132), device=device, dtype=torch.uint8)
    cache.view(1 + pages, -1)[1:, page * 128 :].view(torch.float32).fill_(1)
    table = torch.arange(1, 1 + pages, device=device, dtype=torch.int32)[None]
    for queries in range(1, max_queries + 1):
        context = torch.full((1, queries), 256, device=device, dtype=torch.int32)
        metadata = score.prepare_paged_metadata(
            context,
            num_sms=torch.cuda.get_device_properties(device).multi_processor_count, page_size=page,
        )
        query = torch.zeros(
            (1, queries, 32, 128), device=device, dtype=torch.float8_e4m3fn
        )
        weights = torch.ones((queries, 32), device=device, dtype=torch.float32)
        score.paged(query, cache, weights, context, table, metadata, 256, positions)


@torch.inference_mode()
def warmup_kimi_k31_dsa_jit(model, init_resource):
    if model.kv_cache is None:
        return
    started = time.perf_counter()
    prepared = set()
    prepared_norms = set()
    for layer in model.layers:
        if not init_resource.is_decode_role:
            _warmup_grouped_norm(layer.attention, prepared_norms)
        indexer = getattr(layer.attention, "indexer", None)
        if indexer is None:
            continue
        op = indexer.indexer_op
        for cache in model.kv_cache.get_layer_cache_groups(layer.index):
            page = int(cache.seq_size_per_block)
            indexer_fp8 = False
            if cache.tag == "indexer_kv":
                # Opaque storage is UINT8 for both precisions. Use the same
                # codec as inference, including BF16 draft reinterpretation.
                view = op.cache_view(cache, page)
                indexer_fp8 = view.dtype == torch.uint8
            elif cache.tag in ("full", "default"):
                raw = cache.kv_cache_base
                if raw.dtype in (torch.float8_e4m3fn, torch.uint8):
                    # SFA's packed 656-byte codec is not the 132-byte Indexer
                    # codec. Prefix transport must copy its bytes unchanged.
                    view = raw.view(torch.uint8).view(-1, page, 656)
                else:
                    features = layer.attention.kv_rank + layer.attention.suffix_dim
                    view = raw.view(-1, page, features)
            else:
                continue
            features = view.shape[-1]
            physical_page = int(model.kv_cache.get_seq_size_per_block(cache.tag))
            geometry = (
                view.device,
                view.dtype,
                page,
                features,
                physical_page,
                indexer_fp8,
            )
            if geometry in prepared:
                continue
            parallelism = model.parallelism_config
            queries = max(1, int(model.config.gen_num_per_cycle) + 1)
            _warmup_decode_metadata(
                page,
                physical_page,
                parallelism.tp_size,
                parallelism.tp_rank,
                view.device,
                queries,
            )
            if not init_resource.is_decode_role:
                _warmup_prefix(
                    view,
                    physical_page,
                    parallelism.tp_size,
                    parallelism.tp_rank,
                    indexer_fp8=indexer_fp8,
                )
            if indexer_fp8:
                _warmup_native64(page, view.device)
                _run_deepgemm_warmup_launches_serialized(
                    "K3.1 DSA Indexer",
                    lambda: _warmup_native_score(
                        op.native_score, view.device, queries, physical_page=page
                    ),
                )
            torch.cuda.synchronize(view.device)
            prepared.add(geometry)
    if prepared:
        logging.info(
            "K3.1 DSA Indexer JIT prepared: geometries=%d cost=%.3fs",
            len(prepared),
            time.perf_counter() - started,
        )
