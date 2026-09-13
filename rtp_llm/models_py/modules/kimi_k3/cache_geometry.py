"""Validate K3 cache binding and the supported PageRR execution layout."""

from typing import Any, Sequence

import torch

from rtp_llm.ops import KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import CacheGroupType


def bind_kimi_k3_cache_geometry(
    kv_cache: Any,
    layers: Sequence[Any],
    parallelism: ParallelismConfig,
    *,
    is_decode_role: bool,
) -> tuple[int, int]:
    """Check live layer/group mappings and return the fixed physical/checkpoint spans."""
    page_tokens = int(kv_cache.seq_size_per_block)
    local_shards = int(parallelism.tp_size) if parallelism.kv_page_rr_enabled() else 1
    upstream_shards = local_shards
    if is_decode_role and int(parallelism.prefill_cp_config.prefill_cp_size) > 1:
        upstream_shards = int(parallelism.prefill_cp_config.prefill_cp_size)
    if not 0 <= int(parallelism.tp_rank) < int(parallelism.tp_size):
        raise ValueError("Kimi K3 requires valid physical TP coordinates")
    if int(kv_cache.local_shard_count) != local_shards:
        raise ValueError("Kimi K3 manager/model local shard counts disagree")
    checkpoint_tokens = page_tokens * upstream_shards
    if page_tokens <= 0 or checkpoint_tokens > 2**32 - 1:
        raise ValueError("Kimi K3 has invalid or overflowing page/checkpoint spans")
    spans, kinds = kv_cache.group_seq_size_per_block, kv_cache.layer_group_types
    if len(kinds) != len(layers):
        raise ValueError("Kimi K3 cache requires one attention kind per model layer")
    for layer_idx, layer in enumerate(layers):
        group_id = int(kv_cache.get_layer_cache(layer_idx).group_id)
        if not 0 <= group_id < len(spans):
            raise ValueError(
                f"Kimi K3 cache group is out of bounds for layer {layer_idx}"
            )
        expected_kind = CacheGroupType.LINEAR if layer.is_kda else CacheGroupType.FULL
        expected_span = checkpoint_tokens if layer.is_kda else page_tokens
        if kinds[layer_idx] != expected_kind or int(spans[group_id]) != expected_span:
            raise ValueError(
                f"Kimi K3 cache kind/span disagrees with model layer {layer_idx}"
            )
    return page_tokens, checkpoint_tokens


def validate_kimi_k3_page_rr_target(
    *,
    parallelism: ParallelismConfig,
    model_config: Any,
    kv_cache: Any,
    page_tokens: int,
    checkpoint_tokens: int,
    is_decode_role: bool,
    kda_head_dim: int,
    whole_model_query_budget_tokens: int,
    compute_capability: tuple[int, int],
) -> None:
    """Check layout constraints; the attention factory selects the actual backend."""
    tp_size = int(parallelism.tp_size)
    local_rr = bool(parallelism.kv_page_rr_enabled())
    if (
        tp_size != int(parallelism.ep_size)
        or parallelism.prefill_cp_config.is_enabled()
    ):
        raise ValueError("Kimi K3 PageRR requires TP == EP and Query CP disabled")
    if local_rr == is_decode_role or checkpoint_tokens != page_tokens * tp_size:
        raise ValueError("Kimi K3 PageRR role/checkpoint placement mismatch")
    if int(kv_cache.linear_step) != 1:
        raise ValueError("Kimi K3 compact LINEAR cache requires linear_step=1")
    pages = (
        ((128, 128), (256, 256), (256, 128))
        if is_decode_role
        else ((128, 128), (256, 256))
    )
    if (
        tp_size not in (2, 4, 8)
        or (page_tokens, int(kv_cache.kernel_seq_size_per_block)) not in pages
    ):
        raise ValueError("Unsupported Kimi K3 PageRR page/shard layout")
    attention = model_config.attn_config
    if (
        attention.kv_cache_dtype != KvCacheDataType.BASE
        or model_config.compute_dtype is not torch.bfloat16
    ):
        raise ValueError("Kimi K3 PageRR requires BF16 compute and BASE cache")
    if (attention.kv_lora_rank, attention.rope_head_dim) != (512, 64):
        raise ValueError("Kimi K3 MLA cache requires 512+64 features")
    if not is_decode_role and (
        attention.nope_head_dim + attention.rope_head_dim,
        attention.v_head_dim,
    ) != (192, 128):
        raise ValueError("Unsupported Kimi K3 dense Prefill head dimensions")
    if kda_head_dim != 128 or compute_capability not in ((10, 0), (10, 3)):
        raise ValueError("Kimi K3 cuLA requires head dimension 128 and SM100/SM103")
    budget = whole_model_query_budget_tokens
    if budget > 0 and (budget < checkpoint_tokens or budget % tp_size):
        raise ValueError("Kimi K3 query budget cannot advance an aligned checkpoint")
    if int(attention.mla_prefill_expanded_kv_budget_bytes) < 0:
        raise ValueError("Kimi K3 expanded-KV byte budget must be non-negative")
