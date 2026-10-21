import logging
from math import gcd
from typing import Sequence

from rtp_llm.ops import HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType


def calculate_hybrid_group_layer_num(linear_count: int, full_count: int) -> int:
    if linear_count > 0 and full_count > 0:
        group_layer_num = gcd(linear_count, full_count)
        if group_layer_num < full_count:
            group_layer_num = full_count
    else:
        group_layer_num = max(linear_count, full_count)
    return max(group_layer_num, 1)


def build_hybrid_kv_cache_spec_descs(
    hybrid_attention_types: Sequence[HybridAttentionType],
    full_cache_type: KVCacheSpecType,
    linear_cache_type: KVCacheSpecType = KVCacheSpecType.LINEAR,
) -> list[list[KVCacheSpecDesc]]:
    linear_count = sum(
        attn_type == HybridAttentionType.LINEAR for attn_type in hybrid_attention_types
    )
    full_count = len(hybrid_attention_types) - linear_count
    group_layer_num = calculate_hybrid_group_layer_num(linear_count, full_count)

    full_desc = KVCacheSpecDesc()
    full_desc.tag = "full"
    full_desc.cache_type = full_cache_type

    linear_descs: dict[int, KVCacheSpecDesc] = {}

    def get_linear_desc(group_idx: int) -> KVCacheSpecDesc:
        if full_count == 0:
            group_idx = 0
        desc = linear_descs.get(group_idx)
        if desc is None:
            desc = KVCacheSpecDesc()
            desc.tag = "linear" if full_count == 0 else f"linear{group_idx}"
            desc.cache_type = linear_cache_type
            linear_descs[group_idx] = desc
        return desc

    layer_descs = []
    linear_seen = 0
    for attn_type in hybrid_attention_types:
        if attn_type == HybridAttentionType.LINEAR:
            linear_group_idx = linear_seen // group_layer_num
            layer_descs.append([get_linear_desc(linear_group_idx)])
            linear_seen += 1
        else:
            layer_descs.append([full_desc])
    return layer_descs


def configure_recurrent_cache_capacity(model_config, max_active_requests: int) -> None:
    """Bound recurrent pools by active requests instead of the paged KV count.

    Each request needs the retained state tail, speculative reserve slots and
    room to replace a tail at a block boundary. Keep another request-sized
    allowance for reusable prefixes. FULL pools retain their token-based budget.
    """
    from rtp_llm.ops import CacheCapacityPolicyDesc

    if max_active_requests <= 0:
        raise ValueError("max_active_requests must be positive")
    proposals = max(0, int(model_config.gen_num_per_cycle))
    block_count = max_active_requests * (proposals + 5)
    descs = model_config.kv_cache_spec_descs
    changed = False
    for layer_descs in descs:
        for desc in layer_descs:
            if desc.cache_type != KVCacheSpecType.LINEAR:
                continue
            capacity = desc.capacity
            if capacity is not None and capacity.explicit_block_num:
                continue
            if capacity is None:
                capacity = CacheCapacityPolicyDesc()
            capacity.explicit_block_num = block_count
            capacity.reservable = False
            capacity.charge_to_paged_budget = True
            desc.capacity = capacity
            changed = True
    if changed:
        # pybind vector/optional properties return copies: assign the complete
        # descriptor tree back after updating its policy objects.
        model_config.kv_cache_spec_descs = descs
        logging.info(
            "Recurrent cache capacity: %d blocks per tag, %d active requests, %d proposals",
            block_count,
            max_active_requests,
            proposals,
        )
