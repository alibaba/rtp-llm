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
