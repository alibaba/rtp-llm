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
    full_desc = KVCacheSpecDesc()
    full_desc.tag = "full"
    full_desc.cache_type = full_cache_type

    linear_desc = KVCacheSpecDesc()
    linear_desc.tag = "linear"
    linear_desc.cache_type = linear_cache_type

    layer_descs = []
    for attn_type in hybrid_attention_types:
        if attn_type == HybridAttentionType.LINEAR:
            layer_descs.append([linear_desc])
        else:
            layer_descs.append([full_desc])
    return layer_descs
