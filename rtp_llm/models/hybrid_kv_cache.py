from typing import Optional, Sequence

from rtp_llm.ops import HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType


def build_hybrid_kv_cache_spec_descs(
    hybrid_attention_types: Sequence[HybridAttentionType],
    full_cache_type: KVCacheSpecType,
    linear_cache_type: KVCacheSpecType = KVCacheSpecType.LINEAR,
    *,
    kernel_seq_size_per_block: Optional[int],
) -> list[list[KVCacheSpecDesc]]:
    full_desc = KVCacheSpecDesc()
    full_desc.tag = "full"
    full_desc.cache_type = full_cache_type
    full_desc.kernel_seq_size_per_block = kernel_seq_size_per_block

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
