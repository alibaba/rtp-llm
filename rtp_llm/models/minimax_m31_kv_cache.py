"""Paged NVFP4 cache descriptors for MiniMax-M3.1 sparse attention."""

from rtp_llm.ops import (
    DataType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)

MAIN_KV_TAG = "m31_kv"
INDEXER_KV_TAG = "m31_indexer"


def build_m31_kv_cache_descs(
    layer_num: int, kv_heads: int, head_dim: int, index_dim: int
):
    if layer_num <= 0 or kv_heads <= 0 or head_dim <= 0 or index_dim <= 0:
        raise ValueError(
            "MiniMax-M3.1 cache dimensions and layer count must be positive"
        )
    if head_dim % 16 or index_dim % 16:
        raise ValueError("MiniMax-M3.1 NVFP4 dimensions must be divisible by 16")

    def opaque_page(tag: str, value_count: int) -> KVCacheSpecDesc:
        desc = KVCacheSpecDesc()
        desc.tag = tag
        desc.cache_type = KVCacheSpecType.OPAQUE_KV
        desc.dtype = DataType.TYPE_UINT8
        desc.entry_dtype = DataType.TYPE_UINT8
        desc.entry_count_mode = OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED
        desc.compression_ratio = 1
        # Two E2M1 values per byte and one E4M3 scale per 16 values.
        desc.entry_elems = value_count // 2 + value_count // 16
        return desc

    return [
        [
            opaque_page(MAIN_KV_TAG, 2 * kv_heads * head_dim),
            opaque_page(INDEXER_KV_TAG, index_dim),
        ]
        for _ in range(layer_num)
    ]
