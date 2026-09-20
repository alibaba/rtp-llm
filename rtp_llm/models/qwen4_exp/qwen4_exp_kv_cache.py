"""Qwen3.8-Flash-Next KV cache spec descriptors.

Most layers need only the homogeneous per-layer region that
``build_hybrid_kv_cache_spec_descs`` produces (LINEAR for gated-delta-net
layers, MHA for full-attention layers). The PLE layer additionally needs a
per-sequence fixed state so decode can reproduce its dilated depthwise short
conv, whose receptive field reaches ``(ple_conv_kernel_size - 1) * ngram_size``
positions into the past. That history depends on past hidden states, which are
not otherwise retained, so it must be cached.

The state rides on the generic ``OPAQUE_STATE`` region (``FixedStateCacheSpec``
in C++), exactly as DeepSeek-V4's indexer/CSA states do in ``dsv4_kv_cache.py``;
no new C++ spec type is needed. Emitting a multi-region per-layer desc list also
follows the DSv4 precedent and requires
``hybrid_attention_config.enable_independent_kv_cache_pools``.

The n-gram hash additionally needs the previous ``ngram_size - 1`` token ids,
which cannot be recovered at decode: ``PyModelInputs.input_ids`` then carries
only the newly generated token, and no attention input carries token history.
They therefore get a second, tiny int64 region of their own.
"""

from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.hybrid_kv_cache import build_hybrid_kv_cache_spec_descs
from rtp_llm.ops import (
    CacheGroupType,
    CacheReusePolicyDesc,
    CacheTailPolicyDesc,
    DataType,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)

PLE_STATE_TAG = "ple_conv_state"
PLE_NGRAM_CTX_TAG = "ple_ngram_ctx"
INDEXER_KV_TAG = "indexer_kv"
INDEXER_STATE_TAG = "indexer_state"


def _full_group_state_desc(desc: KVCacheSpecDesc) -> KVCacheSpecDesc:
    """Materialize a paged state chain for the whole request length.

    These regions hold page checkpoints, not a rolling per-block payload: the
    model resumes from the page holding the last processed token. An
    ``OPAQUE_STATE`` desc defaults to the SWA group, whose allocation keeps only
    the tail page(s) once the request disables prefix reuse, which would drop
    the checkpoint a page-crossing decode needs. FULL groups materialize every
    block the request spans and never release them mid-request.
    """
    desc.group_type = CacheGroupType.FULL
    reuse = CacheReusePolicyDesc()
    reuse.enable_prefix_reuse = True
    desc.reuse = reuse
    tail = CacheTailPolicyDesc()
    tail.active_tail_blocks = 0
    desc.tail = tail
    return desc


def ple_short_conv_state_len(ple_conv_kernel_size: int, ngram_size: int) -> int:
    """Positions of conv-input history the dilated short conv reaches back over."""
    return (ple_conv_kernel_size - 1) * ngram_size


def ple_state_desc(
    hc_hidden_size: int,
    ple_conv_kernel_size: int,
    ngram_size: int,
    dtype: DataType = DataType.TYPE_BF16,
) -> KVCacheSpecDesc:
    """Fixed per-sequence conv-activation buffer for the PLE short conv.

    One entry per history position; each entry is one ``hc_hidden_size``-wide
    row of the (already normalized) conv input.
    """
    state_len = ple_short_conv_state_len(ple_conv_kernel_size, ngram_size)
    if state_len <= 0:
        raise ValueError(
            f"ple short conv state length must be positive, got {state_len} "
            f"(kernel={ple_conv_kernel_size}, ngram_size={ngram_size})"
        )
    desc = KVCacheSpecDesc()
    desc.tag = PLE_STATE_TAG
    desc.cache_type = KVCacheSpecType.OPAQUE_STATE
    desc.is_state_cache = True
    desc.dtype = dtype
    desc.entry_dtype = dtype
    desc.entry_elems = hc_hidden_size
    desc.entry_count_mode = OpaqueBlockEntryCountMode.EXPLICIT
    desc.explicit_entry_count = state_len
    return _full_group_state_desc(desc)


def ple_ngram_ctx_desc(ngram_size: int) -> KVCacheSpecDesc:
    """Previous ``ngram_size - 1`` token ids the n-gram hash needs at decode.

    Decode ``input_ids`` carries only the newly generated token, so the hash
    context must be cached. Stored as int64 (matching the hashing math), one id
    per entry.
    """
    context_len = ngram_size - 1
    if context_len <= 0:
        raise ValueError(f"ngram context length must be positive, got {context_len}")
    desc = KVCacheSpecDesc()
    desc.tag = PLE_NGRAM_CTX_TAG
    desc.cache_type = KVCacheSpecType.OPAQUE_STATE
    desc.is_state_cache = True
    desc.dtype = DataType.TYPE_INT64
    desc.entry_dtype = DataType.TYPE_INT64
    desc.entry_elems = 1
    desc.entry_count_mode = OpaqueBlockEntryCountMode.EXPLICIT
    desc.explicit_entry_count = context_len
    # Page-level context chain, restored together with the conv state above.
    return _full_group_state_desc(desc)


def indexer_kv_desc(indexer_head_dim: int, compress_ratio: int) -> KVCacheSpecDesc:
    """Pooled block keys for the QSA indexer: one entry per complete block.

    Mirrors DSv4's ``indexer_kv`` -- ``OPAQUE_KV`` whose entry count is
    ``kernel_block / compression_ratio``, declared ``UINT8`` so ``entry_elems`` is
    a byte count. Qwen4's correctness-first production ABI is exactly one bf16
    key, with no inline scale (256 B for the released 128-wide indexer head).
    Keeping this descriptor single-format is deliberate: the paged scorer reads
    bf16, so advertising DSv4's fp8-plus-scale layout here would silently corrupt
    every score.

    This holds *pooled* keys, not raw ones: compression is a parameter-free mean
    over ``compress_ratio`` raw keys, so the pool is that many times smaller than
    the token count.
    """
    if indexer_head_dim <= 0:
        raise ValueError(f"indexer_head_dim must be positive, got {indexer_head_dim}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    desc = KVCacheSpecDesc()
    desc.tag = INDEXER_KV_TAG
    desc.cache_type = KVCacheSpecType.OPAQUE_KV
    desc.is_state_cache = False
    desc.dtype = DataType.TYPE_UINT8
    desc.entry_dtype = DataType.TYPE_UINT8
    desc.entry_elems = indexer_head_dim * 2
    desc.entry_count_mode = OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED
    desc.compression_ratio = compress_ratio
    reuse = CacheReusePolicyDesc()
    # Complete compressed-block chain: a page-aligned prefix leaves whole
    # pooled blocks in the pool, which the paged scorer reads back on reuse.
    reuse.enable_prefix_reuse = True
    desc.reuse = reuse
    tail = CacheTailPolicyDesc()
    tail.active_tail_blocks = 0
    desc.tail = tail
    return desc


def indexer_state_desc(indexer_head_dim: int, compress_ratio: int) -> KVCacheSpecDesc:
    """Raw keys of the block still being filled, so its mean can be completed.

    One fp32 slot per block position. Staging raw keys rather than a running sum
    is what keeps the pooled key bit-identical to the reference: upstream takes
    ``mean`` over the whole block in fp32, and an incremental sum would not
    reproduce that summation order.

    Page-aligned prefix reuse is supported: the pooled key chain restores the
    prefix's completed blocks, and this ring is tail-sparse, so a boundary that
    lands on a compressed-block boundary needs no ring content restored.
    """
    desc = KVCacheSpecDesc()
    desc.tag = INDEXER_STATE_TAG
    desc.cache_type = KVCacheSpecType.OPAQUE_STATE
    desc.is_state_cache = True
    desc.dtype = DataType.TYPE_FP32
    desc.entry_dtype = DataType.TYPE_FP32
    # STATE_RING already allocates ``(1 + overlap) * ratio`` entries. Each
    # entry is one raw key; multiplying the width by ``ratio`` would allocate a
    # [ring, ratio, head] payload that no writer or reader can address safely.
    desc.entry_elems = indexer_head_dim
    desc.entry_count_mode = OpaqueBlockEntryCountMode.STATE_RING
    desc.compression_ratio = compress_ratio
    desc.state_ring_overlap = 1
    # Page-level record like its sibling opaque-state regions: every page holds
    # the ring content as of that page's end.
    return _full_group_state_desc(desc)


def build_qwen4_exp_kv_cache_spec_descs(
    config: ModelConfig,
    ple_layer_indices: List[int],
    ple_conv_kernel_size: int,
    ngram_size: int,
    indexer_head_dim: int = 0,
    indexer_compress_ratio: int = 0,
) -> List[List[KVCacheSpecDesc]]:
    """Per-layer regions: the attention region, plus PLE state, plus indexer.

    ``ple_layer_indices`` are 0-based decoder layer indices (``ple_layer_ids``
    minus one), so a value of 1 places the state on the second layer.

    Indexer regions go on **full-attention layers only** -- linear-attention
    layers have no ``self_attn`` and therefore no indexer. Passing
    ``indexer_head_dim=0`` skips them, which is what the dense-fallback
    configuration wants.
    """
    hybrid_types = config.hybrid_attention_config.hybrid_attention_types
    layer_descs = build_hybrid_kv_cache_spec_descs(
        hybrid_types,
        KVCacheSpecType.MHA,
    )
    hc_hidden = config.hidden_size * config.hc_mult
    for layer_idx in ple_layer_indices:
        if not 0 <= layer_idx < len(layer_descs):
            raise ValueError(
                f"ple layer index {layer_idx} out of range for {len(layer_descs)} "
                f"layers"
            )
        layer_descs[layer_idx] = layer_descs[layer_idx] + [
            ple_state_desc(hc_hidden, ple_conv_kernel_size, ngram_size),
            ple_ngram_ctx_desc(ngram_size),
        ]
    if indexer_head_dim > 0:
        if indexer_compress_ratio <= 0:
            raise ValueError(
                f"indexer_compress_ratio must be positive when the indexer is "
                f"enabled, got {indexer_compress_ratio}"
            )
        for layer_idx, layer_type in enumerate(hybrid_types):
            if layer_type == HybridAttentionType.LINEAR:
                continue
            layer_descs[layer_idx] = layer_descs[layer_idx] + [
                indexer_kv_desc(indexer_head_dim, indexer_compress_ratio),
                indexer_state_desc(indexer_head_dim, indexer_compress_ratio),
            ]
    return layer_descs
