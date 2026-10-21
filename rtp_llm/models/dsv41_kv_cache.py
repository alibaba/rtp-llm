"""DeepSeek V4.1 pools declared through the main cache specification API.

Global KV/indexer storage belongs only to producer layers. Consumers use the
producer layer id and the same cache tag; no duplicate physical pools or native
model-specific allocator are needed.
"""

from rtp_llm.models.dsv4_kv_cache import (
    _make_dsv4_desc,
    _use_host_pinned_memory,
    apply_dsv4_explicit_pool_blocks,
    resolve_dsv4_tokens_per_block,
)
from rtp_llm.ops import (
    CacheCpPolicyDesc,
    CacheReusePolicyDesc,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    DataType,
    HybridAttentionType,
)


def build_v41_kv_cache_spec_descs(
    layer_num,
    ratios,
    source_layers,
    head_dim,
    *,
    bounded_replay=False,
    draft=False,
    fixed_pool_use_host_memory=False,
):
    """Declare byte-exact FP4 global/indexer and FP8 sliding-window pools.

    Indexer blocks keep one entry per raw token for both ratios, matching the
    kernel's payload/scale planes (ratio two fills the first half). CP sizing
    and placement are delegated to the descriptor's generic policies.
    """
    global_pools = {
        ratio: _make_dsv4_desc(
            f"global_kv_{ratio}", "compressed_kv", 288, DataType.TYPE_UINT8, ratio
        )
        for ratio in (1, 2)
    }
    for desc in global_pools.values():
        desc.block_stride_bytes_alignment = 4608  # lcm(512, 288)
    indexer = _make_dsv4_desc("indexer_kv", "compressed_kv", 68, DataType.TYPE_UINT8, 1)
    state = _make_dsv4_desc(
        "csa_state", "fixed_state", 2 * head_dim, DataType.TYPE_FP32
    )
    state.compression_ratio = 2
    state.state_ring_overlap = 0
    swa = _make_dsv4_desc("swa_kv", "sliding_window_kv", 528, DataType.TYPE_UINT8)
    # Native TMA needs whole token rows as well as the allocator alignment.
    # Main combines this with CP divisibility before slicing the block.
    swa.block_stride_bytes_alignment = 16896  # lcm(512, 528)
    decoder_swa = _make_dsv4_desc(
        "swa_kv", "sliding_window_kv", 528, DataType.TYPE_UINT8
    )
    decoder_swa.tag = "decoder_swa_kv"
    decoder_swa.block_stride_bytes_alignment = 16896
    reuse = CacheReusePolicyDesc()
    reuse.enable_prefix_reuse = False
    decoder_swa.reuse = reuse
    if fixed_pool_use_host_memory:
        for desc in (state, swa, decoder_swa):
            _use_host_pinned_memory(desc)
    sources = set(source_layers)
    decoder_start = max(sources) + 1 if sources else 0
    descs = []
    for layer_id in range(layer_num):
        ratio = ratios[layer_id] if layer_id < len(ratios) else 0
        layer = [
            (
                decoder_swa
                if bounded_replay and (draft or layer_id >= decoder_start)
                else swa
            )
        ]
        if layer_id in sources:
            layer.extend((global_pools[ratio], indexer))
            if ratio == 2:
                layer.append(state)
        descs.append(layer)
    return descs


def configure_v41_kv_cache(model_config, kv_cache_config=None):
    from rtp_llm.models_py.modules.dsv41.bounded_replay import enabled

    attn = model_config.attn_config
    promoted = resolve_dsv4_tokens_per_block(int(attn.tokens_per_block))
    if promoted is not None:
        if int(attn.kernel_tokens_per_block) == int(attn.tokens_per_block):
            attn.kernel_tokens_per_block = promoted
        attn.tokens_per_block = promoted
    bounded = enabled()
    draft = bool(model_config.is_mtp)
    model_config.hybrid_attention_config.hybrid_attention_types = [
        HybridAttentionType.NONE
    ] * model_config.num_layers
    descs = build_v41_kv_cache_spec_descs(
        model_config.num_layers,
        list(attn.layer_compress_ratios),
        list(attn.v41_kv_source_layer_ids),
        int(attn.size_per_head),
        bounded_replay=bounded,
        draft=draft,
        fixed_pool_use_host_memory=bool(
            kv_cache_config and kv_cache_config.dsv4_fixed_pool_use_memory
        ),
    )
    fixed_blocks = (
        int(kv_cache_config.dsv4_fixed_pool_blocks or 0) if kv_cache_config else 0
    )
    if fixed_blocks > 0:
        for tag in ("csa_state", "swa_kv", "decoder_swa_kv"):
            apply_dsv4_explicit_pool_blocks(descs, tag, fixed_blocks)
    model_config.kv_cache_spec_descs = descs
    model_config.cache_min_replay_tokens = 128 if bounded else 1
    # Token-interleaved FlashMLA bytes must never alias the old planar format.
    # Full and bounded replay also need independent encoder checkpoint keys.
    model_config.cache_key_hash_seed = (
        0x4453563431494231 if bounded else 0x4453563431494631
    )
