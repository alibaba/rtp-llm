#include "rtp_llm/cpp/cache/DSV4CacheConfigHelper.h"

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// Kernel block size in tokens. DSV4 attention/compressor kernels and the
// FlashMLA SWA path are compiled assuming 256-token kernel blocks; physical
// blocks (config.seq_size_per_block) may be larger multiples of this, with the
// FULL paged pools auto-expanding via the framework's bpk machinery.
constexpr uint32_t kDsv4KernelTokensPerBlock = 256;
// BF16 pool: head_dim=512 × 2B = 1024B per KV slot, 128 × 2B = 256B per
// indexer slot. FP8 pool packs the same logical KV into a smaller slot
// (canonical fp8_model1_mla layout: 448B fp8 NoPE + 64B bf16 RoPE + 8B
// UE8M0 scales = 584B; indexer is 128B fp8 + 4B fp32 scale = 132B).
// Selected at runtime from ``attn_config.kv_cache_dtype`` — see
// ``buildDSV4PoolDescs``.
constexpr uint32_t kDsv4KvEntryBytesBf16      = 1024;
constexpr uint32_t kDsv4IndexerEntryBytesBf16 = 256;
constexpr uint32_t kDsv4KvEntryBytesFp8       = 584;
constexpr uint32_t kDsv4IndexerEntryBytesFp8  = 132;
constexpr size_t   kDsv4PoolNum               = 7;

struct DSV4LayerSets {
    std::vector<int> csa_layers;
    std::vector<int> hca_layers;
    std::vector<int> swa_only_layers;
    std::vector<int> all_layers;
};

struct DSV4PoolDesc {
    KVCacheRegionName       region_name;
    const std::vector<int>* layer_ids;
    uint32_t                entry_elems;
    uint32_t                entries_per_block;
    DataType                store_dtype;
    bool                    is_paged;
};

DSV4LayerSets classifyDSV4Layers(const std::vector<int>& compress_ratios) {
    DSV4LayerSets sets;
    // ``compress_ratios`` must describe exactly the layers covered by this
    // cache config. The main DSV4 descriptor strips the trailing MTP tail,
    // while the MTP propose descriptor uses ``[0]`` for its SWA-only draft
    // layer. Do not strip a trailing zero here; that would erase the draft.
    const size_t num_layers = compress_ratios.size();

    for (size_t i = 0; i < num_layers; ++i) {
        const int layer_id = static_cast<int>(i);
        const int ratio    = compress_ratios[i];
        sets.all_layers.push_back(layer_id);
        if (ratio == 4) {
            sets.csa_layers.push_back(layer_id);
        } else if (ratio == 128) {
            sets.hca_layers.push_back(layer_id);
        } else if (ratio == 0) {
            sets.swa_only_layers.push_back(layer_id);
        } else {
            RTP_LLM_LOG_WARNING("Unknown DSV4 compress_ratio %d at layer %zu, treating as HCA", ratio, i);
            sets.hca_layers.push_back(layer_id);
        }
    }

    RTP_LLM_LOG_INFO("DSV4 layer classification: %zu total, %zu CSA, %zu HCA, %zu SWA-only",
                     sets.all_layers.size(),
                     sets.csa_layers.size(),
                     sets.hca_layers.size(),
                     sets.swa_only_layers.size());
    return sets;
}

std::vector<DSV4PoolDesc>
buildDSV4PoolDescs(const DSV4LayerSets& sets, const ModelConfig& model_config, uint32_t physical_tokens_per_block) {
    const auto& attn         = model_config.attn_config;
    const auto  head_dim     = static_cast<uint32_t>(attn.size_per_head);
    const auto  idx_head_dim = static_cast<uint32_t>(attn.indexer_head_dim);

    const uint32_t idx_state_dim = 2 * idx_head_dim;
    const uint32_t csa_state_dim = 2 * head_dim;
    const uint32_t hca_state_dim = head_dim;

    // Pick KV / indexer slot byte size from the model's kv_cache_dtype.
    // FP8 paged pools use 584B / 132B (canonical fp8_model1_mla layout
    // shared with the Python writer in
    // dsv4/fp8/_compressor_vllm_triton.py); BF16 stays at 1024B / 256B.
    const bool     fp8_kv              = (attn.kv_cache_dtype == KvCacheDataType::FP8);
    const uint32_t kv_entry_bytes      = fp8_kv ? kDsv4KvEntryBytesFp8 : kDsv4KvEntryBytesBf16;
    const uint32_t indexer_entry_bytes = fp8_kv ? kDsv4IndexerEntryBytesFp8 : kDsv4IndexerEntryBytesBf16;

    // entries_per_block stays at the kernel-block size (256/compress_ratio for
    // paged compressed entries; 256 for state/SWA per-token slots) for every
    // pool. The framework's bpk machinery uniformly expands each physical
    // block into bpk = N/256 contiguous kernel sub-blocks (FULL paged + SWA
    // fixed alike), so kernels always see 256-token blocks and block_table
    // lengths are consistent across regions for the same token range.
    (void)physical_tokens_per_block;  // used for spec.seq_size_per_blk via makeDSV4Spec
    return {
        {KVCacheRegionName::CSA_KV,
         &sets.csa_layers,
         kv_entry_bytes,
         kDsv4KernelTokensPerBlock / 4,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::HCA_KV,
         &sets.hca_layers,
         kv_entry_bytes,
         kDsv4KernelTokensPerBlock / 128,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::INDEXER_KV,
         &sets.csa_layers,
         indexer_entry_bytes,
         kDsv4KernelTokensPerBlock / 4,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::INDEXER_STATE,
         &sets.csa_layers,
         idx_state_dim * 2,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::CSA_STATE,
         &sets.csa_layers,
         csa_state_dim * 2,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::HCA_STATE,
         &sets.hca_layers,
         hca_state_dim * 2,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::SWA_KV,
         &sets.all_layers,
         kv_entry_bytes,
         kDsv4KernelTokensPerBlock,
         DataType::TYPE_UINT8,
         false},
    };
}

KVCacheSpecPtr makeDSV4Spec(const DSV4PoolDesc& pool, uint32_t physical_tokens_per_block) {
    const auto layer_count = static_cast<uint32_t>(pool.layer_ids->size());
    // All pools use the same physical seq_size so cache_keys stay aligned across
    // groups (HybridKVCacheAllocator::reuseCache iterates a single shared keys
    // array). FULL paged pools split each physical block into bpk kernel blocks
    // via the framework's bpk machinery; SWA/state pools have bpk = 1 and
    // entries_per_block scaled to physical_tokens_per_block.
    if (pool.is_paged) {
        return std::make_shared<DSV4KVSpec>(pool.region_name,
                                            layer_count,
                                            pool.entry_elems,
                                            pool.entries_per_block,
                                            pool.store_dtype,
                                            physical_tokens_per_block);
    }
    return std::make_shared<DSV4StateSpec>(pool.region_name,
                                           layer_count,
                                           pool.entry_elems,
                                           pool.entries_per_block,
                                           pool.store_dtype,
                                           physical_tokens_per_block);
}

}  // namespace

void DSV4CacheConfigHelper::applyConfig(CacheConfig&         config,
                                        const ModelConfig&   model_config,
                                        const KVCacheConfig& kv_cache_config) {
    RTP_LLM_LOG_INFO("Creating DSV4 typed hybrid-pool cache config with %zu compress_ratios",
                     model_config.attn_config.layer_compress_ratios.size());

    // Honor user-supplied --seq_size_per_block when it's a positive multiple of
    // the kernel block size; otherwise fall back to the kernel block size. Paged
    // FULL groups split each physical block into integer-many 256-token kernel
    // sub-blocks via the framework's bpk machinery; non-multiples would break it.
    const auto user_seq_size = kv_cache_config.seq_size_per_block;
    uint32_t   physical_tokens_per_block;
    if (user_seq_size > 0 && user_seq_size % kDsv4KernelTokensPerBlock == 0) {
        physical_tokens_per_block = static_cast<uint32_t>(user_seq_size);
    } else {
        if (user_seq_size > 0) {
            RTP_LLM_LOG_WARNING("DSV4 ignoring seq_size_per_block=%d (not a positive multiple of %u); "
                                "using kernel block size %u as physical block size",
                                user_seq_size,
                                kDsv4KernelTokensPerBlock,
                                kDsv4KernelTokensPerBlock);
        }
        physical_tokens_per_block = kDsv4KernelTokensPerBlock;
    }
    RTP_LLM_LOG_INFO("DSV4 physical block = %u tokens, kernel block = %u tokens (bpk = %u)",
                     physical_tokens_per_block,
                     kDsv4KernelTokensPerBlock,
                     physical_tokens_per_block / kDsv4KernelTokensPerBlock);

    const auto sets  = classifyDSV4Layers(model_config.attn_config.layer_compress_ratios);
    const auto pools = buildDSV4PoolDescs(sets, model_config, physical_tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(pools.size() == kDsv4PoolNum, "DSV4 must produce %zu pools", kDsv4PoolNum);

    config.layer_num                                = static_cast<uint32_t>(sets.all_layers.size());
    config.layer_all_num                            = config.layer_num;
    config.use_mla                                  = false;
    config.is_sparse                                = true;
    config.seq_size_per_block                       = physical_tokens_per_block;
    config.kernel_seq_size_per_block                = kDsv4KernelTokensPerBlock;
    config.use_typed_cache_regions                  = true;
    config.use_opaque_kv_cache_store                = true;
    config.disable_decode_first_malloc_device_reuse = true;

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();
    // All groups share the same physical seq_size — required so the global
    // cache_keys array (initCacheKeys uses config.seq_size_per_block) aligns
    // with every group's match() / insertIntoCache() granularity.
    config.group_seq_size_per_block.assign(pools.size(), physical_tokens_per_block);
    config.cache_specs.reserve(pools.size());
    config.global_layer_ids.reserve(pools.size());
    config.layer_ids.reserve(pools.size());
    config.group_types.reserve(pools.size());
    config.group_region_names.reserve(pools.size());
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        const auto& pool = pools[gid];
        auto        spec = makeDSV4Spec(pool, physical_tokens_per_block);

        config.cache_specs.push_back(spec);
        config.global_layer_ids.push_back(*pool.layer_ids);
        config.layer_ids.push_back(*pool.layer_ids);
        config.group_types.push_back(pool.is_paged ? CacheGroupType::FULL : CacheGroupType::SWA);
        config.group_region_names.push_back(pool.region_name);
    }

    // ---- F02 super-block layout (M02-PR1: default OFF, no behaviour change) ----
    // bps[p] == 1 for every DSV4 pool today. `enabled` only flips when the
    // user explicitly opts in via DSV4_UNIFIED_BLOCKS=1 (tri-state -1=auto
    // resolves to OFF until M01-PR5 flips kDsv4UnifiedDefault). num_super_blocks
    // stays 0 in PR-1; HybridPoolConfigCreator populates it in M02-PR2.
    config.super_block_layout.bps.assign(pools.size(), 1u);
    config.super_block_layout.enabled = (kv_cache_config.dsv4_unified_block_count == 1);
}

}  // namespace rtp_llm
