#include "rtp_llm/cpp/cache/DSV4CacheConfigHelper.h"

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr uint32_t kDsv4TokensPerBlock = 256;
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
    uint32_t                fixed_blocks_per_req;
};

DSV4LayerSets classifyDSV4Layers(const std::vector<int>& compress_ratios) {
    DSV4LayerSets sets;
    size_t        num_layers = compress_ratios.size();
    if (num_layers > 0 && compress_ratios.back() == 0) {
        --num_layers;
    }

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

std::vector<DSV4PoolDesc> buildDSV4PoolDescs(const DSV4LayerSets& sets, const ModelConfig& model_config) {
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

    return {
        {KVCacheRegionName::CSA_KV,
         &sets.csa_layers,
         kv_entry_bytes,
         kDsv4TokensPerBlock / 4,
         DataType::TYPE_UINT8,
         true,
         0},
        {KVCacheRegionName::HCA_KV,
         &sets.hca_layers,
         kv_entry_bytes,
         kDsv4TokensPerBlock / 128,
         DataType::TYPE_UINT8,
         true,
         0},
        {KVCacheRegionName::INDEXER_KV,
         &sets.csa_layers,
         indexer_entry_bytes,
         kDsv4TokensPerBlock / 4,
         DataType::TYPE_UINT8,
         true,
         0},
        {KVCacheRegionName::INDEXER_STATE, &sets.csa_layers, idx_state_dim * 2, 256, DataType::TYPE_FP32, false, 2},
        {KVCacheRegionName::CSA_STATE, &sets.csa_layers, csa_state_dim * 2, 256, DataType::TYPE_FP32, false, 2},
        {KVCacheRegionName::HCA_STATE, &sets.hca_layers, hca_state_dim * 2, 256, DataType::TYPE_FP32, false, 2},
        {KVCacheRegionName::SWA_KV,
         &sets.all_layers,
         kv_entry_bytes,
         kDsv4TokensPerBlock,
         DataType::TYPE_UINT8,
         false,
         2},
    };
}

KVCacheSpecPtr makeDSV4Spec(const DSV4PoolDesc& pool) {
    const auto layer_count = static_cast<uint32_t>(pool.layer_ids->size());
    if (pool.is_paged) {
        return std::make_shared<DSV4KVSpec>(pool.region_name,
                                            layer_count,
                                            pool.entry_elems,
                                            pool.entries_per_block,
                                            pool.store_dtype,
                                            kDsv4TokensPerBlock);
    }
    return std::make_shared<DSV4StateSpec>(pool.region_name,
                                           layer_count,
                                           pool.entry_elems,
                                           pool.entries_per_block,
                                           pool.fixed_blocks_per_req,
                                           pool.store_dtype,
                                           kDsv4TokensPerBlock);
}

}  // namespace

void DSV4CacheConfigHelper::applyConfig(CacheConfig& config, const ModelConfig& model_config) {
    RTP_LLM_LOG_INFO("Creating DSV4 typed hybrid-pool cache config with %zu compress_ratios",
                     model_config.attn_config.layer_compress_ratios.size());

    const auto sets  = classifyDSV4Layers(model_config.attn_config.layer_compress_ratios);
    const auto pools = buildDSV4PoolDescs(sets, model_config);
    RTP_LLM_CHECK_WITH_INFO(pools.size() == kDsv4PoolNum, "DSV4 must produce %zu pools", kDsv4PoolNum);

    config.layer_num                                = static_cast<uint32_t>(sets.all_layers.size());
    config.layer_all_num                            = config.layer_num;
    config.use_mla                                  = false;
    config.is_sparse                                = true;
    config.seq_size_per_block                       = kDsv4TokensPerBlock;
    config.kernel_seq_size_per_block                = kDsv4TokensPerBlock;
    config.use_typed_cache_regions                  = true;
    config.use_opaque_kv_cache_store                = true;
    config.disable_decode_first_malloc_device_reuse = true;

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();
    config.group_fixed_blocks_per_req.assign(pools.size(), 0);
    config.cache_specs.reserve(pools.size());
    config.global_layer_ids.reserve(pools.size());
    config.layer_ids.reserve(pools.size());
    config.group_types.reserve(pools.size());
    config.group_region_names.reserve(pools.size());
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        const auto& pool = pools[gid];
        auto        spec = makeDSV4Spec(pool);

        config.cache_specs.push_back(spec);
        config.global_layer_ids.push_back(*pool.layer_ids);
        config.layer_ids.push_back(*pool.layer_ids);
        config.group_types.push_back(pool.is_paged ? CacheGroupType::FULL : CacheGroupType::SWA);
        config.group_region_names.push_back(pool.region_name);
        config.group_fixed_blocks_per_req[gid] = pool.is_paged ? 0 : pool.fixed_blocks_per_req;
    }
}

}  // namespace rtp_llm
