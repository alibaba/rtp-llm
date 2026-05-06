#include "rtp_llm/cpp/cache/DSV4ConfigCreator.h"

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

void DSV4ConfigCreator::classifyLayers(const std::vector<int>& compress_ratios, DSV4CacheConfig& dsv4_config) {
    size_t num_layers = compress_ratios.size();
    if (num_layers > 0 && compress_ratios.back() == 0) {
        num_layers--;  // strip MTP tail
    }

    for (size_t i = 0; i < num_layers; i++) {
        int ratio = compress_ratios[i];
        dsv4_config.all_layer_ids.push_back(static_cast<int>(i));
        if (ratio == 4) {
            dsv4_config.csa_layer_ids.push_back(static_cast<int>(i));
        } else if (ratio == 128) {
            dsv4_config.hca_layer_ids.push_back(static_cast<int>(i));
        } else if (ratio == 0) {
            dsv4_config.swa_only_layer_ids.push_back(static_cast<int>(i));
        } else {
            RTP_LLM_LOG_WARNING("Unknown compress_ratio %d at layer %zu, treating as HCA", ratio, i);
            dsv4_config.hca_layer_ids.push_back(static_cast<int>(i));
        }
    }

    RTP_LLM_LOG_INFO("DSV4 layer classification: %zu total, %zu CSA, %zu HCA, %zu SWA-only",
                     num_layers,
                     dsv4_config.csa_layer_ids.size(),
                     dsv4_config.hca_layer_ids.size(),
                     dsv4_config.swa_only_layer_ids.size());
}

void DSV4ConfigCreator::buildPoolSpecs(DSV4CacheConfig& dsv4_config, const ModelConfig& model_config) {
    const auto& attn         = model_config.attn_config;
    uint32_t    head_dim     = attn.size_per_head;
    uint32_t    idx_head_dim = attn.indexer_head_dim;

    uint32_t num_csa = dsv4_config.num_csa_layers();
    uint32_t num_hca = dsv4_config.num_hca_layers();
    uint32_t num_all = dsv4_config.num_all_layers();

    // All groups use TOKENS_PER_BLOCK = 256
    // Pool 0/1/2: variable num_blocks (large), entries_per_block = 256/ratio
    // Pool 3/4/5/6: fixed 2 blocks per request. HCA_STATE stores its
    // 128-row compressor state in one block.
    //
    // KV pools use TYPE_UINT8 as store_dtype because entry_elems (KV_ENTRY_BYTES /
    // INDEXER_ENTRY_BYTES) is already in bytes. getTypeSize(TYPE_UINT8) = 1, so
    // block_size_bytes = entries_per_block * entry_bytes * 1 = correct byte count.

    // Pool 0: CSA KV (ratio=4, 64 entries per block, bf16 head_dim=512 → 1024 bytes/entry)
    dsv4_config.pool_specs[0] = {
        DSV4CacheType::CSA_KV,
        num_csa,
        DSV4CacheConfig::KV_ENTRY_BYTES,
        DSV4CacheConfig::TOKENS_PER_BLOCK / 4,
        DataType::TYPE_UINT8,
        true,
        0,
    };
    // Pool 1: HCA KV (ratio=128, 2 entries per block, bf16 head_dim=512 → 1024 bytes/entry)
    dsv4_config.pool_specs[1] = {
        DSV4CacheType::HCA_KV,
        num_hca,
        DSV4CacheConfig::KV_ENTRY_BYTES,
        DSV4CacheConfig::TOKENS_PER_BLOCK / 128,
        DataType::TYPE_UINT8,
        true,
        0,
    };
    // Pool 2: Indexer KV (ratio=4, 64 entries per block, bf16 index_head_dim=128 → 256 bytes/entry)
    dsv4_config.pool_specs[2] = {
        DSV4CacheType::INDEXER_KV,
        num_csa,
        DSV4CacheConfig::INDEXER_ENTRY_BYTES,
        DSV4CacheConfig::TOKENS_PER_BLOCK / 4,
        DataType::TYPE_UINT8,
        true,
        0,
    };
    // Pool 3: Indexer State — fixed 2 blocks per request. Each logical
    // block stores the full CSA overlap state: previous 4 + current 4 rows.
    uint32_t idx_coff         = 2;
    uint32_t idx_state_dim    = idx_coff * idx_head_dim;
    dsv4_config.pool_specs[3] = {
        DSV4CacheType::INDEXER_STATE,
        num_csa,
        idx_state_dim * 2,
        256,
        DataType::TYPE_FP32,
        false,
        2,
    };
    // Pool 4: CSA State — fixed 2 blocks per request. Each logical block
    // stores the full CSA overlap state: previous 4 + current 4 rows.
    uint32_t csa_coff         = 2;
    uint32_t csa_state_dim    = csa_coff * head_dim;
    dsv4_config.pool_specs[4] = {
        DSV4CacheType::CSA_STATE,
        num_csa,
        csa_state_dim * 2,
        256,
        DataType::TYPE_FP32,
        false,
        2,
    };
    // Pool 5: HCA State — fixed 2 blocks per request
    uint32_t hca_state_dim    = head_dim;
    dsv4_config.pool_specs[5] = {
        DSV4CacheType::HCA_STATE,
        num_hca,
        hca_state_dim * 2,
        256,
        DataType::TYPE_FP32,
        false,
        2,
    };
    // Pool 6: SWA KV — 256 tokens/block (same as other groups), fixed 2 blocks per request
    dsv4_config.pool_specs[6] = {
        DSV4CacheType::SWA_KV,
        num_all,
        DSV4CacheConfig::KV_ENTRY_BYTES,
        DSV4CacheConfig::TOKENS_PER_BLOCK,
        DataType::TYPE_UINT8,
        false,
        2,
    };
}

void DSV4ConfigCreator::populateCacheConfig(CacheConfig&             config,
                                            const DSV4CacheConfig&   dsv4_config,
                                            const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config) {
    auto     dtype      = MemoryEvaluationHelper::getDataTypeForCache(model_config);
    uint32_t num_layers = dsv4_config.num_all_layers();

    config.layer_num                 = num_layers;
    config.layer_all_num             = num_layers;
    config.dtype                     = dtype;
    config.use_mla                   = false;
    config.is_sparse                 = true;
    config.seq_size_per_block        = DSV4CacheConfig::TOKENS_PER_BLOCK;
    config.kernel_seq_size_per_block = DSV4CacheConfig::TOKENS_PER_BLOCK;

    // 7 groups, each with its own DSV4KVCacheSpec
    // Layer assignments per group:
    //   0: CSA KV      -> csa_layer_ids
    //   1: HCA KV      -> hca_layer_ids
    //   2: Indexer KV   -> csa_layer_ids
    //   3: Indexer State -> csa_layer_ids
    //   4: CSA State    -> csa_layer_ids
    //   5: HCA State    -> hca_layer_ids
    //   6: SWA KV       -> all_layer_ids
    const std::vector<int>* group_layers[DSV4_NUM_POOLS] = {
        &dsv4_config.csa_layer_ids,
        &dsv4_config.hca_layer_ids,
        &dsv4_config.csa_layer_ids,
        &dsv4_config.csa_layer_ids,
        &dsv4_config.csa_layer_ids,
        &dsv4_config.hca_layer_ids,
        &dsv4_config.all_layer_ids,
    };

    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.cache_specs.clear();
    config.group_types.clear();

    // Build per-pool spec and record per-pool strides so BlockPool sizing and
    // block_num calculation both use each pool's real dimensions.
    size_t max_block_stride = 0;
    config.group_kv_block_stride_bytes.assign(DSV4_NUM_POOLS, 0);
    config.group_kv_scale_stride_bytes.assign(DSV4_NUM_POOLS, 0);
    config.group_block_size_bytes.assign(DSV4_NUM_POOLS, 0);
    config.group_seq_size_per_block.assign(DSV4_NUM_POOLS, DSV4CacheConfig::TOKENS_PER_BLOCK);

    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        KVCacheSpecPtr spec;
        const auto&    pool = dsv4_config.pool_specs[i];
        if (pool.is_paged) {
            spec = std::make_shared<DSV4KVSpec>(pool, static_cast<uint32_t>(DSV4CacheConfig::TOKENS_PER_BLOCK));
        } else {
            spec = std::make_shared<DSV4StateSpec>(pool, static_cast<uint32_t>(DSV4CacheConfig::TOKENS_PER_BLOCK));
        }
        config.cache_specs.push_back(spec);
        config.global_layer_ids.push_back(*group_layers[i]);
        config.layer_ids.push_back(*group_layers[i]);
        // Paged pools (0/1/2): FULL. Fixed/tail pools (3/4/5/6): SWA, keeping the last two blocks.
        config.group_types.push_back(pool.is_paged ? CacheGroupType::FULL : CacheGroupType::SWA);

        // Use the TMA-padded per-block stride for FP8 MLA KV pools so that
        // FlashMLA's MODEL1 SM100 kernel sees `k_cache.stride(0) % 576 == 0`.
        // BF16 KV / state / INDEXER pools fall through to natural size.
        const size_t per_layer_stride         = pool.padded_block_size_bytes();
        config.group_kv_block_stride_bytes[i] = per_layer_stride;
        config.group_kv_scale_stride_bytes[i] = spec->scale_block_size_bytes();
        // Aggregate bytes per request-block across all layers in this pool.
        config.group_block_size_bytes[i] = static_cast<size_t>(pool.layer_num) * per_layer_stride;

        max_block_stride = std::max(max_block_stride, per_layer_stride);
    }

    // Map each group to its KVCacheRegionName so kv_cache_base_by_layer_region
    // is populated per-pool (CSA_KV=1, HCA_KV=2, ..., SWA_KV=7).
    static const KVCacheRegionName pool_region_names[DSV4_NUM_POOLS] = {
        KVCacheRegionName::CSA_KV,
        KVCacheRegionName::HCA_KV,
        KVCacheRegionName::INDEXER_KV,
        KVCacheRegionName::INDEXER_STATE,
        KVCacheRegionName::CSA_STATE,
        KVCacheRegionName::HCA_STATE,
        KVCacheRegionName::SWA_KV,
    };
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        config.group_region_names.push_back(pool_region_names[i]);
    }

    // group_layer_num must be the max layer count across all groups,
    // because BlockPool creates group_layer_num layer tensors shared by all groups.
    uint32_t max_group_layers = 0;
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        max_group_layers = std::max(max_group_layers, dsv4_config.pool_specs[i].layer_num);
    }
    config.group_layer_num             = static_cast<int>(max_group_layers);
    config.full_group_num              = 3;  // Pool 0/1/2 (CSA_KV, HCA_KV, INDEXER_KV)
    config.linear_group_num            = 0;
    config.swa_group_num               = 4;  // Pool 3/4/5/6 (INDEXER_STATE, CSA_STATE, HCA_STATE, SWA_KV)
    config.linear_fixed_cap            = 0;
    config.use_independent_block_pools = true;  // DSV4: each group gets its own BlockPool

    // Per-group block counts: fixed pools (3/4/5/6) only need
    // fixed_blocks_per_req * max_batch blocks.
    // Paged pools (0/1/2) use the global block_num (0 = fallback).
    // Use a conservative max_batch estimate; the actual concurrency limit
    // is set later by the scheduler, but we need an upper bound for allocation.
    const uint32_t max_batch = 128;  // DSV4 fixed pools  // conservative upper bound
    config.group_block_nums.resize(DSV4_NUM_POOLS, 0);
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        const auto& pool = dsv4_config.pool_specs[i];
        if (pool.is_paged) {
            config.group_block_nums[i] = 0;  // 0 = use global block_num
        } else {
            config.group_block_nums[i] = pool.fixed_blocks_per_req * max_batch;
        }
    }

    // Global block_size_bytes feeds CacheConfigCreator::createBasicConfig's
    //   block_num = kv_cache_mem_size / block_size_bytes
    // In DSV4 that divisor must match what ONE paged request-block actually
    // costs across the FULL pools (0/1/2), NOT max_stride × all_layers.
    // The fixed pools (3/4/5/6) have their own group_block_nums set above and
    // contribute a fixed memory reservation regardless of block_num.
    size_t paged_bytes_per_block = 0;
    size_t fixed_pool_reserve    = 0;
    for (int i = 0; i < DSV4_NUM_POOLS; i++) {
        const size_t pool_block_bytes = config.group_block_size_bytes[i];
        if (dsv4_config.pool_specs[i].is_paged) {
            paged_bytes_per_block += pool_block_bytes;
        } else {
            fixed_pool_reserve += static_cast<size_t>(config.group_block_nums[i]) * pool_block_bytes;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(paged_bytes_per_block > 0, "DSV4 paged pools produced zero block bytes");
    config.kv_block_stride_bytes = max_block_stride;  // kept for legacy consumers only
    config.kv_block_size_bytes   = paged_bytes_per_block;
    config.kv_scale_stride_bytes = 0;
    config.kv_scale_size_bytes   = 0;
    config.block_size_bytes      = paged_bytes_per_block;
    // Pre-reservation for fixed pools; CacheConfigCreator deducts this from
    // the budget before dividing by block_size_bytes.
    config.fixed_pool_reserve_bytes = fixed_pool_reserve;

    // Per-layer group mapping
    config.layer_to_group_id.assign(num_layers, 6);  // default to SWA group
    config.layer_group_types.assign(num_layers, CacheGroupType::SWA);
    config.layer_to_block_stride_bytes.assign(num_layers, static_cast<int>(max_block_stride));

    // Per-layer region-name-to-group mapping for multi-pool access
    const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    config.layer_region_to_group_id.resize(num_layers);
    config.layer_to_group_ids.resize(num_layers);
    for (size_t i = 0; i < num_layers; i++) {
        config.layer_region_to_group_id[i].assign(region_name_count, -1);
    }
    // Fill from group_layers: for each group, mark its layers
    for (int gid = 0; gid < DSV4_NUM_POOLS; gid++) {
        auto region_name = static_cast<size_t>(pool_region_names[gid]);
        for (int layer_id : *group_layers[gid]) {
            config.layer_region_to_group_id[static_cast<size_t>(layer_id)][region_name] = gid;
            config.layer_to_group_ids[static_cast<size_t>(layer_id)].push_back(gid);
        }
    }
}

DSV4CacheConfig DSV4ConfigCreator::buildDSV4Config(const ModelConfig& model_config) {
    DSV4CacheConfig dsv4_config;
    classifyLayers(model_config.attn_config.layer_compress_ratios, dsv4_config);
    buildPoolSpecs(dsv4_config, model_config);
    return dsv4_config;
}

CacheConfig DSV4ConfigCreator::createConfig(const ModelConfig&       model_config,
                                            const ParallelismConfig& parallelism_config,
                                            bool                     is_mtp) {
    RTP_LLM_LOG_INFO("Creating DSV4 cache config with %zu compress_ratios",
                     model_config.attn_config.layer_compress_ratios.size());

    DSV4CacheConfig dsv4_config = buildDSV4Config(model_config);

    CacheConfig config;
    populateCacheConfig(config, dsv4_config, model_config, parallelism_config);
    config.dsv4_config = dsv4_config;

    return config;
}

}  // namespace rtp_llm
