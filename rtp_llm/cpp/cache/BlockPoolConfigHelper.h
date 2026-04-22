#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    /**
     * Create block pool config from CacheConfig.
     * Supports both single model and MTP (1+N models) configuration.
     * Memory layout is [layout0_kv][layout0_scale][layout1_kv][layout1_scale]...[layoutN_kv][layoutN_scale]
     * Generally Memory layout is [main_kv][main_scale][mtp1_kv][mtp1_scale]...[mtpN_kv][mtpN_scale]
     *
     * @param cache_config The CacheConfig containing main model and optional MTP modules
     */
    static BlockPoolConfig createConfig(const CacheConfig& cache_config) {
        RTP_LLM_CHECK_WITH_INFO(!cache_config.allocator_configs.empty(), "allocator_configs must not be empty");
        const auto& main_alloc = cache_config.getAllocatorConfig(0);
        RTP_LLM_CHECK_WITH_INFO(!main_alloc.cache_specs.empty(), "main allocator cache_specs must not be empty");

        BlockPoolConfig config;
        config.block_num      = main_alloc.block_num;
        const bool  is_hybrid = main_alloc.groupNums() > 1;
        auto        layer_num = is_hybrid ? static_cast<uint32_t>(main_alloc.group_layer_num) : main_alloc.layer_num;
        const auto& main_spec = main_alloc.cache_specs[0];

        // linear block size is same with full block block size
        MemoryLayoutConfig main_layout = createMemoryLayoutConfig(is_hybrid,
                                                                  layer_num,
                                                                  main_alloc.kv_block_stride_bytes,
                                                                  main_alloc.kv_scale_stride_bytes,
                                                                  main_spec,
                                                                  main_alloc.block_num,
                                                                  main_alloc);

        main_layout.kv_cache_offset_bytes = 0;
        main_layout.kv_scale_offset_bytes = main_layout.kv_cache_offset_bytes + main_layout.kv_block_pool_size_bytes;
        size_t current_offset             = main_layout.kv_scale_offset_bytes + main_layout.kv_scale_pool_size_bytes;
        RTP_LLM_LOG_INFO("main_layout.kv_scale_offset_bytes: %zu", main_layout.kv_scale_offset_bytes);
        RTP_LLM_LOG_INFO("main_layout.kv_scale_pool_size_bytes: %zu", main_layout.kv_scale_pool_size_bytes);

        config.memory_layouts.push_back(main_layout);

        // Create sub-model layouts for MTP modules (allocator_configs[1..N])
        for (size_t i = 1; i < cache_config.allocator_configs.size(); ++i) {
            const auto& alloc_cfg = cache_config.allocator_configs[i];
            RTP_LLM_CHECK_WITH_INFO(
                !alloc_cfg.cache_specs.empty(), "allocator_configs[%zu] cache_specs must not be empty", i);

            const auto  sub_layer_num = alloc_cfg.layer_num;
            const auto& sub_spec      = alloc_cfg.cache_specs[0];

            // MTP block size may differ from the main model block size
            MemoryLayoutConfig sub_layout = createMemoryLayoutConfig(false,
                                                                     sub_layer_num,
                                                                     sub_spec->block_size_bytes(),
                                                                     sub_spec->scale_block_size_bytes(),
                                                                     sub_spec,
                                                                     alloc_cfg.block_num,
                                                                     alloc_cfg);

            sub_layout.kv_cache_offset_bytes = current_offset;
            RTP_LLM_LOG_INFO("sub_layout[%zu].kv_block_pool_size_bytes = %ld", i, sub_layout.kv_block_pool_size_bytes);
            current_offset += sub_layout.kv_block_pool_size_bytes;

            if (sub_layout.hasScale()) {
                sub_layout.kv_scale_offset_bytes = current_offset;
                RTP_LLM_LOG_INFO(
                    "sub_layout[%zu].kv_scale_pool_size_bytes = %ld", i, sub_layout.kv_scale_pool_size_bytes);
                current_offset += sub_layout.kv_scale_pool_size_bytes;
            } else {
                sub_layout.kv_scale_offset_bytes = current_offset;
            }

            config.memory_layouts.push_back(sub_layout);
        }

        config.total_size_bytes = current_offset;

        RTP_LLM_LOG_INFO("BlockPoolConfig(memory_layouts=%zu): total_size=%zu bytes",
                         config.memory_layouts.size(),
                         config.total_size_bytes);
        return config;
    }

    // for memory connector
    static BlockPoolConfig
    createConfig(uint32_t layer_num, uint32_t block_num, size_t block_stride_bytes, rtp_llm::DataType dtype) {
        BlockPoolConfig config;
        config.block_num = block_num;

        MemoryLayoutConfig layout_cfg;
        layout_cfg.layer_num = layer_num;
        layout_cfg.block_num = block_num;

        layout_cfg.kv_block_stride_bytes = block_stride_bytes;
        layout_cfg.dtype                 = dtype;

        layout_cfg.kv_cache_offset_bytes = 0;
        layout_cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * block_stride_bytes;
        layout_cfg.kv_scale_offset_bytes    = layout_cfg.kv_cache_offset_bytes + layout_cfg.kv_block_pool_size_bytes;
        layout_cfg.kv_scale_pool_size_bytes = 0;
        layout_cfg.total_size_bytes         = layout_cfg.kv_block_pool_size_bytes;

        config.memory_layouts   = {layout_cfg};
        config.total_size_bytes = layout_cfg.total_size_bytes;
        return config;
    }

private:
    static MemoryLayoutConfig createMemoryLayoutConfig(bool                          enable_hybrid_attention,
                                                       uint32_t                      layer_num,
                                                       size_t                        kv_block_stride_bytes,
                                                       size_t                        kv_scale_stride_bytes,
                                                       KVCacheSpecPtr                spec,
                                                       uint32_t                      block_num,
                                                       const KVCacheAllocatorConfig& alloc_cfg) {
        MemoryLayoutConfig cfg;
        cfg.layer_num             = layer_num;
        cfg.block_num             = block_num;
        cfg.kv_block_stride_bytes = kv_block_stride_bytes;
        cfg.k_block_stride_bytes  = spec->k_block_size_bytes();
        cfg.v_block_stride_bytes  = spec->v_block_size_bytes();
        cfg.kv_scale_stride_bytes = kv_scale_stride_bytes;
        cfg.k_scale_stride_bytes  = spec->k_scale_block_size_bytes();
        cfg.v_scale_stride_bytes  = spec->v_scale_block_size_bytes();

        cfg.enable_kv_scale         = cfg.kv_scale_stride_bytes > 0;
        cfg.dtype                   = alloc_cfg.dtype;
        cfg.local_head_num_kv       = spec->local_head_num_kv;
        cfg.enable_hybrid_attention = enable_hybrid_attention;
        // Scale 3D layout for MLA and indexer; KV 3D only for MLA (concat_and_cache_mla)
        cfg.is_mla             = alloc_cfg.use_mla || alloc_cfg.is_sparse;
        cfg.use_mla            = alloc_cfg.use_mla;
        cfg.seq_size_per_block = static_cast<size_t>(alloc_cfg.seq_size_per_block);

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;

        cfg.kv_scale_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_scale_stride_bytes;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;
        return cfg;
    }
};

}  // namespace rtp_llm
