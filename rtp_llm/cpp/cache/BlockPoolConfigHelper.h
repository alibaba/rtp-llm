#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

#include <string>

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
        RTP_LLM_CHECK_WITH_INFO(cache_config.groupNums() > 0, "cache groups must not be empty");
        BlockPoolConfig config;
        config.pool_name      = "default";
        config.block_num      = cache_config.block_num;
        const bool  is_hybrid = cache_config.groupNums() > 1;
        auto        layer_num = is_hybrid ? cache_config.group_layer_num : cache_config.layer_num;
        const auto& main_spec = cache_config.specForGroup(0);
        // linear block size is same with full block block size
        MemoryLayoutConfig main_layout = createMemoryLayoutConfig(is_hybrid,
                                                                  layer_num,
                                                                  cache_config.kv_block_stride_bytes,
                                                                  cache_config.kv_scale_stride_bytes,
                                                                  main_spec,
                                                                  cache_config,
                                                                  cache_config.localKvHeadNumForGroup(0));

        main_layout.kv_cache_offset_bytes = 0;
        main_layout.kv_scale_offset_bytes = main_layout.kv_cache_offset_bytes + main_layout.kv_block_pool_size_bytes;
        size_t current_offset             = main_layout.kv_scale_offset_bytes + main_layout.kv_scale_pool_size_bytes;
        RTP_LLM_LOG_INFO("main_layout.kv_scale_offset_bytes: %zu", main_layout.kv_scale_offset_bytes);
        RTP_LLM_LOG_INFO("main_layout.kv_scale_pool_size_bytes: %zu", main_layout.kv_scale_pool_size_bytes);

        config.memory_layouts.push_back(main_layout);

        // Create MTP sub-model layouts
        for (size_t i = 0; i < cache_config.mtp_sub_configs.size(); ++i) {
            const auto& mtp_sub_config = cache_config.mtp_sub_configs[i];
            RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%zu] is null", i);
            RTP_LLM_CHECK_WITH_INFO(mtp_sub_config->groupNums() > 0,
                                    "MTP module %zu cache groups must not be empty",
                                    i);

            const auto mtp_layer_num = mtp_sub_config->layer_num;

            size_t real_mtp_gid = 0;
            for (size_t gid = 0; gid < static_cast<size_t>(mtp_sub_config->groupNums()); ++gid) {
                if (!mtp_sub_config->layerIdsForGroup(gid).empty()) {
                    real_mtp_gid = gid;
                    break;
                }
            }
            const auto& mtp_spec = mtp_sub_config->specForGroup(real_mtp_gid);
            // mtp block size is not same with main model block size.  MTP
            // sub-configs may keep target-aligned placeholder groups, so use
            // the first group that owns a real MTP layer instead of gid 0.
            MemoryLayoutConfig mtp_layout = createMemoryLayoutConfig(false,
                                                                     mtp_layer_num,
                                                                     mtp_spec->block_size_bytes(),
                                                                     mtp_spec->scale_block_size_bytes(),
                                                                     mtp_spec,
                                                                     cache_config,
                                                                     mtp_sub_config->localKvHeadNumForGroup(real_mtp_gid));

            mtp_layout.kv_cache_offset_bytes = current_offset;
            RTP_LLM_LOG_INFO("mtp_layout.kv_block_pool_size_bytes = %ld", mtp_layout.kv_block_pool_size_bytes);
            current_offset += mtp_layout.kv_block_pool_size_bytes;

            if (mtp_layout.hasScale()) {
                mtp_layout.kv_scale_offset_bytes = current_offset;
                RTP_LLM_LOG_INFO("mtp_layout.kv_scale_pool_size_bytes = %ld", mtp_layout.kv_scale_pool_size_bytes);
                current_offset += mtp_layout.kv_scale_pool_size_bytes;
            } else {
                mtp_layout.kv_scale_offset_bytes = current_offset;
            }

            config.memory_layouts.push_back(mtp_layout);
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
        config.pool_name = "memory_connector";
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
    static MemoryLayoutConfig createMemoryLayoutConfig(bool           enable_hybrid_attention,
                                                       uint32_t       layer_num,
                                                       size_t         kv_block_stride_bytes,
                                                       size_t         kv_scale_stride_bytes,
                                                       KVCacheSpecPtr spec,
                                                       CacheConfig    cache_config,
                                                       uint32_t       local_kv_head_num) {
        MemoryLayoutConfig cfg;
        cfg.layer_num             = layer_num;
        cfg.block_num             = cache_config.block_num;
        cfg.kv_block_stride_bytes = kv_block_stride_bytes;
        cfg.k_block_stride_bytes  = spec->k_block_size_bytes();
        cfg.v_block_stride_bytes  = spec->v_block_size_bytes();
        cfg.kv_scale_stride_bytes = kv_scale_stride_bytes;
        cfg.k_scale_stride_bytes  = spec->k_scale_block_size_bytes();
        cfg.v_scale_stride_bytes  = spec->v_scale_block_size_bytes();

        cfg.enable_kv_scale         = cfg.kv_scale_stride_bytes > 0;
        cfg.dtype                   = spec->memoryLayoutDType();
        cfg.local_head_num_kv       = local_kv_head_num;
        cfg.enable_hybrid_attention = enable_hybrid_attention;
        // Scale 3D layout for MLA and indexer; KV 3D only for MLA (concat_and_cache_mla)
        cfg.is_mla             = cache_config.use_mla || cache_config.is_sparse;
        cfg.use_mla            = cache_config.use_mla;
        cfg.seq_size_per_block = static_cast<size_t>(cache_config.seq_size_per_block);

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;

        cfg.kv_scale_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_scale_stride_bytes;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;
        return cfg;
    }
};

}  // namespace rtp_llm
