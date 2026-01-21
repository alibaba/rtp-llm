#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    static inline bool needKvScale(rtp_llm::DataType dtype) {
        return dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3;
    }

    static BlockPoolConfig createLayerFirstConfig(uint32_t layer_num, uint32_t block_num, size_t block_stride_bytes) {
        BlockPoolConfig config;
        config.block_num = block_num;

        MemoryLayoutConfig layout_cfg;
        layout_cfg.layer_num             = layer_num;
        layout_cfg.block_num             = block_num;
        layout_cfg.layout                = LAYER_FIRST;
        layout_cfg.kv_block_stride_bytes = block_stride_bytes;

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

    /**
     * Create a Layer-First layout configuration from CacheConfig.
     * Supports both single model and MTP (1+N models) configuration.
     * Memory layout is [layout0_kv][layout0_scale][layout1_kv][layout1_scale]...[layoutN_kv][layoutN_scale]
     * Generally Memory layout is [main_kv][main_scale][mtp1_kv][mtp1_scale]...[mtpN_kv][mtpN_scale]
     *
     * @param cache_config The CacheConfig containing main model and optional MTP modules
     */
    static BlockPoolConfig createLayerFirstConfig(const CacheConfig& cache_config) {
        RTP_LLM_CHECK_WITH_INFO(!cache_config.cache_specs.empty(), "cache_specs must not be empty");
        BlockPoolConfig config;
        config.block_num = cache_config.block_num;

        // Create main model layout directly from CacheConfig
        MemoryLayoutConfig main_layout    = createMemoryLayoutFromCacheConfig(cache_config, cache_config.block_num);
        main_layout.kv_cache_offset_bytes = 0;
        main_layout.kv_scale_offset_bytes = main_layout.kv_cache_offset_bytes + main_layout.kv_block_pool_size_bytes;

        size_t current_offset = main_layout.kv_scale_offset_bytes + main_layout.kv_scale_pool_size_bytes;

        config.memory_layouts.push_back(main_layout);

        // Create MTP sub-model layouts
        for (size_t i = 0; i < cache_config.mtp_sub_configs.size(); ++i) {
            const auto& mtp_sub_config = cache_config.mtp_sub_configs[i];
            RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%zu] is null", i);
            RTP_LLM_CHECK_WITH_INFO(
                !mtp_sub_config->cache_specs.empty(), "MTP module %zu cache_specs must not be empty", i);

            MemoryLayoutConfig mtp_layout = createMemoryLayoutFromCacheConfig(*mtp_sub_config, cache_config.block_num);
            mtp_layout.kv_cache_offset_bytes = current_offset;
            current_offset += mtp_layout.kv_block_pool_size_bytes;

            if (mtp_layout.hasScale()) {
                mtp_layout.kv_scale_offset_bytes = current_offset;
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

private:
    /**
     * Create MemoryLayoutConfig directly from CacheConfig.
     * This method avoids recalculating values that are already computed in CacheConfig.
     *
     * @param cache_config The CacheConfig with all pre-computed values
     * @param block_num Number of blocks
     * @return MemoryLayoutConfig with values directly from CacheConfig
     */
    static MemoryLayoutConfig createMemoryLayoutFromCacheConfig(const CacheConfig& cache_config, uint32_t block_num) {
        RTP_LLM_CHECK_WITH_INFO(!cache_config.cache_specs.empty(), "cache_specs must not be empty");

        const auto& spec           = cache_config.cache_specs[0];
        const auto  layer_num      = cache_config.layer_num;
        const auto  local_head_num = spec->local_head_num_kv;
        const auto  seq_size       = spec->seq_size_per_block;

        MemoryLayoutConfig cfg;
        cfg.layer_num = layer_num;
        cfg.block_num = block_num;
        cfg.layout    = LAYER_FIRST;
        cfg.dtype     = cache_config.dtype;

        // Directly use values from CacheConfig (already computed in CacheConfigCreator)
        cfg.kv_block_stride       = cache_config.kv_block_stride;
        cfg.kv_block_stride_bytes = cache_config.kv_block_stride_bytes;
        cfg.kv_scale_stride       = cache_config.kv_scale_stride;
        cfg.kv_scale_stride_bytes = cache_config.kv_scale_stride_bytes;
        cfg.block_stride          = cache_config.block_stride;
        cfg.block_stride_bytes    = cache_config.block_stride_bytes;

        // Calculate per-block and pool sizes
        cfg.kv_block_size       = cache_config.kv_block_size;
        cfg.kv_block_size_bytes = cache_config.kv_block_size_bytes;
        cfg.kv_scale_size       = cache_config.kv_scale_size;
        cfg.kv_scale_size_bytes = cache_config.kv_scale_size_bytes;
        cfg.block_size          = cache_config.block_size;
        cfg.block_size_bytes    = cache_config.block_size_bytes;

        cfg.kv_block_pool_size_bytes =
            cfg.kv_block_stride_bytes * static_cast<size_t>(layer_num) * static_cast<size_t>(block_num);

        // For K/V separation (needed by some kernels)
        cfg.k_block_stride       = spec->k_block_size();
        cfg.v_block_stride       = spec->v_block_size();
        cfg.k_block_stride_bytes = spec->k_block_size_bytes();
        cfg.v_block_stride_bytes = spec->v_block_size_bytes();
        cfg.k_block_size         = cfg.k_block_stride * static_cast<size_t>(layer_num);
        cfg.v_block_size         = cfg.v_block_stride * static_cast<size_t>(layer_num);
        cfg.k_block_size_bytes   = cfg.k_block_stride_bytes * static_cast<size_t>(layer_num);
        cfg.v_block_size_bytes   = cfg.v_block_stride_bytes * static_cast<size_t>(layer_num);
        cfg.k_token_size         = spec->k_token_size();
        cfg.v_token_size         = spec->v_token_size();

        cfg.seq_size_per_block = seq_size;
        cfg.local_head_num_kv  = local_head_num;
        cfg.is_mla             = cache_config.use_mla;

        // Handle kv_scale configuration
        bool has_scale = (cfg.kv_scale_stride_bytes > 0) && (cfg.kv_scale_size_bytes > 0);
        if (has_scale) {
            cfg.enable_kv_scale = true;
            cfg.kv_scale_pool_size_bytes =
                cfg.kv_scale_stride_bytes * static_cast<size_t>(layer_num) * static_cast<size_t>(block_num);

            // For INT8/FP8 quantization: k_scale and v_scale are separated
            bool is_quantization_scale = needKvScale(cache_config.dtype);
            if (is_quantization_scale) {
                cfg.k_scale_stride_bytes = local_head_num * seq_size * sizeof(float);
                cfg.v_scale_stride_bytes = cfg.k_scale_stride_bytes;
            } else {
                // For MLA is_sparse: indexer storage, no separate k/v scales
                cfg.k_scale_stride_bytes = 0;
                cfg.v_scale_stride_bytes = 0;
            }
        } else {
            cfg.enable_kv_scale          = false;
            cfg.kv_scale_pool_size_bytes = 0;
            cfg.k_scale_stride_bytes     = 0;
            cfg.v_scale_stride_bytes     = 0;
        }

        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;

        return cfg;
    }
};

}  // namespace rtp_llm
