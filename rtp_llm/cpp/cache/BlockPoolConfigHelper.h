#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    static inline bool needKvScale(rtp_llm::DataType dtype) {
        return dtype == rtp_llm::TYPE_INT8 || dtype == rtp_llm::TYPE_FP8_E4M3;
    }

    /**
     * Create block pool config from CacheConfig.
     * Supports both single model and MTP (1+N models) configuration.
     * Memory layout is [layout0_kv][layout0_scale][layout1_kv][layout1_scale]...[layoutN_kv][layoutN_scale]
     * Generally Memory layout is [main_kv][main_scale][mtp1_kv][mtp1_scale]...[mtpN_kv][mtpN_scale]
     *
     * @param cache_config The CacheConfig containing main model and optional MTP modules
     */
    static BlockPoolConfig createConfig(const CacheConfig& cache_config) {
        RTP_LLM_CHECK_WITH_INFO(!cache_config.cache_specs.empty(), "cache_specs must not be empty");
        BlockPoolConfig config;
        config.block_num = cache_config.block_num;
        MemoryLayoutConfig main_layout;
        const bool         is_hybrid = cache_config.groupNums() > 1;
        if (!is_hybrid) {
            const auto& main_spec = cache_config.cache_specs[0];
            main_layout =
                createMemoryLayoutConfig(cache_config.layer_num, cache_config.block_num, main_spec, cache_config);
        } else {
            main_layout = createHybridMemoryLayoutConfig(cache_config.group_layer_num,
                                                         cache_config.block_num,
                                                         cache_config.kv_block_stride_bytes,
                                                         cache_config.kv_scale_stride_bytes,
                                                         cache_config.cache_specs[0]->dtype);
        }
        main_layout.kv_cache_offset_bytes = 0;
        main_layout.kv_scale_offset_bytes = main_layout.kv_cache_offset_bytes + main_layout.kv_block_pool_size_bytes;

        size_t current_offset = main_layout.kv_scale_offset_bytes + main_layout.kv_scale_pool_size_bytes;

        config.memory_layouts.push_back(main_layout);

        for (size_t i = 0; i < cache_config.mtp_sub_configs.size(); ++i) {
            const auto& mtp_sub_config = cache_config.mtp_sub_configs[i];
            RTP_LLM_CHECK_WITH_INFO(mtp_sub_config != nullptr, "mtp_sub_configs[%zu] is null", i);
            RTP_LLM_CHECK_WITH_INFO(
                !mtp_sub_config->cache_specs.empty(), "MTP module %zu cache_specs must not be empty", i);

            const auto& mtp_spec      = mtp_sub_config->cache_specs[0];
            const auto  mtp_layer_num = mtp_sub_config->layer_num;

            MemoryLayoutConfig mtp_layout =
                createMemoryLayoutConfig(mtp_layer_num, cache_config.block_num, mtp_spec, *mtp_sub_config);
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
    static MemoryLayoutConfig createMemoryLayoutConfig(uint32_t                            layer_num,
                                                       uint32_t                            block_num,
                                                       const std::shared_ptr<KVCacheSpec>& spec,
                                                       const CacheConfig& cache_config = CacheConfig{}) {
        MemoryLayoutConfig cfg;

        const size_t kv_block_stride       = spec->block_size();
        const size_t k_block_stride        = spec->k_block_size();
        const size_t v_block_stride        = spec->v_block_size();
        const size_t kv_block_stride_bytes = spec->block_size_bytes();
        const size_t k_block_stride_bytes  = spec->k_block_size_bytes();
        const size_t v_block_stride_bytes  = spec->v_block_size_bytes();
        const size_t local_head_num_kv     = spec->local_head_num_kv;
        const size_t seq_size_per_block    = spec->seq_size_per_block;
        const size_t k_dim                 = spec->k_dim();
        const size_t v_dim                 = spec->v_dim();

        cfg.layer_num = layer_num;
        cfg.block_num = block_num;
        cfg.dtype     = spec->dtype;

        cfg.kv_block_stride       = kv_block_stride;
        cfg.k_block_stride        = k_block_stride;
        cfg.v_block_stride        = v_block_stride;
        cfg.kv_block_stride_bytes = kv_block_stride_bytes;
        cfg.k_block_stride_bytes  = k_block_stride_bytes;
        cfg.v_block_stride_bytes  = v_block_stride_bytes;

        cfg.seq_size_per_block = seq_size_per_block;
        cfg.local_head_num_kv  = local_head_num_kv;
        cfg.is_mla             = spec->type == KVCacheSpecType::MultiHeadLatentAttention;
        cfg.k_dim              = k_dim;
        cfg.v_dim              = v_dim;

        if (!cfg.is_mla) {
            RTP_LLM_CHECK_WITH_INFO(cfg.k_dim == cfg.v_dim, "k_dim and v_dim are not equal");
        }

        // Use pre-computed values from CacheConfig if available
        if (cache_config.kv_block_stride_bytes > 0) {
            cfg.kv_block_stride_bytes = cache_config.kv_block_stride_bytes;
        }
        if (cache_config.kv_scale_stride_bytes > 0) {
            cfg.kv_scale_stride_bytes = cache_config.kv_scale_stride_bytes;
        }

        cfg.kv_block_size_bytes = cfg.kv_block_stride_bytes * static_cast<size_t>(layer_num);
        cfg.k_block_size_bytes  = k_block_stride_bytes * static_cast<size_t>(layer_num);
        cfg.v_block_size_bytes  = v_block_stride_bytes * static_cast<size_t>(layer_num);

        if (needKvScale(spec->dtype)) {
            cfg.enable_kv_scale       = true;
            cfg.k_scale_stride_bytes  = local_head_num_kv * seq_size_per_block * sizeof(float);
            cfg.v_scale_stride_bytes  = cfg.k_scale_stride_bytes;
            cfg.kv_scale_stride_bytes = 2 * cfg.k_scale_stride_bytes;
            cfg.kv_scale_stride       = 2 * local_head_num_kv * seq_size_per_block;

            // Use pre-computed values from CacheConfig if available
            if (cache_config.kv_scale_stride_bytes > 0) {
                cfg.kv_scale_stride_bytes = cache_config.kv_scale_stride_bytes;
                cfg.kv_scale_stride       = cfg.kv_scale_stride_bytes / sizeof(float);  // Assuming float scale
            }

            cfg.kv_scale_pool_size_bytes =
                static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * cfg.kv_scale_stride_bytes;
            cfg.kv_scale_size_bytes = static_cast<size_t>(layer_num) * cfg.kv_scale_stride_bytes;
            cfg.kv_scale_size       = static_cast<size_t>(layer_num) * cfg.kv_scale_stride;
        } else {
            cfg.kv_scale_pool_size_bytes = 0;
            cfg.kv_scale_stride          = 0;
            cfg.kv_scale_stride_bytes    = 0;
            cfg.kv_scale_size            = 0;
            cfg.kv_scale_size_bytes      = 0;
        }

        cfg.block_stride       = cfg.kv_block_stride + cfg.kv_scale_stride;
        cfg.block_stride_bytes = cfg.kv_block_stride_bytes + cfg.kv_scale_stride_bytes;
        cfg.block_size         = cfg.kv_block_size + cfg.kv_scale_size;

        // Use pre-computed values from CacheConfig if available
        if (cache_config.block_size_bytes > 0) {
            cfg.block_size_bytes = cache_config.block_size_bytes;
        } else {
            cfg.block_size_bytes = cfg.kv_block_size_bytes + cfg.kv_scale_size_bytes;
        }

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * cfg.kv_block_stride_bytes;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;

        return cfg;
    }

    // for hybrid attention model with both linear and full attentions
    static MemoryLayoutConfig createHybridMemoryLayoutConfig(uint32_t          layer_num,
                                                             uint32_t          block_num,
                                                             size_t            block_stride_bytes,
                                                             size_t            scale_stride_bytes,
                                                             rtp_llm::DataType dtype) {
        MemoryLayoutConfig cfg;
        cfg.layer_num = layer_num;
        cfg.block_num = block_num;

        // Hybrid KV pool is still byte-addressed internally, but expose to upper layer with spec[0].dtype
        // (typically bf16/fp16/fp8) to reduce Python-side dtype handling.
        cfg.dtype                   = dtype;
        cfg.enable_hybrid_attention = true;

        cfg.kv_block_stride_bytes = block_stride_bytes;
        cfg.kv_scale_stride_bytes = scale_stride_bytes;
        cfg.enable_kv_scale       = (scale_stride_bytes > 0);

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * cfg.kv_block_stride_bytes;
        cfg.kv_scale_pool_size_bytes =
            cfg.enable_kv_scale ?
                static_cast<size_t>(layer_num) * static_cast<size_t>(block_num) * cfg.kv_scale_stride_bytes :
                0;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;
        return cfg;
    }
};

}  // namespace rtp_llm
