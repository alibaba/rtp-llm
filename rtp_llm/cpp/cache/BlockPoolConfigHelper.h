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
        RTP_LLM_CHECK_WITH_INFO(!cache_config.cache_specs.empty(), "cache_specs must not be empty");
        BlockPoolConfig config;
        config.pool_name      = "default";
        config.block_num      = cache_config.block_num;
        const bool  is_hybrid = cache_config.groupNums() > 1;
        auto        layer_num = is_hybrid ? cache_config.group_layer_num : cache_config.layer_num;
        const auto& main_spec = cache_config.cache_specs[0];
        // linear block size is same with full block block size
        MemoryLayoutConfig main_layout = createMemoryLayoutConfig(is_hybrid,
                                                                  layer_num,
                                                                  cache_config.kv_block_stride_bytes,
                                                                  cache_config.kv_scale_stride_bytes,
                                                                  main_spec,
                                                                  cache_config);

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
            RTP_LLM_CHECK_WITH_INFO(
                !mtp_sub_config->cache_specs.empty(), "MTP module %zu cache_specs must not be empty", i);

            const auto mtp_layer_num = mtp_sub_config->layer_num;

            const auto& mtp_spec = mtp_sub_config->cache_specs[0];
            // mtp block size is not same with main model block size
            MemoryLayoutConfig mtp_layout = createMemoryLayoutConfig(false,
                                                                     mtp_layer_num,
                                                                     mtp_spec->block_size_bytes(),
                                                                     mtp_spec->scale_block_size_bytes(),
                                                                     mtp_spec,
                                                                     cache_config);

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

    static BlockPoolConfig createConfigForGroup(const CacheConfig& cache_config, size_t group_id) {
        RTP_LLM_CHECK_WITH_INFO(group_id < cache_config.cache_specs.size(),
                                "group_id %zu out of range, cache_specs.size=%zu",
                                group_id,
                                cache_config.cache_specs.size());
        RTP_LLM_CHECK_WITH_INFO(group_id < cache_config.global_layer_ids.size(),
                                "group_id %zu out of range, global_layer_ids.size=%zu",
                                group_id,
                                cache_config.global_layer_ids.size());
        const auto& spec = cache_config.cache_specs[group_id];
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache_specs[%zu] is null", group_id);

        BlockPoolConfig config;
        config.pool_name = "group_" + std::to_string(group_id);
        if (group_id < cache_config.group_region_names.size()) {
            const auto region_name = cache_config.group_region_names[group_id];
            if (region_name != KVCacheRegionName::DEFAULT) {
                config.pool_name = cacheRegionName(region_name);
            }
        }
        const bool has_group_blocks =
            group_id < cache_config.group_block_nums.size() && cache_config.group_block_nums[group_id] > 0;
        config.block_num = has_group_blocks ? cache_config.group_block_nums[group_id] : cache_config.block_num;
        RTP_LLM_LOG_INFO("createConfigForGroup: pool_name=%s gid=%zu block_num=%d (has_group_blocks=%d, "
                         "group_block_nums.size=%zu, global_block_num=%d)",
                         config.pool_name.c_str(),
                         group_id,
                         config.block_num,
                         has_group_blocks,
                         cache_config.group_block_nums.size(),
                         cache_config.block_num);

        const uint32_t layer_num = static_cast<uint32_t>(cache_config.global_layer_ids[group_id].size());
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "group %zu has no layers", group_id);

        const size_t kv_stride    = (group_id < cache_config.group_kv_block_stride_bytes.size()
                                  && cache_config.group_kv_block_stride_bytes[group_id] > 0) ?
                                        cache_config.group_kv_block_stride_bytes[group_id] :
                                        spec->block_size_bytes();
        const size_t scale_stride = (group_id < cache_config.group_kv_scale_stride_bytes.size()) ?
                                        cache_config.group_kv_scale_stride_bytes[group_id] :
                                        spec->scale_block_size_bytes();

        CacheConfig group_cache_config = cache_config;
        group_cache_config.block_num   = config.block_num;
        if (group_id < cache_config.group_seq_size_per_block.size()
            && cache_config.group_seq_size_per_block[group_id] > 0) {
            group_cache_config.seq_size_per_block = cache_config.group_seq_size_per_block[group_id];
        }

        MemoryLayoutConfig layout =
            createMemoryLayoutConfig(false, layer_num, kv_stride, scale_stride, spec, group_cache_config);
        RTP_LLM_CHECK_WITH_INFO(group_id < cache_config.group_types.size(),
                                "missing cache group type for group %zu (group_types.size=%zu)",
                                group_id,
                                cache_config.group_types.size());
        const bool is_full_group          = cache_config.group_types[group_id] == CacheGroupType::FULL;
        layout.kernel_blocks_per_kv_block = is_full_group ? cache_config.kernelBlocksPerKvBlock() : 1;
        layout.kv_cache_offset_bytes      = 0;
        layout.kv_scale_offset_bytes      = layout.kv_cache_offset_bytes + layout.kv_block_pool_size_bytes;

        config.memory_layouts.push_back(layout);
        config.total_size_bytes = layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes;
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
                                                       CacheConfig    cache_config) {
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
        cfg.dtype                   = spec->dtype;
        cfg.local_head_num_kv       = spec->local_head_num_kv;
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
