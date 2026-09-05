#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>

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
        RTP_LLM_CHECK_WITH_INFO(cache_config.groupNums() == 1,
                                "group-agnostic block pool config requires exactly one cache group, got %d",
                                cache_config.groupNums());
        return createConfigForGroup(cache_config, cache_config.groups().front().tag);
    }

    static BlockPoolConfig createConfigForGroup(const CacheConfig& cache_config, std::string_view tag) {
        const auto& group = cache_config.group(tag);
        const auto& spec  = group.spec;
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec for group tag=%s is null", group.tag.c_str());

        BlockPoolConfig config;
        config.pool_name            = group.tag;
        config.block_num            = group.block_num;
        const bool has_group_blocks = config.block_num != cache_config.block_num;
        RTP_LLM_LOG_INFO("createConfigForGroup: pool_name=%s tag=%s block_num=%d (has_group_blocks=%d, "
                         "groupNums=%d, global_block_num=%d)",
                         config.pool_name.c_str(),
                         group.tag.c_str(),
                         config.block_num,
                         has_group_blocks,
                         cache_config.groupNums(),
                         cache_config.block_num);

        size_t     total_layout_layers = 0;
        size_t     current_offset      = 0;
        const auto append_layout =
            [&](const CacheConfig& source_config, const CacheGroup& source_group, uint32_t layer_num) {
                RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "group tag=%s layout has no layers", group.tag.c_str());
                const auto& layout_spec = source_group.spec;
                RTP_LLM_CHECK_WITH_INFO(
                    layout_spec != nullptr, "cache spec for group tag=%s is null", source_group.tag.c_str());
                auto layout                  = createMemoryLayoutConfig(source_config.enable_hybrid_attention,
                                                       layer_num,
                                                       source_group.kv_block_stride_bytes,
                                                       source_group.kv_scale_stride_bytes,
                                                       layout_spec,
                                                       cache_config,
                                                       config.block_num,
                                                       source_group.local_kv_head_num,
                                                       source_group.seqSizePerBlock(),
                                                       kernelBlocksPerKvBlock(source_group));
                layout.kv_cache_offset_bytes = current_offset;
                current_offset += layout.kv_block_pool_size_bytes;
                layout.kv_scale_offset_bytes = current_offset;
                current_offset += layout.kv_scale_pool_size_bytes;
                total_layout_layers += layer_num;
                config.memory_layouts.push_back(std::move(layout));
            };

        const auto& group_layer_ids = cache_config.groupLayerIds(group.tag);
        const auto  main_layer_num  = static_cast<uint32_t>(
            std::count_if(group_layer_ids.begin(), group_layer_ids.end(), [&cache_config](int layer_id) {
                return layer_id >= 0 && static_cast<uint32_t>(layer_id) < cache_config.layer_num;
            }));
        if (main_layer_num > 0) {
            append_layout(cache_config, group, main_layer_num);
        }

        for (size_t module_index = 0; module_index < cache_config.mtp_sub_configs.size(); ++module_index) {
            const auto& mtp_config = cache_config.mtp_sub_configs[module_index];
            RTP_LLM_CHECK_WITH_INFO(mtp_config != nullptr, "mtp_sub_configs[%zu] is null", module_index);
            const auto& mtp_group     = mtp_config->group(tag);
            const auto  mtp_layer_num = static_cast<uint32_t>(mtp_config->groupLayerIds(tag).size());
            if (mtp_layer_num > 0) {
                append_layout(*mtp_config, mtp_group, mtp_layer_num);
            }
        }

        RTP_LLM_CHECK_WITH_INFO(total_layout_layers == group_layer_ids.size(),
                                "group tag=%s layout layer count=%zu does not match topology layers=%zu",
                                group.tag.c_str(),
                                total_layout_layers,
                                group_layer_ids.size());
        RTP_LLM_CHECK_WITH_INFO(!config.memory_layouts.empty(), "group tag=%s has no layers", group.tag.c_str());
        config.total_size_bytes = current_offset;
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
    static size_t kernelBlocksPerKvBlock(const CacheGroup& group) {
        return group.kernelBlocksPerKvBlock();
    }

    static MemoryLayoutConfig createMemoryLayoutConfig(bool                               enable_hybrid_attention,
                                                       uint32_t                           layer_num,
                                                       size_t                             kv_block_stride_bytes,
                                                       size_t                             kv_scale_stride_bytes,
                                                       std::shared_ptr<const KVCacheSpec> spec,
                                                       const CacheConfig&                 cache_config,
                                                       uint32_t                           block_num,
                                                       uint32_t                           local_kv_head_num,
                                                       size_t                             seq_size_per_block,
                                                       size_t                             kernel_blocks_per_kv_block) {
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
        cfg.dtype                   = spec->memoryLayoutDType();
        cfg.local_head_num_kv       = local_kv_head_num;
        cfg.enable_hybrid_attention = enable_hybrid_attention;
        // Scale 3D layout for MLA and indexer; KV 3D only for MLA (concat_and_cache_mla)
        cfg.is_mla                     = cache_config.use_mla || cache_config.is_sparse;
        cfg.use_mla                    = cache_config.use_mla;
        cfg.seq_size_per_block         = seq_size_per_block;
        cfg.kernel_blocks_per_kv_block = kernel_blocks_per_kv_block;

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;

        cfg.kv_scale_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_scale_stride_bytes;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;
        return cfg;
    }
};

}  // namespace rtp_llm
