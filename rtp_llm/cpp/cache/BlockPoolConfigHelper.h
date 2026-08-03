#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

#include <algorithm>
#include <string>

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    static BlockPoolConfig createConfigForGroup(const CacheConfig& cache_config, std::string_view tag) {
        const auto& group = cache_config.group(tag);
        const auto& spec  = group.spec;
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec tag=%s is null", group.tag.c_str());

        BlockPoolConfig config;
        config.pool_name            = group.tag;
        config.block_num            = group.block_num;
        const bool has_group_blocks = config.block_num != cache_config.block_num;
        RTP_LLM_LOG_INFO("createConfigForGroup: pool_name=%s block_num=%d (has_group_blocks=%d, "
                         "groupNums=%d, global_block_num=%d)",
                         config.pool_name.c_str(),
                         config.block_num,
                         has_group_blocks,
                         cache_config.groupNums(),
                         cache_config.block_num);

        size_t   current_offset = 0;
        uint32_t layout_layers  = 0;
        auto append_layout = [&](const CacheConfig& source_config, const GroupBase& source_group, uint32_t layer_num) {
            if (layer_num == 0) {
                return;
            }
            CacheConfig resolved_config  = source_config;
            resolved_config.block_num    = config.block_num;
            auto layout                  = createMemoryLayoutConfig(false,
                                                   layer_num,
                                                   source_group.kv_block_stride_bytes,
                                                   source_group.kv_scale_stride_bytes,
                                                   source_group.spec,
                                                   resolved_config,
                                                   source_group.local_kv_head_num,
                                                   source_group.seq_size_per_block,
                                                   source_config.kernelBlocksPerKvBlockForGroup(source_group.tag));
            layout.kv_cache_offset_bytes = current_offset;
            current_offset += layout.kv_block_pool_size_bytes;
            layout.kv_scale_offset_bytes = current_offset;
            current_offset += layout.kv_scale_pool_size_bytes;
            layout_layers += layer_num;
            config.memory_layouts.push_back(std::move(layout));
        };

        const auto main_layer_num =
            static_cast<uint32_t>(std::count_if(group.layer_ids.begin(), group.layer_ids.end(), [&](int layer_id) {
                return layer_id >= 0 && static_cast<uint32_t>(layer_id) < cache_config.layer_num;
            }));
        append_layout(cache_config, group, main_layer_num);

        for (const auto& mtp_config : cache_config.mtp_sub_configs) {
            if (mtp_config == nullptr || !mtp_config->topology().contains(tag)) {
                continue;
            }
            const auto& mtp_group = mtp_config->group(tag);
            append_layout(*mtp_config, mtp_group, static_cast<uint32_t>(mtp_group.layer_ids.size()));
        }

        RTP_LLM_CHECK_WITH_INFO(layout_layers == group.layer_ids.size(),
                                "group tag=%s resolved layout layers=%u do not match mapped layers=%zu",
                                group.tag.c_str(),
                                layout_layers,
                                group.layer_ids.size());
        RTP_LLM_CHECK_WITH_INFO(
            !config.memory_layouts.empty(), "group tag=%s has no physical layouts", group.tag.c_str());
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
    static MemoryLayoutConfig createMemoryLayoutConfig(bool                               enable_hybrid_attention,
                                                       uint32_t                           layer_num,
                                                       size_t                             kv_block_stride_bytes,
                                                       size_t                             kv_scale_stride_bytes,
                                                       std::shared_ptr<const KVCacheSpec> spec,
                                                       CacheConfig                        cache_config,
                                                       uint32_t                           local_kv_head_num,
                                                       size_t                             seq_size_per_block,
                                                       size_t                             kernel_blocks_per_kv_block) {
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
