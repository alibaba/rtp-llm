#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

#include <string>

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
    static size_t sharedPoolKvBlockStrideBytes(const CacheConfig& cache_config) {
        return sharedPoolStrideBytes(cache_config, /*scale=*/false);
    }

    static size_t sharedPoolKvScaleStrideBytes(const CacheConfig& cache_config) {
        return sharedPoolStrideBytes(cache_config, /*scale=*/true);
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
        RTP_LLM_CHECK_WITH_INFO(cache_config.groupNums() > 0, "cache groups must not be empty");
        BlockPoolConfig config;
        config.pool_name       = "default";
        config.block_num       = cache_config.blockNum();
        const bool  is_hybrid  = cache_config.groupNums() > 1;
        auto        layer_num  = is_hybrid ? cache_config.groupLayerNum() : cache_config.layer_num;
        const auto& main_group = requireSharedPoolLayoutGroup(cache_config);
        const auto& main_spec  = main_group.spec;
        // linear block size is same with full block block size
        MemoryLayoutConfig main_layout =
            createMemoryLayoutConfig(is_hybrid,
                                     layer_num,
                                     sharedPoolKvBlockStrideBytes(cache_config),
                                     sharedPoolKvScaleStrideBytes(cache_config),
                                     config.block_num,
                                     main_spec,
                                     cache_config,
                                     main_group.local_kv_head_num,
                                     main_group.spec->seq_size_per_block,
                                     cache_config.kernelBlocksPerKvBlockForGroup(main_group.tag));

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
                mtp_sub_config->groupNums() > 0, "MTP module %zu cache groups must not be empty", i);

            const auto mtp_layer_num = mtp_sub_config->layer_num;

            const auto& mtp_group = requireSingleGroupForLayer(*mtp_sub_config, 0);
            const auto& mtp_spec  = mtp_group.spec;
            // MTP block size may differ from the main model. Use the real
            // MTP group that owns a layer; target-aligned placeholder groups
            // must not affect the sub-model memory layout.
            MemoryLayoutConfig mtp_layout =
                createMemoryLayoutConfig(false,
                                         mtp_layer_num,
                                         mtp_sub_config->kvBlockStrideBytesForGroup(mtp_group.tag),
                                         mtp_sub_config->kvScaleStrideBytesForGroup(mtp_group.tag),
                                         config.block_num,
                                         mtp_spec,
                                         cache_config,
                                         mtp_group.local_kv_head_num,
                                         mtp_group.spec->seq_size_per_block,
                                         mtp_sub_config->kernelBlocksPerKvBlockForGroup(mtp_group.tag));

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

    static BlockPoolConfig createConfigForGroup(const CacheConfig& cache_config, std::string_view tag) {
        const auto& group = cache_config.group(tag);
        const auto& spec  = group.spec;
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec tag=%s is null", group.tag.c_str());

        BlockPoolConfig config;
        config.pool_name = group.tag;
        config.block_num = group.block_num;
        RTP_LLM_LOG_INFO("createConfigForGroup: pool_name=%s block_num=%d groupNums=%d",
                         config.pool_name.c_str(),
                         config.block_num,
                         cache_config.groupNums());

        const uint32_t layer_num = static_cast<uint32_t>(group.layer_ids.size());
        RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "group tag=%s has no layers", group.tag.c_str());

        const size_t kv_stride    = group.kv_block_stride_bytes;
        const size_t scale_stride = group.kv_scale_stride_bytes;

        MemoryLayoutConfig layout    = createMemoryLayoutConfig(false,
                                                             layer_num,
                                                             kv_stride,
                                                             scale_stride,
                                                             config.block_num,
                                                             spec,
                                                             cache_config,
                                                             group.local_kv_head_num,
                                                             group.spec->seq_size_per_block,
                                                             cache_config.kernelBlocksPerKvBlockForGroup(group.tag));
        layout.kv_cache_offset_bytes = 0;
        layout.kv_scale_offset_bytes = layout.kv_cache_offset_bytes + layout.kv_block_pool_size_bytes;

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
    static size_t sharedPoolStrideBytes(const CacheConfig& cache_config, bool scale) {
        size_t result = 0;
        for (const auto& group : cache_config.topology().groups()) {
            result = std::max(result, scale ? group.kv_scale_stride_bytes : group.kv_block_stride_bytes);
        }
        return result;
    }

    static const GroupBase& requireSingleGroupForLayer(const CacheConfig& cache_config, int layer_id) {
        const auto groups = cache_config.groupsForLayer(layer_id);
        RTP_LLM_CHECK_WITH_INFO(groups.size() == 1,
                                "BlockPoolConfigHelper layer=%d requires exactly one group, got %zu",
                                layer_id,
                                groups.size());
        for (const auto& group : groups) {
            return group.get();
        }
        RTP_LLM_CHECK_WITH_INFO(false, "BlockPoolConfigHelper layer=%d has no group", layer_id);
        return cache_config.group("");
    }

    static const GroupBase& requireSharedPoolLayoutGroup(const CacheConfig& cache_config) {
        const auto       kv_stride    = sharedPoolKvBlockStrideBytes(cache_config);
        const auto       scale_stride = sharedPoolKvScaleStrideBytes(cache_config);
        const GroupBase* layout_group = nullptr;
        for (const auto& group : cache_config.topology().groups()) {
            if (group.spec == nullptr || group.kv_block_stride_bytes != kv_stride
                || group.kv_scale_stride_bytes != scale_stride) {
                continue;
            }
            if (layout_group == nullptr || group.tag < layout_group->tag) {
                layout_group = &group;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(layout_group != nullptr,
                                "BlockPoolConfigHelper has no group matching shared pool strides kv=%zu scale=%zu",
                                kv_stride,
                                scale_stride);
        return *layout_group;
    }

    static MemoryLayoutConfig createMemoryLayoutConfig(bool                               enable_hybrid_attention,
                                                       uint32_t                           layer_num,
                                                       size_t                             kv_block_stride_bytes,
                                                       size_t                             kv_scale_stride_bytes,
                                                       uint32_t                           block_num,
                                                       std::shared_ptr<const KVCacheSpec> spec,
                                                       const CacheConfig&                 cache_config,
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
