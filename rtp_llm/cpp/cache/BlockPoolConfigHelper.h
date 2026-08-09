#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/BlockPoolConfig.h"

#include <string>

namespace rtp_llm {

class BlockPoolConfigHelper {
public:
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

        const bool         is_hybrid = cache_config.groupNums() > 1;
        MemoryLayoutConfig layout    = createMemoryLayoutConfig(is_hybrid,
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
