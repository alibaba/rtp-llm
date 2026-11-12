#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

#include <algorithm>
#include <string>
#include <utility>

namespace rtp_llm {

class DeviceBlockPoolConfigHelper {
public:
    /**
     * Create the per-(module, tag) block pool config from a single module's
     * config. Memory layout is one segment: [kv][scale] for this module's own
     * layers in local layer order.
     *
     * @param module_config The module (main or one MTP module) this pool serves
     * @param group The cache group this pool serves
     */
    static DeviceBlockPoolConfig createConfigForGroup(const CacheModuleConfig& module_config, const GroupBase& group) {
        const auto& spec = group.spec;
        RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec for group tag=%s is null", group.tag.c_str());

        DeviceBlockPoolConfig config;
        config.pool_type            = BlockPoolType::DEVICE;
        config.pool_name            = group.tag.empty() ? "group" : group.tag;
        config.physical_block_count = group.block_num;
        RTP_LLM_CHECK_WITH_INFO(group.block_num > 0, "group tag=%s requires positive pool capacity", group.tag.c_str());
        RTP_LLM_LOG_INFO("createConfigForGroup: pool_name=%s block_num=%zu groupNums=%d",
                         config.pool_name.c_str(),
                         config.physical_block_count,
                         module_config.groupNums());

        const auto group_layer_ids = module_config.layerIdsForGroup(group.tag);
        RTP_LLM_CHECK_WITH_INFO(!group_layer_ids.empty(), "group tag=%s has no layers", group.tag.c_str());
        const auto layer_num = static_cast<uint32_t>(group_layer_ids.size());

        auto layout = createMemoryLayoutConfig(false,
                                              layer_num,
                                              group.kvBlockStrideBytes(),
                                              group.kvScaleStrideBytes(),
                                              group.spec,
                                              module_config,
                                              group.block_num,
                                              group.localKvHeadNum(),
                                              group.seqSizePerBlock(),
                                              group.kernelBlocksPerKvBlock());
        layout.kv_cache_offset_bytes  = 0;
        layout.kv_scale_offset_bytes  = layout.kv_block_pool_size_bytes;
        config.memory_layouts.push_back(std::move(layout));
        config.total_size_bytes = layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes;
        return config;
    }

private:
    static MemoryLayoutConfig createMemoryLayoutConfig(bool                               enable_hybrid_attention,
                                                       uint32_t                           layer_num,
                                                       size_t                             kv_block_stride_bytes,
                                                       size_t                             kv_scale_stride_bytes,
                                                       std::shared_ptr<const KVCacheSpec> spec,
                                                       const CacheModuleConfig&           module_config,
                                                       uint32_t                           physical_block_count,
                                                       uint32_t                           local_kv_head_num,
                                                       size_t                             seq_size_per_block,
                                                       size_t                             kernel_blocks_per_kv_block) {
        MemoryLayoutConfig cfg;
        cfg.layer_num             = layer_num;
        cfg.block_num             = physical_block_count;
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
        // Hybrid models share the model-level MLA flag with their state pools.
        // Linear state has no token axis and must retain its physical block view.
        const bool is_linear_state = spec->type == KVCacheSpecType::LinearAttention;
        cfg.is_mla                     = !is_linear_state && (module_config.use_mla || module_config.is_sparse);
        cfg.use_mla                    = !is_linear_state && module_config.use_mla;
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
