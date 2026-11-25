#pragma once

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DeviceBlockPool.h"

#include <algorithm>
#include <string>
#include <utility>

namespace rtp_llm {

class DeviceBlockPoolConfigHelper {
public:
    static DeviceBlockPoolConfig createConfigForGroup(const CacheConfig& config, const GroupBase& group) {
        return createConfigForGroup(config.topology(config.moduleOfTag(group.tag)), group);
    }

    /**
     * Create the per-(module, tag) block pool config from a single module's
     * immutable topology. Memory layout is one segment: [kv][scale] for this
     * module's own layers in local layer order.
     *
     * @param module_topology The topology of the module (main or one MTP module) this pool serves
     * @param group The cache group this pool serves
     */
    static DeviceBlockPoolConfig createConfigForGroup(const CacheTopology& module_topology, const GroupBase& group) {
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
                         static_cast<int>(module_topology.groups().size()));

        const auto layer_num = static_cast<uint32_t>(module_topology.layerIdsForGroup(group.tag).size());
        config.allocate_storage = layer_num > 0;
        if (layer_num == 0) {
            // This stage allocates block IDs for the group without backing KV memory.
            return config;
        }

        auto layout                  = createMemoryLayoutConfig(group, layer_num, group.block_num);
        layout.kv_cache_offset_bytes = 0;
        layout.kv_scale_offset_bytes = layout.kv_block_pool_size_bytes;
        config.total_size_bytes      = layout.total_size_bytes;
        config.memory_layouts.push_back(std::move(layout));
        return config;
    }

private:
    static MemoryLayoutConfig
    createMemoryLayoutConfig(const GroupBase& group, uint32_t layer_num, uint32_t physical_block_count) {
        const auto&        spec = group.spec;
        MemoryLayoutConfig cfg;
        cfg.layer_num             = layer_num;
        cfg.block_num             = physical_block_count;
        cfg.kv_block_stride_bytes = group.kvBlockStrideBytes();
        cfg.k_block_stride_bytes  = spec->k_block_size_bytes();
        cfg.v_block_stride_bytes  = spec->v_block_size_bytes();
        cfg.kv_scale_stride_bytes = group.kvScaleStrideBytes();
        cfg.k_scale_stride_bytes  = spec->k_scale_block_size_bytes();
        cfg.v_scale_stride_bytes  = spec->v_scale_block_size_bytes();

        cfg.enable_kv_scale         = cfg.kv_scale_stride_bytes > 0;
        cfg.dtype                   = spec->memoryLayoutDType();
        cfg.cache_type                 = spec->type;
        cfg.local_head_num_kv          = group.localKvHeadNum();
        cfg.seq_size_per_block         = group.seqSizePerBlock();
        cfg.kernel_blocks_per_kv_block = group.kernelBlocksPerKvBlock();

        cfg.kv_block_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_block_stride_bytes;

        cfg.kv_scale_pool_size_bytes =
            static_cast<size_t>(layer_num) * static_cast<size_t>(cfg.block_num) * cfg.kv_scale_stride_bytes;
        cfg.total_size_bytes = cfg.kv_block_pool_size_bytes + cfg.kv_scale_pool_size_bytes;
        return cfg;
    }
};

}  // namespace rtp_llm
