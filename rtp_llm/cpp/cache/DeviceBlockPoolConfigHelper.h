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
     * Create the per-group block pool config from a merged CacheConfig.
     * Memory layout is [layout0_kv][layout0_scale][layout1_kv][layout1_scale]...[layoutN_kv][layoutN_scale]
     * Generally Memory layout is [main_kv][main_scale][mtp1_kv][mtp1_scale]...[mtpN_kv][mtpN_scale]
     *
     * @param cache_config The merged CacheConfig (topology owns every main and MTP layer)
     * @param group The cache group this pool serves
     */
    static DeviceBlockPoolConfig createConfigForGroup(const CacheConfig& cache_config, const GroupBase& group) {
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
                         cache_config.groupNums());

        size_t     total_layout_layers = 0;
        size_t     current_offset      = 0;
        const auto&    mla_plan            = cache_config.mla_host_cache;
        const bool     mla_group           = group.tag == "default";
        const uint32_t device_blocks       = mla_group ? mla_plan.decodeDeviceBlocks(group.block_num) : group.block_num;
        const auto append_layout = [&](const GroupBase& source_group, uint32_t layer_num, bool prefill_host = false) {
            RTP_LLM_CHECK_WITH_INFO(layer_num > 0, "group tag=%s layout has no layers", group.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(
                source_group.spec != nullptr, "cache spec for group tag=%s is null", source_group.tag.c_str());
            auto layout                  = createMemoryLayoutConfig(source_group, layer_num, group.block_num);
            if (prefill_host) {
                layout.host_block_count = group.block_num;
            } else if (device_blocks < group.block_num) {
                RTP_LLM_CHECK_WITH_INFO(source_group.kvScaleStrideBytes() == 0,
                                        "GLM5 split MLA blocks must use a packed layout");
                layout.device_block_count  = device_blocks;
                layout.device_spare_blocks = mla_plan.decode_resident_blocks;
                layout.host_block_count    = group.block_num - device_blocks;
            }
            layout.kv_cache_offset_bytes = current_offset;
            current_offset += layout.kv_block_pool_size_bytes;
            layout.kv_scale_offset_bytes = current_offset;
            current_offset += layout.kv_scale_pool_size_bytes;
            total_layout_layers += layer_num;
            config.memory_layouts.push_back(std::move(layout));
        };

        const auto& group_layer_ids = cache_config.layerIdsForGroup(group.tag);
        config.allocate_storage     = !group_layer_ids.empty();
        const auto  main_layer_num  = static_cast<uint32_t>(
            std::count_if(group_layer_ids.begin(), group_layer_ids.end(), [&cache_config](int layer_id) {
                return layer_id >= 0 && static_cast<uint32_t>(layer_id) < cache_config.layer_num;
            }));
        if (mla_group && mla_plan.layeredPrefill()) {
            RTP_LLM_CHECK_WITH_INFO(mla_plan.prefill_hbm_layers > 0 && mla_plan.prefill_hbm_layers < main_layer_num,
                                    "layered GLM5 prefill requires resident and host MLA layers");
            append_layout(group, mla_plan.prefill_hbm_layers);
            append_layout(group, main_layer_num - mla_plan.prefill_hbm_layers, true);
        } else if (main_layer_num > 0) {
            append_layout(group, main_layer_num);
        }

        for (size_t module_index = 0; module_index < cache_config.mtp_sub_configs.size(); ++module_index) {
            const auto& mtp_config = cache_config.mtp_sub_configs[module_index];
            RTP_LLM_CHECK_WITH_INFO(mtp_config != nullptr, "mtp_sub_configs[%zu] is null", module_index);
            const auto& mtp_group     = mtp_config->topology().group(group.tag);
            const auto  mtp_layer_num = static_cast<uint32_t>(mtp_config->layerIdsForGroup(mtp_group.tag).size());
            if (mtp_layer_num > 0) {
                append_layout(mtp_group, mtp_layer_num);
            }
        }

        RTP_LLM_CHECK_WITH_INFO(total_layout_layers == group_layer_ids.size(),
                                "group tag=%s layout layer count=%zu does not match topology layers=%zu",
                                group.tag.c_str(),
                                total_layout_layers,
                                group_layer_ids.size());
        config.total_size_bytes = current_offset;
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
