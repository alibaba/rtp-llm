#include "rtp_llm/cpp/cache/KVCacheLayerRegionUtils.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

size_t groupStrideBytes(const CacheConfig& cache_config, int gid, int layer_id) {
    if (gid >= 0 && static_cast<size_t>(gid) < cache_config.group_kv_block_stride_bytes.size()) {
        const size_t kv_stride    = cache_config.group_kv_block_stride_bytes[static_cast<size_t>(gid)];
        const size_t scale_stride = static_cast<size_t>(gid) < cache_config.group_kv_scale_stride_bytes.size() ?
                                        cache_config.group_kv_scale_stride_bytes[static_cast<size_t>(gid)] :
                                        0;
        if (kv_stride + scale_stride > 0) {
            return kv_stride + scale_stride;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0
                                && static_cast<size_t>(layer_id) < cache_config.layer_to_block_stride_bytes.size(),
                            "missing block stride bytes for layer=%d group=%d",
                            layer_id,
                            gid);
    return static_cast<size_t>(cache_config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)]);
}

}  // namespace

std::vector<LayerRegionSlot> buildLayerRegionSlots(const CacheConfig& cache_config, size_t layer_num) {
    std::vector<LayerRegionSlot> slots;
    const size_t                 region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    const bool                   has_typed_mapping = !cache_config.layer_region_to_group_id.empty();

    if (has_typed_mapping) {
        RTP_LLM_CHECK_WITH_INFO(cache_config.layer_region_to_group_id.size() >= layer_num,
                                "layer_region_to_group_id size %zu < layer_num %zu",
                                cache_config.layer_region_to_group_id.size(),
                                layer_num);
        RTP_LLM_CHECK_WITH_INFO(cache_config.group_region_names.size() == static_cast<size_t>(cache_config.groupNums()),
                                "group_region_names size %zu != group num %d for typed layer-region mapping",
                                cache_config.group_region_names.size(),
                                cache_config.groupNums());
    }

    for (size_t layer = 0; layer < layer_num; ++layer) {
        bool has_typed_slot = false;
        if (layer < cache_config.layer_region_to_group_id.size()) {
            const auto&  dense = cache_config.layer_region_to_group_id[layer];
            const size_t n     = std::min(region_name_count, dense.size());
            for (size_t region = 0; region < n; ++region) {
                const int gid = dense[region];
                if (gid < 0) {
                    continue;
                }
                RTP_LLM_CHECK_WITH_INFO(gid < cache_config.groupNums(),
                                        "invalid group id %d for layer %zu region_name %zu",
                                        gid,
                                        layer,
                                        region);
                slots.push_back(LayerRegionSlot{static_cast<int>(layer),
                                                static_cast<KVCacheRegionName>(region),
                                                gid,
                                                groupStrideBytes(cache_config, gid, static_cast<int>(layer))});
                has_typed_slot = true;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(
            !has_typed_mapping || has_typed_slot, "missing typed layer-region mapping for layer %zu", layer);
        if (!has_typed_slot) {
            int gid = 0;
            if (layer < cache_config.layer_to_group_id.size() && cache_config.layer_to_group_id[layer] >= 0) {
                gid = cache_config.layer_to_group_id[layer];
            }
            RTP_LLM_CHECK_WITH_INFO(
                gid < cache_config.groupNums(), "invalid default group id %d for layer %zu", gid, layer);
            slots.push_back(LayerRegionSlot{static_cast<int>(layer),
                                            KVCacheRegionName::DEFAULT,
                                            gid,
                                            groupStrideBytes(cache_config, gid, static_cast<int>(layer))});
        }
    }
    return slots;
}

bool hasTypedLayerRegionSlots(const std::vector<LayerRegionSlot>& slots, size_t layer_num) {
    if (slots.size() != layer_num) {
        return true;
    }
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].layer_id != static_cast<int>(i) || slots[i].region_name != KVCacheRegionName::DEFAULT) {
            return true;
        }
    }
    return false;
}

CacheGroupType cacheGroupTypeForGroup(const CacheConfig& cache_config, size_t group_id) {
    RTP_LLM_CHECK_WITH_INFO(group_id < static_cast<size_t>(cache_config.groupNums()),
                            "group id %zu out of range %d",
                            group_id,
                            cache_config.groupNums());
    if (group_id < cache_config.group_types.size()) {
        return cache_config.group_types[group_id];
    }
    RTP_LLM_CHECK_WITH_INFO(
        cache_config.groupNums() == 1, "missing group type for group %zu in multi-group cache config", group_id);
    return CacheGroupType::FULL;
}

}  // namespace rtp_llm
