#pragma once

#include <cstddef>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct LayerRegionSlot {
    int               layer_id{-1};
    KVCacheRegionName region_name{KVCacheRegionName::DEFAULT};
    int               group_id{-1};
    size_t            stride_bytes{0};
};

std::vector<LayerRegionSlot> buildLayerRegionSlots(const CacheConfig& cache_config, size_t layer_num);

bool hasTypedLayerRegionSlots(const std::vector<LayerRegionSlot>& slots, size_t layer_num);

CacheGroupType cacheGroupTypeForGroup(const CacheConfig& cache_config, size_t group_id);

}  // namespace rtp_llm
