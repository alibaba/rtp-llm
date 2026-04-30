#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

bool useFullBlockRemoteLoad(const CacheConfig& cache_config);

std::vector<size_t> blockPositionsForRpc(
    size_t block_num, size_t reuse_block_size, bool use_hybrid, CacheGroupType group_type, bool hybrid_full_from_begin);

std::string layerRegionRequestKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name);

}  // namespace rtp_llm
