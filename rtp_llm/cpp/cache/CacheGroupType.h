#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (PD cache-store transfer keeps the last block)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - SWA: sliding-window attention group (PD cache-store transfer keeps the last two blocks)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
    SWA    = 2,
};

// Region name for models where one logical layer owns multiple cache entries.
// CacheGroupType describes allocation/reuse policy, while KVCacheRegionName
// describes which cache object a layer wants to access.
enum class KVCacheRegionName : int8_t {
    DEFAULT       = 0,
    CSA_KV        = 1,
    HCA_KV        = 2,
    INDEXER_KV    = 3,
    INDEXER_STATE = 4,
    CSA_STATE     = 5,
    HCA_STATE     = 6,
    SWA_KV        = 7,
    REGION_COUNT  = 8,
};

}  // namespace rtp_llm
