#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (only last block is needed for cache-store transfer)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - FIXED: fixed-allocation group (not paged, not prefix-cacheable; e.g. DSV4 state pools)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
    FIXED  = 2,
};

// Cache identity for models where one logical layer owns multiple cache entries.
// CacheGroupType describes allocation/reuse policy, while KVCacheAttnType
// describes which cache object a layer wants to access.
enum class KVCacheAttnType : int8_t {
    DEFAULT       = 0,
    CSA_KV        = 1,
    HCA_KV        = 2,
    INDEXER_KV    = 3,
    INDEXER_STATE = 4,
    CSA_STATE     = 5,
    HCA_STATE     = 6,
    SWA_KV        = 7,
    TYPE_COUNT    = 8,
};

}  // namespace rtp_llm
