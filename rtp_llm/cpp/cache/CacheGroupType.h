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

}  // namespace rtp_llm
