#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (only last block is needed for cache-store transfer)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - SLIDING_WINDOW: sliding window MHA group (full blocks needed, but with different KV dims)
enum class CacheGroupType : int8_t {
    LINEAR         = 0,
    FULL           = 1,
    SLIDING_WINDOW = 2,
};

}  // namespace rtp_llm
