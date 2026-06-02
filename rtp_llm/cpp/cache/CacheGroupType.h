#pragma once

#include <cstdint>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (only last block is needed for cache-store transfer)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
};

}  // namespace rtp_llm
