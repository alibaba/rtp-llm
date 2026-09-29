#pragma once

#include <cstddef>
#include <cstdint>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

/**
 * Cross-stage agreed block counts, expressed as construction inputs: a count
 * per tag, plus the top-level yardstick of the pools that follow the budget.
 */
struct NegotiatedCapacity {
    uint32_t            paged_block_num                  = 0;
    size_t              block_tree_combined_stride_bytes = 0;
    PPBlockNumOverrides block_num_overrides;
};

/**
 * Cross-stage capacity negotiation hook, invoked by engine initialization
 * after intra-stage alignment and before KVCacheManager construction.
 * Implementations abort startup on failure, so a returned value is usable.
 */
class CacheCapacityNegotiator {
public:
    virtual ~CacheCapacityNegotiator() = default;

    /**
     * topology is this stage's unfinalized local topology; local_block_num is the
     * stage-aligned device capacity, and local_block_tree_combined_stride_bytes comes
     * from the already-planned local BlockTree layout.
     */
    virtual NegotiatedCapacity negotiate(const CacheConfig&   topology,
                                         uint32_t             local_block_num,
                                         size_t               local_block_tree_combined_stride_bytes,
                                         const RuntimeConfig& runtime_config) = 0;
};

}  // namespace rtp_llm
