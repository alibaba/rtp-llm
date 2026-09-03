#pragma once

#include <cstdint>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

// Cross-stage agreed block counts, expressed as construction inputs: a count
// per tag, plus the top-level yardstick of the pools that follow the budget.
struct NegotiatedCapacity {
    uint32_t            paged_block_num = 0;
    PPBlockNumOverrides block_num_overrides;
};

/* Capacity negotiation hook for CacheConfigCreator, consulted between local
   measurement and sizing so the agreement enters the config as an input.
   Implementations abort startup on failure, so a returned value is always usable. */
class CacheCapacityNegotiator {
public:
    virtual ~CacheCapacityNegotiator() = default;

    // topology is this stage's unsized skeleton; local_block_num is what this
    // machine alone can afford.
    virtual NegotiatedCapacity
    negotiate(const CacheConfig& topology, uint32_t local_block_num, const RuntimeConfig& runtime_config) = 0;

    virtual void validateComposed(const CacheConfig& composed, const NegotiatedCapacity& agreed) = 0;
};

}  // namespace rtp_llm
