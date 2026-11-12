#pragma once

#include <cstdint>
#include <optional>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

struct BlockTreeCacheReuseTimeMetrics {
    Tier                   tier{Tier::DEVICE};
    CacheGroupType         group_type{CacheGroupType::FULL};
    std::optional<int64_t> reuse_block_first_hit_idletime_ms;
    int64_t                reuse_block_hit_idletime_ms{0};
};

}  // namespace rtp_llm
