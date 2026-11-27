#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

// Each endpoint is already aligned to this group's page geometry.
inline size_t reusableBlockCount(CacheGroupType type, size_t page_tokens, size_t window_tokens, size_t rows) {
    assert(page_tokens > 0);
    switch (type) {
        case CacheGroupType::FULL:
            return rows;
        case CacheGroupType::LINEAR:
            return rows != 0;
        case CacheGroupType::SWA:
            return window_tokens == 0 ? rows : std::min(rows, 1 + (window_tokens - 1) / page_tokens);
    }
    assert(false && "Unknown BlockTree group type");
    return 0;
}

}  // namespace rtp_llm
