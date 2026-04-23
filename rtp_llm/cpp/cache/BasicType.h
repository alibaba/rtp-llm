#pragma once

#include <cstdint>
#include <vector>

namespace rtp_llm {

using CacheKeyType = uint64_t;
using BlockIdxType = int32_t;
using GroupIdType  = int32_t;
using LayerIdsType = std::vector<int>;

}  // namespace rtp_llm
