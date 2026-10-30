#pragma once

#include <cstddef>
#include <optional>

#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

class CPSlotMapper {
public:
    static std::optional<size_t> physicalBlockPosition(const CacheGroupPolicy& policy,
                                                       size_t                  logical_position,
                                                       size_t                  logical_count,
                                                       int                     cp_rank,
                                                       int                     cp_size);
};

}  // namespace rtp_llm
