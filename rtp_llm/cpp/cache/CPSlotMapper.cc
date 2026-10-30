#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <stdexcept>

namespace rtp_llm {

std::optional<size_t> CPSlotMapper::physicalBlockPosition(const CacheGroupPolicy& policy,
                                                          size_t                  logical_position,
                                                          size_t                  logical_count,
                                                          int                     cp_rank,
                                                          int                     cp_size) {
    if (cp_size <= 0) {
        throw std::invalid_argument("CPSlotMapper cp_size must be positive");
    }
    if (cp_rank < 0 || cp_rank >= cp_size) {
        throw std::invalid_argument("CPSlotMapper cp_rank out of range");
    }

    const size_t cp_scale = static_cast<size_t>(cp_size);
    if (policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN && cp_scale > 1) {
        if (logical_position % cp_scale != static_cast<size_t>(cp_rank)) {
            return std::nullopt;
        }
        return logical_position / cp_scale;
    }
    if (policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK && cp_scale > 1) {
        const bool is_segment_tail = logical_position % cp_scale == cp_scale - 1;
        const bool is_final_key    = logical_position + 1 == logical_count;
        if (!is_segment_tail && !is_final_key) {
            return std::nullopt;
        }
        return logical_position / cp_scale;
    }
    return logical_position;
}

}  // namespace rtp_llm
