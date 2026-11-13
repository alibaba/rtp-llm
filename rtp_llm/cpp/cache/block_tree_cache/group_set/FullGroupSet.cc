#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"

namespace rtp_llm {

std::unique_ptr<MatchValidator> FullGroupSet::createMatchValidator() {
    return std::make_unique<FullMatchValidator>();
}

size_t FullGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return group(groupTags().front()).reuseBlockCount(matched_block_count);
}

// FullMatchValidator
bool FullMatchValidator::validate(const GroupSetResource& resource) {
    // Unavailable sources act as holes; complete DEVICE demotion sources remain
    // matchable. The latch keeps FULL reuse contiguous from the root.
    prefix_valid_ = prefix_valid_ && resource.isMatchUsable() && !resource.is_empty();
    return prefix_valid_;
}

}  // namespace rtp_llm
