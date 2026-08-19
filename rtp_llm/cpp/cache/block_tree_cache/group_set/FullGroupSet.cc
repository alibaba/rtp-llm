#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"

namespace rtp_llm {

std::unique_ptr<MatchValidator> FullGroupSet::createMatchValidator() {
    return std::make_unique<FullMatchValidator>();
}

size_t FullGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return groupAt(0).reuseBlockCount(matched_block_count);
}

// FullMatchValidator
bool FullMatchValidator::validate(const GroupSetResource& resource) {
    // A busy resource (DEMOTING/LOAD_PENDING) is as unusable as a hole; the prefix
    // latch keeps FULL reuse contiguous from the root.
    prefix_valid_ = prefix_valid_ && resource.isMatchUsable() && !resource.is_empty();
    return prefix_valid_;
}

}  // namespace rtp_llm
