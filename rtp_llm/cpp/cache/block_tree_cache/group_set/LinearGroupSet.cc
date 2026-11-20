#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"

namespace rtp_llm {

std::unique_ptr<MatchValidator> LinearGroupSet::createMatchValidator() {
    return std::make_unique<LinearMatchValidator>();
}

size_t LinearGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return group(groupTags().front()).reuseBlockCount(matched_block_count);
}

bool LinearMatchValidator::validate(const GroupSetResource& resource) {
    return resource.isMatchUsable();
}

}  // namespace rtp_llm
