#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"

namespace rtp_llm {

LinearGroupSet::LinearGroupSet() = default;

std::unique_ptr<MatchValidator> LinearGroupSet::createMatchValidator() {
    return std::make_unique<LinearMatchValidator>();
}

size_t LinearGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return matched_block_count == 0 ? 0 : 1;
}

bool LinearMatchValidator::validate(const GroupSetResource& resource) {
    // Point-state: each node is judged on its own; a busy resource is unusable.
    return resource.isMatchUsable() && !resource.is_empty();
}

}  // namespace rtp_llm
