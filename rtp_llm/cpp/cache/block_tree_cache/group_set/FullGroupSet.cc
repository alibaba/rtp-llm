#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/PrefixCoverage.h"

namespace rtp_llm {

std::unique_ptr<MatchValidator> FullGroupSet::createMatchValidator() {
    return std::make_unique<FullMatchValidator>();
}

size_t FullGroupSet::computeReuseBlockCount(size_t matched_block_count) const {
    return reusableBlockCount(groupType(), pageTokens(), windowTokens(), matched_block_count);
}

// FullMatchValidator
bool FullMatchValidator::validate(const GroupSetResource& resource) {
    prefix_valid_ = prefix_valid_ && resource.isMatchUsable();
    return prefix_valid_;
}

}  // namespace rtp_llm
