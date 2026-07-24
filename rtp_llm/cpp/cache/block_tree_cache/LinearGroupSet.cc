#include "rtp_llm/cpp/cache/block_tree_cache/LinearGroupSet.h"

namespace rtp_llm {

LinearGroupSet::LinearGroupSet() = default;

std::unique_ptr<MatchValidator> LinearGroupSet::createMatchValidator() {
    return std::make_unique<LinearMatchValidator>();
}

size_t LinearGroupSet::computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>&) const {
    return matched_block_count == 0 ? 0 : 1;
}

bool LinearMatchValidator::validate(const TreeNode* node, const GroupSetResource& slot) {
    return !slot.is_empty();
}

}  // namespace rtp_llm
