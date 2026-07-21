#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"

namespace rtp_llm {

LinearComponentGroup::LinearComponentGroup() {
    group_type = CacheGroupType::LINEAR;
}

std::unique_ptr<MatchValidator> LinearComponentGroup::createMatchValidator() {
    return std::make_unique<LinearMatchValidator>();
}

size_t LinearComponentGroup::computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>&) const {
    return matched_block_count == 0 ? 0 : 1;
}

bool LinearMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    return !slot.is_empty();
}

}  // namespace rtp_llm
