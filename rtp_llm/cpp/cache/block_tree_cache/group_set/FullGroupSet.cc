#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"

namespace rtp_llm {

FullGroupSet::FullGroupSet() = default;

std::unique_ptr<MatchValidator> FullGroupSet::createMatchValidator() {
    return std::make_unique<FullMatchValidator>();
}

size_t FullGroupSet::computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>&) const {
    return matched_block_count;
}

bool FullGroupSet::isSlotEvictable(const TreeNode& node, Tier tier) const {
    if (groupSetId() >= node.group_set_resources.size()) {
        return false;
    }
    if (!isLeafAtTier(&node, tier)) {
        return false;
    }
    return GroupSet::isSlotEvictable(node, tier);
}

// FullMatchValidator
bool FullMatchValidator::validate(const TreeNode* node, const GroupSetResource& slot) {
    // A busy slot (DEMOTING/LOAD_PENDING) is as unusable as a hole; the prefix
    // latch keeps FULL reuse contiguous from the root.
    prefix_valid_ = prefix_valid_ && slot.isMatchUsable() && !slot.is_empty();
    return prefix_valid_;
}

}  // namespace rtp_llm
