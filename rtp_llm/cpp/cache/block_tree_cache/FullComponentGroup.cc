#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"

namespace rtp_llm {

FullComponentGroup::FullComponentGroup() {
    group_type = CacheGroupType::FULL;
}

std::unique_ptr<MatchValidator> FullComponentGroup::createMatchValidator() {
    return std::make_unique<FullMatchValidator>();
}

size_t FullComponentGroup::computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>&) const {
    return matched_block_count;
}

bool FullComponentGroup::isSlotEvictable(const TreeNode& node, Tier tier) const {
    if (component_group_id < 0 || static_cast<size_t>(component_group_id) >= node.group_slots.size()) {
        return false;
    }
    if (!isLeafAtTier(&node, component_group_id, tier)) {
        return false;
    }
    return ComponentGroup::isSlotEvictable(node, tier);
}

// FullMatchValidator
bool FullMatchValidator::validate(const TreeNode* node, const GroupSlot& slot) {
    prefix_valid_ = prefix_valid_ && !slot.is_empty();
    return prefix_valid_;
}

}  // namespace rtp_llm
