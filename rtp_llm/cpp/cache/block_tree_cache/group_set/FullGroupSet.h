#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

// FullGroupSet: manages full KV cache indices.
// Uses Leaf-based heaps: only nodes without children having data at
// the same tier are eligible for eviction.
class FullGroupSet: public GroupSet {
public:
    FullGroupSet();

    std::unique_ptr<MatchValidator> createMatchValidator() override;
    size_t computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const override;

    // Full-specific: the base block/refcount checks plus tier-leaf topology.
    bool isSlotEvictable(const TreeNode& node, Tier tier) const override;
};

class FullMatchValidator: public MatchValidator {
public:
    bool validate(const TreeNode* node, const GroupSetResource& slot) override;

private:
    bool prefix_valid_{true};
};

}  // namespace rtp_llm
