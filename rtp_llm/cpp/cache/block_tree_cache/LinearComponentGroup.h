#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// LinearComponentGroup: manages linear attention / SSM hidden states.
// Point-state: only specific nodes hold state snapshots.
// Uses Any-node heaps like SWA.
class LinearComponentGroup: public ComponentGroup {
public:
    LinearComponentGroup();

    std::unique_ptr<MatchValidator> createMatchValidator() override;
    size_t computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const override;
};

class LinearMatchValidator: public MatchValidator {
public:
    bool validate(const TreeNode* node, const GroupSlot& slot) override;
};

}  // namespace rtp_llm
