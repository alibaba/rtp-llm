#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

// LinearGroupSet: manages linear attention / SSM hidden states.
// Point-state: only specific nodes hold state snapshots.
// Uses Any-node heaps like SWA.
class LinearGroupSet: public GroupSet {
public:
    using GroupSet::GroupSet;

    std::unique_ptr<MatchValidator> createMatchValidator() override;
    size_t computeReuseBlockCount(size_t matched_block_count) const override;
};

class LinearMatchValidator: public MatchValidator {
public:
    bool validate(const GroupSetResource& resource) override;
};

}  // namespace rtp_llm
