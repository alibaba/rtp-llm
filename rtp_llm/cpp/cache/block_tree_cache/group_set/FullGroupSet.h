#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

class FullGroupSet: public GroupSet {
public:
    using GroupSet::GroupSet;

    std::unique_ptr<MatchValidator> createMatchValidator() override;
    size_t computeReuseBlockCount(size_t matched_block_count) const override;
};

class FullMatchValidator: public MatchValidator {
public:
    bool validate(const GroupSetResource& resource) override;

private:
    bool prefix_valid_{true};
};

}  // namespace rtp_llm
