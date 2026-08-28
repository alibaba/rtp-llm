#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

struct EvictionTimingSnapshot {
    EvictionTimingSnapshot() = default;
    explicit EvictionTimingSnapshot(const CandidateMeta& candidate_meta):
        tier_enter_time_us(candidate_meta.tier_enter_time_us),
        insert_time_us(candidate_meta.insert_time_us),
        last_access_time_us(candidate_meta.last_access_time_us),
        selected_time_us(currentTimeUs()) {}

    int64_t tier_enter_time_us{0};
    int64_t insert_time_us{0};
    int64_t last_access_time_us{0};
    int64_t selected_time_us{0};
};

struct EvictionTask {
    TransferDescriptor                  primary_desc;
    EvictionTimingSnapshot              primary_timing;
    std::vector<TransferDescriptor>     cascade_descs;
    std::vector<EvictionTimingSnapshot> cascade_timings;
    // FULL prune closure only. Every dependent descriptor targets NONE,
    // and nodes stay valid because task activation is synchronous.
    std::vector<TransferDescriptor>     dependent_prune_descs;
    std::vector<EvictionTimingSnapshot> dependent_prune_timings;
    std::vector<TreeNode*>              full_prune_nodes_bottom_up;
    bool                                business_credit_acquired{false};

    bool needsCopy() const {
        return primary_desc.target_tier != Tier::NONE
               || std::any_of(cascade_descs.begin(), cascade_descs.end(), [](const TransferDescriptor& cascade_desc) {
                      return cascade_desc.target_tier != Tier::NONE;
                  });
    }

    bool hasFullPrune() const {
        return !full_prune_nodes_bottom_up.empty();
    }
};

struct EvictionTaskResult {
    bool              primary_success{false};
    std::vector<bool> cascade_success;
};

}  // namespace rtp_llm
