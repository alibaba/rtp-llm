#pragma once

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

struct EvictionTransferTask {
    std::vector<TransferDescriptor>     descs;
    std::vector<EvictionTimingSnapshot> timings;
    int64_t                             enqueue_time_us{0};
};

struct EvictionDropTask {
    TransferDescriptor                  primary_desc;
    EvictionTimingSnapshot              primary_timing;
    std::vector<TransferDescriptor>     cascade_descs;
    std::vector<EvictionTimingSnapshot> cascade_timings;
    // FULL prune closure only. Every dependent descriptor targets NONE,
    // and nodes stay valid because task activation is synchronous.
    std::vector<TransferDescriptor>     dependent_prune_descs;
    std::vector<EvictionTimingSnapshot> dependent_prune_timings;
    std::vector<TreeNode*>              full_prune_nodes_bottom_up;

    bool hasFullPrune() const {
        return !full_prune_nodes_bottom_up.empty();
    }
};

}  // namespace rtp_llm
