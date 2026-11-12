#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/EvictionMetrics.h"

namespace rtp_llm {

struct EvictionTransferTask {
    explicit EvictionTransferTask(TransferTask task): transfer_task(std::move(task)) {}
    EvictionTransferTask(TransferTask task, std::vector<EvictionTimingMetrics> timings):
        transfer_task(std::move(task)), timings(std::move(timings)) {}

    const std::vector<TransferDescriptor>& descriptors() const {
        return transfer_task.descriptors();
    }

    std::vector<TransferDescriptor>& mutableDescriptorsForPreparation() {
        return transfer_task.mutableDescriptorsForPreparation();
    }

    TransferTask                        transfer_task;
    std::vector<EvictionTimingMetrics>  timings;
};

struct EvictionDropTask {
    TransferDescriptor                  primary_desc;
    EvictionTimingMetrics               primary_timing;
    std::vector<TransferDescriptor>     cascade_descs;
    std::vector<EvictionTimingMetrics>  cascade_timings;
    // FULL prune closure only. Every dependent descriptor targets NONE,
    // and nodes stay valid because task activation is synchronous.
    std::vector<TransferDescriptor>     dependent_prune_descs;
    std::vector<EvictionTimingMetrics>  dependent_prune_timings;
    std::vector<TreeNode*>              full_prune_nodes_bottom_up;

    bool hasFullPrune() const {
        return !full_prune_nodes_bottom_up.empty();
    }
};

}  // namespace rtp_llm
