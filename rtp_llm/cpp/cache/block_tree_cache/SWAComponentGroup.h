#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// SWAComponentGroup: manages sliding-window attention KV cache.
// Uses Any-node heaps: any node with data can be evicted.
class SWAComponentGroup: public ComponentGroup {
public:
    explicit SWAComponentGroup(int            sliding_window_size = 0,
                               int            seq_size_per_block  = 1,
                               EvictionPolicy device_policy       = EvictionPolicy::LRU,
                               EvictionPolicy host_policy         = EvictionPolicy::LRU,
                               EvictionPolicy disk_policy         = EvictionPolicy::FIFO);

    std::unique_ptr<MatchValidator> createMatchValidator() override;

    // SWA: any node with data can enter heap.
    void tryAddToDeviceHeap(TreeNode* node) override;

    // SWA window lock: only lock nodes within sliding_window_size from path tail.
    size_t computeReferenceCount(size_t matched_blocks, const std::vector<TreeNode*>& path) const override;

    int slidingWindowSize() const {
        return sliding_window_size_;
    }
    int seqSizePerBlock() const {
        return seq_size_per_block_;
    }

private:
    int sliding_window_size_;
    int seq_size_per_block_;
};

// SWA window match validator.
class SWAMatchValidator: public MatchValidator {
public:
    explicit SWAMatchValidator(int sliding_window_size, int seq_size_per_block);

    bool validate(const TreeNode* node, const GroupSlot& slot) override;

    bool connectedToRoot() const {
        return connected_to_root_;
    }
    size_t accumulatedLength() const {
        return accumulated_length_;
    }

private:
    int    sliding_window_size_;
    int    seq_size_per_block_;
    bool   connected_to_root_{true};
    size_t accumulated_length_{0};
};

}  // namespace rtp_llm
