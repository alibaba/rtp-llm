#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// SWAComponentGroup: manages sliding-window attention KV cache.
// Uses Any-node heaps: any node with data can be evicted.
class SWAComponentGroup: public ComponentGroup {
public:
    explicit SWAComponentGroup(size_t sliding_window_size = 0, size_t seq_size_per_block = 1);

    std::unique_ptr<MatchValidator> createMatchValidator() override;

    // SWA window lock: only lock nodes within sliding_window_size from path tail.
    size_t computeReuseBlockCount(size_t matched_block_count, const std::vector<TreeNode*>& path) const override;

    size_t slidingWindowSize() const {
        return sliding_window_size_;
    }
    size_t seqSizePerBlock() const {
        return seq_size_per_block_;
    }

private:
    size_t sliding_window_size_;
    size_t seq_size_per_block_;
};

// SWA window match validator.
class SWAMatchValidator: public MatchValidator {
public:
    explicit SWAMatchValidator(size_t sliding_window_size, size_t seq_size_per_block);

    bool validate(const TreeNode* node, const GroupSlot& slot) override;

    bool connectedToRoot() const {
        return connected_to_root_;
    }
    size_t accumulatedLength() const {
        return accumulated_length_;
    }

private:
    size_t sliding_window_size_;
    size_t seq_size_per_block_;
    bool   connected_to_root_{true};
    size_t accumulated_length_{0};
};

}  // namespace rtp_llm
