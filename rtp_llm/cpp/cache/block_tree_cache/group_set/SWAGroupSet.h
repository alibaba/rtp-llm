#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

// SWAGroupSet: manages sliding-window attention KV cache.
// Uses Any-node heaps: any node with data can be evicted.
class SWAGroupSet: public GroupSet {
public:
    SWAGroupSet(size_t                         sliding_window_size,
                size_t                         seq_size_per_block,
                std::vector<DeviceBlockPoolPtr> device_pools,
                std::shared_ptr<HostBlockPool> host_pool,
                BlockTreeDiskBlockPoolPtr      disk_pool);

    std::unique_ptr<MatchValidator> createMatchValidator() override;

    // SWA window lock: only lock nodes within sliding_window_size from path tail.
    size_t computeReuseBlockCount(size_t matched_block_count) const override;

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

    bool validate(const GroupSetResource& resource) override;

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
