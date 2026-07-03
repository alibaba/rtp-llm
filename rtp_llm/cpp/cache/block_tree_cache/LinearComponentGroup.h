#pragma once

#include "rtp_llm/cpp/cache/block_tree_cache/ComponentGroup.h"

namespace rtp_llm {

// LinearComponentGroup: manages linear attention / SSM hidden states.
// Point-state: only specific nodes hold state snapshots.
// Uses Any-node heaps like SWA.
class LinearComponentGroup: public ComponentGroup {
public:
    explicit LinearComponentGroup(EvictionPolicy device_policy = EvictionPolicy::LRU,
                                  EvictionPolicy host_policy   = EvictionPolicy::LRU,
                                  EvictionPolicy disk_policy   = EvictionPolicy::FIFO);

    std::unique_ptr<MatchValidator> createMatchValidator() override;

    // LINEAR: any node with data can enter heap.
    void tryAddToDeviceHeap(TreeNode* node) override;
};

class LinearMatchValidator: public MatchValidator {
public:
    bool validate(const TreeNode* node, const GroupSlot& slot) override;
};

}  // namespace rtp_llm
